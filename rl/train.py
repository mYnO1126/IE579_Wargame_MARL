# train.py
# IPPO (parameter-shared PPO) — milestone 3 v1.
# 한 팀(--team)은 공유 정책망으로 학습, 상대 팀은 스크립트(최근접 적 교전 + goal/적 방향 이동).
# 중앙 critic(MAPPO/CTDE)은 다음 업그레이드. (DESIGN §7)
#
# 실행:  conda run -n wargame python -m rl.train --iters 50 --team red

import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn

from rl.env import WargameParallelEnv, _MOVES
from rl.policy import ActorCritic
from rl import obs as obsmod


# ----- 스크립트 상대 정책 -----
def _dir_to_move(dx, dy):
    """(dx,dy) 방향에 가장 가까운 이동 인덱스(1~8). 0이면 정지(0)."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0
    best, best_dot = 0, -1e9
    for idx in range(1, 9):
        mx, my = _MOVES[idx]
        norm = (mx * mx + my * my) ** 0.5
        dot = (dx * mx + dy * my) / norm
        if dot > best_dot:
            best_dot, best = dot, idx
    return best


def scripted_action(env, aid):
    tr = env.troops_by_id[aid]
    order = env._enemy_order.get(aid, [])
    if tr.fixed_dest is not None:                      # goal 있으면 goal로
        dx, dy = tr.fixed_dest.x - tr.coord.x, tr.fixed_dest.y - tr.coord.y
    elif order:                                        # 없으면 최근접 적으로
        dx, dy = order[0].coord.x - tr.coord.x, order[0].coord.y - tr.coord.y
    else:
        dx = dy = 0.0
    return np.array([_dir_to_move(dx, dy), 0, 1], dtype=np.int64)  # target=0(최근접), engage=1


# ----- GAE -----
def gae(rew, val, last_val, gamma=0.99, lam=0.95):
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        nextv = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nextv - val[t]
        g = delta + gamma * lam * g
        adv[t] = g
    return adv, adv + np.asarray(val, dtype=np.float32)


def collect(env, policy, team, roll_steps, device):
    """roll_steps(학습팀 agent-step) 이상 모일 때까지 에피소드 수집. 반환: 배치 + 통계."""
    B = {"obs": [], "act": [], "logp": [], "adv": [], "ret": []}
    ep_returns, wins, games = [], 0, 0
    steps = 0
    while steps < roll_steps:
        obs, _ = env.reset()
        cur = obs
        traj = {}   # aid -> dict(lists) + term flag
        last_obs = {}
        while env.agents:
            learn = [a for a in env.agents if env.troops_by_id[a].team == team]
            opp = [a for a in env.agents if env.troops_by_id[a].team != team]
            actions = {}
            if learn:
                ob = torch.as_tensor(np.stack([cur[a] for a in learn]), dtype=torch.float32, device=device)
                acts, logp, val = policy.act(ob)
                acts_np = acts.cpu().numpy()
                for i, a in enumerate(learn):
                    actions[a] = acts_np[i]
                    d = traj.setdefault(a, {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "term": False})
                    d["obs"].append(cur[a]); d["act"].append(acts_np[i])
                    d["logp"].append(float(logp[i])); d["val"].append(float(val[i]))
            for a in opp:
                actions[a] = scripted_action(env, a)
            nobs, rews, terms, truncs, _ = env.step(actions)
            for a in learn:
                traj[a]["rew"].append(float(rews.get(a, 0.0)))
                traj[a]["term"] = bool(terms.get(a, False))
                if a in nobs:
                    last_obs[a] = nobs[a]
            cur = nobs

        # 승패 판정
        blue = any(t.alive for t in env.troop_list.blue_troops)
        red = any(t.alive for t in env.troop_list.red_troops)
        won = (team == "blue" and not red) or (team == "red" and not blue)
        games += 1; wins += int(won)

        # 트래젝토리 마감 + GAE
        for a, d in traj.items():
            if d["term"] or a not in last_obs:
                last_val = 0.0                      # 사망/종료 → bootstrap 0
            else:
                with torch.no_grad():               # 시간초과 생존 → V(마지막 obs)
                    lo = torch.as_tensor(last_obs[a], dtype=torch.float32, device=device)
                    last_val = float(policy.critic(policy.trunk(lo)).squeeze(-1))
            adv, ret = gae(d["rew"], d["val"], last_val)
            B["obs"].extend(d["obs"]); B["act"].extend(d["act"])
            B["logp"].extend(d["logp"]); B["adv"].extend(adv); B["ret"].extend(ret)
            ep_returns.append(float(np.sum(d["rew"])))
            steps += len(d["rew"])
    stats = {"ep_return": float(np.mean(ep_returns)), "win_rate": wins / max(1, games),
             "games": games, "agent_steps": steps}
    return B, stats


def ppo_update(policy, opt, B, device, epochs=4, mb=256, clip=0.2, ent_c=0.01, vf_c=0.5):
    obs = torch.as_tensor(np.array(B["obs"]), dtype=torch.float32, device=device)
    act = torch.as_tensor(np.array(B["act"]), dtype=torch.int64, device=device)
    logp_old = torch.as_tensor(np.array(B["logp"]), dtype=torch.float32, device=device)
    adv = torch.as_tensor(np.array(B["adv"]), dtype=torch.float32, device=device)
    ret = torch.as_tensor(np.array(B["ret"]), dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    N = obs.shape[0]
    pl = vl = 0.0
    for _ in range(epochs):
        idx = torch.randperm(N, device=device)
        for s in range(0, N, mb):
            b = idx[s:s + mb]
            logp, entropy, val = policy.evaluate(obs[b], act[b])
            ratio = torch.exp(logp - logp_old[b])
            s1 = ratio * adv[b]
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]
            ploss = -torch.min(s1, s2).mean()
            vloss = ((val - ret[b]) ** 2).mean()
            loss = ploss + vf_c * vloss - ent_c * entropy.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            pl += ploss.item(); vl += vloss.item()
    return pl, vl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--team", choices=["red", "blue"], default="red")
    ap.add_argument("--roll", type=int, default=2048, help="iteration당 학습팀 agent-step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=os.path.join(_ROOT, "rl", "policies", "ippo.pt"))
    args = ap.parse_args()

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    env = WargameParallelEnv(seed=args.seed)
    policy = ActorCritic(obsmod.OBS_DIM, env.action_space("x").nvec).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    print(f"device={device}  team={args.team}  obs_dim={obsmod.OBS_DIM}  params={sum(p.numel() for p in policy.parameters())}")

    for it in range(1, args.iters + 1):
        B, st = collect(env, policy, args.team, args.roll, device)
        pl, vl = ppo_update(policy, opt, B, device)
        print(f"iter {it:3d} | steps {st['agent_steps']:5d} games {st['games']:3d} | "
              f"ep_return {st['ep_return']:+6.2f} | win {st['win_rate']:.2f} | "
              f"ploss {pl:+.3f} vloss {vl:.3f}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({"model": policy.state_dict(), "obs_dim": obsmod.OBS_DIM,
                "act_nvec": policy.act_nvec, "team": args.team}, args.save)
    print(f"saved policy -> {args.save}")


if __name__ == "__main__":
    main()
