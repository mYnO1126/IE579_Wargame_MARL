# rollout.py
# 롤아웃 수집 로직(단일/병렬 공용). 한 팀=정책, 상대=스크립트. 에피소드 수집 + 에이전트별 GAE.

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from rl.env import _MOVES


# ----- 스크립트 상대 정책 -----
def dir_to_move(dx, dy):
    """(dx,dy)에 가장 가까운 이동 인덱스(1~8). 0이면 정지."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0
    best, best_dot = 0, -1e9
    for idx in range(1, 9):
        mx, my = _MOVES[idx]
        dot = (dx * mx + dy * my) / ((mx * mx + my * my) ** 0.5)
        if dot > best_dot:
            best_dot, best = dot, idx
    return best


def scripted_action(env, aid):
    tr = env.troops_by_id[aid]
    order = env._enemy_order.get(aid, [])
    if tr.fixed_dest is not None:
        dx, dy = tr.fixed_dest.x - tr.coord.x, tr.fixed_dest.y - tr.coord.y
    elif order:
        dx, dy = order[0].coord.x - tr.coord.x, order[0].coord.y - tr.coord.y
    else:
        dx = dy = 0.0
    return np.array([dir_to_move(dx, dy), 0, 1], dtype=np.int64)


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


def collect(env, policy, team, roll_steps, device="cpu"):
    """roll_steps(학습팀 agent-step) 이상 모일 때까지 에피소드 수집. 반환: (batch, stats)."""
    B = {"obs": [], "act": [], "logp": [], "adv": [], "ret": []}
    ep_returns, wins, games, steps = [], 0, 0, 0
    while steps < roll_steps:
        obs, _ = env.reset()
        cur = obs
        traj, last_obs = {}, {}
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

        blue = any(t.alive for t in env.troop_list.blue_troops)
        red = any(t.alive for t in env.troop_list.red_troops)
        won = (team == "blue" and not red) or (team == "red" and not blue)
        games += 1; wins += int(won)

        for a, d in traj.items():
            if d["term"] or a not in last_obs:
                last_val = 0.0
            else:
                with torch.no_grad():
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
