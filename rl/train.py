# train.py
# IPPO (parameter-shared PPO) — milestone 3.
# 한 팀(--team)은 공유 정책망으로 학습, 상대 팀은 스크립트.
# --workers N > 1 이면 멀티프로세스 병렬 롤아웃 수집(코어 수만큼 빠름).
# 중앙 critic(MAPPO/CTDE)은 다음 업그레이드. (DESIGN §7)
#
# 실행:
#   conda run -n wargame python -m rl.train --iters 50 --team red
#   conda run -n wargame python -m rl.train --iters 50 --team red --workers 8

import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn

from rl.env import WargameParallelEnv
from rl.policy import ActorCritic
from rl import obs as obsmod
from rl.rollout import collect


def ppo_update(policy, opt, B, device, epochs=4, mb=256, clip=0.2, ent_c=0.01, vf_c=0.5):
    obs = torch.as_tensor(np.array(B["obs"]), dtype=torch.float32, device=device)
    glob = torch.as_tensor(np.array(B["glob"]), dtype=torch.float32, device=device)
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
            logp, entropy, val = policy.evaluate(obs[b], glob[b], act[b])
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
    ap.add_argument("--workers", type=int, default=1, help="병렬 롤아웃 worker 프로세스 수 (1=단일 env)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="기본: rl/policies/ippo_<team>.pt (팀별 분리)")
    args = ap.parse_args()

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    env = WargameParallelEnv(seed=args.seed)   # action_space nvec용(+단일 워커면 롤아웃에도 사용)
    nvec = list(env.action_space("x").nvec)
    policy = ActorCritic(obsmod.OBS_DIM, nvec, obsmod.GLOBAL_DIM).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    collector = None
    if args.workers > 1:
        from rl.parallel import ParallelCollector
        collector = ParallelCollector(args.workers, args.team, obsmod.OBS_DIM,
                                      nvec, obsmod.GLOBAL_DIM, seed=args.seed)

    print(f"device={device}  team={args.team}  workers={args.workers}  obs_dim={obsmod.OBS_DIM}  "
          f"params={sum(p.numel() for p in policy.parameters())}")

    import time
    for it in range(1, args.iters + 1):
        t0 = time.perf_counter()
        if collector is not None:
            B, st = collector.collect(policy, args.roll)
        else:
            B, st = collect(env, policy, args.team, args.roll, device)
        pl, vl = ppo_update(policy, opt, B, device)
        dt = time.perf_counter() - t0
        print(f"iter {it:3d} | steps {st['agent_steps']:5d} games {st['games']:3d} | "
              f"ep_return {st['ep_return']:+6.2f} | win {st['win_rate']:.2f} | "
              f"ploss {pl:+.3f} vloss {vl:7.3f} | {st['agent_steps']/dt:6.0f} steps/s")

    save = args.save or os.path.join(_ROOT, "rl", "policies", f"ippo_{args.team}.pt")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    torch.save({"model": policy.state_dict(), "obs_dim": obsmod.OBS_DIM,
                "act_nvec": policy.act_nvec, "global_dim": obsmod.GLOBAL_DIM,
                "team": args.team}, save)
    print(f"saved policy -> {save}")
    if collector is not None:
        collector.close()


if __name__ == "__main__":
    main()
