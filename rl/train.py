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


# in-process 커리큘럼: 한 번의 학습에서 맵 크기·유닛 수를 단계적으로 키워 배포(풀 시뮬,
# blue 65 : red 267 ≈ 1:4)에 가까운 규모로 마무리한다. 정책/옵티마이저는 유지하고
# 스테이지 경계마다 env(+병렬 collector)만 재생성. frac = 전체 iters 중 그 스테이지 비중.
# blue/red = DEFAULT_COMP 배수(편성 4종 × 배수). 마지막 스테이지가 목표 규모/비율.
CURRICULUM = [
    {"map": 200, "blue": 2,  "red": 4,  "frac": 0.15},   # 8 : 16   (1:2)  기본 교전·생존
    {"map": 320, "blue": 4,  "red": 12, "frac": 0.20},   # 16 : 48  (1:3)
    {"map": 480, "blue": 8,  "red": 32, "frac": 0.25},   # 32 : 128 (1:4)
    {"map": 640, "blue": 16, "red": 64, "frac": 0.40},   # 64 : 256 (1:4)  목표 = 풀 시뮬(65:267)과 거의 동일 규모
]


def _make_comp(bu, ru):
    """팀별 배수로 편성 dict 생성. 둘 다 1이면 기본 편성(None)."""
    from rl.spawn import DEFAULT_COMP
    if bu <= 1 and ru <= 1:
        return None
    return {"blue": DEFAULT_COMP["blue"] * max(1, bu), "red": DEFAULT_COMP["red"] * max(1, ru)}


def _build_collection(args, map_size, comp, nvec):
    """현재 스테이지의 (env, collector) 생성. workers>1 이면 collector, 아니면 단일 env."""
    if args.workers > 1:
        from rl.parallel import ParallelCollector
        col = ParallelCollector(args.workers, args.team, obsmod.OBS_DIM, nvec,
                                obsmod.GLOBAL_DIM, seed=args.seed, map_size=map_size, comp=comp)
        return None, col
    return WargameParallelEnv(map_size=map_size, comp=comp, seed=args.seed), None


def _stage_ends(stages, iters):
    """각 스테이지가 끝나는 iter 번호(누적). 마지막은 항상 iters."""
    ends, acc = [], 0.0
    for s in stages:
        acc += s["frac"]
        ends.append(max(1, round(acc / sum(x["frac"] for x in stages) * iters)))
    ends[-1] = iters
    return ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--team", choices=["red", "blue"], default="red")
    ap.add_argument("--roll", type=int, default=2048, help="iteration당 학습팀 agent-step")
    ap.add_argument("--workers", type=int, default=1, help="병렬 롤아웃 worker 프로세스 수 (1=단일 env)")
    ap.add_argument("--curriculum", action="store_true",
                    help="in-process 커리큘럼: 한 번의 학습에서 맵·유닛을 단계적으로 키워 목표 규모(1:4)로 마무리 "
                         "(CURRICULUM 스케줄 사용; --map_size/--blue_units/--red_units 무시)")
    ap.add_argument("--map_size", type=int, default=160, help="학습 크롭 맵 크기(px). 클수록 풀맵에 가까움")
    ap.add_argument("--units", type=int, default=1, help="유닛 수 배수(커리큘럼: 기본 편성×배수, 양 팀 공통)")
    ap.add_argument("--blue_units", type=int, default=None, help="blue 전용 배수(미지정 시 --units)")
    ap.add_argument("--red_units", type=int, default=None, help="red 전용 배수(미지정 시 --units). "
                    "예: --blue_units 4 --red_units 16 → blue:red ≈ 1:4 수적 열세 방어 학습")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="기본: rl/policies/ippo_<team>.pt (팀별 분리)")
    args = ap.parse_args()

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # 스테이지 구성: --curriculum 이면 점진 스케줄, 아니면 단일 스테이지(기존 동작).
    if args.curriculum:
        stages = CURRICULUM
    else:
        bu = args.blue_units if args.blue_units is not None else args.units
        ru = args.red_units if args.red_units is not None else args.units
        stages = [{"map": args.map_size, "blue": bu, "red": ru, "frac": 1.0}]
    ends = _stage_ends(stages, args.iters)

    # nvec 은 스케일과 무관 → 임시 env 하나로 얻고 정책/옵티마이저 생성(스테이지 간 유지).
    probe = WargameParallelEnv(map_size=stages[0]["map"],
                               comp=_make_comp(stages[0]["blue"], stages[0]["red"]), seed=args.seed)
    nvec = list(probe.action_space("x").nvec)
    del probe
    policy = ActorCritic(obsmod.OBS_DIM, nvec, obsmod.GLOBAL_DIM).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print(f"device={device}  team={args.team}  workers={args.workers}  obs_dim={obsmod.OBS_DIM}  "
          f"params={sum(p.numel() for p in policy.parameters())}  "
          f"{'curriculum '+str(len(stages))+'-stage' if args.curriculum else 'single-stage'}")

    import time
    env, collector, cur_stage = None, None, -1
    for it in range(1, args.iters + 1):
        si = next(i for i, e in enumerate(ends) if it <= e)        # 현재 스테이지 인덱스
        if si != cur_stage:                                        # 스테이지 진입 → env 재생성
            if collector is not None:
                collector.close()
            s = stages[si]
            comp = _make_comp(s["blue"], s["red"])
            env, collector = _build_collection(args, s["map"], comp, nvec)
            cur_stage = si
            nb = len(comp["blue"]) if comp else 4       # DEFAULT_COMP 팀당 4종 → 배수×4
            nr = len(comp["red"]) if comp else 4
            print(f"--- stage {si+1}/{len(stages)} @iter {it}: map {s['map']}px  "
                  f"blue {nb} : red {nr}  (~1:{nr/max(1,nb):.0f}) ---")

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
