# visualize_episode.py
# 1 에피소드를 렌더 ON 으로 돌려 프레임을 저장한다. (학습과 분리된 일회성 실행)
# 학습 루프는 env.render() 를 호출하지 않으므로 학습 속도에는 영향이 없다.
#
# 실행:
#   conda run -n wargame python -m rl.visualize_episode                       # 기본: scripted 정책
#   conda run -n wargame python -m rl.visualize_episode --policy random --mode tactical
#   # 학습 정책으로 렌더(eval 과 동일한 8v32 구성). 상대만 규칙이면 한쪽 ckpt만:
#   conda run -n wargame python -m rl.visualize_episode --blue rl/policies/ippo_blue.pt \
#       --red rl/policies/ippo_red.pt --mode board --seed 3

import os
import sys
import glob
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from rl.env import WargameParallelEnv


def random_policy(env, rng):
    return {a: env.action_space(a).sample(rng) for a in env.agents}


def scripted_policy(env, rng):
    """데모용: BLUE(방어)는 정지, RED(공격)만 BLUE 진영 쪽(-x)으로 접근 + 최근접 적(tgt=0) + 항상 교전.
    (blue 좌측 / red 우측 스폰 가정)"""
    actions = {}
    for a in env.agents:
        tr = env.troops_by_id[a]
        move = 0 if tr.team == "blue" else 7   # blue=정지(0), red=좌측 접근(7=(-1,0))
        actions[a] = np.array([move, 0, 1], dtype=np.int64)
    return actions


def learned_actions(env, obs, policies):
    """정책 있는 팀=학습 정책(deterministic), 없는 팀=규칙 기반(scripted_action). eval 과 동일."""
    import torch
    from rl.rollout import scripted_action
    actions = {}
    for tm in ("blue", "red"):
        members = [a for a in env.agents if env.troops_by_id[a].team == tm]
        if not members:
            continue
        if policies.get(tm) is not None:
            ob = torch.as_tensor(np.stack([obs[a] for a in members]), dtype=torch.float32)
            acts, _ = policies[tm].act(ob, deterministic=True)
            acts = acts.cpu().numpy()
            for i, a in enumerate(members):
                actions[a] = acts[i]
        else:
            for a in members:
                actions[a] = scripted_action(env, a)
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["scripted", "random"], default="scripted",
                    help="--blue/--red 미지정 시 사용할 데모 정책")
    ap.add_argument("--blue", default=None, help="blue 정책 ckpt (지정 시 학습 정책으로 렌더)")
    ap.add_argument("--red", default=None, help="red 정책 ckpt (없으면 그 팀은 규칙기반)")
    ap.add_argument("--mode", choices=["board", "tactical"], default="board")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map_size", type=int, default=160)
    ap.add_argument("--blue_units", type=int, default=2, help="학습 렌더 시 blue 편성 배수(기본 2=8기)")
    ap.add_argument("--red_units", type=int, default=4, help="학습 렌더 시 red 편성 배수(기본 4=16기)")
    ap.add_argument("--every", type=int, default=1,
                    help="N 시뮬분마다 1프레임 (1=매분, 5=결정마다)")
    ap.add_argument("--legend", action="store_true",
                    help="그림 안에 범례 표시 (기본: 표시 안 함)")
    args = ap.parse_args()

    learned = bool(args.blue or args.red)
    if learned:
        from rl.evaluate import _load
        from rl.spawn import DEFAULT_COMP
        policies = {"blue": _load(args.blue), "red": _load(args.red)}
        comp = {"blue": DEFAULT_COMP["blue"] * args.blue_units,
                "red": DEFAULT_COMP["red"] * args.red_units}     # 데모용 편성(기본 8:16)
        map_size = 240 if args.map_size == 160 else args.map_size
        tag = "learned"
    else:
        policies, comp = None, None
        map_size = args.map_size
        tag = args.policy

    save_dir = os.path.join(_ROOT, "rl", "viz", f"ep_{tag}_seed{args.seed}")
    if os.path.isdir(save_dir):        # 이전 프레임 정리
        for f in glob.glob(os.path.join(save_dir, "*.png")):
            os.remove(f)

    np.random.seed(args.seed)   # 전투 RNG(Ph/Pk, 전역 np.random) 고정 → 렌더 재현성 (evaluate 와 동일)
    env = WargameParallelEnv(map_size=map_size, decision_interval=5,
                             max_decisions=60, comp=comp, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=args.seed)
    demo = scripted_policy if args.policy == "scripted" else random_policy

    # step() 내부에서 매 args.every 시뮬분마다 프레임 자동 저장(1분 단위면 전투가 부드럽게 보임)
    env.set_recording(save_dir, mode=args.mode, every=args.every, show_legend=args.legend)
    env.render(save_dir, mode=args.mode, show_legend=args.legend)   # 초기 프레임 (t=0)
    step = 0
    while env.agents:
        actions = learned_actions(env, obs, policies) if learned else demo(env, rng)
        obs, *_ = env.step(actions)
        step += 1

    n = len(glob.glob(os.path.join(save_dir, "*.png")))
    b = sum(t.alive for t in env.troop_list.blue_troops)
    r = sum(t.alive for t in env.troop_list.red_troops)
    print(f"decisions={step}  survivors blue={b} red={r}")
    print(f"saved {n} frames ({args.mode}) -> {save_dir}")


if __name__ == "__main__":
    main()
