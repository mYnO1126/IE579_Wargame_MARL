# visualize_episode.py
# 1 에피소드를 렌더 ON 으로 돌려 프레임을 저장한다. (학습과 분리된 일회성 실행)
# 학습 루프는 env.render() 를 호출하지 않으므로 학습 속도에는 영향이 없다.
#
# 실행:
#   conda run -n wargame python -m rl.visualize_episode                 # 기본: scripted 정책
#   conda run -n wargame python -m rl.visualize_episode --policy random --seed 1 --mode tactical

import os
import sys
import glob
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from rl.env import WargameParallelEnv, _MOVES


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    ap.add_argument("--mode", choices=["board", "tactical"], default="board")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map_size", type=int, default=120)
    ap.add_argument("--every", type=int, default=1,
                    help="N 시뮬분마다 1프레임 (1=매분, 5=결정마다)")
    args = ap.parse_args()

    save_dir = os.path.join(_ROOT, "rl", "viz", f"ep_{args.policy}_seed{args.seed}")
    # 이전 프레임 정리
    if os.path.isdir(save_dir):
        for f in glob.glob(os.path.join(save_dir, "*.png")):
            os.remove(f)

    env = WargameParallelEnv(map_size=args.map_size, decision_interval=5,
                             max_decisions=60, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)
    policy = scripted_policy if args.policy == "scripted" else random_policy

    # step() 내부에서 매 args.every 시뮬분마다 프레임 자동 저장(1분 단위면 전투가 부드럽게 보임)
    env.set_recording(save_dir, mode=args.mode, every=args.every)
    env.render(save_dir, mode=args.mode)   # 초기 프레임 (t=0)
    step = 0
    while env.agents:
        env.step(policy(env, rng))
        step += 1

    n = len(glob.glob(os.path.join(save_dir, "*.png")))
    b = sum(t.alive for t in env.troop_list.blue_troops)
    r = sum(t.alive for t in env.troop_list.red_troops)
    print(f"decisions={step}  survivors blue={b} red={r}")
    print(f"saved {n} frames ({args.mode}) -> {save_dir}")


if __name__ == "__main__":
    main()
