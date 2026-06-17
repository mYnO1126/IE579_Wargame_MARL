# random_rollout.py
# 1단계 수용 기준: 랜덤 정책으로 1 에피소드를 끝까지 돌려본다 (환경 골격 동작 확인).
# 실행:  conda run -n wargame python -m rl.random_rollout

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from rl.env import WargameParallelEnv


def main(seed=0):
    rng = np.random.default_rng(seed)
    env = WargameParallelEnv(map_size=120, decision_interval=5, max_decisions=60, seed=seed)
    obs, infos = env.reset(seed=seed)

    print(f"[reset] agents={len(env.agents)}  obs_dim={env.observation_space(env.agents[0]).shape}  "
          f"act_nvec={env.action_space(env.agents[0]).nvec}")
    print(f"        blue={env._init_count['blue']}  red={env._init_count['red']}  map={env.map_size}px")

    ep_return = {a: 0.0 for a in env.possible_agents}
    step = 0
    while env.agents:
        actions = {a: env.action_space(a).sample(rng) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for a, r in rewards.items():
            ep_return[a] = ep_return.get(a, 0.0) + r
        step += 1
        if step % 10 == 0 or not env.agents:
            blue = sum(1 for t in env.troop_list.blue_troops if t.alive)
            red = sum(1 for t in env.troop_list.red_troops if t.alive)
            print(f"  decision {step:3d} | t={env.t:6.0f}min | alive blue={blue} red={red} | active agents={len(env.agents)}")

    blue = sum(1 for t in env.troop_list.blue_troops if t.alive)
    red = sum(1 for t in env.troop_list.red_troops if t.alive)
    total = sum(ep_return.values())
    print("\n=== EPISODE DONE ===")
    print(f"decisions={step}  sim_time={env.t:.0f}min  survivors blue={blue} red={red}")
    print(f"return: sum={total:+.2f}  "
          f"blue_sum={sum(ep_return[a] for a in ep_return if a.startswith('B')):+.2f}  "
          f"red_sum={sum(ep_return[a] for a in ep_return if a.startswith('R')):+.2f}")
    print("OK: 환경 골격이 1 에피소드를 정상 종료함." )


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
