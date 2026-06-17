# evaluate.py
# 베이스라인 비교 하니스 (DESIGN §10). 한 팀에 대해 [학습 정책] vs [rule-based(scripted)]를
# 같은 상대(반대편 scripted)·같은 seed들로 N 에피소드 돌려 승률/생존/사상자 개선폭을 비교.
#
# 성공 기준은 절대 승률이 아니라 "rule-based 대비 개선"(특히 BLUE 수비 개선이 중요).
#
# 실행:  conda run -n wargame python -m rl.evaluate --ckpt rl/policies/ippo_red.pt --episodes 200

import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from rl.env import WargameParallelEnv
from rl.policy import ActorCritic
from rl.rollout import scripted_action


def play_episode(env, team, policy, seed, device="cpu"):
    # 같은 seed → 같은 크롭/스폰/goal. combat 난수도 시드해 실행 재현성 확보(모드 간은 행동이 달라 분기).
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    cur = obs
    while env.agents:
        learn = [a for a in env.agents if env.troops_by_id[a].team == team]
        opp = [a for a in env.agents if env.troops_by_id[a].team != team]
        actions = {}
        if policy is not None and learn:
            ob = torch.as_tensor(np.stack([cur[a] for a in learn]), dtype=torch.float32, device=device)
            acts, _ = policy.act(ob, deterministic=True)
            acts = acts.cpu().numpy()
            for i, a in enumerate(learn):
                actions[a] = acts[i]
        else:
            for a in learn:
                actions[a] = scripted_action(env, a)
        for a in opp:
            actions[a] = scripted_action(env, a)
        nobs, _, _, _, _ = env.step(actions)
        cur = nobs

    opp_team = "blue" if team == "red" else "red"
    own = env.troop_list.blue_troops if team == "blue" else env.troop_list.red_troops
    enem = env.troop_list.blue_troops if opp_team == "blue" else env.troop_list.red_troops
    own_alive = sum(t.alive for t in own)
    en_alive = sum(t.alive for t in enem)
    return {
        "own_frac": own_alive / max(1, env._init_count[team]),
        "en_frac": en_alive / max(1, env._init_count[opp_team]),
        "win": int(en_alive == 0 and own_alive > 0),
        "loss": int(own_alive == 0),
    }


def run(env, team, policy, episodes, seed0, device="cpu"):
    rows = [play_episode(env, team, policy, seed0 + s, device) for s in range(episodes)]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=10000)   # 학습 seed와 겹치지 않게
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    team = ckpt["team"]
    policy = ActorCritic(ckpt["obs_dim"], ckpt["act_nvec"], ckpt["global_dim"])
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    env = WargameParallelEnv(seed=0)
    print(f"평가: team={team}  episodes={args.episodes}  ckpt={os.path.basename(args.ckpt)}")
    rule = run(env, team, None, args.episodes, args.seed0)      # rule-based(scripted)
    marl = run(env, team, policy, args.episodes, args.seed0)    # 학습 정책

    print(f"\n{'metric':<22}{'rule-based':>12}{'MARL':>12}{'Δ(MARL-rule)':>16}")
    print("-" * 62)
    def line(name, r, m, better_up=True):
        d = m - r
        mark = "↑" if (d > 0) == better_up else "↓"
        print(f"{name:<22}{r:>12.3f}{m:>12.3f}{d:>+14.3f} {mark}")
    line("win rate", rule["win"], marl["win"])
    line("loss rate", rule["loss"], marl["loss"], better_up=False)
    line("아군 생존율", rule["own_frac"], marl["own_frac"])
    line("적 생존율", rule["en_frac"], marl["en_frac"], better_up=False)
    print(f"\n해석: {team} 정책이 rule-based 대비 win↑/loss↓/아군생존↑/적생존↓ 이면 개선.")
    print("주의: combat 난수는 모드 간 행동 차이로 분기 → seed별 완전 페어링은 아님(다중 seed 평균으로 비교).")


if __name__ == "__main__":
    main()
