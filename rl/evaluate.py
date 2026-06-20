# evaluate.py
# 베이스라인 비교 하니스 (크롭, 통계용; DESIGN §10). --blue/--red 로 한쪽 또는 양쪽에 정책을 넣고,
# all-rule(both scripted) 베이스라인 대비 각 팀의 전투효율을 N 에피소드로 비교한다.
#  - 한쪽만 정책: 그 팀 정책 vs 규칙(상대 고정).
#  - 양쪽 정책: 두 팀 다 정책 vs all-rule.
# ★ 성공 기준 = 절대 승률이 아니라 LER(교환비=적손실/아군손실)·생존·손실의 rule 대비 개선.
#
# 실행:
#   conda run -n wargame python -m rl.evaluate --blue rl/policies/ippo_blue.pt --episodes 200
#   conda run -n wargame python -m rl.evaluate --blue rl/policies/ippo_blue.pt --red rl/policies/ippo_red.pt

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


def _load(path):
    if not path:
        return None
    ck = torch.load(path, map_location="cpu")
    p = ActorCritic(ck["obs_dim"], ck["act_nvec"], ck["global_dim"])
    p.load_state_dict(ck["model"]); p.eval()
    return p


def play_episode(env, policies, seed, device="cpu"):
    """policies={"blue":p|None,"red":p|None}. 정책 있는 팀=정책, 없으면 scripted. 동일 seed로 재현성."""
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    cur = obs
    while env.agents:
        actions = {}
        for tm in ("blue", "red"):
            members = [a for a in env.agents if env.troops_by_id[a].team == tm]
            if not members:
                continue
            if policies.get(tm) is not None:
                ob = torch.as_tensor(np.stack([cur[a] for a in members]), dtype=torch.float32, device=device)
                acts, _ = policies[tm].act(ob, deterministic=True)
                acts = acts.cpu().numpy()
                for i, a in enumerate(members):
                    actions[a] = acts[i]
            else:
                for a in members:
                    actions[a] = scripted_action(env, a)
        nobs, _, _, _, _ = env.step(actions)
        cur = nobs

    b = sum(t.alive for t in env.troop_list.blue_troops)
    r = sum(t.alive for t in env.troop_list.red_troops)
    return {"blue": b, "red": r, "blue0": env._init_count["blue"], "red0": env._init_count["red"]}


def run(env, policies, episodes, seed0, device="cpu"):
    rows = [play_episode(env, policies, seed0 + s, device) for s in range(episodes)]
    out = {}
    for tm in ("blue", "red"):
        opp = "red" if tm == "blue" else "blue"
        own_loss = np.array([r[tm + "0"] - r[tm] for r in rows], dtype=float)
        en_loss = np.array([r[opp + "0"] - r[opp] for r in rows], dtype=float)
        out[tm] = {
            "LER": float(en_loss.sum() / max(1.0, own_loss.sum())),       # pooled 교환비 ★
            "win": float(np.mean([1.0 if (r[opp] == 0 and r[tm] > 0) else 0.0 for r in rows])),
            "생존율": float(np.mean([r[tm] / max(1, r[tm + "0"]) for r in rows])),
            "아군손실": float(own_loss.mean()),
            "적손실": float(en_loss.mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue", default=None, help="blue 정책 ckpt (없으면 blue=규칙기반)")
    ap.add_argument("--red", default=None, help="red 정책 ckpt (없으면 red=규칙기반)")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=10000)   # 학습 seed와 겹치지 않게
    args = ap.parse_args()
    if not args.blue and not args.red:
        ap.error("최소 한 쪽(--blue/--red) 체크포인트가 필요합니다.")

    policies = {"blue": _load(args.blue), "red": _load(args.red)}
    pol_sides = [t for t in ("blue", "red") if policies[t] is not None]

    env = WargameParallelEnv(seed=0)
    print(f"크롭 평가: 정책팀={pol_sides}  episodes={args.episodes}  (비교 대상=all-rule 베이스라인)")
    base = run(env, {}, args.episodes, args.seed0)           # both rule
    cfg = run(env, policies, args.episodes, args.seed0)      # 지정 정책

    for tm in ("blue", "red"):
        tag = "정책" if tm in pol_sides else "규칙(고정)"
        print(f"\n[{tm} — {tag}]   {'metric':<8}{'rule-based':>12}{'config':>12}{'Δ':>10}")
        print("-" * 50)
        for k in base[tm]:
            r, c = base[tm][k], cfg[tm][k]
            up = k not in ("아군손실",)   # 아군손실만 낮을수록 좋음
            mark = "↑" if (c - r > 0) == up else "↓"
            print(f"{'':<11}{k:<8}{r:>12.3f}{c:>12.3f}{c - r:>+9.3f} {mark}")
    print("\n해석: 정책팀의 LER↑/생존율↑/적손실↑/아군손실↓ 이면 규칙 대비 개선. (승률은 참고용)")


if __name__ == "__main__":
    main()
