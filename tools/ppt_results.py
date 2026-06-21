# ppt_results.py
# 제출 PPT용 결과 숫자 + 그래프 생성기 (크롭 통계 평가).
# all-rule 베이스라인 대비 [blue 정책] / [red 정책] / [both 정책] 구성을 동일 seed로 비교하고,
#   - ppt_assets/crop_results.json   : 전체 수치(표·재현용)
#   - ppt_assets/crop_results.md     : PPT에 붙일 표
#   - ppt_assets/*.png               : 발표용 그래프
# 를 저장한다.  실행: conda run -n wargame python -m tools.ppt_results --episodes 500
#
# ★ 핵심 지표 = LER(교환비=적손실/아군손실). 승률은 비대칭/제로섬이라 오해를 부른다(참고용).
# 그래프 정책: 차트/축 제목 외 텍스트 최소화(값 라벨·각주 제거) → 설명은 슬라이드에. 파스텔 팀컬러.

import os
import sys
import json
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from rl.env import WargameParallelEnv
from rl.evaluate import _load, play_episode
from tools.plot_crop import draw          # 그래프(fig1/2/3)는 plot_crop 로 분리(평가 없이 재생성 가능)

OUT = os.path.join(_ROOT, "ppt_assets")


def collect(env, policies, episodes, seed0=10000):
    """동일 seed 묶음으로 N 에피소드 → 팀별 per-episode 손실/생존 배열을 모은다."""
    rows = [play_episode(env, policies, seed0 + s) for s in range(episodes)]
    out = {}
    for tm in ("blue", "red"):
        opp = "red" if tm == "blue" else "blue"
        own_loss = np.array([r[tm + "0"] - r[tm] for r in rows], dtype=float)
        en_loss = np.array([r[opp + "0"] - r[opp] for r in rows], dtype=float)
        surv = np.array([r[tm] / max(1, r[tm + "0"]) for r in rows], dtype=float)
        win = np.array([1.0 if (r[opp] == 0 and r[tm] > 0) else 0.0 for r in rows])
        out[tm] = {
            "LER": float(en_loss.sum() / max(1.0, own_loss.sum())),  # pooled 교환비
            "win": float(win.mean()),
            "survival": float(surv.mean()),
            "own_loss": float(own_loss.mean()),
            "enemy_loss": float(en_loss.mean()),
            "_own_loss_ep": own_loss,        # 분포 그래프용
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pb = _load(os.path.join(_ROOT, "rl/policies/ippo_blue.pt"))
    pr = _load(os.path.join(_ROOT, "rl/policies/ippo_red.pt"))
    # 학습 정책의 학습/배포와 동일한 1:4 비대칭 구성으로 크롭 평가 (blue 8 : red 32, map 240)
    from rl.spawn import DEFAULT_COMP
    COMP = {"blue": DEFAULT_COMP["blue"] * 2, "red": DEFAULT_COMP["red"] * 8}
    env = WargameParallelEnv(map_size=240, comp=COMP, seed=0)

    print(f"[ppt_results] episodes={args.episodes} (동일 seed pairing)")
    base = collect(env, {}, args.episodes, args.seed0)                       # all-rule
    blue = collect(env, {"blue": pb}, args.episodes, args.seed0)             # blue 정책
    red = collect(env, {"red": pr}, args.episodes, args.seed0)              # red 정책
    both = collect(env, {"blue": pb, "red": pr}, args.episodes, args.seed0)  # 양쪽 정책

    cfg = {"blue": blue["blue"], "red": red["red"]}
    keys = ["LER", "win", "survival", "own_loss", "enemy_loss"]

    # ---- JSON 저장 ----
    def clean(d):
        return {tm: {k: v[k] for k in keys} for tm, v in d.items()}
    payload = {
        "episodes": args.episodes,
        "baseline_all_rule": clean(base),
        "blue_policy": clean(blue), "red_policy": clean(red), "both_policy": clean(both),
        "summary_vs_baseline": {tm: {k: cfg[tm][k] for k in keys} for tm in ("blue", "red")},
    }
    with open(os.path.join(OUT, "crop_results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # ---- Markdown 표 저장 ----
    lines = [f"# Crop Evaluation Results (learned policy, blue 8 : red 32 = 1:4, N={args.episodes} episodes, paired seeds)\n",
             "Higher is better for all except **Own Loss**. Compared vs all-rule baseline.\n",
             "| Team | Method | LER | Survival | Own Loss | Enemy kills |",
             "|------|--------|-----|----------|----------|-------------|"]
    for tm in ("Blue", "Red"):
        t = tm.lower(); b, c = base[t], cfg[t]
        lines.append(f"| {tm} | Rule-based | {b['LER']:.2f} | {b['survival']:.2f} | {b['own_loss']:.2f} | {b['enemy_loss']:.2f} |")
        lines.append(f"| {tm} | **MARL** | **{c['LER']:.2f}** | **{c['survival']:.2f}** | **{c['own_loss']:.2f}** | **{c['enemy_loss']:.2f}** |")
    lines += ["", f"Both-policy config (both teams MARL vs all-rule): "
              f"blue LER {both['blue']['LER']:.2f}, red LER {both['red']['LER']:.2f}."]
    with open(os.path.join(OUT, "crop_results.md"), "w") as f:
        f.write("\n".join(lines))

    # ===== 그래프 (fig1/2/3) — plot_crop.draw 로 생성(스타일 변경 시 평가 없이 재실행 가능) =====
    draw(base, cfg, OUT)

    # ---- 콘솔 요약 ----
    print("\n저장 위치:", OUT)
    for fn in sorted(os.listdir(OUT)):
        print("  -", fn)
    print("\n[요약 vs all-rule 베이스라인]")
    for tm in ("blue", "red"):
        b, c = base[tm], cfg[tm]
        print(f"  {tm:4s}: LER {b['LER']:.2f}→{c['LER']:.2f}  kills {b['enemy_loss']:.2f}→{c['enemy_loss']:.2f}  "
              f"surv {b['survival']:.2f}→{c['survival']:.2f}  ownloss {b['own_loss']:.2f}→{c['own_loss']:.2f}")


if __name__ == "__main__":
    main()
