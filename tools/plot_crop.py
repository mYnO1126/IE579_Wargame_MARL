# plot_crop.py
# 크롭 평가 결과(ppt_assets/crop_results.json)를 읽어 crop 그래프(fig1/2/3)를 생성한다.
# ★ 평가(ppt_results.py)와 분리: 스타일만 바꿔 다시 그릴 땐 평가 재실행 없이 이 스크립트만 돌리면 됨.
#   conda run -n wargame python -m tools.plot_crop
# (ppt_results.py 도 평가 직후 draw() 를 호출해 같은 그래프를 만든다 → 플롯 코드 단일 출처)
#
# 그래프 정책: 차트/축 제목 외 텍스트 최소화, 파스텔 팀컬러(연=Rule, 진=MARL).

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "ppt_assets")

BLUE_L, BLUE_D = "#AFCBE8", "#5E89BE"     # Blue 팀(파랑 계열)
RED_L, RED_D = "#ECB7AF", "#CD7E74"       # Red 팀(빨강 계열)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "Liberation Sans", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlepad": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999999", "axes.labelcolor": "#333333", "axes.labelsize": 12,
    "text.color": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def _g2(ax, rv, mv, title, ylabel, leg=False, legloc="upper left"):
    """두 팀(Blue,Red) × (Rule,MARL) 묶은 막대. 팀컬러(연=Rule, 진=MARL)."""
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, rv, w, color=[BLUE_L, RED_L])
    ax.bar(x + w / 2, mv, w, color=[BLUE_D, RED_D])
    ax.set_xticks(x); ax.set_xticklabels(["Blue", "Red"]); ax.set_title(title); ax.set_ylabel(ylabel)
    if leg:
        ax.legend(handles=[Patch(color="#bdbdbd", label="Rule-based"),
                           Patch(color="#6f6f6f", label="MARL")], frameon=False, loc=legloc)


def draw(base, cfg, out=OUT):
    """base/cfg = {team: {LER, survival, own_loss, enemy_loss, ...}} → fig1/2/3 저장."""
    bb, cb = base["blue"], cfg["blue"]
    rr, cr = base["red"], cfg["red"]

    # 1) LER — 두 팀, 팀컬러(연=Rule, 진=MARL). 점선 1.0 = 손익분기
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, [bb["LER"], rr["LER"]], w, color=[BLUE_L, RED_L])
    ax.bar(x + w / 2, [cb["LER"], cr["LER"]], w, color=[BLUE_D, RED_D])
    ax.axhline(1.0, color="#aaaaaa", lw=0.9, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(["Blue (Israeli)", "Red (Syrian)"])
    ax.set_ylabel("Loss-Exchange Ratio  (enemy losses / own losses)")
    ax.set_title("Loss-Exchange Ratio")
    ax.legend(handles=[Patch(color="#bdbdbd", label="Rule-based"),
                       Patch(color="#6f6f6f", label="MARL")], frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_ler_comparison.png")); plt.close(fig)

    # 2) 평균 상대사살수 (두 팀) — 승률 대신 사용(비대칭에서 승률은 floor/ceiling에 포화되어 무의미)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _g2(ax, [bb["enemy_loss"], rr["enemy_loss"]], [cb["enemy_loss"], cr["enemy_loss"]],
        "Average enemies killed per episode", "enemies killed  (mean per episode)",
        leg=True, legloc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_enemy_kills.png")); plt.close(fig)

    # 3) 효율 분해(두 팀): Blue=적을 더 잡아서 / Red=아군을 덜 잃어서 개선
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 4.6))
    def _loss(ax, r, c, light, dark, title):
        x = np.arange(2); w = 0.36
        ax.bar(x - w / 2, [r["own_loss"], r["enemy_loss"]], w, color=light, label="Rule-based")
        ax.bar(x + w / 2, [c["own_loss"], c["enemy_loss"]], w, color=dark, label="MARL")
        ax.set_xticks(x); ax.set_xticklabels(["Own losses", "Enemy losses"])
        ax.set_title(title); ax.set_ylabel("units per episode")
        ax.legend(frameon=False, loc="upper left")
    _loss(a1, bb, cb, BLUE_L, BLUE_D, "Blue (defender)")
    _loss(a2, rr, cr, RED_L, RED_D, "Red (attacker)")
    fig.suptitle("How each side improves: Blue kills more, Red loses fewer",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out, "fig3_crop_breakdown.png")); plt.close(fig)


if __name__ == "__main__":
    d = json.load(open(os.path.join(OUT, "crop_results.json")))
    draw(d["baseline_all_rule"], d["summary_vs_baseline"])
    print("[saved] ppt_assets/fig1_ler_comparison.png, fig2_enemy_kills.png, fig3_crop_breakdown.png")
