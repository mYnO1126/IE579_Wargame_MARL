# plot_fullsim.py
# 학습 정책(crowd obs + 1:4)으로 얻은 blue/red 각각의 full-sim 결과를 한 그래프로.
#   conda run -n wargame python -m tools.plot_fullsim
# 입력: ppt_assets/full_eval_blue_s8.json, full_eval_red_s8.json  (full_eval --out 산출)
# 산출: ppt_assets/fig6_fullsim_ler.png (LER, 팀별 패널) + fig7_fullsim_survival.png (생존율)
# 그래프 정책: 차트/축 제목 외 텍스트 최소화, 파스텔 팀컬러(연=Rule, 진=MARL).

import os
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BLUE_L, BLUE_D = "#AFCBE8", "#5E89BE"
RED_L, RED_D = "#ECB7AF", "#CD7E74"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "Liberation Sans", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlepad": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999999", "axes.labelcolor": "#333333", "axes.labelsize": 12,
    "text.color": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.8, "axes.axisbelow": True,
})

OUT = os.path.join(_ROOT, "ppt_assets")
blue = json.load(open(os.path.join(OUT, "full_eval_blue_s8.json")))["summary"]["blue"]
red = json.load(open(os.path.join(OUT, "full_eval_red_s8.json")))["summary"]["red"]


def _bars(ax, m, light, dark, ylabel, title):
    ax.bar([0], [m["rule_mean"]], 0.5, yerr=[m["rule_std"]], capsize=5,
           color=light, label="Rule-based", ecolor="#888888")
    ax.bar([1], [m["cfg_mean"]], 0.5, yerr=[m["cfg_std"]], capsize=5,
           color=dark, label="MARL", ecolor="#888888")
    ax.set_xticks([]); ax.set_ylabel(ylabel); ax.set_title(title)
    top = max(m["rule_mean"] + m["rule_std"], m["cfg_mean"] + m["cfg_std"])   # 두 막대 오차 상단 모두 포함
    ax.set_ylim(0, top * 1.18)


# === fig6: LER (팀별 패널, 스케일 다름) ===
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4.6))
_bars(a1, blue["LER"], BLUE_L, BLUE_D, "Loss-Exchange Ratio  (enemy / own losses)",
      "Blue (defender)")
a1.legend(frameon=False, loc="upper left")
_bars(a2, red["LER"], RED_L, RED_D, "Loss-Exchange Ratio  (enemy / own losses)",
      "Red (attacker)")
a2.legend(frameon=False, loc="upper left")
fig.suptitle("Full-scenario LER — MARL vs Rule-based (8 seeds)",
             fontweight="bold", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "fig6_fullsim_ler.png")); plt.close(fig)

# === fig7: 생존율 (양 팀 동일 0~1 축 → 한 패널) ===
fig, ax = plt.subplots(figsize=(7, 4.6))
x = [0, 1]; w = 0.36
ax.bar([0 - w / 2, 1 - w / 2], [blue["생존율"]["rule_mean"], red["생존율"]["rule_mean"]], w,
       yerr=[blue["생존율"]["rule_std"], red["생존율"]["rule_std"]], capsize=5,
       color=[BLUE_L, RED_L], ecolor="#888888")
ax.bar([0 + w / 2, 1 + w / 2], [blue["생존율"]["cfg_mean"], red["생존율"]["cfg_mean"]], w,
       yerr=[blue["생존율"]["cfg_std"], red["생존율"]["cfg_std"]], capsize=5,
       color=[BLUE_D, RED_D], ecolor="#888888")
ax.set_xticks(x); ax.set_xticklabels(["Blue (defender)", "Red (attacker)"])
ax.set_ylabel("Survival rate  (survivors / initial)")
ax.set_title("Full-scenario survival rate — MARL vs Rule-based (8 seeds)")
ax.legend(handles=[Patch(color="#bdbdbd", label="Rule-based"),
                   Patch(color="#6f6f6f", label="MARL")], frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_fullsim_survival.png")); plt.close(fig)

print("[saved] ppt_assets/fig6_fullsim_ler.png, fig7_fullsim_survival.png")
for tm, s in (("blue", blue), ("red", red)):
    print(f"  {tm}: LER {s['LER']['rule_mean']:.2f}->{s['LER']['cfg_mean']:.2f}  "
          f"survival {s['생존율']['rule_mean']:.2f}->{s['생존율']['cfg_mean']:.2f}")
