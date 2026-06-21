# plot_full_eval.py
# full_eval --out 로 저장한 JSON을 읽어 풀 시나리오 비교 그래프를 만든다.
#   conda run -n wargame python -m tools.plot_full_eval ppt_assets/full_eval_blue.json
# 산출: ppt_assets/fig5_full_transfer.png (정책팀의 rule vs MARL, 전이 격차)
# 그래프 정책: 차트/축 제목 외 텍스트 최소화. 파스텔 팀컬러(연=Rule, 진=MARL). 설명은 슬라이드에.

import os
import sys
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 파스텔 팀컬러 (ppt_results.py 와 동일)
TEAM = {
    "blue": ("#AFCBE8", "#5E89BE"),
    "red": ("#ECB7AF", "#CD7E74"),
}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "Liberation Sans", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlepad": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999999", "axes.labelcolor": "#333333", "axes.labelsize": 12,
    "text.color": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "ppt_assets/full_eval_blue.json")
    d = json.load(open(path))
    tm = d["policy_sides"][0]
    s = d["summary"][tm]
    light, dark = TEAM[tm]

    panels = [("LER", "Loss-Exchange Ratio  (enemy / own losses)", "LER"),
              ("생존율", "Blue survival rate  (survivors / initial)", "Survival rate")]
    fig, axes = plt.subplots(1, len(panels), figsize=(8, 4.4))
    for i, (ax, (key, ylab, title)) in enumerate(zip(axes, panels)):
        m = s[key]
        ax.bar([0], [m["rule_mean"]], 0.5, yerr=[m["rule_std"]], capsize=5,
               color=light, label="Rule-based", ecolor="#888888")
        ax.bar([1], [m["cfg_mean"]], 0.5, yerr=[m["cfg_std"]], capsize=5,
               color=dark, label="MARL", ecolor="#888888")
        ax.set_xticks([]); ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_ylim(0, (m["rule_mean"] + m["rule_std"]) * 1.25)
        if i == len(panels) - 1:
            ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out = os.path.join(_ROOT, "ppt_assets/fig5_full_transfer.png")
    fig.savefig(out)
    print("[saved]", out)


if __name__ == "__main__":
    main()
