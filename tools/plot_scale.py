# plot_scale.py
# 핵심 발견 시각화: blue 학습 규모가 클수록 full-sim 전이가 나빠진다(scale backfire).
#   8:32(작음) → 16:64 → 64:256(≈풀 규모) 의 full-sim LER 비교.
#   conda run -n wargame python -m tools.plot_scale
# 입력: ppt_assets/full_eval_blue_{s8,16v64_s6,64v256_s8}.json
# 산출: ppt_assets/fig8_scale_backfire.png

import os
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE_D = "#5E89BE"
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
runs = [("8 : 32", "full_eval_blue_s8"),
        ("16 : 64", "full_eval_blue_16v64_s6"),
        ("64 : 256", "full_eval_blue_64v256_s8")]
labels, means, stds, rule = [], [], [], []
for lab, f in runs:
    d = json.load(open(os.path.join(OUT, "%s.json" % f)))
    le = d["summary"]["blue"]["LER"]
    labels.append(lab); means.append(le["cfg_mean"]); stds.append(le["cfg_std"]); rule.append(le["rule_mean"])
rule_ref = sum(rule) / len(rule)

fig, ax = plt.subplots(figsize=(7, 4.6))
x = range(len(labels))
ax.bar(x, means, 0.55, yerr=stds, capsize=5, color=BLUE_D, ecolor="#888888")
ax.axhline(rule_ref, color="#c0392b", lw=1.6, ls="--", label=f"Rule-based (≈{rule_ref:.1f})")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("Full-scenario LER  (enemy / own losses)")
ax.set_xlabel("Training scale (blue : red)")
ax.set_title("Bigger training scale → worse transfer (Blue)")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
out = os.path.join(OUT, "fig8_scale_backfire.png")
fig.savefig(out); print("[saved]", out)
print("  LER:", {l.split(chr(10))[0]: round(m, 2) for l, m in zip(labels, means)}, " rule≈", round(rule_ref, 2))
