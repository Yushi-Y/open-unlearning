"""Plot autoresearch progress: DISCO iterations on Llama-3.2-3B WMDP-Bio."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

# All iterations: (iter, robustness, status, label)
# (iter, robustness, status, label, epochs_before_break)
experiments = [
    (0,  0.115, "keep",    "baseline DISCO",                 "2/10"),
    (1,  0.114, "discard", "Mahalanobis subtraction",        "2/10"),
    (2,  0.114, "discard", "diagonal Σ_r",                   "2/10"),
    (3,  0.096, "keep",    "+ DISCO grad collapse",          "7/10"),
    (4,  0.113, "discard", "skip zero-active → raw grads",   "2/10"),
    (5,  0.111, "discard", "no double projection",           "9/10"),
    (6,  0.113, "discard", "act all + grad selective",       "2/10"),
    (7,  0.066, "keep",    "+ KL token masking",             "6/10"),
    (8,  0.037, "keep",    "+ LoRA adversary",               "7/10"),
    (9,  0.071, "discard", "post-hoc weight projection",     "2/10"),
    (10, 0.115, "discard", "adaptive threshold",             "2/10"),
    (11, 0.082, "discard", "Joint Kronecker",                "4/10"),
    (12, 0.082, "discard", "token selectivity weighting",    "4/10"),
    (13, 0.115, "discard", "soft Mahalanobis",               "2/10"),
]

target = 0.031  # RepCollapse

fig, ax = plt.subplots(figsize=(14, 6))

# Track running best
kept = [(e[0], e[1], e[4]) for e in experiments if e[2] == "keep"]
discarded = [(e[0], e[1], e[3], e[4]) for e in experiments if e[2] == "discard"]

# Plot discarded as light red dots with labels + epochs
for x, y, label, epochs in discarded:
    ax.scatter(x, y, color="#e74c3c", alpha=0.35, s=60, zorder=3, edgecolors="none")
    ax.annotate(f"{label} ({epochs})", (x, y), fontsize=6, alpha=0.5, color="#e74c3c",
                xytext=(5, 5), textcoords="offset points", rotation=0)

# Plot kept as green dots with labels and staircase
prev_x, prev_y = None, None
for x, y, epochs in kept:
    ax.scatter(x, y, color="#27ae60", s=100, zorder=5, edgecolors="white", linewidths=1.5)
    if prev_x is not None:
        # Staircase: horizontal then vertical
        ax.plot([prev_x, x], [prev_y, prev_y], color="#27ae60", linewidth=2, alpha=0.7)
        ax.plot([x, x], [prev_y, y], color="#27ae60", linewidth=2, alpha=0.7)
    prev_x, prev_y = x, y

# Extend staircase to the end
ax.plot([prev_x, len(experiments) - 1], [prev_y, prev_y], color="#27ae60",
        linewidth=2, alpha=0.7)

# Label kept points
labels_kept = {0: "baseline DISCO (2/10)", 3: "+ grad collapse (7/10)", 7: "+ KL masking (6/10)", 8: "+ LoRA adversary (7/10)"}
for x, y, epochs in kept:
    if x in labels_kept:
        offset = (-10, -18) if x == 0 else (8, -15)
        ax.annotate(labels_kept[x], (x, y), fontsize=8, fontweight="bold",
                    color="#27ae60", xytext=offset, textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#27ae60", alpha=0.5) if x > 0 else None)

# Target line
ax.axhline(y=target, color="#3498db", linestyle="--", linewidth=1.5, alpha=0.7)
ax.annotate(f"RepCollapse target = {target}", xy=(len(experiments) - 1, target),
            fontsize=9, color="#3498db", ha="right", va="bottom",
            xytext=(0, 5), textcoords="offset points")

# Formatting
ax.set_xlabel("Experiment #", fontsize=12)
ax.set_ylabel("Robustness (lower is better)", fontsize=12)
ax.set_title("DISCO Autoresearch: Llama-3.2-3B, WMDP-Bio\n"
             f"{len(experiments)} experiments, {len(kept)} kept improvements",
             fontsize=13, fontweight="bold")
ax.set_xlim(-0.5, len(experiments) - 0.5)
ax.set_ylim(0.02, 0.13)
ax.set_xticks(range(len(experiments)))

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
           markersize=10, label='Kept'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
           markersize=8, alpha=0.4, label='Discarded'),
    Line2D([0], [0], color='#27ae60', linewidth=2, label='Running best'),
    Line2D([0], [0], color='#3498db', linewidth=1.5, linestyle='--', label='RepCollapse target'),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

ax.grid(True, alpha=0.2)

# Add params text box
params_text = (
    "Llama-3.2-3B · WMDP-Bio · SGD\n"
    "DISCO: lr=0.2, n_pcs=200, mom=0.97, lora_lr=0.1, rank=32\n"
    "RepCollapse: lr=0.2, n_pcs=400, mom=0.97, lora_lr=0.1, rank=32 (10/10 epochs, no break)"
)
ax.text(0.02, 0.02, params_text, transform=ax.transAxes, fontsize=7,
        verticalalignment="bottom", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.savefig("autoresearch_disco_progress.png", dpi=150, bbox_inches="tight")
plt.savefig("autoresearch_disco_progress.pdf", bbox_inches="tight")
print("Saved autoresearch_disco_progress.png and .pdf")
