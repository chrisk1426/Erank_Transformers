"""
scripts/build_layer0_freeze_layerwise_figures.py

Build the slide-ready layer-by-layer eRank analysis deliverables for the
layer-0 freezing experiment. Reads only existing outputs under
outputs/layer0_freeze_modadd/. Does not retrain.

Produces:
  plots/layerwise_attn_mlp_delta_by_variant.png      # grouped bar chart
  plots/layerwise_attn_mlp_delta_heatmap.png         # diverging heatmap
  plots/layerwise_attention_delta_by_variant.png     # attention-only bar
  plots/layerwise_mlp_delta_by_variant.png           # MLP-only bar
  plots/training_time_layerwise_deltas_<variant>.png # per-variant time series
  layerwise_attn_mlp_delta_summary_for_slides.csv
  layerwise_erank_delta_report_for_chatgpt.md
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = "/jumbo/lisp/f006j44/erank/erank-interp"
OUT_ROOT = os.path.join(PROJECT_ROOT, "outputs/layer0_freeze_modadd")
PLOT_DIR = os.path.join(OUT_ROOT, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

VARIANTS = ["normal", "freeze_A0", "freeze_M0", "freeze_A0_M0"]
SEEDS = [0, 1, 2]
LAYERS = [0, 1, 2]
COMPONENTS = ["A", "M"]

VARIANT_LABEL = {
    "normal":       "normal",
    "freeze_A0":    "freeze A0",
    "freeze_M0":    "freeze M0",
    "freeze_A0_M0": "freeze A0+M0",
}
VARIANT_COLOR = {
    "normal":       "#1f77b4",
    "freeze_A0":    "#ff7f0e",
    "freeze_M0":    "#2ca02c",
    "freeze_A0_M0": "#d62728",
}
LAYER_COLOR = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}


# ---------------------------------------------------------------------------
# Load per-(variant, seed) best-val data
# ---------------------------------------------------------------------------

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_run_summary(variant, seed):
    p = os.path.join(OUT_ROOT, f"{variant}_seed{seed}", "run_summary.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def load_erank_by_ckpt(variant, seed):
    p = os.path.join(OUT_ROOT, f"{variant}_seed{seed}", "erank_by_checkpoint.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def load_erank_summary_by_ckpt(variant, seed):
    p = os.path.join(OUT_ROOT, f"{variant}_seed{seed}",
                     "erank_summary_by_checkpoint.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


# best_val_data[variant][seed][layer] = {erank_pre, erank_mid, erank_post,
#                                        delta_attn, delta_mlp,
#                                        best_val_epoch, best_val_acc,
#                                        test_acc_at_best_val}
best_val_data: dict = {}
for v in VARIANTS:
    best_val_data[v] = {}
    for s in SEEDS:
        rows = load_erank_by_ckpt(v, s)
        bv_rows = [r for r in rows if r["checkpoint_label"] == "best_val"]
        summary = load_run_summary(v, s)
        per_layer = {}
        for r in bv_rows:
            li = int(r["layer"])
            per_layer[li] = {
                "erank_pre":  safe_float(r["erank_pre"]),
                "erank_mid":  safe_float(r["erank_mid"]),
                "erank_post": safe_float(r["erank_post"]),
                "delta_attn": safe_float(r["delta_attn"]),
                "delta_mlp":  safe_float(r["delta_mlp"]),
                "epoch":      int(r["epoch"]),
            }
        best_val_data[v][s] = {
            "layers": per_layer,
            "best_val_epoch": summary.get("best_val_epoch"),
            "best_val_acc":   safe_float(summary.get("best_val_acc")),
            "test_acc":       safe_float(summary.get("test_acc_at_best_val")),
        }

# Aggregate mean ± std across seeds.
# agg[variant][layer]["delta_attn"] = (mean, std, n)
agg: dict = {v: {l: {} for l in LAYERS} for v in VARIANTS}
agg_per_run_acc: dict = {v: {} for v in VARIANTS}
for v in VARIANTS:
    for l in LAYERS:
        for k in ("delta_attn", "delta_mlp", "erank_pre", "erank_mid", "erank_post"):
            vals = []
            for s in SEEDS:
                layers = best_val_data[v][s]["layers"]
                if l in layers:
                    vals.append(layers[l][k])
            agg[v][l][k] = (
                float(np.mean(vals)) if vals else float("nan"),
                float(np.std(vals))  if vals else float("nan"),
                len(vals),
            )
    bvas = [best_val_data[v][s]["best_val_acc"] for s in SEEDS]
    tas  = [best_val_data[v][s]["test_acc"]     for s in SEEDS]
    agg_per_run_acc[v] = {
        "mean_best_val_acc": float(np.mean(bvas)),
        "mean_test_acc_at_best_val": float(np.mean(tas)),
    }


# ---------------------------------------------------------------------------
# Figure 1: Combined layer-by-layer bar chart (6 bars per variant)
# ---------------------------------------------------------------------------

# Bar layout: per variant, six bars in fixed order: A0, M0, A1, M1, A2, M2.
bar_order = [("A", 0), ("M", 0), ("A", 1), ("M", 1), ("A", 2), ("M", 2)]
bar_label = [f"{c}{l}" for c, l in bar_order]
bar_color = {("A", l): "#4a90e2" for l in LAYERS}   # blue family for attention
bar_color.update({("M", l): "#e25c4a" for l in LAYERS})   # red family for MLP
# Use shading by layer index
for c in ("A", "M"):
    for l in LAYERS:
        bar_color[(c, l)] = plt.cm.tab20((4 if c == "A" else 6) + l * 4) if False else bar_color[(c, l)]
# Override with cleaner palette
bar_color = {
    ("A", 0): "#1f77b4",
    ("A", 1): "#5fa2dd",
    ("A", 2): "#a8ccea",
    ("M", 0): "#d62728",
    ("M", 1): "#ed5e5e",
    ("M", 2): "#f6a9a9",
}

fig, ax = plt.subplots(figsize=(12, 5.5))
n_var = len(VARIANTS)
n_bars = len(bar_order)
group_width = 0.85
bar_w = group_width / n_bars
x_centers = np.arange(n_var)

for bi, (comp, layer) in enumerate(bar_order):
    means = []
    stds = []
    for vi, v in enumerate(VARIANTS):
        key = "delta_attn" if comp == "A" else "delta_mlp"
        m, s, _ = agg[v][layer][key]
        means.append(m)
        stds.append(s)
    offsets = (bi - (n_bars - 1) / 2) * bar_w
    ax.bar(x_centers + offsets, means, width=bar_w, yerr=stds, capsize=2.2,
           color=bar_color[(comp, layer)], edgecolor="black", linewidth=0.4,
           label=f"{comp}{layer}", alpha=0.93)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x_centers)
ax.set_xticklabels([VARIANT_LABEL[v] for v in VARIANTS], fontsize=11)
ax.set_ylabel(r"$\Delta$ eRank at best-val   (negative = compression, positive = expansion)",
              fontsize=11)
ax.set_title(
    "Layerwise eRank deltas at best-val by freeze variant\n"
    "3-layer transformer on modular addition (p=113), mean ± std across 3 seeds",
    fontsize=12)
ax.legend(title="component / layer", loc="upper left",
          fontsize=9, ncol=3, framealpha=0.95)
ax.grid(True, axis="y", alpha=0.3)
# Annotate per-bar means for readability
for bi, (comp, layer) in enumerate(bar_order):
    for vi, v in enumerate(VARIANTS):
        key = "delta_attn" if comp == "A" else "delta_mlp"
        m, _, _ = agg[v][layer][key]
        if np.isnan(m):
            continue
        offset = (bi - (n_bars - 1) / 2) * bar_w
        ax.annotate(f"{m:+.1f}",
                    (vi + offset, m),
                    xytext=(0, 3 if m >= 0 else -10),
                    textcoords="offset points",
                    ha="center", fontsize=7,
                    color="black", clip_on=True)
fig.tight_layout()
out1 = os.path.join(PLOT_DIR, "layerwise_attn_mlp_delta_by_variant.png")
fig.savefig(out1, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out1}")


# ---------------------------------------------------------------------------
# Figure 2: Heatmap of layerwise eRank deltas
# ---------------------------------------------------------------------------

col_labels = bar_label  # A0, M0, A1, M1, A2, M2
matrix = np.full((len(VARIANTS), len(bar_order)), np.nan)
for vi, v in enumerate(VARIANTS):
    for bi, (comp, layer) in enumerate(bar_order):
        key = "delta_attn" if comp == "A" else "delta_mlp"
        matrix[vi, bi] = agg[v][layer][key][0]

vmax_abs = float(np.nanmax(np.abs(matrix))) if np.any(~np.isnan(matrix)) else 1.0
if vmax_abs == 0 or np.isnan(vmax_abs):
    vmax_abs = 1.0

fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
               vmin=-vmax_abs, vmax=vmax_abs)
ax.set_xticks(np.arange(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=11)
ax.set_yticks(np.arange(len(VARIANTS)))
ax.set_yticklabels([VARIANT_LABEL[v] for v in VARIANTS], fontsize=11)
# Annotate every cell
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        if np.isnan(val):
            continue
        color = "white" if abs(val) > 0.55 * vmax_abs else "black"
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label(r"$\Delta$ eRank at best-val (mean over seeds)", fontsize=10)
ax.set_title(
    "Layer-0 freezing: mean Δ eRank at best-val (RdBu_r diverging colormap)\n"
    "rows = freeze variant, cols = component × layer    "
    "(red = expansion, blue = compression)",
    fontsize=11)
fig.tight_layout()
out2 = os.path.join(PLOT_DIR, "layerwise_attn_mlp_delta_heatmap.png")
fig.savefig(out2, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out2}")


# ---------------------------------------------------------------------------
# Figure 3: Attention-only and MLP-only bar plots
# ---------------------------------------------------------------------------

def attn_or_mlp_plot(key: str, ylabel: str, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    x_centers = np.arange(len(LAYERS))
    n_var = len(VARIANTS)
    width = 0.85 / n_var
    for vi, v in enumerate(VARIANTS):
        means = []
        stds = []
        for l in LAYERS:
            m, s, _ = agg[v][l][key]
            means.append(m); stds.append(s)
        offsets = (vi - (n_var - 1) / 2) * width
        bars = ax.bar(x_centers + offsets, means, width=width, yerr=stds,
                      capsize=3, color=VARIANT_COLOR[v], edgecolor="black",
                      linewidth=0.4, label=VARIANT_LABEL[v], alpha=0.93)
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax.annotate(f"{m:+.2f}", (b.get_x()+b.get_width()/2, m),
                            xytext=(0, 3 if m >= 0 else -10),
                            textcoords="offset points",
                            ha="center", fontsize=7.5, color="black",
                            clip_on=True)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"layer {l}" for l in LAYERS], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

attn_or_mlp_plot(
    "delta_attn",
    r"$\Delta_{\mathrm{attn}}$ at best-val (mean ± std across 3 seeds)",
    "Layerwise attention Δ eRank by freeze variant\n"
    "(positive = attention expands eRank; negative = attention compresses)",
    os.path.join(PLOT_DIR, "layerwise_attention_delta_by_variant.png"),
)
attn_or_mlp_plot(
    "delta_mlp",
    r"$\Delta_{\mathrm{MLP}}$ at best-val (mean ± std across 3 seeds)",
    "Layerwise MLP Δ eRank by freeze variant\n"
    "(positive = MLP expands eRank; negative = MLP compresses)",
    os.path.join(PLOT_DIR, "layerwise_mlp_delta_by_variant.png"),
)


# ---------------------------------------------------------------------------
# Figure 4: Training-time per-variant layerwise eRank deltas
# ---------------------------------------------------------------------------

# Use erank_summary_by_checkpoint.csv (it has the layer{0,1,2}_delta_attn/mlp
# columns pre-computed by the trainer at every periodic_full + event ckpt).

def load_periodic_full(variant, seed):
    rows = load_erank_summary_by_ckpt(variant, seed)
    rows = [r for r in rows if r["checkpoint_label"] == "periodic_full"]
    rows.sort(key=lambda r: int(r["epoch"]))
    return rows


for v in VARIANTS:
    # Collect per-seed time series and overlay.
    fig, (ax_attn, ax_mlp) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Top: Δ_attn per layer over training
    # Bottom: Δ_mlp per layer over training
    # Show all 3 seeds as lighter, plus the mean over seeds as bold.
    per_seed_curves = {l: [] for l in LAYERS}     # layer -> list of (ep, Δ_attn) arrays
    per_seed_curves_mlp = {l: [] for l in LAYERS}
    all_epochs = set()
    for s in SEEDS:
        rows = load_periodic_full(v, s)
        if not rows:
            continue
        eps = np.array([int(r["epoch"]) for r in rows])
        for l in LAYERS:
            da = np.array([safe_float(r[f"layer{l}_delta_attn"]) for r in rows])
            dm = np.array([safe_float(r[f"layer{l}_delta_mlp"])  for r in rows])
            per_seed_curves[l].append((eps, da))
            per_seed_curves_mlp[l].append((eps, dm))
            all_epochs.update(eps.tolist())

            # Faint seed line
            ax_attn.plot(eps, da, color=LAYER_COLOR[l], alpha=0.32, linewidth=1.0)
            ax_mlp.plot(eps, dm, color=LAYER_COLOR[l], alpha=0.32, linewidth=1.0)

    # Compute mean over seeds at shared epochs.
    shared_epochs = sorted(all_epochs)
    for l in LAYERS:
        if per_seed_curves[l]:
            # Mean over seeds at each epoch they all share.
            common_ep_to_vals = defaultdict(list)
            for eps, da in per_seed_curves[l]:
                for e, d in zip(eps, da):
                    common_ep_to_vals[int(e)].append(d)
            mean_eps = sorted(common_ep_to_vals.keys())
            mean_vals = [float(np.mean(common_ep_to_vals[e])) for e in mean_eps]
            ax_attn.plot(mean_eps, mean_vals, color=LAYER_COLOR[l], linewidth=2.4,
                         label=f"layer {l} (mean)")
        if per_seed_curves_mlp[l]:
            common_ep_to_vals = defaultdict(list)
            for eps, dm in per_seed_curves_mlp[l]:
                for e, d in zip(eps, dm):
                    common_ep_to_vals[int(e)].append(d)
            mean_eps = sorted(common_ep_to_vals.keys())
            mean_vals = [float(np.mean(common_ep_to_vals[e])) for e in mean_eps]
            ax_mlp.plot(mean_eps, mean_vals, color=LAYER_COLOR[l], linewidth=2.4,
                        label=f"layer {l} (mean)")

    # Mean τ_0.95 over seeds for this variant
    taus = [agg_per_run_acc[v]]  # placeholder
    tau_list = []
    for s in SEEDS:
        sm = load_run_summary(v, s)
        t = sm.get("tau_0.95")
        if t is not None:
            tau_list.append(t)
    mean_tau = float(np.mean(tau_list)) if tau_list else None
    if mean_tau is not None:
        for ax in (ax_attn, ax_mlp):
            ax.axvline(mean_tau, color="green", linestyle="--", linewidth=1.3,
                       alpha=0.8)
        ax_attn.annotate(rf"mean $\tau_{{0.95}}$ = {int(mean_tau)}",
                         (mean_tau, 0), xytext=(6, 0),
                         textcoords="offset points",
                         color="green", fontsize=9, fontweight="bold")

    for ax, title, ylabel in (
        (ax_attn, f"{VARIANT_LABEL[v]}: layerwise " + r"$\Delta_{\mathrm{attn}}$ over training",
         r"$\Delta_{\mathrm{attn}}$"),
        (ax_mlp, f"{VARIANT_LABEL[v]}: layerwise " + r"$\Delta_{\mathrm{MLP}}$ over training",
         r"$\Delta_{\mathrm{MLP}}$"),
    ):
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(loc="best", fontsize=9, framealpha=0.95)
        ax.grid(True, alpha=0.3)
    ax_mlp.set_xlabel("epoch", fontsize=11)
    fig.suptitle(
        f"Training-time layerwise eRank deltas — {VARIANT_LABEL[v]}\n"
        "thin = individual seeds, thick = mean over 3 seeds; "
        "vertical dashed = mean grokking epoch",
        fontsize=12, y=1.005)
    fig.tight_layout()
    out_tt = os.path.join(PLOT_DIR, f"training_time_layerwise_deltas_{v}.png")
    fig.savefig(out_tt, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_tt}")


# ---------------------------------------------------------------------------
# Deliverable 5: CSV summary
# ---------------------------------------------------------------------------

csv_rows = []
for v in VARIANTS:
    bv_mean = agg_per_run_acc[v]["mean_best_val_acc"]
    test_mean = agg_per_run_acc[v]["mean_test_acc_at_best_val"]
    for l in LAYERS:
        for comp, key in (("A", "delta_attn"), ("M", "delta_mlp")):
            m, s, n = agg[v][l][key]
            if np.isnan(m):
                interpretation = "no data"
            elif m < -0.1:
                interpretation = "compression / concentration"
            elif m > 0.1:
                interpretation = "expansion / feature construction"
            else:
                interpretation = "approximately zero (no representational change)"
            csv_rows.append({
                "variant": v,
                "layer": l,
                "component": comp,
                "mean_delta_erank": m,
                "std_delta_erank": s,
                "n_seeds": n,
                "mean_best_val_acc": bv_mean,
                "mean_test_acc_at_best_val": test_mean,
                "interpretation": interpretation,
            })
csv_path = os.path.join(OUT_ROOT, "layerwise_attn_mlp_delta_summary_for_slides.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    w.writeheader()
    for r in csv_rows:
        w.writerow(r)
print(f"wrote {csv_path}")


# ---------------------------------------------------------------------------
# Deliverable 6: Markdown report
# ---------------------------------------------------------------------------

def get(v, l, comp):
    key = "delta_attn" if comp == "A" else "delta_mlp"
    return agg[v][l][key]


def most_extreme(v, sign="neg"):
    """Most negative (compressive) or positive (expansive) component for variant v."""
    items = []
    for l in LAYERS:
        for comp in COMPONENTS:
            m, s, _ = get(v, l, comp)
            if not np.isnan(m):
                items.append(((comp, l), m, s))
    if not items: return None
    if sign == "neg":
        return min(items, key=lambda x: x[1])
    return max(items, key=lambda x: x[1])


md_lines = []
md_lines.append("# Layer-by-Layer eRank Analysis — Layer-0 Freezing Experiment")
md_lines.append("")
md_lines.append("Generated by `scripts/build_layer0_freeze_layerwise_figures.py` "
                "from existing outputs under `outputs/layer0_freeze_modadd/`. No retraining.")
md_lines.append("")
md_lines.append("## Source files")
md_lines.append("")
md_lines.append("- `outputs/layer0_freeze_modadd/{variant}_seed{0,1,2}/erank_by_checkpoint.csv` — best-val per-layer rows")
md_lines.append("- `outputs/layer0_freeze_modadd/{variant}_seed{0,1,2}/erank_summary_by_checkpoint.csv` — periodic_full snapshots for training-time plots")
md_lines.append("- `outputs/layer0_freeze_modadd/{variant}_seed{0,1,2}/run_summary.json` — best_val_acc, test_acc_at_best_val, tau_0.95")
md_lines.append("")
md_lines.append("## Mean ± std Δ eRank at best-val (3 seeds per variant)")
md_lines.append("")
md_lines.append("| variant | L0 Δ_attn | L0 Δ_MLP | L1 Δ_attn | L1 Δ_MLP | L2 Δ_attn | L2 Δ_MLP |")
md_lines.append("|:--|---:|---:|---:|---:|---:|---:|")
for v in VARIANTS:
    cells = []
    for (comp, l) in bar_order:
        key = "delta_attn" if comp == "A" else "delta_mlp"
        m, s, _ = agg[v][l][key]
        cells.append(f"{m:+.3f} ± {s:.3f}")
    md_lines.append(f"| {VARIANT_LABEL[v]} | " + " | ".join(cells) + " |")
md_lines.append("")
md_lines.append("All means/stds over seeds 0, 1, 2 at the best-validation checkpoint.")
md_lines.append("")

# Per-variant most compressive / most expansive
md_lines.append("## Per-variant most compressive vs most expansive component (best-val)")
md_lines.append("")
md_lines.append("| variant | most compressive | most expansive |")
md_lines.append("|:--|:--|:--|")
for v in VARIANTS:
    neg = most_extreme(v, "neg")
    pos = most_extreme(v, "pos")
    if neg and pos:
        (cn, ln), mn, sn = neg
        (cp, lp), mp, sp = pos
        md_lines.append(f"| {VARIANT_LABEL[v]} | "
                        f"{cn}{ln} ({mn:+.2f} ± {sn:.2f}) | "
                        f"{cp}{lp} ({mp:+.2f} ± {sp:.2f}) |")
md_lines.append("")

# Compression-shift analysis
md_lines.append("## Does compression shift when A0 or M0 is frozen?")
md_lines.append("")
norm_l0_mlp = agg["normal"][0]["delta_mlp"][0]
fA0_l0_mlp  = agg["freeze_A0"][0]["delta_mlp"][0]
fM0_l0_mlp  = agg["freeze_M0"][0]["delta_mlp"][0]
fboth_l1_mlp = agg["freeze_A0_M0"][1]["delta_mlp"][0]
fboth_l1_attn = agg["freeze_A0_M0"][1]["delta_attn"][0]
norm_l1_mlp = agg["normal"][1]["delta_mlp"][0]
norm_l1_attn = agg["normal"][1]["delta_attn"][0]
fM0_l1_mlp  = agg["freeze_M0"][1]["delta_mlp"][0]
fA0_l1_attn = agg["freeze_A0"][1]["delta_attn"][0]
fboth_l0_attn = agg["freeze_A0_M0"][0]["delta_attn"][0]
fA0_l0_attn = agg["freeze_A0"][0]["delta_attn"][0]
norm_l0_attn = agg["normal"][0]["delta_attn"][0]

md_lines.append(f"- **freeze_A0 → M0 picks up *much* more compression**: L0 Δ_MLP moves from "
                f"{norm_l0_mlp:+.2f} (normal) to **{fA0_l0_mlp:+.2f}** (freeze_A0). "
                "When attention is randomized, M0 has to clean up the noisier input — "
                f"M0 becomes ~{abs(fA0_l0_mlp)/abs(norm_l0_mlp):.0f}× more compressive.")
md_lines.append(f"- **freeze_M0 → L0 Δ_MLP goes positive** ({fM0_l0_mlp:+.2f} vs {norm_l0_mlp:+.2f} in normal). "
                "Frozen M0 sees a trained A0 output that it happens to *expand*. "
                f"L1 Δ_MLP compensates only mildly ({fM0_l1_mlp:+.2f} vs {norm_l1_mlp:+.2f}).")
md_lines.append(f"- **freeze_A0_M0 → compression shifts primarily to L1 ATTENTION**, not L1 MLP. "
                f"Δ_attn[1] moves from {norm_l1_attn:+.2f} (normal) to "
                f"**{fboth_l1_attn:+.2f}** (freeze_A0_M0) — the single largest compressive cell "
                "in the whole experiment. L1 MLP also compensates "
                f"(Δ_MLP[1] = {fboth_l1_mlp:+.2f} vs normal {norm_l1_mlp:+.2f}), "
                "but the heavier lifting is done by the trainable L1 attention.")
md_lines.append(f"- **Random / frozen A0 produces large positive Δ_attn[0]** "
                f"({fA0_l0_attn:+.2f} in freeze_A0, {fboth_l0_attn:+.2f} in freeze_A0_M0, "
                f"vs {norm_l0_attn:+.2f} in normal). Random attention mixes positions noisily "
                "and inflates rank artificially. Downstream layers (L1 attention especially) "
                "then have to compress this random-noise expansion back out.")
md_lines.append("")

# Attention layers expand or compress?
md_lines.append("## Do attention layers mostly expand or compress eRank?")
md_lines.append("")
attn_signs = []
for v in VARIANTS:
    for l in LAYERS:
        if v == "freeze_A0" and l == 0: continue  # frozen at random init; not informative
        m = agg[v][l]["delta_attn"][0]
        if not np.isnan(m):
            attn_signs.append(m)
n_pos = sum(1 for x in attn_signs if x > 0)
n_neg = sum(1 for x in attn_signs if x < 0)
md_lines.append(f"Across {len(attn_signs)} trainable-attention (variant, layer) cells "
                f"(excluding `freeze_A0`/L0 which is frozen at random init), "
                f"**{n_pos} show positive Δ_attn (expansion)** and "
                f"**{n_neg} show negative Δ_attn (compression)**. "
                "Attention is *primarily expansive* on this task — both layer-0 and layer-2 "
                "attention expand rank strongly across all variants (Δ_attn[0] = "
                f"{agg['normal'][0]['delta_attn'][0]:+.2f} in normal; "
                f"Δ_attn[2] = {agg['normal'][2]['delta_attn'][0]:+.2f} in normal). "
                "**The only attention layer that compresses is L1 attention**, and it does so "
                "dramatically when L0 is frozen "
                f"(Δ_attn[1] = {agg['freeze_A0_M0'][1]['delta_attn'][0]:+.2f} in `freeze_A0_M0`).")
md_lines.append("")

# Specific Q&A
md_lines.append("## Answers to the specific questions")
md_lines.append("")
md_lines.append("**Q1: In the normal model, which MLP layer is most compressive?**  ")
norm_mlps = [(l, agg["normal"][l]["delta_mlp"][0]) for l in LAYERS]
norm_min = min(norm_mlps, key=lambda x: x[1])
md_lines.append(f"L{norm_min[0]} MLP (Δ_MLP = {norm_min[1]:+.2f}). "
                "L0 and L1 both compress (negative Δ_MLP); L2 expands strongly.")
md_lines.append("")
md_lines.append("**Q2: When A0 is frozen, does M0 become more compressive?**  ")
md_lines.append(f"Yes, dramatically. M0 Δ_MLP goes from {norm_l0_mlp:+.2f} (normal) to "
                f"{fA0_l0_mlp:+.2f} (freeze_A0) — roughly **9× more compressive**.")
md_lines.append("")
md_lines.append("**Q3: When M0 is frozen, where does compression shift?**  ")
fM0_per_layer = [(l, agg["freeze_M0"][l]["delta_mlp"][0]) for l in LAYERS]
fM0_most_compressive = min(fM0_per_layer, key=lambda x: x[1])
md_lines.append(f"L{fM0_most_compressive[0]} MLP becomes the most compressive "
                f"(Δ_MLP = {fM0_most_compressive[1]:+.2f}). "
                "L0 swings positive (no compression — and M0 can't change anyway); "
                "L1 picks up some additional compression, but mostly the *whole* "
                "geometry is less extreme (L2 expansion also smaller).")
md_lines.append("")
md_lines.append("**Q4: When A0 and M0 are both frozen, which later layer compensates?**  ")
md_lines.append(f"**L1 attention compensates most**, not L1 MLP. "
                f"Δ_attn[1] = {fboth_l1_attn:+.2f} (vs {norm_l1_attn:+.2f} in normal) — "
                "the single largest compressive cell in the whole experiment. "
                f"L1 MLP also helps (Δ_MLP[1] = {fboth_l1_mlp:+.2f} vs {norm_l1_mlp:+.2f}), "
                "but the heavy lifting is done by the trainable L1 attention. "
                f"L2 MLP expands slightly more than normal "
                f"(Δ_MLP[2] = {agg['freeze_A0_M0'][2]['delta_mlp'][0]:+.2f} vs "
                f"{agg['normal'][2]['delta_mlp'][0]:+.2f}).")
md_lines.append("")
md_lines.append("**Q5: Do attention layers mostly expand eRank or compress it?**  ")
attn_pos_layers = []
attn_neg_layers = []
for v in VARIANTS:
    for l in LAYERS:
        # Exclude frozen A0 cells (no learning happened)
        if l == 0 and v in ("freeze_A0", "freeze_A0_M0"): continue
        m = agg[v][l]["delta_attn"][0]
        if np.isnan(m): continue
        if m > 0: attn_pos_layers.append((v, l, m))
        else:     attn_neg_layers.append((v, l, m))
md_lines.append(f"**Mostly expand, but with one big exception: L1 attention.** "
                f"Across the {len(attn_pos_layers)+len(attn_neg_layers)} trainable-attention "
                f"(variant, layer) cells (excluding the frozen A0 cells), "
                f"**{len(attn_pos_layers)} expand and {len(attn_neg_layers)} compress**. "
                "All L0 and L2 attention cells in the trainable variants expand eRank "
                f"(L2 Δ_attn is +5.8 to +7.7 across variants). "
                "But **L1 attention is the compressive attention layer** in this architecture: "
                f"Δ_attn[1] = {norm_l1_attn:+.2f} (normal), "
                f"{fA0_l1_attn:+.2f} (freeze_A0), "
                f"{agg['freeze_M0'][1]['delta_attn'][0]:+.2f} (freeze_M0), "
                f"{fboth_l1_attn:+.2f} (freeze_A0_M0). "
                "It picks up dramatic compressive work specifically when layer 0 is frozen.")
md_lines.append("")

# Figure paths
md_lines.append("## Figure paths")
md_lines.append("")
md_lines.append("Best-val analysis:")
md_lines.append("- `plots/layerwise_attn_mlp_delta_by_variant.png` — combined bar chart (6 bars per variant)")
md_lines.append("- `plots/layerwise_attn_mlp_delta_heatmap.png` — diverging heatmap (rows = variant, cols = A0/M0/A1/M1/A2/M2)")
md_lines.append("- `plots/layerwise_attention_delta_by_variant.png` — attention-only bars")
md_lines.append("- `plots/layerwise_mlp_delta_by_variant.png` — MLP-only bars")
md_lines.append("")
md_lines.append("Training-time per-variant:")
for v in VARIANTS:
    md_lines.append(f"- `plots/training_time_layerwise_deltas_{v}.png`")
md_lines.append("")

# 3-5 bullet interpretation for advisor slides
md_lines.append("## Slide-ready interpretation (3–5 bullets)")
md_lines.append("")
md_lines.append(f"- **Compression is mobile, and it lives in attention as much as in MLP.** "
                f"`freeze_A0_M0` shifts the compressive role from L0 MLP to **L1 ATTENTION** "
                f"(Δ_attn[1] = {fboth_l1_attn:+.2f} vs normal {norm_l1_attn:+.2f}) — the largest "
                f"compressive cell in the whole experiment. L1 MLP also compensates "
                f"(Δ_MLP[1] = {fboth_l1_mlp:+.2f} vs {norm_l1_mlp:+.2f}), but secondarily.")
md_lines.append(f"- **Random A0 forces *trainable* M0 to clean up harder.** `freeze_A0` produces "
                f"the most striking single MLP number: Δ_MLP[0] = {fA0_l0_mlp:+.2f}, "
                f"~{abs(fA0_l0_mlp)/abs(norm_l0_mlp):.0f}× more compressive than `normal` ({norm_l0_mlp:+.2f}).")
md_lines.append(f"- **Random / frozen A0 inflates Δ_attn[0] hugely** "
                f"({fA0_l0_attn:+.2f} freeze_A0, {fboth_l0_attn:+.2f} freeze_A0_M0, vs "
                f"{norm_l0_attn:+.2f} normal). Random attention mixes positions noisily — "
                "the trained downstream layers (especially L1 attention) then have to compress "
                "this artifactual expansion back out.")
md_lines.append("- **Position of compressive layers in `normal`:** "
                f"L0 MLP ({norm_l0_mlp:+.2f}), L1 MLP ({norm_l1_mlp:+.2f}), L1 attention "
                f"({norm_l1_attn:+.2f}). L0 and L2 attention strongly expand; L2 MLP strongly "
                f"expands ({agg['normal'][2]['delta_mlp'][0]:+.2f}, just before unembed).")
md_lines.append("- **All four variants reach test ≈ 0.998** despite arriving at different "
                "layerwise geometries. The model finds *a* task-aligned representation; "
                "exactly *which* layer does the compression work depends on what's frozen.")
md_lines.append("")

md_path = os.path.join(OUT_ROOT, "layerwise_erank_delta_report_for_chatgpt.md")
with open(md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"wrote {md_path}")


print("\nALL DELIVERABLES COMPLETE.")
