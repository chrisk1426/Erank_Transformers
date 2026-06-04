"""Two-panel thesis figure: singular value spectrum (pre/post grok) +
eRank-over-training with the two snapshots marked.

Model: 2-layer modular-addition transformer (modadd_periodic_checkpoints/seed_0).
Hook: layer 1 resid_post (final residual stream that feeds the unembed).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/jumbo/lisp/f006j44/erank/erank-interp")
GEOM = REPO / "outputs" / "modadd_grokking_geometry"
SPECTRA = GEOM / "spectra"
CSV = GEOM / "training_time_erank.csv"
OUT_LOG = REPO / "outputs" / "advisor_core_figures" / "modadd_spectrum_pre_post_grok.png"
OUT_LIN = REPO / "outputs" / "advisor_core_figures" / "modadd_spectrum_pre_post_grok_linear.png"
OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

PRE_EPOCH = 1500
POST_EPOCH = 2974

df = pd.read_csv(CSV)
df = df[(df["seed"] == 0) & (df["layer"] == 1) & (df["hook"] == "resid_post")]
df = df.sort_values("epoch").reset_index(drop=True)

# tau = epoch where relative_epoch_to_tau == 0
tau_row = df.loc[df["relative_epoch_to_tau"].abs().idxmin()]
tau = int(tau_row["epoch"])

pre_row = df[df["epoch"] == PRE_EPOCH].iloc[0]
post_row = df[df["epoch"] == POST_EPOCH].iloc[0]

sv_pre = np.load(SPECTRA / f"seed_0_epoch_{PRE_EPOCH}_layer1_resid_post_singular_values.npy")
sv_post = np.load(SPECTRA / f"seed_0_epoch_{POST_EPOCH}_layer1_resid_post_singular_values.npy")

idx_pre = np.arange(1, len(sv_pre) + 1)
idx_post = np.arange(1, len(sv_post) + 1)

def draw_spectrum(ax, log_scale: bool):
    plotter = ax.semilogy if log_scale else ax.plot
    plotter(
        idx_pre, sv_pre,
        color="#d62728", linewidth=2.2,
        marker="o", markersize=3.5, markevery=4,
        label=f"Pre-grok  (epoch {PRE_EPOCH}, val acc {pre_row['val_acc']*100:.1f}%)",
    )
    plotter(
        idx_post, sv_post,
        color="#1f77b4", linewidth=2.2,
        marker="s", markersize=3.5, markevery=4,
        label=f"Post-grok (epoch {POST_EPOCH}, val acc {post_row['val_acc']*100:.1f}%)",
    )
    ax.set_xlabel("Singular-value index $i$", fontsize=12)
    scale_label = "log scale" if log_scale else "linear scale"
    ax.set_ylabel(rf"Singular value $\sigma_i$  ({scale_label})", fontsize=12)
    ax.set_title(
        "Final residual-stream spectrum\n(layer-1 resid_post, modular addition)",
        fontsize=12,
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)


def draw_erank(ax):
    ax.plot(
        df["epoch"], df["erank"],
        color="black", linewidth=2.0, marker="o", markersize=3.5,
        label="eRank, layer-1 resid_post",
    )
    ax.axvline(tau, color="gray", linestyle=":", linewidth=1.4)
    ax.text(
        tau, ax.get_ylim()[1] * 0.97,
        rf" grokking $\tau$ = {tau}",
        color="gray", fontsize=10, va="top", ha="left",
    )
    for epoch, color, marker, label in [
        (PRE_EPOCH, "#d62728", "o", f"Pre-grok (epoch {PRE_EPOCH})"),
        (POST_EPOCH, "#1f77b4", "s", f"Post-grok (epoch {POST_EPOCH})"),
    ]:
        row = df[df["epoch"] == epoch].iloc[0]
        ax.scatter(
            [epoch], [row["erank"]],
            color=color, marker=marker, s=140, zorder=5,
            edgecolor="black", linewidth=1.2, label=label,
        )
        ax.annotate(
            f"eRank = {row['erank']:.1f}",
            xy=(epoch, row["erank"]),
            xytext=(12, 12 if epoch == PRE_EPOCH else -22),
            textcoords="offset points",
            fontsize=10, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
        )
    ax.set_xlabel("Training epoch", fontsize=12)
    ax.set_ylabel("Effective rank (eRank)", fontsize=12)
    ax.set_title(
        "eRank trajectory across grokking\n(snapshots used in left panel marked)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)


for log_scale, out_path in [(True, OUT_LOG), (False, OUT_LIN)]:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    draw_spectrum(axes[0], log_scale=log_scale)
    draw_erank(axes[1])
    fig.suptitle(
        "Modular addition: spectrum compression at the grokking transition",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")

print(f"tau = {tau}")
print(f"pre-grok  epoch {PRE_EPOCH}: erank = {pre_row['erank']:.3f}, val_acc = {pre_row['val_acc']:.4f}")
print(f"post-grok epoch {POST_EPOCH}: erank = {post_row['erank']:.3f}, val_acc = {post_row['val_acc']:.4f}")
