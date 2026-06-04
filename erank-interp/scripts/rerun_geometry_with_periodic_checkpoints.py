"""
scripts/rerun_geometry_with_periodic_checkpoints.py

Rerun the modular-addition grokking geometry analysis using the dense
periodic checkpoints from retrain_modadd_with_periodic_checkpoints.py.

Produces updated versions of all plots with proper "over time" resolution.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.erank import compute_erank
from analysis.extract_activations import extract_activations
from models.transformer import create_standard_transformer
from data.modular_addition import generate_modular_addition_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
P = 113
SEED = 0
CKPT_DIR = PROJECT_ROOT / "outputs" / "modadd_periodic_checkpoints" / f"seed_{SEED}" / "checkpoints"
OUT_DIR = PROJECT_ROOT / "outputs" / "modadd_grokking_geometry"
PLOT_DIR = OUT_DIR / "plots"
SPECTRA_DIR = OUT_DIR / "spectra"

COLOR_TRAIN_ACC = "#1f77b4"
COLOR_VAL_ACC = "#ff7f0e"
COLOR_ERANK = "#2ca02c"
COLOR_MLP_NEG = "#C44E52"
COLOR_MLP_POS = "#E65100"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 12, "axes.titlesize": 14,
    "axes.labelsize": 13, "legend.fontsize": 10,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
})


# ===========================================================================
# Utilities
# ===========================================================================

def get_all_epoch_checkpoints() -> list[tuple[int, str, Path]]:
    """Return sorted list of (epoch, label, path) for all epoch-numbered checkpoints."""
    results = []
    for f in sorted(CKPT_DIR.glob("model_epoch_*.pt")):
        epoch = int(f.stem.split("_")[-1])
        results.append((epoch, f"epoch_{epoch}", f))

    # Add milestone checkpoints that may not have epoch-numbered duplicates
    for f in sorted(CKPT_DIR.glob("model_first_*.pt")):
        ckpt = torch.load(str(f), map_location="cpu")
        epoch = ckpt["epoch"]
        label = f.stem.replace("model_", "")
        # Check if we already have this epoch
        if not any(e == epoch for e, _, _ in results):
            results.append((epoch, label, f))

    # Add random_init
    ri = CKPT_DIR / "model_random_init.pt"
    if ri.exists() and not any(e == 0 for e, _, _ in results):
        results.append((0, "random_init", ri))

    # Add best_val
    bv = CKPT_DIR / "model_best_val.pt"
    if bv.exists():
        ckpt = torch.load(str(bv), map_location="cpu")
        epoch = ckpt["epoch"]
        if not any(e == epoch for e, _, _ in results):
            results.append((epoch, "best_val", bv))

    # Add final
    fn = CKPT_DIR / "model_final.pt"
    if fn.exists():
        ckpt = torch.load(str(fn), map_location="cpu")
        epoch = ckpt["epoch"]
        if not any(e == epoch for e, _, _ in results):
            results.append((epoch, "final", fn))

    results.sort(key=lambda x: x[0])
    return results


def load_model_from_checkpoint(ckpt_path: Path, device: str = "cuda"):
    """Load model weights from a checkpoint."""
    model = create_standard_transformer(
        "modular_addition", cfg_overrides={"mod_p": P}, seed=SEED,
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


def get_full_grid(model, device: str = "cuda"):
    """Run full 113^2 grid, return activations, a, b, sum_classes."""
    inputs, labels = generate_modular_addition_data(p=P, seed=0)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)
    acts = extract_activations(model, loader, "modular_addition", n_examples=P*P, device=device)
    return acts, inputs[:, 0], inputs[:, 1], labels


def compute_singular_values(z: torch.Tensor) -> np.ndarray:
    A = z.detach().float().cpu().numpy()
    A = A - A.mean(axis=0)
    return np.linalg.svd(A, full_matrices=False, compute_uv=False)


def compute_topk_mass(sigma: np.ndarray, k: int) -> float:
    total = sigma.sum()
    return float(sigma[:k].sum() / total) if total > 0 else 0.0


def compute_clustering(z: torch.Tensor, sum_classes: torch.Tensor):
    """Compute D_within, D_between, rho for centered activations."""
    z_np = z.float().cpu().numpy()
    z_c = z_np - z_np.mean(axis=0)
    sc = sum_classes.numpy()

    centroids = np.zeros((P, z_c.shape[1]))
    counts = np.zeros(P)
    for c in range(P):
        mask = sc == c
        counts[c] = mask.sum()
        if counts[c] > 0:
            centroids[c] = z_c[mask].mean(axis=0)

    # D_within
    N = len(z_c)
    d_within = 0.0
    for i in range(N):
        diff = z_c[i] - centroids[sc[i]]
        d_within += np.dot(diff, diff)
    d_within /= N

    # D_between
    d_between = 0.0
    n_pairs = 0
    for c1 in range(P):
        for c2 in range(c1 + 1, P):
            diff = centroids[c1] - centroids[c2]
            d_between += np.dot(diff, diff)
            n_pairs += 1
    d_between /= n_pairs

    rho = d_within / d_between if d_between > 0 else float("inf")
    return d_within, d_between, rho, z_c


# ===========================================================================
# Main analysis loop
# ===========================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    SPECTRA_DIR.mkdir(parents=True, exist_ok=True)

    checkpoints = get_all_epoch_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints")
    for epoch, label, path in checkpoints:
        print(f"  epoch {epoch:6d} — {label}")

    # Get training history from final checkpoint for accuracy curves
    _, final_ckpt = load_model_from_checkpoint(CKPT_DIR / "model_final.pt", device=device)
    th = final_ckpt["train_history"]
    all_epochs = th["epoch"]
    train_accs = th["train_acc"]
    val_accs = th["val_acc"]

    # Find tau_s
    tau_s = None
    for ep, va in zip(all_epochs, val_accs):
        if va >= 0.95:
            tau_s = ep
            break
    print(f"tau_s = {tau_s}")

    # ------------------------------------------------------------------
    # Loop over all checkpoints and compute everything
    # ------------------------------------------------------------------
    erank_rows = []
    cluster_rows = []
    spectrum_rows = []
    delta_rows = []
    pca_checkpoints = {}  # label -> (z_centered, sum_classes)

    for epoch, label, ckpt_path in checkpoints:
        print(f"\n--- Epoch {epoch} ({label}) ---")
        model, ckpt = load_model_from_checkpoint(ckpt_path, device=device)
        acts, a_vals, b_vals, sum_classes = get_full_grid(model, device=device)

        # Get accuracy at this epoch
        if epoch == 0:
            train_acc, val_acc, test_acc = 0.0, 0.0, 0.0
        else:
            ep_idx = min(epoch - 1, len(train_accs) - 1)
            train_acc = train_accs[ep_idx]
            val_acc = val_accs[ep_idx]
            test_acc = th["test_acc"][ep_idx] if "test_acc" in th else None

        rel_epoch = (epoch - tau_s) if tau_s else None

        # ---- eRank at all hooks ----
        for layer in [0, 1]:
            eranks = {}
            for hook in ["pre", "mid", "post"]:
                key = f"blocks.{layer}.hook_resid_{hook}"
                eranks[hook] = compute_erank(acts[key])

            delta_attn = eranks["mid"] - eranks["pre"]
            delta_mlp = eranks["post"] - eranks["mid"]

            for hook in ["pre", "mid", "post"]:
                erank_rows.append({
                    "seed": SEED, "epoch": epoch,
                    "relative_epoch_to_tau": rel_epoch,
                    "train_acc": train_acc, "val_acc": val_acc,
                    "test_acc_if_available": test_acc,
                    "layer": layer, "hook": f"resid_{hook}",
                    "erank": eranks[hook],
                    "delta_attn_if_layer_hook": delta_attn if hook == "pre" else None,
                    "delta_mlp_if_layer_hook": delta_mlp if hook == "pre" else None,
                    "checkpoint_label": label,
                    "checkpoint_path": str(ckpt_path),
                })

            # Layer delta summary (for Analysis 4)
            delta_rows.append({
                "seed": SEED, "epoch": epoch, "checkpoint_label": label,
                "train_acc": train_acc, "val_acc": val_acc,
                "layer": layer, "erank_mid": eranks["mid"],
                "erank_post": eranks["post"], "delta_mlp": delta_mlp,
            })

        # ---- Clustering on layer 1 resid_post ----
        z_l1_post = acts["blocks.1.hook_resid_post"]
        d_within, d_between, rho, z_centered = compute_clustering(z_l1_post, sum_classes)

        cluster_rows.append({
            "seed": SEED, "epoch": epoch,
            "relative_epoch_to_tau": rel_epoch,
            "train_acc": train_acc, "val_acc": val_acc,
            "checkpoint_label": label,
            "hook": "resid_post", "layer": 1,
            "D_within": d_within, "D_between": d_between,
            "within_between_ratio": rho,
            "num_examples": P * P,
            "examples_per_sum_class": P,
            "analysis_set_description": "diagnostic full-grid analysis (all 113^2 pairs)",
        })

        # ---- Spectrum / top-k mass on layer 1 resid_post ----
        sigma = compute_singular_values(z_l1_post)
        erank_l1 = compute_erank(z_l1_post)

        npy_path = SPECTRA_DIR / f"seed_{SEED}_epoch_{epoch}_layer1_resid_post_singular_values.npy"
        np.save(str(npy_path), sigma)

        spectrum_rows.append({
            "seed": SEED, "epoch": epoch,
            "relative_epoch_to_tau": rel_epoch,
            "checkpoint_label": label,
            "train_acc": train_acc, "val_acc": val_acc,
            "hook": "resid_post", "layer": 1,
            "erank": erank_l1,
            "rank_nonzero": int((sigma > 1e-10).sum()),
            "top1_mass": compute_topk_mass(sigma, 1),
            "top5_mass": compute_topk_mass(sigma, 5),
            "top10_mass": compute_topk_mass(sigma, 10),
            "top20_mass": compute_topk_mass(sigma, 20),
            "singular_values_path": str(npy_path),
        })

        # Store PCA data for key checkpoints
        if label in ("random_init", "first_train_acc_99", "first_val_acc_50",
                      "first_val_acc_95", "best_val", "final") or epoch == 0:
            pca_checkpoints[label] = (z_centered, sum_classes.numpy())

        print(f"  eRank L1 post={erank_l1:.2f}, D_within={d_within:.1f}, "
              f"D_between={d_between:.1f}, rho={rho:.6f}, top5={spectrum_rows[-1]['top5_mass']:.4f}")

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    df_erank = pd.DataFrame(erank_rows)
    df_erank.to_csv(OUT_DIR / "training_time_erank.csv", index=False)
    print(f"\nSaved training_time_erank.csv ({len(df_erank)} rows)")

    df_cluster = pd.DataFrame(cluster_rows)
    df_cluster.to_csv(OUT_DIR / "same_sum_clustering_metrics.csv", index=False)
    print(f"Saved same_sum_clustering_metrics.csv ({len(df_cluster)} rows)")

    df_spectrum = pd.DataFrame(spectrum_rows)
    df_spectrum.to_csv(OUT_DIR / "spectrum_metrics.csv", index=False)
    print(f"Saved spectrum_metrics.csv ({len(df_spectrum)} rows)")

    df_delta = pd.DataFrame(delta_rows)
    df_delta.to_csv(OUT_DIR / "layerwise_mlp_delta_summary.csv", index=False)
    print(f"Saved layerwise_mlp_delta_summary.csv ({len(df_delta)} rows)")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    epochs_arr = df_cluster["epoch"].values
    step = max(1, len(all_epochs) // 500)

    # ---- Plot 1: training_time_erank_accuracy_seedwise ----
    df_l1_post = df_erank[(df_erank["layer"] == 1) & (df_erank["hook"] == "resid_post")]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(all_epochs[::step], train_accs[::step], color=COLOR_TRAIN_ACC, alpha=0.7,
            label="Train acc", linewidth=1)
    ax.plot(all_epochs[::step], val_accs[::step], color=COLOR_VAL_ACC, alpha=0.7,
            label="Val acc", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    ax2 = ax.twinx()
    ax2.plot(df_l1_post["epoch"].values, df_l1_post["erank"].values,
             color=COLOR_ERANK, marker="o", markersize=4, linewidth=2,
             label="eRank(R_post^(1))")
    ax2.set_ylabel("eRank(R_post^(1))", color=COLOR_ERANK)
    ax2.tick_params(axis="y", labelcolor=COLOR_ERANK)

    if tau_s:
        ax.axvline(tau_s, color="red", linestyle="--", alpha=0.7, label=f"τ_s={tau_s}")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax.set_title("Training-Time eRank & Accuracy (Seed 0, Periodic Checkpoints)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "training_time_erank_accuracy_seedwise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved training_time_erank_accuracy_seedwise.png")

    # ---- Plot 2: training_time_erank_aligned_by_tau ----
    if tau_s:
        fig, ax = plt.subplots(figsize=(10, 5))
        rel = df_l1_post["epoch"].values - tau_s
        ax.plot(rel, df_l1_post["erank"].values, marker="o", markersize=4, label="Seed 0")
        ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="τ_s")
        ax.set_xlabel("Epoch − τ_s")
        ax.set_ylabel("eRank(R_post^(1))")
        ax.set_title("eRank Aligned by Grokking Time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "training_time_erank_aligned_by_tau.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved training_time_erank_aligned_by_tau.png")

    # ---- Plot 3: layer1_delta_mlp_over_time ----
    df_delta_l1 = df_delta[df_delta["layer"] == 1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_delta_l1["epoch"].values, df_delta_l1["delta_mlp"].values,
            marker="o", markersize=4, color=COLOR_MLP_POS, label="Δ_mlp^(1)")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    if tau_s:
        ax.axvline(tau_s, color="red", linestyle="--", alpha=0.5, label=f"τ_s={tau_s}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ_mlp^(1)")
    ax.set_title("Layer 1 MLP Delta over Training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "layer1_delta_mlp_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved layer1_delta_mlp_over_time.png")

    # ---- Plot 4: same_sum_distance_over_time ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(epochs_arr, df_cluster["D_within"].values, marker="o", markersize=4, color=COLOR_TRAIN_ACC)
    axes[0].set_title("D_within over epoch")
    axes[0].set_ylabel("D_within")

    axes[1].plot(epochs_arr, df_cluster["D_between"].values, marker="o", markersize=4, color=COLOR_VAL_ACC)
    axes[1].set_title("D_between over epoch")
    axes[1].set_ylabel("D_between")

    axes[2].plot(epochs_arr, df_cluster["within_between_ratio"].values, marker="o", markersize=4, color=COLOR_ERANK)
    axes[2].set_title("ρ = D_within / D_between")
    axes[2].set_ylabel("ρ")

    for ax in axes:
        ax.set_xlabel("Epoch")
        if tau_s:
            ax.axvline(tau_s, color="red", linestyle="--", alpha=0.5)

    fig.suptitle("Same-Sum Clustering over Training (R_post^(1), Seed 0)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "same_sum_distance_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved same_sum_distance_over_time.png")

    # ---- Plot 5: same_sum_ratio_aligned_by_tau ----
    if tau_s:
        fig, ax = plt.subplots(figsize=(8, 5))
        rel = epochs_arr - tau_s
        ax.plot(rel, df_cluster["within_between_ratio"].values, marker="o", markersize=4)
        ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="τ_s")
        ax.set_xlabel("Epoch − τ_s")
        ax.set_ylabel("ρ = D_within / D_between")
        ax.set_title("Same-Sum Ratio Aligned by Grokking Time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "same_sum_ratio_aligned_by_tau.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved same_sum_ratio_aligned_by_tau.png")

    # ---- Plot 6: PCA before/after ----
    if "random_init" in pca_checkpoints and ("best_val" in pca_checkpoints or "final" in pca_checkpoints):
        pre_label = "random_init"
        post_label = "best_val" if "best_val" in pca_checkpoints else "final"
        pre_z, sc = pca_checkpoints[pre_label]
        post_z, _ = pca_checkpoints[post_label]

        combined = np.vstack([pre_z, post_z])
        pca = PCA(n_components=2)
        pca.fit(combined)
        pre_pca = pca.transform(pre_z)
        post_pca = pca.transform(post_z)

        # All classes
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        cmap = plt.cm.hsv
        for ax, data, title in [
            (axes[0], pre_pca, "Random Init (Pre-Grokking)"),
            (axes[1], post_pca, "Best Val (Post-Grokking)"),
        ]:
            scatter = ax.scatter(data[:, 0], data[:, 1], c=sc, cmap=cmap,
                                 s=2, alpha=0.4, vmin=0, vmax=P - 1)
            ax.set_title(title)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        fig.colorbar(scatter, ax=axes, label="Sum class c = (a+b) mod 113", shrink=0.8)
        fig.suptitle("PCA of R_post^(1) — Seed 0", fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "same_sum_pca_before_after_all_classes.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Subset classes
        subset_classes = list(range(0, P, P // 10))[:10]
        colors_sub = plt.cm.tab10(np.linspace(0, 1, len(subset_classes)))
        class_to_color = {c: colors_sub[i] for i, c in enumerate(subset_classes)}

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, data, title in [
            (axes[0], pre_pca, "Random Init (Pre-Grokking)"),
            (axes[1], post_pca, "Best Val (Post-Grokking)"),
        ]:
            for i, c in enumerate(subset_classes):
                cmask = sc == c
                ax.scatter(data[cmask, 0], data[cmask, 1], c=[class_to_color[c]],
                           s=8, alpha=0.6, label=f"c={c}")
            ax.set_title(title)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(fontsize=7, ncol=2, loc="upper right")
        fig.suptitle("PCA of R_post^(1) — Subset Classes — Seed 0", fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "same_sum_pca_before_after_subset_classes.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved PCA before/after plots")

    # ---- Plot 7: singular_spectrum_pre_vs_post ----
    fig, ax = plt.subplots(figsize=(10, 6))
    key_labels = ["random_init", "first_train_acc_99", "first_val_acc_50",
                   "first_val_acc_95", "best_val", "final"]
    for _, row in df_spectrum.iterrows():
        if row["checkpoint_label"] in key_labels:
            sigma = np.load(row["singular_values_path"])
            p_norm = sigma / sigma.sum()
            ax.plot(np.arange(len(p_norm)), p_norm,
                    label=f"{row['checkpoint_label']} (ep {int(row['epoch'])})", alpha=0.8)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalized singular value (p_i)")
    ax.set_title("Singular Spectrum at Key Checkpoints (R_post^(1))")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "singular_spectrum_pre_vs_post.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved singular_spectrum_pre_vs_post.png")

    # ---- Plot 8: topk_mass_over_time ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ep_arr = df_spectrum["epoch"].values
    for k, color in [(1, "#e41a1c"), (5, "#377eb8"), (10, "#4daf4a"), (20, "#984ea3")]:
        ax.plot(ep_arr, df_spectrum[f"top{k}_mass"].values, marker="o", markersize=3,
                color=color, label=f"top-{k}")
    if tau_s:
        ax.axvline(tau_s, color="red", linestyle="--", alpha=0.5, label=f"τ_s={tau_s}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-k spectral mass")
    ax.set_title("Top-k Mass over Training (R_post^(1))")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "topk_mass_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved topk_mass_over_time.png")

    # ---- Plot 9: erank_vs_topk_mass_over_time ----
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df_l1_post["epoch"].values, df_l1_post["erank"].values,
             color=COLOR_ERANK, marker="o", markersize=3, label="eRank(R_post^(1))")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("eRank(R_post^(1))", color=COLOR_ERANK)

    ax2 = ax1.twinx()
    ax2.plot(ep_arr, df_spectrum["top5_mass"].values, color="#377eb8",
             marker="s", markersize=3, linestyle="--", label="top-5 mass")
    ax2.set_ylabel("Top-5 mass", color="#377eb8")

    if tau_s:
        ax1.axvline(tau_s, color="red", linestyle="--", alpha=0.5, label=f"τ_s={tau_s}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.suptitle("eRank vs Top-5 Mass (R_post^(1))", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "erank_vs_topk_mass_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved erank_vs_topk_mass_over_time.png")

    # ---- Plot 10: layer0_vs_layer1_mlp_delta (bar at best_val) ----
    # Use best_val checkpoint data
    bv_epoch = final_ckpt["train_history"]["best_val_epoch"]
    df_bv = df_delta[df_delta["epoch"] == bv_epoch]
    if not df_bv.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        l0_delta = df_bv[df_bv["layer"] == 0]["delta_mlp"].values[0]
        l1_delta = df_bv[df_bv["layer"] == 1]["delta_mlp"].values[0]
        colors = [COLOR_MLP_NEG, COLOR_MLP_POS]
        bars = ax.bar([0, 1], [l0_delta, l1_delta], color=colors, edgecolor="black",
                       width=0.5, alpha=0.85)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Layer 0", "Layer 1"])
        ax.set_ylabel("Δ_mlp (eRank_post − eRank_mid)")
        ax.set_title(f"MLP Delta: Layer 0 vs Layer 1 (Best Val, Epoch {bv_epoch})")
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        for i, v in enumerate([l0_delta, l1_delta]):
            ax.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=11)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "layer0_vs_layer1_mlp_delta.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved layer0_vs_layer1_mlp_delta.png")

    # ---- Plot 11: layer0 AND layer1 delta_mlp over time ----
    df_delta_l0 = df_delta[df_delta["layer"] == 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_delta_l0["epoch"].values, df_delta_l0["delta_mlp"].values,
            marker="o", markersize=4, color=COLOR_MLP_NEG, label="Δ_mlp^(0) (Layer 0)")
    ax.plot(df_delta_l1["epoch"].values, df_delta_l1["delta_mlp"].values,
            marker="o", markersize=4, color=COLOR_MLP_POS, label="Δ_mlp^(1) (Layer 1)")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    if tau_s:
        ax.axvline(tau_s, color="red", linestyle="--", alpha=0.5, label=f"τ_s={tau_s}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ_mlp")
    ax.set_title("Layer 0 vs Layer 1 MLP Delta over Training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "layerwise_delta_mlp_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved layerwise_delta_mlp_over_time.png")

    print("\n" + "=" * 60)
    print("All analyses complete with periodic checkpoints!")
    print(f"Outputs: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
