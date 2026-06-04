"""
scripts/run_graded_ablation_core.py

Graded (soft) ablation sensitivity analysis for thesis.

Instead of fully zeroing a component, scales its output by (1-r) for
ablation strengths r in [0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0].

This produces dose-response curves showing how sensitive each component is
to partial weakening.

Runs on the four most important task/component pairs from full-ablation:
  - key_value      L1 attention  (main routing component)
  - key_value      L0 MLP        (surprising strong full-ablation effect)
  - modular_addition L1 attention (causally important in full ablation)
  - modular_addition L1 MLP      (processing hypothesis / late MLP)

Uses seeds 0, 1, 2 for each.

Outputs:
  outputs/graded_ablation_core/graded_ablation_all_runs.csv
  outputs/graded_ablation_core/graded_ablation_summary.csv
  outputs/graded_ablation_core/plots/*.png
  outputs/graded_ablation_core/graded_ablation_report_for_chatgpt.md
"""

from __future__ import annotations

import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

from data.splits import (
    get_key_value_dataloaders_3way,
    get_modular_addition_dataloaders_3way,
)
from models.transformer import create_standard_transformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.join(PROJECT_ROOT, "outputs", "thesis_additions")
OUT_DIR    = os.path.join(PROJECT_ROOT, "outputs", "graded_ablation_core")
PLOTS_DIR  = os.path.join(OUT_DIR, "plots")
SEEDS      = [0, 1, 2]

ABLATION_STRENGTHS = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]

# (task, layer, component, short_label)
COMPONENTS = [
    ("key_value",        1, "attn", "kv_l1_attention"),
    ("key_value",        0, "mlp",  "kv_l0_mlp"),
    ("modular_addition", 1, "attn", "modular_l1_attention"),
    ("modular_addition", 1, "mlp",  "modular_l1_mlp"),
]

CHANCE = {
    "key_value": 1.0 / 30,        # d_vocab_out = 30
    "modular_addition": 1.0 / 113, # d_vocab_out = 113
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_logits_at_pred(logits: torch.Tensor, task_name: str) -> torch.Tensor:
    if task_name == "modular_addition":
        return logits[:, 2, :]
    return logits[:, -1, :]


def _eval_accuracy_and_loss(model, loader, task_name: str, device: torch.device,
                            fwd_hooks=None) -> tuple[float, float]:
    """Return (accuracy, avg_loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    n_correct = 0
    n_total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if fwd_hooks:
                logits = model.run_with_hooks(inputs, fwd_hooks=fwd_hooks)
            else:
                logits = model(inputs)
            pred_logits = _get_logits_at_pred(logits, task_name)
            loss = criterion(pred_logits, labels)
            total_loss += loss.item()
            n_correct += (pred_logits.argmax(dim=-1) == labels).sum().item()
            n_total += labels.size(0)
    acc = n_correct / n_total if n_total > 0 else 0.0
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    return acc, avg_loss


def _make_scale_hook(layer_idx: int, component: str, scale: float):
    """Return (hook_name, hook_fn) that scales component output by `scale`."""
    if component == "attn":
        name = f"blocks.{layer_idx}.hook_attn_out"
    else:
        name = f"blocks.{layer_idx}.hook_mlp_out"

    def hook_fn(value, hook):
        return value * scale

    return name, hook_fn


def _build_dataloaders(task_name: str, cfg: dict, seed: int):
    if task_name == "key_value":
        return get_key_value_dataloaders_3way(
            num_keys=cfg["kv_num_keys"],
            vocab_size=cfg["kv_vocab_size"],
            train_frac=cfg["kv_train_frac"],
            batch_size=cfg["batch_size"],
            seed=seed,
        )
    elif task_name == "modular_addition":
        return get_modular_addition_dataloaders_3way(
            p=cfg["mod_p"],
            train_frac=cfg["mod_train_frac"],
            batch_size=cfg["batch_size"],
            seed=seed,
        )
    raise ValueError(task_name)


def _load_cfg():
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_cfg()
    print(f"Device: {device}")
    print(f"Ablation strengths: {ABLATION_STRENGTHS}")
    print(f"Components: {len(COMPONENTS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Total evaluations: {len(COMPONENTS) * len(SEEDS) * len(ABLATION_STRENGTHS)}")

    rows = []

    for task, layer_idx, component, short_label in COMPONENTS:
        print(f"\n{'='*60}")
        print(f"  {short_label}: {task} layer {layer_idx} {component}")
        print(f"{'='*60}")

        for seed in SEEDS:
            ckpt_path = os.path.join(
                BASE_DIR, f"seed_{seed}", f"standard_{task}",
                "checkpoints", "model_best_val.pt"
            )
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] checkpoint not found: {ckpt_path}")
                continue

            print(f"\n  seed={seed}")

            # Build model and load weights
            if task == "key_value":
                overrides = {"kv_num_keys": cfg["kv_num_keys"],
                             "kv_vocab_size": cfg["kv_vocab_size"]}
            else:
                overrides = {"mod_p": cfg["mod_p"]}

            model = create_standard_transformer(task, cfg_overrides=overrides, seed=seed)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            model.eval()

            # Build data loaders
            _, val_loader, test_loader, _ = _build_dataloaders(task, cfg, seed)

            # Base accuracy (r=0.0, scale=1.0 — no hooks needed)
            base_test_acc, base_test_loss = _eval_accuracy_and_loss(
                model, test_loader, task, device)
            base_val_acc, _ = _eval_accuracy_and_loss(
                model, val_loader, task, device)
            print(f"    base: test_acc={base_test_acc:.4f} val_acc={base_val_acc:.4f}")

            # Graded ablation
            for r in ABLATION_STRENGTHS:
                scale = 1.0 - r

                if r == 0.0:
                    # No ablation — use base results directly
                    abl_test_acc = base_test_acc
                    abl_test_loss = base_test_loss
                    abl_val_acc = base_val_acc
                else:
                    hook_name, hook_fn = _make_scale_hook(layer_idx, component, scale)
                    abl_test_acc, abl_test_loss = _eval_accuracy_and_loss(
                        model, test_loader, task, device,
                        fwd_hooks=[(hook_name, hook_fn)])
                    abl_val_acc, _ = _eval_accuracy_and_loss(
                        model, val_loader, task, device,
                        fwd_hooks=[(hook_name, hook_fn)])

                delta = base_test_acc - abl_test_acc
                delta_val = base_val_acc - abl_val_acc

                print(f"    r={r:.2f} scale={scale:.2f}  "
                      f"test_acc={abl_test_acc:.4f} Δ={delta:+.4f}  "
                      f"val_acc={abl_val_acc:.4f}")

                rows.append({
                    "task": task,
                    "architecture": "standard",
                    "seed": seed,
                    "checkpoint_path": ckpt_path,
                    "layer": layer_idx,
                    "component": component,
                    "ablation_strength_r": r,
                    "scale": scale,
                    "base_test_acc": base_test_acc,
                    "ablated_test_acc": abl_test_acc,
                    "delta_test_acc": delta,
                    "base_test_loss": base_test_loss,
                    "ablated_test_loss": abl_test_loss,
                    "base_val_acc": base_val_acc,
                    "ablated_val_acc": abl_val_acc,
                    "delta_val_acc": delta_val,
                    "status": "ok",
                })

    # -----------------------------------------------------------------------
    # Save all-runs CSV
    # -----------------------------------------------------------------------
    all_csv = os.path.join(OUT_DIR, "graded_ablation_all_runs.csv")
    fieldnames = [
        "task", "architecture", "seed", "checkpoint_path",
        "layer", "component", "ablation_strength_r", "scale",
        "base_test_acc", "ablated_test_acc", "delta_test_acc",
        "base_test_loss", "ablated_test_loss",
        "base_val_acc", "ablated_val_acc", "delta_val_acc",
        "status",
    ]
    with open(all_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {all_csv}")

    # -----------------------------------------------------------------------
    # Summary CSV: aggregate across seeds
    # -----------------------------------------------------------------------
    summary_rows = []
    for task, layer_idx, component, short_label in COMPONENTS:
        for r in ABLATION_STRENGTHS:
            matching = [row for row in rows
                        if row["task"] == task
                        and row["layer"] == layer_idx
                        and row["component"] == component
                        and row["ablation_strength_r"] == r]
            if not matching:
                continue
            accs = [row["ablated_test_acc"] for row in matching]
            deltas = [row["delta_test_acc"] for row in matching]
            summary_rows.append({
                "task": task,
                "architecture": "standard",
                "layer": layer_idx,
                "component": component,
                "ablation_strength_r": r,
                "scale": 1.0 - r,
                "mean_ablated_test_acc": float(np.mean(accs)),
                "std_ablated_test_acc": float(np.std(accs)),
                "mean_delta_test_acc": float(np.mean(deltas)),
                "std_delta_test_acc": float(np.std(deltas)),
                "n_seeds": len(matching),
            })

    summary_csv = os.path.join(OUT_DIR, "graded_ablation_summary.csv")
    sum_fields = [
        "task", "architecture", "layer", "component",
        "ablation_strength_r", "scale",
        "mean_ablated_test_acc", "std_ablated_test_acc",
        "mean_delta_test_acc", "std_delta_test_acc",
        "n_seeds",
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sum_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_csv}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    _make_dose_response_plots(rows, summary_rows)
    _make_combined_plot(summary_rows)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    _write_report(rows, summary_rows)

    print(f"\nDone. All outputs in: {OUT_DIR}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_dose_response_plots(rows, summary_rows):
    """One plot per component showing per-seed curves + mean."""
    seed_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

    for task, layer_idx, component, short_label in COMPONENTS:
        fig, ax = plt.subplots(figsize=(7, 5))

        # Per-seed curves
        for seed in SEEDS:
            seed_rows = [r for r in rows
                         if r["task"] == task and r["layer"] == layer_idx
                         and r["component"] == component and r["seed"] == seed]
            if not seed_rows:
                continue
            seed_rows.sort(key=lambda x: x["ablation_strength_r"])
            rs = [r["ablation_strength_r"] for r in seed_rows]
            accs = [r["ablated_test_acc"] for r in seed_rows]
            ax.plot(rs, accs, "o-", color=seed_colors[seed], alpha=0.5,
                    markersize=4, label=f"seed {seed}")

        # Mean curve
        comp_summary = [r for r in summary_rows
                        if r["task"] == task and r["layer"] == layer_idx
                        and r["component"] == component]
        if comp_summary:
            comp_summary.sort(key=lambda x: x["ablation_strength_r"])
            rs = [r["ablation_strength_r"] for r in comp_summary]
            means = [r["mean_ablated_test_acc"] for r in comp_summary]
            stds = [r["std_ablated_test_acc"] for r in comp_summary]
            ax.plot(rs, means, "s-", color="black", linewidth=2,
                    markersize=6, label="mean", zorder=5)
            ax.fill_between(rs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color="gray", alpha=0.2)

        # Chance line
        chance = CHANCE.get(task, 0)
        ax.axhline(chance, color="red", linestyle=":", linewidth=1,
                   label=f"chance ({chance:.4f})")

        ax.set_xlabel("Ablation strength r")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"{task} — L{layer_idx} {component}\nGraded ablation dose-response")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, f"{short_label}_dose_response.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Plot: {path}")


def _make_combined_plot(summary_rows):
    """Combined 2x2 plot of all four components."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for idx, (task, layer_idx, component, short_label) in enumerate(COMPONENTS):
        ax = axes_flat[idx]
        comp_summary = [r for r in summary_rows
                        if r["task"] == task and r["layer"] == layer_idx
                        and r["component"] == component]
        if not comp_summary:
            continue
        comp_summary.sort(key=lambda x: x["ablation_strength_r"])
        rs = [r["ablation_strength_r"] for r in comp_summary]
        means = [r["mean_ablated_test_acc"] for r in comp_summary]
        stds = [r["std_ablated_test_acc"] for r in comp_summary]

        ax.plot(rs, means, "s-", color="black", linewidth=2, markersize=5)
        ax.fill_between(rs,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color="gray", alpha=0.2)

        chance = CHANCE.get(task, 0)
        ax.axhline(chance, color="red", linestyle=":", linewidth=1)

        ax.set_xlabel("Ablation strength r")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"{task}\nL{layer_idx} {component}")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Graded Ablation Dose-Response (mean +/- std, n=3 seeds)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "combined_dose_response.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(rows, summary_rows):
    lines = []
    a = lines.append

    a("# Graded Ablation Sensitivity Report")
    a("")

    # 1. Executive summary
    a("## 1. Executive summary")
    a("")
    a("Graded output-scaling ablation was applied to the four most important")
    a("task/component pairs identified by full ablation. Each component's output")
    a("was scaled by (1-r) for r in [0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0].")
    a("The resulting dose-response curves reveal whether each component's contribution")
    a("degrades smoothly, has a critical threshold, or collapses immediately.")
    a("")

    # Summarize key findings
    for task, layer_idx, component, short_label in COMPONENTS:
        comp_summary = [r for r in summary_rows
                        if r["task"] == task and r["layer"] == layer_idx
                        and r["component"] == component]
        if not comp_summary:
            continue
        comp_summary.sort(key=lambda x: x["ablation_strength_r"])
        base_acc = comp_summary[0]["mean_ablated_test_acc"]  # r=0
        r50_row = next((r for r in comp_summary if r["ablation_strength_r"] == 0.50), None)
        full_row = next((r for r in comp_summary if r["ablation_strength_r"] == 1.0), None)
        if r50_row and full_row:
            a(f"- **{short_label}**: base={base_acc:.4f}, "
              f"r=0.50 acc={r50_row['mean_ablated_test_acc']:.4f}, "
              f"r=1.0 acc={full_row['mean_ablated_test_acc']:.4f}")
    a("")

    # 2. Why graded ablation
    a("## 2. Why graded ablation was run")
    a("")
    a("Full ablation tests whether a component is **necessary** by completely removing it.")
    a("Graded ablation tests **sensitivity**: how robust the model's performance is to")
    a("partial weakening of a component. A component can be necessary (full ablation drops")
    a("accuracy) yet robust to partial weakening (accuracy stays high until r is large),")
    a("or it can be extremely fragile (even small r causes collapse).")
    a("")

    # 3. Method
    a("## 3. Method")
    a("")
    a("Component output scaling:")
    a("")
    a("For attention at layer l:")
    a("```")
    a("A_l(x) -> (1-r) * A_l(x)")
    a("```")
    a("")
    a("For MLP at layer l:")
    a("```")
    a("M_l(x) -> (1-r) * M_l(x)")
    a("```")
    a("")
    a("Where r is the ablation strength:")
    a("- r = 0.0: no ablation (scale = 1.0)")
    a("- r = 1.0: full ablation (scale = 0.0)")
    a("")
    a(f"Ablation strengths tested: {ABLATION_STRENGTHS}")
    a("")
    a("Components tested:")
    a("")
    a("| Task | Layer | Component | Reason |")
    a("|---|---:|---|---|")
    a("| key_value | 1 | attention | main routing component |")
    a("| key_value | 0 | MLP | surprising strong full-ablation effect |")
    a("| modular_addition | 1 | attention | causally important in full ablation |")
    a("| modular_addition | 1 | MLP | processing hypothesis / late MLP contribution |")
    a("")
    a(f"Seeds: {SEEDS}")
    a("")
    a("Checkpoints: best-validation from `outputs/thesis_additions/seed_N/standard_TASK/`")
    a("")

    # 4. Runs and checkpoints
    a("## 4. Runs and checkpoints")
    a("")
    a("| Task | Seed | Checkpoint | Base test acc |")
    a("|---|---:|---|---:|")
    seen = set()
    for r in rows:
        key = (r["task"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
        # Shorten checkpoint path for readability
        short_ckpt = r["checkpoint_path"].replace(PROJECT_ROOT + "/", "")
        a(f"| {r['task']} | {r['seed']} | `{short_ckpt}` | {r['base_test_acc']:.4f} |")
    a("")

    # 5. Results by component
    a("## 5. Results by component")
    a("")
    for task, layer_idx, component, short_label in COMPONENTS:
        a(f"### {short_label}")
        a("")

        comp_summary = [r for r in summary_rows
                        if r["task"] == task and r["layer"] == layer_idx
                        and r["component"] == component]
        if not comp_summary:
            a("No data.")
            a("")
            continue

        comp_summary.sort(key=lambda x: x["ablation_strength_r"])

        a("| r | scale | mean test acc | std | mean delta | std delta |")
        a("|---:|---:|---:|---:|---:|---:|")
        for sr in comp_summary:
            a(f"| {sr['ablation_strength_r']:.2f} | {sr['scale']:.2f} | "
              f"{sr['mean_ablated_test_acc']:.4f} | {sr['std_ablated_test_acc']:.4f} | "
              f"{sr['mean_delta_test_acc']:+.4f} | {sr['std_delta_test_acc']:.4f} |")
        a("")
        a(f"Plot: `plots/{short_label}_dose_response.png`")
        a("")

    # 6. Interpretation
    a("## 6. Interpretation")
    a("")

    for task, layer_idx, component, short_label in COMPONENTS:
        comp_summary = [r for r in summary_rows
                        if r["task"] == task and r["layer"] == layer_idx
                        and r["component"] == component]
        if not comp_summary:
            continue
        comp_summary.sort(key=lambda x: x["ablation_strength_r"])

        base_acc = comp_summary[0]["mean_ablated_test_acc"]
        r05 = next((r for r in comp_summary if r["ablation_strength_r"] == 0.05), None)
        r10 = next((r for r in comp_summary if r["ablation_strength_r"] == 0.10), None)
        r50 = next((r for r in comp_summary if r["ablation_strength_r"] == 0.50), None)
        full = next((r for r in comp_summary if r["ablation_strength_r"] == 1.0), None)

        a(f"### {short_label}")
        a("")

        if full:
            full_drop = full["mean_delta_test_acc"]
            if r05 and r05["mean_delta_test_acc"] > 0.5 * full_drop and full_drop > 0.01:
                a(f"**Immediate collapse**: Even r=0.05 causes a large accuracy drop "
                  f"({r05['mean_delta_test_acc']:+.4f}), indicating extreme sensitivity.")
            elif r50 and r50["mean_delta_test_acc"] < 0.1 * base_acc and full_drop > 0.1:
                a(f"**Threshold collapse**: Accuracy stays high until ~r=0.50, then collapses. "
                  f"The component has a critical strength threshold.")
            elif full_drop < 0.01:
                a(f"**No effect**: Even full ablation (r=1.0) causes minimal accuracy drop "
                  f"({full_drop:+.4f}). The component may not be necessary.")
            else:
                a(f"**Smooth degradation**: Accuracy decreases gradually from r=0 to r=1. "
                  f"The component contributes continuously.")
        a("")

    a("### Cross-component comparison")
    a("")
    a("Which component is most sensitive to partial ablation?")
    a("")
    for task, layer_idx, component, short_label in COMPONENTS:
        r10_row = next((r for r in summary_rows
                        if r["task"] == task and r["layer"] == layer_idx
                        and r["component"] == component
                        and r["ablation_strength_r"] == 0.10), None)
        if r10_row:
            a(f"- {short_label} at r=0.10: delta={r10_row['mean_delta_test_acc']:+.4f}")
    a("")
    a("Do the graded curves agree with full-ablation results? Compare r=1.0 values with")
    a("the earlier full-ablation report.")
    a("")

    # 7. Thesis implications
    a("## 7. Thesis implications")
    a("")
    a("Graded ablation adds a third layer of evidence to the causal story:")
    a("")
    a("1. **eRank** shows where representational geometry changes (observational)")
    a("2. **Full ablation** shows whether a component is necessary (binary causal)")
    a("3. **Graded ablation** shows how robust/sensitive the learned circuit is (continuous causal)")
    a("")
    a("Together, these three analyses create a progression from observation to mechanism:")
    a("eRank identifies candidate components, full ablation confirms necessity, and graded")
    a("ablation characterises the nature of the dependency (fragile vs robust).")
    a("")

    # 8. Caveats
    a("## 8. Caveats")
    a("")
    a("- Output scaling is deterministic but not identical to neuron deletion or head pruning.")
    a("- Scaling can push activations off-distribution, so large ablation strengths may")
    a("  produce effects beyond simple capacity reduction.")
    a("- This is evaluation-only; no retraining was performed.")
    a("- Only the four most important components (from full ablation) were tested.")
    a("- Models are 2-layer standard transformers; results may not generalise to deeper models.")
    a("")

    # 9. Recommended next action
    a("## 9. Recommended next action")
    a("")
    a("Choose exactly one:")
    a("")
    a("- If curves are informative: add one figure to advisor/thesis deck.")
    a("- If results are noisy: rerun with validation split or more seeds.")
    a("- If curves contradict full ablation: inspect implementation and compare with full-ablation code.")
    a("")

    report_path = os.path.join(OUT_DIR, "graded_ablation_report_for_chatgpt.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
