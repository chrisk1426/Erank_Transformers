"""
scripts/aggregate_novel_experiments.py

Part D — final consolidated report for the novel architecture + probe + FFN-head
experiment package.

Reads:
    outputs/modadd_arch_matrix_7gpu/    (Part A)
    outputs/hybrid_probe_diagnostics/   (Part B1)
    outputs/hybrid_ffn_head/            (Part B2)

Writes:
    outputs/novel_architecture_and_probe_results_for_chatgpt.md
    outputs/novel_architecture_and_probe_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import yaml


def _read_csv(p: str) -> list[dict]:
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def _read_yaml(p: str) -> dict:
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def collect_arch(arch_root: str) -> list[dict]:
    """One row per arch run."""
    summary_csv = os.path.join(arch_root, "architecture_matrix_summary.csv")
    erank_csv = os.path.join(arch_root, "architecture_matrix_erank_summary.csv")
    summary = {r["run_name"]: r for r in _read_csv(summary_csv)}
    erank = {r["run_name"]: r for r in _read_csv(erank_csv)}
    runs: list[dict] = []
    for name in sorted(summary.keys()):
        s = summary[name]
        e = erank.get(name, {})
        runs.append({
            "run_name": name,
            "schedule": s.get("schedule", ""),
            "n_params": _safe_float(s.get("n_params")),
            "best_val_acc": _safe_float(s.get("best_val_acc")),
            "best_val_epoch": _safe_float(s.get("best_val_epoch")),
            "test_at_best_val": _safe_float(s.get("test_at_best_val")),
            "tau": _safe_float(s.get("grokking_tau")),
            "status": s.get("status", ""),
            "final_erank": _safe_float(e.get("final_erank")),
            "sum_delta_attn": _safe_float(e.get("sum_delta_attn")),
            "sum_delta_mlp": _safe_float(e.get("sum_delta_mlp")),
            "topk1": _safe_float(e.get("topk_mass_1")),
            "topk5": _safe_float(e.get("topk_mass_5")),
            "topk10": _safe_float(e.get("topk_mass_10")),
            "topk20": _safe_float(e.get("topk_mass_20")),
            "same_sum_ratio": _safe_float(e.get("same_sum_ratio")),
        })
    return runs


def collect_probes(probe_root: str) -> list[dict]:
    csv_path = os.path.join(probe_root, "probe_all_results.csv")
    rows = _read_csv(csv_path)
    out: list[dict] = []
    for r in rows:
        out.append({
            "model_name": r.get("model_name", ""),
            "layers": _safe_float(r.get("layers")),
            "probe_type": r.get("probe_type", ""),
            "target": r.get("target", ""),
            "train_acc": _safe_float(r.get("train_acc")),
            "val_acc": _safe_float(r.get("val_acc")),
            "test_acc": _safe_float(r.get("test_acc")),
        })
    return out


def collect_ffn(ffn_root: str) -> list[dict]:
    if not os.path.isdir(ffn_root):
        return []
    runs: list[dict] = []
    for name in sorted(os.listdir(ffn_root)):
        run_dir = os.path.join(ffn_root, name)
        if not os.path.isdir(run_dir):
            continue
        cfg = _read_yaml(os.path.join(run_dir, "config_used.yaml"))
        if not cfg:
            continue
        log = _read_csv(os.path.join(run_dir, "training_log.csv"))
        best_val = -1.0; best_ep = None; test_at_best = None
        for r in log:
            va = _safe_float(r.get("val_acc"))
            ta = _safe_float(r.get("test_acc"))
            if va is None:
                continue
            if va > best_val:
                best_val = va; best_ep = int(r["epoch"]); test_at_best = ta
        # Endpoint eRank
        erows = _read_csv(os.path.join(run_dir, "endpoint_erank.csv"))
        sum_da = sum_dm = None
        for r in erows:
            if r.get("layer") == "sum_delta_attn":
                sum_da = _safe_float(r.get("erank_pre"))  # value column for trailing rows
            if r.get("layer") == "sum_delta_mlp":
                sum_dm = _safe_float(r.get("erank_pre"))
        status_path = os.path.join(run_dir, "status.txt")
        status = ""
        if os.path.exists(status_path):
            with open(status_path) as f:
                status = f.read().strip()
        runs.append({
            "run_name": name,
            "n_layers": cfg.get("n_layers"),
            "head_hidden": cfg.get("output_head_hidden_dim"),
            "weight_decay": cfg.get("weight_decay"),
            "best_val_acc": best_val if best_val >= 0 else None,
            "best_val_epoch": best_ep,
            "test_at_best_val": test_at_best,
            "sum_delta_attn": sum_da,
            "sum_delta_mlp": sum_dm,
            "status": status,
        })
    return runs


def fmt(x, fmt_str=".4f", na="n/a"):
    if x is None:
        return na
    try:
        return f"{x:{fmt_str}}"
    except (ValueError, TypeError):
        return na


def write_summary_csv(out_path: str, arch: list[dict], probes: list[dict], ffn: list[dict]) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "row_kind", "key1", "key2", "key3", "metric", "value"])
        # Arch
        for r in arch:
            for metric in ("best_val_acc", "test_at_best_val", "tau",
                           "final_erank", "sum_delta_attn", "sum_delta_mlp",
                           "topk1", "topk5", "topk10", "topk20", "same_sum_ratio"):
                w.writerow(["arch", "run", r["run_name"], r["schedule"], "",
                            metric, r.get(metric)])
        # Probes
        for r in probes:
            for metric in ("train_acc", "val_acc", "test_acc"):
                w.writerow(["probe", "result", r["model_name"], r["target"], r["probe_type"],
                            metric, r.get(metric)])
        # FFN-head
        for r in ffn:
            for metric in ("best_val_acc", "test_at_best_val",
                           "sum_delta_attn", "sum_delta_mlp"):
                w.writerow(["ffn_head", "run", r["run_name"], "", "", metric, r.get(metric)])
    print(f"  wrote {out_path}")


def write_report(out_path: str, arch: list[dict], probes: list[dict], ffn: list[dict],
                 arch_root: str, probe_root: str, ffn_root: str) -> None:
    lines: list[str] = []
    lines.append("# Novel Architecture and Probe Experiment Results")
    lines.append("")
    lines.append("## 1. Executive summary")
    grokked = [r for r in arch if (r.get("best_val_acc") or 0) >= 0.95]
    failed = [r for r in arch if r.get("best_val_acc") is not None and r["best_val_acc"] < 0.5]
    bullets: list[str] = []
    if arch:
        bullets.append(
            f"- Architecture matrix: {len(grokked)}/{len(arch)} runs reached val_acc ≥ 0.95 "
            f"(grokked); {len(failed)}/{len(arch)} stayed below 0.50."
        )
        if grokked:
            best = max(grokked, key=lambda r: r["best_val_acc"])
            bullets.append(
                f"- Best arch: `{best['schedule']}` "
                f"(val={best['best_val_acc']:.4f}, test@best={fmt(best['test_at_best_val'])}, "
                f"τ={int(best['tau']) if best['tau'] is not None else 'NA'})."
            )
    else:
        bullets.append("- Architecture matrix: no runs completed.")
    if probes:
        models = sorted({p["model_name"] for p in probes})
        bullets.append(f"- Probes: ran across {len(models)} hybrid model(s).")
    else:
        bullets.append("- Probes: none completed.")
    if ffn:
        ffr = ffn[0]
        bullets.append(
            f"- FFN-head: {len(ffn)} run(s); first run val={fmt(ffr.get('best_val_acc'))} "
            f"test@best={fmt(ffr.get('test_at_best_val'))}."
        )
    else:
        bullets.append("- FFN-head: not completed.")
    lines.extend(bullets)
    lines.append("")

    lines.append("## 2. Experiments completed")
    lines.append("- Architecture-order runs (Part A):")
    for r in arch:
        lines.append(f"  - `{r['run_name']}` ({r['schedule']}) — status: {r['status']}")
    lines.append("- Frozen probe runs (Part B1):")
    if probes:
        for m in sorted({p["model_name"] for p in probes}):
            lines.append(f"  - `{m}`")
    else:
        lines.append("  - (none)")
    lines.append("- FFN-head runs (Part B2):")
    if ffn:
        for r in ffn:
            lines.append(f"  - `{r['run_name']}` — status: {r['status']}")
    else:
        lines.append("  - (none)")
    lines.append("")

    lines.append("## 3. GPU/tmux status")
    lines.append("Logs are under each output dir's `logs/` subfolder. The master dispatcher")
    lines.append("ran a 2-GPU job queue (default GPUs 4 and 6) under tmux session `novel_experiments`.")
    lines.append("")

    lines.append("## 4. Modular-addition architecture matrix")
    lines.append("| run | schedule | params | best val | test@best | τ | status |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in arch:
        lines.append(
            f"| {r['run_name']} | `{r['schedule']}` | "
            f"{int(r['n_params']) if r['n_params'] is not None else 'n/a'} | "
            f"{fmt(r['best_val_acc'])} | {fmt(r['test_at_best_val'])} | "
            f"{int(r['tau']) if r['tau'] is not None else 'NA'} | {r['status']} |"
        )
    lines.append("")

    lines.append("## 5. Architecture eRank results")
    lines.append("| run | final eRank | Σ Δ_attn | Σ Δ_mlp | top-1 | top-5 | top-10 | top-20 | same-sum |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in arch:
        lines.append(
            f"| {r['run_name']} | {fmt(r['final_erank'])} | {fmt(r['sum_delta_attn'])} | "
            f"{fmt(r['sum_delta_mlp'])} | {fmt(r['topk1'])} | {fmt(r['topk5'])} | "
            f"{fmt(r['topk10'])} | {fmt(r['topk20'])} | {fmt(r['same_sum_ratio'])} |"
        )
    lines.append("")

    lines.append("## 6. Interpretation of architecture-order results")
    if not arch:
        lines.append("(No runs to interpret yet.)")
    else:
        solved = [r for r in arch if r.get("best_val_acc") and r["best_val_acc"] >= 0.95]
        not_solved = [r for r in arch if not (r.get("best_val_acc") and r["best_val_acc"] >= 0.95)]
        lines.append(f"- **Solved** (val_acc ≥ 0.95): " +
                     (", ".join(f"`{r['schedule']}`" for r in solved) or "none"))
        lines.append(f"- **Did not solve**: " +
                     (", ".join(f"`{r['schedule']}`" for r in not_solved) or "none"))

        # route-vs-process
        def _by(sched_str):
            return next((r for r in arch if r["schedule"].replace("-", ",") == sched_str), None)
        rbp = _by("A,A,M,M"); pbr = _by("M,M,A,A")
        if rbp and pbr:
            lines.append(
                f"- Route-then-process (`AAMM`) val={fmt(rbp.get('best_val_acc'))} vs "
                f"process-then-route (`MMAA`) val={fmt(pbr.get('best_val_acc'))}."
            )
        a_only = _by("A,A,A,A"); m_only = _by("M,M,M,M")
        if a_only:
            lines.append(f"- Attention-only (`AAAA`) val={fmt(a_only.get('best_val_acc'))}.")
        if m_only:
            lines.append(f"- MLP-only (`MMMM`) val={fmt(m_only.get('best_val_acc'))}.")

        if solved:
            comp = [(r["schedule"], r["final_erank"]) for r in solved if r["final_erank"] is not None]
            if comp:
                comp.sort(key=lambda t: t[1])
                lines.append("- Among grokked runs, eRank(R_final) ranking (low → high compression first):")
                for s, v in comp:
                    lines.append(f"    - `{s}`: {v:.3f}")
    lines.append("")

    lines.append("## 7. Frozen probe diagnostics")
    if not probes:
        lines.append("(No probe results.)")
    else:
        models = sorted({p["model_name"] for p in probes})
        lines.append("| model | linear sum | MLP sum | linear r1 | MLP r1 | linear r2 | MLP r2 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for m in models:
            def _acc(t, p):
                return next((r["test_acc"] for r in probes
                             if r["model_name"] == m and r["target"] == t and r["probe_type"] == p), None)
            cells = [
                fmt(_acc("sum", "linear"), ".3f"),
                fmt(_acc("sum", "mlp"),    ".3f"),
                fmt(_acc("r1",  "linear"), ".3f"),
                fmt(_acc("r1",  "mlp"),    ".3f"),
                fmt(_acc("r2",  "linear"), ".3f"),
                fmt(_acc("r2",  "mlp"),    ".3f"),
            ]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Decision tree (per the handoff doc):")
        lines.append("- r1/r2 probes fail → no retrieval information in residual.")
        lines.append("- r1/r2 succeed, sum fails → retrieval present, addition not computed.")
        lines.append("- MLP sum > linear sum → information not linearly accessible.")
        lines.append("- linear sum succeeds → readout bottleneck.")
    lines.append("")

    lines.append("## 8. FFN output-head experiments")
    if not ffn:
        lines.append("(No FFN-head runs.)")
    else:
        lines.append("| run | best val | test@best | best epoch | Σ Δ_attn | Σ Δ_mlp | status |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for r in ffn:
            lines.append(
                f"| {r['run_name']} | {fmt(r['best_val_acc'])} | {fmt(r['test_at_best_val'])} | "
                f"{r['best_val_epoch']} | {fmt(r['sum_delta_attn'])} | "
                f"{fmt(r['sum_delta_mlp'])} | {r['status']} |"
            )
    lines.append("")

    lines.append("## 9. Thesis implications")
    lines.append("(Auto-summary — refine before sharing.)")
    if arch and grokked:
        amam = next((r for r in arch if r["schedule"].replace("-", ",") == "AM,AM,AM,AM"), None)
        if amam and amam.get("best_val_acc") and amam["best_val_acc"] >= 0.95:
            lines.append("- The standard `AMAMAMAM` baseline groks, replicating prior modular-addition results.")
        late_mlp = [r for r in grokked if "M" in r["schedule"].split("-")[-1]]
        if late_mlp:
            lines.append("- At least one grokking architecture has MLP at the final position, consistent with late-MLP computation.")
        attn_first = [r for r in grokked if r["schedule"].split("-")[0] == "A"]
        if attn_first:
            lines.append("- Routing-before-processing variants reach grokking, supporting an `attention-first` interpretation.")
    if probes:
        # Check whether MLP probes substantially exceed linear probes for sum.
        improvements = []
        for m in sorted({p["model_name"] for p in probes}):
            lin = next((p["test_acc"] for p in probes
                        if p["model_name"] == m and p["target"] == "sum" and p["probe_type"] == "linear"), None)
            mlp = next((p["test_acc"] for p in probes
                        if p["model_name"] == m and p["target"] == "sum" and p["probe_type"] == "mlp"), None)
            if lin is not None and mlp is not None:
                improvements.append((m, mlp - lin))
        if improvements:
            big = [m for m, d in improvements if d > 0.2]
            if big:
                lines.append(
                    "- For " + ", ".join(f"`{m}`" for m in big) +
                    ": MLP sum probe materially outperforms linear (Δ > 0.2), suggesting the sum is encoded but not linearly accessible."
                )
    lines.append("")

    lines.append("## 10. Caveats")
    lines.append("- Single seed (seed=0) per architecture; no multi-seed reliability check.")
    lines.append("- Architecture variants are NOT parameter-matched (A-only and M-only blocks are smaller).")
    lines.append("- Architectures use a custom `BlockScheduleTransformer`, not HookedTransformer; same residual-only design but minor numerical differences are possible.")
    lines.append("- `same-sum` ratio uses centroid-based between-class distance; sensitive to outliers.")
    lines.append("- Probes can overfit; we report best-val checkpoint to mitigate but do not run full hyperparameter search.")
    lines.append("- Failed-model eRank statistics describe a model that did not learn the task — they should not be read as compression evidence.")
    lines.append("")

    lines.append("## 11. Recommended next actions")
    lines.append("- Re-run any single-seed grokking with seeds 1, 2 to confirm reliability.")
    lines.append("- For schedules that grok, compute training-time eRank trajectory at finer granularity to localise the compression event.")
    lines.append("- If MLP sum probe » linear sum probe, run a small FFN-head training experiment per checkpoint to test whether a learned non-linear readout closes the gap.")
    lines.append("- For attention-only (`AAAA`) failure (if observed), test whether a wider MLP-only or deeper schedule recovers performance.")
    lines.append("- Consider an attention-only FFN-head hybrid follow-up only if the standard FFN-head run shows promising behavior on the hybrid task.")
    lines.append("")

    lines.append("## Appendix — output locations")
    lines.append(f"- arch matrix: `{arch_root}/`")
    lines.append(f"- probes: `{probe_root}/`")
    lines.append(f"- ffn-head: `{ffn_root}/`")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_root", default="outputs/modadd_arch_matrix_7gpu")
    parser.add_argument("--probe_root", default="outputs/hybrid_probe_diagnostics")
    parser.add_argument("--ffn_root", default="outputs/hybrid_ffn_head")
    parser.add_argument("--out_md", default="outputs/novel_architecture_and_probe_results_for_chatgpt.md")
    parser.add_argument("--out_csv", default="outputs/novel_architecture_and_probe_summary.csv")
    args = parser.parse_args()
    arch_root = os.path.join(PROJECT_ROOT, args.arch_root)
    probe_root = os.path.join(PROJECT_ROOT, args.probe_root)
    ffn_root = os.path.join(PROJECT_ROOT, args.ffn_root)
    out_md = os.path.join(PROJECT_ROOT, args.out_md)
    out_csv = os.path.join(PROJECT_ROOT, args.out_csv)

    arch = collect_arch(arch_root)
    probes = collect_probes(probe_root)
    ffn = collect_ffn(ffn_root)
    write_summary_csv(out_csv, arch, probes, ffn)
    write_report(out_md, arch, probes, ffn, arch_root, probe_root, ffn_root)
    print("Done.")


if __name__ == "__main__":
    main()
