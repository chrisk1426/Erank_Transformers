"""
scripts/generate_arch_order_manifest.py

Generate schedule_manifest.csv for the balanced A/M architecture-order sweep:
  - depth 2: C(4,2) = 6 balanced schedules (2 A + 2 M)
  - depth 3: C(6,3) = 20 balanced schedules (3 A + 3 M)
  - depth 4: C(8,4) = 70 balanced schedules (4 A + 4 M)
  - + 6 optional unbalanced controls (all-A, all-M for each depth)

Writes:
    outputs/arch_order_sweep_6gpu/schedule_manifest.csv

with columns:
    run_id, depth, num_blocks, schedule, num_attention, num_mlp,
    is_balanced, is_control, first_block, last_block,
    num_alternations, a_before_m_score
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def generate_balanced_schedules(depth: int) -> list[str]:
    """All unique sequences with exactly `depth` A's and `depth` M's, in lexical order."""
    n = 2 * depth
    seen: set[str] = set()
    out: list[str] = []
    for a_positions in combinations(range(n), depth):
        s = ["M"] * n
        for i in a_positions:
            s[i] = "A"
        joined = "".join(s)
        if joined not in seen:
            seen.add(joined)
            out.append(joined)
    return out


def a_before_m_score(schedule: str) -> float:
    """
    Average over (A,M) block pairs of the indicator [i_A < i_M].
    1.0 = all A's before all M's; 0.0 = all M's before all A's.
    Defined only when num_A == num_M; for unbalanced returns the same formula
    normalised by num_A*num_M (NA-safe).
    """
    a_idx = [i for i, c in enumerate(schedule) if c == "A"]
    m_idx = [i for i, c in enumerate(schedule) if c == "M"]
    if not a_idx or not m_idx:
        return float("nan")
    total = len(a_idx) * len(m_idx)
    hits = sum(1 for i in a_idx for j in m_idx if i < j)
    return hits / total


def num_alternations(schedule: str) -> int:
    return sum(1 for i in range(1, len(schedule)) if schedule[i] != schedule[i - 1])


def is_balanced(schedule: str) -> bool:
    return schedule.count("A") == schedule.count("M")


def row_for_schedule(schedule: str, depth: int, *, is_control: bool) -> dict:
    nA = schedule.count("A")
    nM = schedule.count("M")
    prefix = "control" if is_control else "bal"
    run_id = f"depth{depth}_{prefix}_{schedule}"
    return {
        "run_id": run_id,
        "depth": depth,
        "num_blocks": len(schedule),
        "schedule": schedule,
        "num_attention": nA,
        "num_mlp": nM,
        "is_balanced": int(is_balanced(schedule)),
        "is_control": int(is_control),
        "first_block": schedule[0],
        "last_block": schedule[-1],
        "num_alternations": num_alternations(schedule),
        "a_before_m_score": a_before_m_score(schedule),
    }


def build_manifest(include_controls: bool, depths: list[int]) -> list[dict]:
    rows: list[dict] = []
    for d in depths:
        for s in generate_balanced_schedules(d):
            rows.append(row_for_schedule(s, d, is_control=False))
    if include_controls:
        for d in depths:
            n = 2 * d
            rows.append(row_for_schedule("A" * n, d, is_control=True))
            rows.append(row_for_schedule("M" * n, d, is_control=True))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/arch_order_sweep_6gpu")
    parser.add_argument("--no_controls", action="store_true",
                        help="Omit the 6 unbalanced all-A / all-M controls.")
    parser.add_argument("--depths", default="2,3,4",
                        help="Comma-separated list of depths to generate (default: 2,3,4).")
    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    rows = build_manifest(include_controls=not args.no_controls, depths=depths)

    out_dir = os.path.join(PROJECT_ROOT, args.output_root)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "schedule_manifest.csv")

    fieldnames = [
        "run_id", "depth", "num_blocks", "schedule",
        "num_attention", "num_mlp", "is_balanced", "is_control",
        "first_block", "last_block", "num_alternations", "a_before_m_score",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    by_depth_balanced: dict[int, int] = {}
    by_depth_control: dict[int, int] = {}
    for r in rows:
        if r["is_control"]:
            by_depth_control[r["depth"]] = by_depth_control.get(r["depth"], 0) + 1
        else:
            by_depth_balanced[r["depth"]] = by_depth_balanced.get(r["depth"], 0) + 1

    print(f"Wrote {out_csv}")
    print(f"Total schedules: {len(rows)}")
    for d in depths:
        b = by_depth_balanced.get(d, 0)
        c = by_depth_control.get(d, 0)
        print(f"  depth {d}: {b} balanced + {c} controls = {b + c}")


if __name__ == "__main__":
    main()
