"""
Block World Solver — Automated Experiment Script (Sections ה2 & ה3)
====================================================================
Generates 12 valid (start, goal) pairs of varying sizes using a documented
random seed, evaluates A* and IDA* across all 3 heuristics, computes mean and
standard deviation for all required metrics, verifies cost equality between
A* and IDA*, and exports results to CSV files.
"""

from __future__ import annotations

import csv
import os
import random
import statistics
import sys
from typing import Dict, List, Tuple

# Ensure solver.py can be imported whether run from root or from src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from solver import (
    SearchResult,
    State,
    get_algorithms,
    get_heuristics,
    make_state,
    run_search,
    state_to_sorted_lists,
    validate_pair,
)

# ---------------------------------------------------------------------------
# Experiment Configuration (Documented Seed & Problem Sizes)
# ---------------------------------------------------------------------------

SEED = 42
BLOCK_POOL = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# 12 problem instances with varying sizes within the allowed range (1 <= n <= 10)
INSTANCE_SIZES = [2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9]


# ---------------------------------------------------------------------------
# Random Valid Instance Generation
# ---------------------------------------------------------------------------

def _random_stacks(rng: random.Random, blocks: List[str]) -> List[List[str]]:
    """Partition a shuffled copy of `blocks` into k non-empty stacks."""
    shuffled = list(blocks)
    rng.shuffle(shuffled)
    n = len(shuffled)
    num_stacks = rng.randint(1, n)
    if num_stacks == 1:
        return [shuffled]

    split_points = sorted(rng.sample(range(1, n), num_stacks - 1))
    stacks: List[List[str]] = []
    prev = 0
    for idx in split_points:
        stacks.append(shuffled[prev:idx])
        prev = idx
    stacks.append(shuffled[prev:])
    return stacks


def generate_test_pairs(
    seed: int = SEED, sizes: List[int] = INSTANCE_SIZES
) -> List[Tuple[int, int, State, State]]:
    """
    Generate valid (start, goal) pairs with a fixed random seed.
    Returns a list of tuples: (instance_id, num_blocks, start_state, goal_state).
    """
    rng = random.Random(seed)
    pairs: List[Tuple[int, int, State, State]] = []

    for idx, n in enumerate(sizes, start=1):
        blocks = BLOCK_POOL[:n]
        while True:
            start_stacks = _random_stacks(rng, blocks)
            goal_stacks = _random_stacks(rng, blocks)
            if make_state(start_stacks) != make_state(goal_stacks):
                break

        start_state, goal_state = validate_pair(start_stacks, goal_stacks)
        pairs.append((idx, n, start_state, goal_state))

    return pairs


def format_state(state: State) -> str:
    """Format canonical state into a compact readable string."""
    return " | ".join("-".join(stack) for stack in state_to_sorted_lists(state))


# ---------------------------------------------------------------------------
# Statistics Helpers
# ---------------------------------------------------------------------------

def mean_and_std(values: List[float]) -> Tuple[float, float]:
    """Return (mean, sample_std). Returns (0.0, 0.0) if list is empty."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.mean(values), statistics.stdev(values)


# ---------------------------------------------------------------------------
# Main Experiment Runner
# ---------------------------------------------------------------------------

def run_experiments() -> None:
    print("=" * 80)
    print(f"ניסוי השוואתי בעולם הקוביות (סעיפים ה2 ו-ה3) | Seed = {SEED}")
    print("=" * 80)

    pairs = generate_test_pairs(SEED, INSTANCE_SIZES)
    algorithms = get_algorithms()   # e.g. ["IDA*", "A*"]
    heuristics = get_heuristics()   # ["h=0 (ללא היוריסטיקה)", "h1", "h2"]

    # Order algorithms cleanly as A* then IDA* if both exist
    ordered_algos = [a for a in ["A*", "IDA*"] if a in algorithms]
    for a in algorithms:
        if a not in ordered_algos:
            ordered_algos.append(a)

    # Print generated instances
    print("\n--- 12 זוגות הבדיקה שנוצרו ---")
    for inst_id, n, start, goal in pairs:
        print(
            f"מקרה {inst_id:2d} (n={n}): "
            f"התחלה: [{format_state(start)}]  -->  יעד: [{format_state(goal)}]"
        )

    # Store all runs: results[(algo, heur)] = list of dicts
    results: Dict[Tuple[str, str], List[Dict]] = {
        (algo, heur): [] for algo in ordered_algos for heur in heuristics
    }

    print("\n--- מריץ חיפושים (אנא המתן, ריצות ארוכות נעצרות במגבלת הזמן) ---")
    for inst_id, n, start, goal in pairs:
        print(f"\n[מקרה {inst_id:2d}/{len(pairs)} | n={n}]")
        for algo in ordered_algos:
            for heur in heuristics:
                res: SearchResult = run_search(algo, heur, start, goal)
                timed_out = (not res.success) and ("מגבלה" in res.error_message)
                status_str = (
                    f"הצלחה (עלות={res.cost})"
                    if res.success
                    else f"נכשל ({res.error_message})"
                )
                print(
                    f"  {algo:<5} | {heur:<22} -> {status_str:<26} | "
                    f"זמן: {res.metrics.runtime_seconds:.4f}s | "
                    f"פיתוחים: {res.metrics.states_expanded}"
                )

                results[(algo, heur)].append({
                    "instance_id": inst_id,
                    "n": n,
                    "start": format_state(start),
                    "goal": format_state(goal),
                    "algorithm": algo,
                    "heuristic": heur,
                    "success": res.success,
                    "timed_out": timed_out,
                    "error_message": res.error_message,
                    "cost": res.cost if res.success else None,
                    "runtime_seconds": res.metrics.runtime_seconds,
                    "states_expanded": res.metrics.states_expanded,
                    "states_generated": res.metrics.states_generated,
                    "max_queue_size": res.metrics.max_queue_size,
                    "ida_iterations": res.metrics.ida_iterations,
                })

    # -----------------------------------------------------------------------
    # Section ה2: Summary Table (Means & Standard Deviations)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("סיכום תוצאות (סעיף ה2) — ממוצע ± סטיית תקן")
    print("=" * 115)
    header = (
        f"{'אלגוריתם':<8} | {'היוריסטיקה':<22} | {'הצלחות':<8} | {'מגבלת זמן':<10} | "
        f"{'זמן ריצה (שנ׳)':<20} | {'פיתוחים':<22} | {'עלות פתרון (בהצלחה)':<22}"
    )
    print(header)
    print("-" * 115)

    summary_rows: List[Dict] = []

    for algo in ordered_algos:
        for heur in heuristics:
            records = results[(algo, heur)]
            total = len(records)
            successes = sum(1 for r in records if r["success"])
            timeouts = sum(1 for r in records if r["timed_out"])

            runtimes = [r["runtime_seconds"] for r in records]
            expansions = [float(r["states_expanded"]) for r in records]
            solved_costs = [float(r["cost"]) for r in records if r["success"] and r["cost"] is not None]

            rt_mean, rt_std = mean_and_std(runtimes)
            exp_mean, exp_std = mean_and_std(expansions)
            cost_mean, cost_std = mean_and_std(solved_costs)

            cost_display = (
                f"{cost_mean:.2f} ± {cost_std:.2f} (n={len(solved_costs)})"
                if solved_costs
                else "אין פתרונות"
            )

            print(
                f"{algo:<8} | {heur:<22} | {f'{successes}/{total}':<8} | {f'{timeouts}/{total}':<10} | "
                f"{f'{rt_mean:.4f} ± {rt_std:.4f}':<20} | "
                f"{f'{exp_mean:.1f} ± {exp_std:.1f}':<22} | "
                f"{cost_display:<22}"
            )

            summary_rows.append({
                "seed": SEED,
                "algorithm": algo,
                "heuristic": heur,
                "total_instances": total,
                "success_count": successes,
                "timeout_count": timeouts,
                "runtime_mean": round(rt_mean, 6),
                "runtime_std": round(rt_std, 6),
                "expansions_mean": round(exp_mean, 2),
                "expansions_std": round(exp_std, 2),
                "solved_cost_mean": round(cost_mean, 4) if solved_costs else "",
                "solved_cost_std": round(cost_std, 4) if solved_costs else "",
                "solved_cost_count": len(solved_costs),
            })

    # -----------------------------------------------------------------------
    # Section ה3: Verify Cost Equality Between A* and IDA*
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("אימות שוויון עלויות פתרון בין A* ל-IDA* (סעיף ה3)")
    print("=" * 80)

    if "A*" in results and "IDA*" in results:
        comparisons = 0
        mismatches = 0
        for heur in heuristics:
            astar_recs = {r["instance_id"]: r for r in results[("A*", heur)]}
            idastar_recs = {r["instance_id"]: r for r in results[("IDA*", heur)]}

            for inst_id in astar_recs:
                r_a = astar_recs[inst_id]
                r_i = idastar_recs[inst_id]
                if r_a["success"] and r_i["success"]:
                    comparisons += 1
                    if r_a["cost"] != r_i["cost"]:
                        mismatches += 1
                        print(
                            f"[שגיאה] מקרה {inst_id} ({heur}): "
                            f"עלות A*={r_a['cost']} != עלות IDA*={r_i['cost']}"
                        )

        if mismatches == 0:
            print(
                f"אומת בהצלחה! בכל {comparisons} ההשוואות שבהן שני האלגוריתמים "
                f"סיימו בהצלחה, התקבלה עלות פתרון זהה לחלוטין."
            )
        else:
            print(f"נמצאו {mismatches} אי-התאמות בעלות הפתרון!")

    # -----------------------------------------------------------------------
    # Export Results to CSV
    # -----------------------------------------------------------------------
    raw_csv_path = os.path.join(CURRENT_DIR, "experiment_raw_results.csv")
    summary_csv_path = os.path.join(CURRENT_DIR, "experiment_summary.csv")

    all_raw_rows = [
        row
        for algo in ordered_algos
        for heur in heuristics
        for row in results[(algo, heur)]
    ]

    if all_raw_rows:
        with open(raw_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_raw_rows)

    if summary_rows:
        with open(summary_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\nהתוצאות נשמרו בהצלחה לקבצים:\n  1. {raw_csv_path}\n  2. {summary_csv_path}")


if __name__ == "__main__":
    run_experiments()