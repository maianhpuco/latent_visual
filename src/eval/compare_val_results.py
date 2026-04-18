"""
CLI utility to compare V* validation results across DIMV-ROI checkpoints.

Usage:
    python -m src.eval.compare_val_results checkpoints_dimv_roi/val_results/

Reads all step_N.json files in the given directory and prints a summary table
with per-step accuracy, plus samples that improved or regressed between steps.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional


def load_all_results(results_dir: str) -> Dict[int, dict]:
    """Load all step_N.json files from results_dir, keyed by step number."""
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if not (fname.startswith("step_") and fname.endswith(".json")):
            continue
        step_str = fname[len("step_"):-len(".json")]
        try:
            step = int(step_str)
        except ValueError:
            continue
        with open(os.path.join(results_dir, fname)) as f:
            results[step] = json.load(f)
    return results


def print_summary_table(results: Dict[int, dict]) -> None:
    """Print a per-step accuracy table sorted by step."""
    print(f"\n{'Step':>8}  {'Accuracy':>10}  {'Correct':>8}  {'Total':>8}")
    print("-" * 44)
    for step in sorted(results.keys()):
        r = results[step]
        acc = r.get("accuracy", 0.0)
        n_correct = r.get("n_correct", 0)
        n_samples = r.get("n_samples", 0)
        print(f"{step:>8}  {acc:>10.4f}  {n_correct:>8}  {n_samples:>8}")
    print()


def analyse_errors(
    results: Dict[int, dict],
    step_a: int,
    step_b: int,
) -> None:
    """
    Compare per-sample correctness between step_a and step_b.
    Reports samples that improved (wrong → correct) and regressed (correct → wrong).
    """
    if step_a not in results or step_b not in results:
        print(f"Steps {step_a} or {step_b} not found in results.", file=sys.stderr)
        return

    by_idx_a = {s["idx"]: s for s in results[step_a].get("per_sample", [])}
    by_idx_b = {s["idx"]: s for s in results[step_b].get("per_sample", [])}

    improved, regressed = [], []
    for idx in sorted(set(by_idx_a) & set(by_idx_b)):
        a_correct = by_idx_a[idx]["correct"]
        b_correct = by_idx_b[idx]["correct"]
        if not a_correct and b_correct:
            improved.append(by_idx_b[idx])
        elif a_correct and not b_correct:
            regressed.append(by_idx_b[idx])

    print(f"\nStep {step_a} → step {step_b}")
    print(f"  Improved  (wrong→correct): {len(improved)}")
    for s in improved[:5]:
        print(f"    idx={s['idx']}  q={s['question'][:60]!r}  pred={s['prediction']!r}")
    print(f"  Regressed (correct→wrong): {len(regressed)}")
    for s in regressed[:5]:
        print(f"    idx={s['idx']}  q={s['question'][:60]!r}  pred={s['prediction']!r}")
    print()


def save_summary_json(results: Dict[int, dict], output_path: str) -> None:
    """Save a compact accuracy-over-steps summary to output_path."""
    summary = [
        {
            "step": step,
            "accuracy": results[step].get("accuracy", 0.0),
            "n_correct": results[step].get("n_correct", 0),
            "n_samples": results[step].get("n_samples", 0),
        }
        for step in sorted(results.keys())
    ]
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare DIMV-ROI V* validation results.")
    parser.add_argument("results_dir", help="Directory containing step_N.json files.")
    parser.add_argument(
        "--compare", nargs=2, type=int, metavar=("STEP_A", "STEP_B"),
        help="Compare per-sample correctness between two steps.",
    )
    parser.add_argument(
        "--save-summary", metavar="OUTPUT_JSON",
        help="Save accuracy-over-steps summary to a JSON file.",
    )
    args = parser.parse_args(argv)

    results = load_all_results(args.results_dir)
    if not results:
        print(f"No step_N.json files found in {args.results_dir!r}.", file=sys.stderr)
        sys.exit(1)

    print_summary_table(results)

    if args.compare:
        analyse_errors(results, args.compare[0], args.compare[1])

    if args.save_summary:
        save_summary_json(results, args.save_summary)


if __name__ == "__main__":
    main()
