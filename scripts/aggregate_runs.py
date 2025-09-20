#!/usr/bin/env python3
"""Aggregate metrics across multiple run directories and report mean ± 95% CI."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials inclusive")
    z = 1.96 if confidence == 0.95 else math.sqrt(2) * math.erf_inv(confidence)
    phat = successes / trials
    denom = 1 + z ** 2 / trials
    centre = phat + z ** 2 / (2 * trials)
    margin = z * math.sqrt((phat * (1 - phat) + z ** 2 / (4 * trials)) / trials)
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def _choose_metrics(summary: Dict[str, Any], prefer: str) -> Dict[str, Any]:
    prefer = prefer.lower()
    if prefer == "raw":
        metrics = summary.get("metrics_raw") or summary.get("metrics")
    elif prefer == "ema":
        metrics = summary.get("metrics_ema")
    else:  # auto
        metrics = summary.get("metrics_raw") or summary.get("metrics")
        if metrics is None:
            metrics = summary.get("metrics_ema")
    if metrics is None:
        available = [k for k in summary.keys() if k.startswith("metrics")]
        raise ValueError(f"No metrics matching preference '{prefer}' (available: {available})")
    return metrics


def load_summary(run_dir: Path, prefer: str = "raw") -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    with summary_path.open() as f:
        summary = json.load(f)
    metrics = _choose_metrics(summary, prefer)
    metrics.setdefault("episodes", metrics.get("episodes", 0))
    metrics.setdefault("wins", metrics.get("wins", 0))
    metrics.setdefault("win_rate", metrics.get("win_rate", 0.0))
    metrics.setdefault("avg_steps", metrics.get("avg_steps", float("nan")))
    metrics.setdefault("guesses_per_episode", metrics.get("guesses_per_episode", float("nan")))
    metrics.setdefault("guess_success_rate", metrics.get("guess_success_rate", float("nan")))
    metrics.setdefault("belief_auroc", metrics.get("belief_auroc", float("nan")))
    metrics.setdefault("belief_ece", metrics.get("belief_ece", float("nan")))
    return metrics


def weighted_average(values: Iterable[Tuple[float, float]]) -> float:
    total_weight = 0.0
    total_value = 0.0
    for value, weight in values:
        if not math.isfinite(value) or weight <= 0:
            continue
        total_value += value * weight
        total_weight += weight
    if total_weight == 0:
        return float("nan")
    return total_value / total_weight


def mean_and_ci(values: Iterable[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(var / n)
    ci = 1.96 * se
    return mean, ci


def aggregate_runs(run_dirs: Iterable[Path], prefer: str = "raw") -> Dict[str, Any]:
    metrics_list = [load_summary(Path(run), prefer=prefer) for run in run_dirs]
    total_episodes = sum(int(m.get("episodes", 0)) for m in metrics_list)
    total_wins = sum(int(m.get("wins", 0)) for m in metrics_list)
    win_rate = total_wins / total_episodes if total_episodes else float("nan")
    ci_low, ci_high = wilson_interval(total_wins, total_episodes) if total_episodes else (float("nan"), float("nan"))

    avg_steps = weighted_average((m.get("avg_steps", float("nan")), m.get("episodes", 0)) for m in metrics_list)
    guesses_ep = weighted_average((m.get("guesses_per_episode", float("nan")), m.get("episodes", 0)) for m in metrics_list)
    guess_success = weighted_average((m.get("guess_success_rate", float("nan")), m.get("total_guess_attempts", 0)) for m in metrics_list)
    auroc_mean, auroc_ci = mean_and_ci(m.get("belief_auroc", float("nan")) for m in metrics_list)
    ece_mean, ece_ci = mean_and_ci(m.get("belief_ece", float("nan")) for m in metrics_list)

    return {
        "runs": [str(Path(run)) for run in run_dirs],
        "episodes": total_episodes,
        "wins": total_wins,
        "win_rate": win_rate,
        "win_rate_ci": [ci_low, ci_high],
        "avg_steps": avg_steps,
        "guesses_per_episode": guesses_ep,
        "guess_success_rate": guess_success,
        "belief_auroc_mean": auroc_mean,
        "belief_auroc_ci": auroc_ci,
        "belief_ece_mean": ece_mean,
        "belief_ece_ci": ece_ci,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Minesweeper run metrics")
    parser.add_argument("runs", nargs="+", help="Paths to run directories (one per seed)")
    parser.add_argument("--json", action="store_true", help="Emit JSON only (suitable for scripting)")
    parser.add_argument(
        "--metrics",
        choices=["raw", "ema", "auto"],
        default="raw",
        help="Which summary metrics to use (default: raw)",
    )
    args = parser.parse_args()

    result = aggregate_runs(args.runs, prefer=args.metrics)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("Aggregated results for:")
    for run in result["runs"]:
        print(f"  - {run}")
    print(f"Total episodes: {result['episodes']}  |  Wins: {result['wins']}")
    ci_low, ci_high = result["win_rate_ci"]
    print(f"Win rate: {result['win_rate']*100:.2f}% (95% CI: {ci_low*100:.2f}%, {ci_high*100:.2f}%)")
    print(f"Avg steps: {result['avg_steps']:.3f}")
    print(f"Guesses/episode: {result['guesses_per_episode']:.3f}")
    print(f"Guess success rate: {result['guess_success_rate']:.3f}")
    print(
        f"Belief AUROC: {result['belief_auroc_mean']:.4f} ± {result['belief_auroc_ci']:.4f}"
    )
    print(
        f"Belief ECE: {result['belief_ece_mean']:.4f} ± {result['belief_ece_ci']:.4f}"
    )


if __name__ == "__main__":
    main()
