"""Publication figures from frozen validation/test summaries, without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("outputs/models"))
    parser.add_argument("--evaluation", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--output", type=Path, default=Path("reports/2026-10-03"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((args.evaluation / "selection.json").read_text(encoding="utf-8"))
    test = json.loads((args.evaluation / "final_test.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(11, 5))
    for run in sorted(args.models.glob("*/history.jsonl")):
        records = [json.loads(line) for line in run.read_text(encoding="utf-8").splitlines()]
        epochs = [r["epoch"] for r in records]
        values = np.array([r["early_validation_loss"] for r in records])
        ax.plot(epochs, values / values[0], marker="o", markersize=3,
                label=run.parent.name.replace("_seed20261003", ""))
    ax.axhline(1, color="#777777", lw=.7)
    ax.set(xlabel="Epoch", ylabel="Early-validation loss / first-epoch loss",
           title="Validation loss by objective and architecture")
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(args.output / "training_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, rec in selection["candidates"].items():
        if name not in test["candidates"]:
            continue
        x = rec["predictive"]["calibrated_execution_correlation_selection"]
        y = test["candidates"][name]["calibrated_execution_correlation"]
        ax.scatter(x, y, s=60 if name == selection["predictive_leader"] else 28)
        label = name.replace("_seed20261003", "")
        offset = (-4, 4) if x > .06 else (4, 4)
        alignment = "right" if x > .06 else "left"
        ax.annotate(label, (x, y), xytext=offset, ha=alignment,
                    textcoords="offset points", fontsize=7)
    ax.axhline(0, color="#777777", lw=.8)
    ax.axvline(0, color="#777777", lw=.8)
    ax.set_xlim(-.01, .085)
    ax.set_ylim(-.052, .027)
    ax.set(xlabel="Selection-period correlation", ylabel="Test-period correlation",
           title="Predicted vs executable 4h return")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "validation_vs_test.png", dpi=180)
    plt.close(fig)

    leader = selection["trading_winner"]
    sensitivity = test["candidates"][leader]["cost_sensitivity"]
    costs = sorted((float(k), v["mean_sharpe"]) for k, v in sensitivity.items())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([10000 * cost for cost, _ in costs], [sharpe for _, sharpe in costs], marker="o")
    ax.axhline(0, color="#777777", lw=.8)
    ax.set(xlabel="One-way assumed cost (bp)", ylabel="Mean four-phase test Sharpe",
           title="Frozen validation winner: cost sensitivity")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "cost_sensitivity.png", dpi=180)
    plt.close(fig)

    monthly = test["candidates"][leader]["execution_correlation_by_month"]
    fig, ax = plt.subplots(figsize=(8, 4))
    months = list(monthly)
    values = [monthly[month] for month in months]
    ax.bar(months, values, color=["#4477aa" if value >= 0 else "#bb5544" for value in values])
    ax.axhline(0, color="#444444", lw=.8)
    ax.set(xlabel="Test month (UTC)", ylabel="Score vs executable return correlation",
           title="Frozen validation winner: monthly prediction stability")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / "monthly_correlation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
