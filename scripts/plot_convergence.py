#!/usr/bin/env python3
"""Plot measured validation progress and the learning-rate schedule."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / "validation.jsonl").read_text().splitlines()]
    best = json.loads((args.run / "best.json").read_text())
    steps = [row["step"] for row in rows]
    fig, (score, rate) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    for key, label, color in (("score", "Selection score", "#177e89"),
                              ("training_velocity_ratio", "Training velocity error", "#9c89b8"),
                              ("validation_velocity_ratio", "Validation velocity error", "#f4a261"),
                              ("validation_endpoint_ratio", "Validation endpoint error", "#e76f51")):
        score.plot(steps, [r[key] for r in rows], label=label, color=color, linewidth=1.7)
    score.axvline(best["step"], color="#444444", linestyle="--", label=f"Selected step {best['step']}")
    score.set(ylabel="Residual relative to the base model's teacher gap",
              title="Supra final boss — validation convergence")
    score.legend(fontsize=9)
    score.grid(alpha=0.2)
    rate.step(steps, [row["lr"] for row in rows], where="post", color="#177e89")
    rate.set(yscale="log", xlabel="Total optimizer updates", ylabel="Learning rate")
    rate.grid(alpha=0.2)
    fig.savefig(args.run / "convergence.png", dpi=180)
    fig.savefig(args.run / "convergence.svg")
    print(args.run / "convergence.png")


if __name__ == "__main__":
    main()
