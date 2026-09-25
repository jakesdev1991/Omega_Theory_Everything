# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""Benchmark: does RCOD governance improve loss recovery after a label-noise shock?

This is the empirical test the Omega $OMEGA whitepaper asks for ("test
whether the proposed geometry-informed metrics improve ... outcomes
compared with simpler baselines"), applied to the RCOD optimizer governor.

Teacher-student synthetic task: a fixed teacher MLP labels Gaussian
inputs; a student MLP trains with AdamW on mini-batches. During a fixed
step window, a fraction of each batch's labels is corrupted (the shock);
afterwards labels are clean again. Methods compared:

- ``baseline``        plain AdamW
- ``rcod-default``    RCOD wrapper with the design-document thresholds
                      (FLOW 0.15 / SHOCK 0.70 / dissonance 0.35)
- ``rcod-calibrated`` RCOD wrapper whose thresholds are set from
                      clean-phase quantiles of the observed metrics, so
                      the mechanism actually binds at this scale

Metrics: pre-noise best test loss, peak test loss inside the noise
window, recovery steps after the noise window, final test loss/accuracy,
and the RCOD regime/telemetry counters. The benchmark is honest: if RCOD
does not help, the table says so.

Requires: torch (CPU is sufficient), numpy. Not in requirements.txt by
default because torch is a heavy optional dependency::

    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import argparse
import csv
import statistics
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RCOD label-noise recovery benchmark (teacher-student)"
    )
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds")
    parser.add_argument("--steps", type=int, default=3000, help="total steps")
    parser.add_argument(
        "--noise-start", type=int, default=1200, help="first noisy step"
    )
    parser.add_argument(
        "--noise-length", type=int, default=500, help="number of noisy steps"
    )
    parser.add_argument(
        "--flip-prob", type=float, default=0.5, help="label flip probability"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW lr")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "baseline",
            "rcod-default",
            "rcod-updates-default",
            "rcod-updates-calibrated",
        ],
        choices=[
            "baseline",
            "rcod-default",
            "rcod-updates-default",
            "rcod-updates-calibrated",
        ],
    )
    parser.add_argument("--csv", type=str, default="", help="optional CSV output path")
    return parser.parse_args(argv)


def _build_mlp(torch: Any, seed: int, dims: list[int], activation: str) -> Any:
    torch.manual_seed(seed)
    layers: list[Any] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if activation == "relu":
                layers.append(torch.nn.ReLU())
            else:
                layers.append(torch.nn.Tanh())
    model = torch.nn.Sequential(*layers)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def _evaluate(torch: Any, model: Any, x: Any, y: Any) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    model.train()
    return loss, acc


def run_experiment(
    args: argparse.Namespace, method: str, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from rcod_optimizer import RCODMultiScaleOptimizer

    dim_in, classes = 32, 4
    teacher = _build_mlp(torch, 10_000 + seed, [dim_in, 64, classes], "tanh")
    student = _build_mlp(torch, 20_000 + seed, [dim_in, 128, 128, classes], "relu")

    data_gen = torch.Generator().manual_seed(30_000 + seed)
    x_train = torch.randn(args.n_train, dim_in, generator=data_gen)
    x_test = torch.randn(args.n_test, dim_in, generator=data_gen)
    with torch.no_grad():
        y_train = teacher(x_train).argmax(dim=1)
        y_test = teacher(x_test).argmax(dim=1)

    inner = torch.optim.AdamW(student.parameters(), lr=args.lr)
    rcod: RCODMultiScaleOptimizer | None = None
    if method == "rcod-default":
        rcod = RCODMultiScaleOptimizer(inner)
    elif method == "rcod-updates-default":
        rcod = RCODMultiScaleOptimizer(inner, state_mode="updates")
    elif method == "rcod-updates-calibrated":
        rcod = RCODMultiScaleOptimizer(inner, state_mode="updates", auto_calibrate=True)

    batch_gen = torch.Generator().manual_seed(40_000 + seed)
    noise_end = args.noise_start + args.noise_length
    trajectory: list[dict[str, Any]] = []
    mu_clean: list[float] = []
    mu_noise: list[float] = []
    shocks_clean = 0
    shocks_noise = 0
    shocks_post = 0

    for step in range(args.steps):
        idx = torch.randint(0, args.n_train, (args.batch_size,), generator=batch_gen)
        x_batch = x_train[idx]
        y_batch = y_train[idx]

        if args.noise_start <= step < noise_end and args.flip_prob > 0:
            flip = torch.rand(args.batch_size, generator=batch_gen)
            wrong = torch.randint(1, classes, (args.batch_size,), generator=batch_gen)
            y_batch = torch.where(
                flip < args.flip_prob,
                (y_batch + wrong) % classes,
                y_batch,
            )

        logits = student(x_batch)
        loss = torch.nn.functional.cross_entropy(logits, y_batch)
        if rcod is not None:
            rcod.zero_grad()
        else:
            inner.zero_grad()
        loss.backward()
        if rcod is not None:
            _, telemetry = rcod.step()
            if telemetry["regime"] == "SHOCK":
                if step < args.noise_start:
                    shocks_clean += 1
                elif step < noise_end:
                    shocks_noise += 1
                else:
                    shocks_post += 1
            if telemetry["regime"] not in ("WARMUP", "CALIBRATION"):
                if step < args.noise_start:
                    mu_clean.append(telemetry["mu_bar"])
                elif step < noise_end:
                    mu_noise.append(telemetry["mu_bar"])
        else:
            inner.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            test_loss, test_acc = _evaluate(torch, student, x_test, y_test)
            trajectory.append(
                {
                    "method": method,
                    "seed": seed,
                    "step": step,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "in_noise": int(args.noise_start <= step < noise_end),
                }
            )

    pre_best = min(
        (row["test_loss"] for row in trajectory if row["step"] < args.noise_start),
        default=float("nan"),
    )
    peak_noise = max(
        (row["test_loss"] for row in trajectory if row["in_noise"] == 1),
        default=float("nan"),
    )
    recovery: int | None = None
    for row in trajectory:
        if row["step"] >= noise_end and row["test_loss"] <= 1.10 * pre_best:
            recovery = row["step"] - noise_end
            break
    final = trajectory[-1]

    result: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "pre_best": pre_best,
        "peak_noise": peak_noise,
        "damage": peak_noise - pre_best,
        "recovery": recovery,
        "final_loss": final["test_loss"],
        "final_acc": final["test_acc"],
        "mu_clean_mean": statistics.fmean(mu_clean) if mu_clean else float("nan"),
        "mu_noise_mean": statistics.fmean(mu_noise) if mu_noise else float("nan"),
    }
    if rcod is not None:
        result["regime_counts"] = dict(rcod.regime_counts)
        result["reversals"] = rcod.reversal_count
        result["shocks_clean"] = shocks_clean
        result["shocks_noise"] = shocks_noise
        result["shocks_post"] = shocks_post
        result["mean_alpha"] = (
            statistics.fmean(rcod.alpha_history) if rcod.alpha_history else 0.0
        )
    return result, trajectory


def _mean_range(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.fmean(values):.4f} [{min(values):.4f}, "
    f"{max(values):.4f}]"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    all_results: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for method in args.methods:
        for seed in range(args.seeds):
            result, trajectory = run_experiment(args, method, seed)
            all_results.append(result)
            all_rows.extend(trajectory)
            rec = result["recovery"]
            rec_str = "not recovered" if rec is None else f"{rec} steps"
            print(
                f"[{method} seed={seed}] pre_best="
                f"{result['pre_best']:.4f} peak_noise="
                f"{result['peak_noise']:.4f} damage="
                f"{result['damage']:+.4f} recovery={rec_str} "
                f"final={result['final_loss']:.4f}/"
                f"{result['final_acc'] * 100:.1f}%"
            )

    print("\n=== Aggregate over seeds (mean [min, max]) ===")
    header = (
        f"{'method':<17} {'pre-best':<24} {'peak-in-noise':<24} "
        f"{'damage':<24} {'recovery (steps)':<22} {'final loss':<24} "
        f"{'final acc':<18}"
    )
    print(header)
    print("-" * len(header))
    for method in args.methods:
        rows = [r for r in all_results if r["method"] == method]
        pre = [r["pre_best"] for r in rows]
        peak = [r["peak_noise"] for r in rows]
        dmg = [r["damage"] for r in rows]
        rec = [r["recovery"] for r in rows if r["recovery"] is not None]
        fin = [r["final_loss"] for r in rows]
        acc = [r["final_acc"] for r in rows]
        rec_str = (
            _mean_range([float(v) for v in rec])
            if len(rec) == len(rows)
            else f"{len(rec)}/{len(rows)} recovered"
        )
        print(
            f"{method:<17} {_mean_range(pre):<24} {_mean_range(peak):<24} "
            f"{_mean_range(dmg):<24} {rec_str:<22} {_mean_range(fin):<24} "
            f"{_mean_range(acc):<18}"
        )

    print("\n=== RCOD telemetry ===")
    for method in args.methods:
        rows = [r for r in all_results if r["method"] == method]
        rcod_rows = [r for r in rows if "regime_counts" in r]
        if not rcod_rows:
            continue
        counts: dict[str, int] = {}
        for r in rcod_rows:
            for k, v in r["regime_counts"].items():
                counts[k] = counts.get(k, 0) + v
        reversals = sum(r["reversals"] for r in rcod_rows)
        alphas = [r["mean_alpha"] for r in rcod_rows if r["reversals"] > 0]
        mu_c = _mean_range([r["mu_clean_mean"] for r in rcod_rows])
        mu_n = _mean_range([r["mu_noise_mean"] for r in rcod_rows])
        s_clean = sum(r["shocks_clean"] for r in rcod_rows)
        s_noise = sum(r["shocks_noise"] for r in rcod_rows)
        s_post = sum(r["shocks_post"] for r in rcod_rows)
        print(
            f"{method}: regimes={counts} reversals={reversals} "
            f"mean_alpha={_mean_range(alphas) if alphas else 'n/a'} "
            f"mu_bar(clean)={mu_c} mu_bar(noise)={mu_n}"
        )
        print(
            f"    shocks: clean-phase={s_clean} noise-window={s_noise} "
            f"post-noise={s_post}"
        )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nTrajectory CSV written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
