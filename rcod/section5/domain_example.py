# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Run the pre-registered protocol on a series that is not the §5 series.

`experiment.py` is the frozen archival instance of the experiment (synthetic
AR(1) cosmology-side shocks). This file is the portable entry point: the same
protocol from `../harness.py`, driven either by a synthetic family that is
deliberately unlike the AR(1) one, or by your own CSV.

    # synthetic non-AR(1) family: drifting baseline, variance ramps, step faults
    python -m rcod.section5.domain_example

    # your own data: one value column, one onset column (0/1 flags or indices)
    python -m rcod.section5.domain_example --csv my_telemetry.csv --value sensor --onset fault

What it demonstrates is the part that transfers: metrics, arms, split, metric
choice, paired test and kill criterion are all reusable; only the data source
and the feature extractors change. What it does *not* claim is that the RCOD
arms win anywhere -- on the §5 family they did not, and a harness that can only
report wins is a marketing device, not an instrument.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rcod.harness import Arm, Protocol, Series, render_report, run_benchmark  # noqa: E402
from rcod.section5.swarm import swarm_metrics  # noqa: E402

T = 24_000
LAGS = (0, 5, 10, 20)
MACRO_W = 252


def lagged_stack(columns, idx, lags=LAGS):
    return np.column_stack([c[idx - lag] for c in columns for lag in lags])


def telemetry_family(seed: int, n_faults: int = 8) -> Series:
    """A family with no relation to the §5 generator.

    A slow random-walk baseline with a diurnal component and multiplicative
    noise, interrupted by `n_faults` step faults. Each fault is preceded by a
    120-step variance ramp -- the same *phenomenology* the §5 design needed
    (a precursor), on different mechanics (drift + heteroscedastic noise rather
    than AR(1) shocks), so the comparison tests the metrics rather than the
    generator.
    """
    rng = np.random.default_rng(10_000 + seed)
    t = np.arange(T)
    drift = np.cumsum(rng.standard_normal(T)) * 0.02
    daily = 0.6 * np.sin(2 * np.pi * t / 1_440)
    vol = 1.0 + 0.25 * np.sin(2 * np.pi * t / 7_200 + rng.uniform(0, 6.28))
    x = rng.standard_normal(T) * vol + drift + daily

    onsets: list[int] = []
    cursor = 2_000 + int(rng.integers(0, 600))
    ramp = 120
    while len(onsets) < n_faults and cursor + ramp + 500 < T - 1_000:
        # precursor: volatility climbs before the break
        x[cursor : cursor + ramp] *= np.linspace(1.0, 2.5, ramp)
        magnitude = rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 4.0)
        x[cursor + ramp : cursor + ramp + 500] += magnitude
        onsets.append(cursor + ramp)
        cursor += ramp + 500 + 1_500 + int(rng.integers(0, 1_200))
    return Series(values=x, onsets=np.array(onsets), name=f"telemetry-{seed}")


def rcod_trio(series: Series, warmup: int, normalize: str = "zscore"):
    m = swarm_metrics(
        series.values, macro_w=MACRO_W, seed=0, normalize=normalize, backend="prefix"
    )
    idx = np.arange(warmup, len(series) - 30)
    feats = lagged_stack([m["median_mu"], m["macro_mu"], m["dissonance"]], idx)
    return ("median_mu", "macro_mu", "dissonance"), feats


def rolling_stats(series: Series, warmup: int, window: int = 60):
    """The simplest credible baseline: the one a claim has to beat."""
    x = series.values
    n = len(x)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    ac1 = np.full(n, np.nan)
    maxd = np.full(n, np.nan)
    for t in range(max(warmup - max(LAGS), window), n - 30):
        s = x[t - window + 1 : t + 1]
        mean[t] = s.mean()
        std[t] = s.std()
        ac1[t] = np.corrcoef(s[:-1], s[1:])[0, 1] if s.std() > 1e-12 else 0.0
        maxd[t] = np.abs(np.diff(s)).max()
    idx = np.arange(warmup, n - 30)
    return ("mean", "std", "ac1", "maxd"), lagged_stack([mean, std, ac1, maxd], idx)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=pathlib.Path, default=None)
    parser.add_argument("--value", default="value")
    parser.add_argument("--onset", default="onset")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument(
        "--normalize",
        choices=("zscore", "none"),
        default="zscore",
        help="'none' makes the metric level-shift sensitive (see DESIGN.md)",
    )
    args = parser.parse_args(argv)

    protocol = Protocol(horizon=args.horizon)
    warmup = 3 * MACRO_W

    if args.csv:
        series = Series.from_csv(args.csv, value=args.value, onset=args.onset)
        factory = lambda _seed: series  # noqa: E731 - one dataset, no seed variation
        seeds = [0]
        if series.onsets.size < 2:
            print(
                f"{series.name}: only {series.onsets.size} labelled onset(s); the "
                "operational metrics (lead time, false alarms) need at least two.",
                file=sys.stderr,
            )
        if len(series) < warmup + protocol.horizon + 100:
            print(
                f"{series.name}: {len(series)} samples is too short for warmup "
                f"{warmup} at the default macro window {MACRO_W}; pass a smaller "
                "macro window in code.",
                file=sys.stderr,
            )
            return 2
    else:
        factory = telemetry_family
        seeds = list(range(args.seeds))

    arms = [
        Arm(
            "rcod_trio",
            lambda s, w: rcod_trio(s, w, normalize=args.normalize),
            description="RCOD swarm metrics (median/macro μ, dissonance) + lags",
            warmup=warmup,
        ),
        Arm(
            "rolling",
            rolling_stats,
            description="rolling mean/std/AC1/max|Δ| over 60 samples + lags",
            warmup=warmup,
        ),
    ]

    summary = run_benchmark(factory, arms, seeds, protocol)
    print(render_report(summary, arms, baseline="rolling", treatment="rcod_trio"))

    # The harness is only as honest as its labels: show the positive rate the
    # arms were actually scored on, so a 1% base rate cannot be mistaken for a
    # 50/50 task.
    if summary["per_seed"]:
        first = summary["per_seed"][0]
        rows = first["rolling"]["n_test_rows"]
        pos = first["rolling"]["n_test_pos"]
        print(
            f"\nbase rate (seed {first['seed']}): {pos}/{rows} = "
            f"{100 * pos / max(1, rows):.2f}% positive in the test slice"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
