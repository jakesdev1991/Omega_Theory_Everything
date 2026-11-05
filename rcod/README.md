<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT -->

# RCOD multi-scale optimizer governor — research prototype

Implementation and empirical evaluation of the RCOD (Reverse Chain Overlap
Density) optimizer governor from the Omega informational-geometry research
program. The governor wraps any `torch.optim` optimizer, tracks the
parameter (or update) trajectory on the unit sphere, computes multi-window
overlap `Φ_w = s_{t-w}·s_t` and matter density `μ_w = √(1−Φ_w²)`, and gates
each step into FLOW / VISCOSITY / SHOCK regimes with a Reverse-With-Matter
partial rollback `w_rev = (1−α)·w_cand + α·w_clean`.

**Read [`RESULTS.md`](RESULTS.md) before using this for anything.** The
headline: on the label-noise-recovery benchmark, the spec-as-written
governor never engages, and the variants that do engage do not improve
recovery over plain AdamW. This directory exists to make that claim
checkable, per the repo's own standard ("no formula becomes valid merely
because it is written mathematically"; the $OMEGA whitepaper asks that
geometry-informed metrics be tested against simpler baselines).

## Files

| File | Purpose |
|---|---|
| `rcod_optimizer.py` | The wrapper class (`RCODMultiScaleOptimizer`) |
| `benchmark_noise_recovery.py` | Teacher-student label-noise shock benchmark |
| `RESULTS.md` | Findings, honest verdict, next experiments |
| `results_noise_recovery.csv` | Per-eval-step trajectories (evidence) |
| `harness.py` | **Portable**: pre-registered evaluation of an onset-prediction claim |
| `section5/swarm.py` | **Portable**: the Φ/μ metrics (batch, prefix-sum, streaming) |
| `section5/experiment.py` | The frozen §5 experiment (pre-registered; not a library) |
| `section5/domain_example.py` | **Portable**: run the protocol on another family or your own CSV |

## Defects fixed relative to the design draft

The circulated draft class contained bugs that any production use would
have hit:

1. **Training was silently frozen.** `self.optimizer.step(closure) if
   closure else None` never calls the inner optimizer's `step()` in the
   standard no-closure path (backward → step), so the model never updated.
   Found empirically: μ printed as exactly 0 and weights never moved.
2. **`leaf_windows or [13-16]`** — Python evaluates `13-16` to `-3`, giving
   a window of `[-3]` that indexes the wrong end of the history buffer.
   Default is now `(5, 10, 20, 40)` per the document's own text.
3. **`macro_mu = mu_values`** returned the whole window list as the macro
   anchor; now returns the macro window's own μ.
4. **No warm-up.** Early training always shows high trajectory novelty, so
   the draft would fire SHOCK reversals toward the *raw initialization*
   during the first steps. A `WARMUP` regime (record + refresh the clean
   checkpoint only) now runs until every window is populated.
5. History list with `pop(0)` → `collections.deque(maxlen=…)`.
6. The commented-out FLOW prune is now an explicit opt-in
   (`prune_in_flow`, off by default) — pruning interacts with Adam's moment
   state and can silently stall learning.
7. Telemetry typing corrected (regime is a string, not a float); typing
   modernized for current tooling.
8. Guards added: zero-length update vectors in `updates` mode, single-leaf
   σ, input validation, and a quantile helper for calibration.

Additions beyond the draft: `state_mode` (`"weights"` = spec-as-written,
`"updates"` = same math on step directions) and `auto_calibrate`
(thresholds from clean-phase quantiles) — both motivated by the benchmark
findings in `RESULTS.md`.

## Usage

```python
import torch
from rcod_optimizer import RCODMultiScaleOptimizer

inner = torch.optim.AdamW(model.parameters(), lr=1e-3)
gov = RCODMultiScaleOptimizer(inner, state_mode="updates", auto_calibrate=True)

for batch in loader:
    loss = criterion(model(batch.x), batch.y)
    gov.zero_grad()
    loss.backward()
    _, telemetry = gov.step()  # telemetry: regime, mu_bar, sigma_mu, ...
```

Benchmark:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python benchmark_noise_recovery.py --csv results_noise_recovery.csv
```

## Using the metrics on your own series

Nothing in `section5/swarm.py` is cosmology-specific, and the §5 protocol in
`harness.py` is not either — that is the part of this directory worth reusing
elsewhere. See [`../docs/DOMAIN_AGNOSTIC_CORE.md`](../docs/DOMAIN_AGNOSTIC_CORE.md)
for what is portable, what is not, and the measurements behind each claim.

```python
from rcod import SwarmTracker, swarm_metrics, pool_channels

m = swarm_metrics(x)  # (T,) or (T, channels)
m = pool_channels(m)  # one alarm signal across channels
tracker = SwarmTracker()  # online: O(1) memory, ~97 us/sample
state = tracker.update(value)  # {'median_mu', 'macro_mu', 'dissonance'}
```

```python
from rcod.harness import Arm, Protocol, Series, run_benchmark, render_report

summary = run_benchmark(
    lambda seed: Series.from_csv("telemetry.csv"),
    [Arm("rcod", my_extractor), Arm("rolling", baseline)],
    seeds=range(10),
    protocol=Protocol(horizon=30),
)
print(render_report(summary, arms, baseline="rolling"))
```

```bash
python -m rcod.section5.domain_example                     # synthetic non-AR(1) family
python -m rcod.section5.domain_example --csv yours.csv --value sensor --onset fault
```

Three implementations of the same arithmetic exist on purpose: the reference
path produced the committed §5 numbers, the prefix-sum path is the same
arithmetic in O(T·W) memory, and the tracker is the online form. `tests/test_rcod.py`
is the proof that they agree (max difference 4e-12), and re-running Run 5 after
that refactor reproduces `section5/results_dyadic.json` byte for byte.

(`torch` is deliberately **not** in the repo `requirements.txt` — it is a
heavy optional dependency for this directory only.)

## Known limitations

- `_get_flat_weights` flattens and copies all parameters every step
  (O(N) per step, several allocations). Fine at research scale; production
  telemetry needs a fused/native ring-buffer path.
- The clean checkpoint refreshes on every VISCOSITY step, which pollutes it
  during slow-developing noise regimes (see RESULTS.md F4).
- Thresholds are workload-dependent; there is no universal constant (F2).

## Next step (telemetry pipeline)

The per-step flatten/normalize/metrics work is the natural consumer of the
high-performance telemetry path. In this repository the planned home for
that is the **Rust `telemetry` crate** in the blueprint's workspace layout
(`rust/crates/` per the tri-token blueprint §10), which would stream
governor state off the training process without stalling it. A C++23 SPSC
lock-free ring buffer is a reasonable alternative if the governor ends up
living inside an existing C++ training stack — decide based on where the
governor is deployed, not before.

## License

MIT (science side of this repository). This is research tooling for
falsifying/improving the Omega metrics; it is not product code. If it is
ever moved into a product path (e.g., the C.A.R.E. protocol stack), re-scope
the license deliberately first.
