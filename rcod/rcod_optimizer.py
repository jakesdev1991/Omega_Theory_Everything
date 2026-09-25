# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: MIT
"""RCOD multi-scale optimizer governor (research prototype).

Wraps any ``torch.optim`` optimizer and governs its updates with Reverse
Chain Overlap Density (RCOD) metrics from the Omega informational-geometry
research program:

- state trajectory   ``s_t = w_t / ||w_t||_2``
- chain overlap      ``Phi_w = s_{t-w} . s_t``       (per window ``w``)
- matter density     ``mu_w = sqrt(1 - Phi_w**2)``   (per window ``w``)
- democratic truth   ``mu_bar = median(mu_w)``       (across all windows)
- swarm dissonance   ``sigma_mu = std(mu_leaves)``   (across leaf windows)

Regimes:

- ``FLOW``       ``mu_bar < flow_threshold``   redundant step (optional prune)
- ``VISCOSITY``  below shock thresholds       accept and refresh ``w_clean``
- ``SHOCK``      ``mu_bar >= shock_threshold`` or ``sigma_mu >= dissonance``:
                 partial rollback toward ``w_clean`` with the
                 Reverse-With-Matter schedule::

                     alpha = clip(gamma1 * mu_bar**2 + gamma2 * sigma_mu,
                                  0.0, max_alpha)
                     w_rev = (1 - alpha) * w_candidate + alpha * w_clean

``state_mode`` selects what trajectory the metrics ride on:

- ``"weights"`` (default, spec-as-written): ``s_t = w_t / ||w_t||``.
  Measured ``mu_bar`` runs small (~0.01–0.1 at benchmark scale) because
  the accumulated weight norm dominates per-step rotations; whether the
  document's thresholds engage is workload-dependent (see RESULTS.md).
- ``"updates"``: ``s_t = (w_t - w_{t-1}) / ||w_t - w_{t-1}||`` -- the
  same formulas applied to the step direction, which decoheres
  measurably under distribution shocks such as label noise.

This module fixes several defects in the draft implementation it replaces
(see ``README.md``): the ``[13-16]`` default that Python evaluates to
``[-3]`` (a window indexed from the wrong end of history), the macro-anchor
value that returned the whole window list, and the missing warm-up phase
that would have fired SHOCK reversals toward the *initial* weights during
early training.

Research status: the metrics are well defined; whether governing an
optimizer with them improves anything is an empirical question, answered
per workload by ``benchmark_noise_recovery.py``. No performance claim is
made by this module alone.

License note: research tooling on the science side of this repository
(MIT). Re-scope before any product/commercial use.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Iterable
from typing import Any

import torch

__all__ = ["RCODMultiScaleOptimizer"]


def _quantile(sorted_data: list[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted list."""
    if not sorted_data:
        raise ValueError("quantile of empty sample")
    if q <= 0.0:
        return sorted_data[0]
    if q >= 1.0:
        return sorted_data[-1]
    pos = q * (len(sorted_data) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_data[lo]
    frac = pos - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


class RCODMultiScaleOptimizer:
    """Govern a wrapped torch optimizer with multi-scale RCOD metrics.

    Parameters mirror the design document. ``step`` must be called *after*
    ``loss.backward()`` and ``zero_grad()`` must be called on this wrapper
    (it delegates to the inner optimizer).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        macro_window: int = 100,
        leaf_windows: Iterable[int] = (5, 10, 20, 40),
        *,
        flow_threshold: float = 0.15,
        shock_threshold: float = 0.70,
        dissonance_threshold: float = 0.35,
        reversal_gamma1: float = 0.80,
        reversal_gamma2: float = 0.50,
        max_alpha: float = 0.85,
        prune_in_flow: bool = False,
        state_mode: str = "weights",
        auto_calibrate: bool = False,
        calibration_steps: int = 300,
        calibration_quantiles: tuple[float, float, float] = (0.50, 0.975, 0.975),
    ) -> None:
        if macro_window <= 0:
            raise ValueError("macro_window must be positive")
        leaves = sorted({int(w) for w in leaf_windows})
        if not leaves or leaves[0] <= 0:
            raise ValueError("leaf_windows must contain positive integers")
        if state_mode not in ("weights", "updates"):
            raise ValueError("state_mode must be 'weights' or 'updates'")

        self.optimizer = optimizer
        self.state_mode = state_mode
        self.macro_window = int(macro_window)
        self.leaf_windows: list[int] = leaves
        self.flow_threshold = float(flow_threshold)
        self.shock_threshold = float(shock_threshold)
        self.dissonance_threshold = float(dissonance_threshold)
        self.gamma1 = float(reversal_gamma1)
        self.gamma2 = float(reversal_gamma2)
        self.max_alpha = float(max_alpha)
        self.prune_in_flow = bool(prune_in_flow)

        self.auto_calibrate = bool(auto_calibrate)
        self.calibration_steps = int(calibration_steps)
        if self.auto_calibrate and self.calibration_steps < 20:
            raise ValueError(
                "auto_calibrate requires calibration_steps >= 20 "
                "(quantiles need a sample)"
            )
        self._calibration_quantiles = tuple(float(q) for q in calibration_quantiles)

        self.max_buffer_size = max(self.macro_window, self.leaf_windows[-1]) + 1
        self.history_buffer: deque[torch.Tensor] = deque(maxlen=self.max_buffer_size)
        self.last_clean_weights: torch.Tensor | None = None
        self.step_counter = 0

        self._calibration_remaining = (
            self.calibration_steps if self.auto_calibrate else 0
        )
        self._calibration_mu: list[float] = []
        self._calibration_sigma: list[float] = []
        self._calibrated = not self.auto_calibrate

        self.regime_counts: dict[str, int] = {}
        self.reversal_count = 0
        self.alpha_history: list[float] = []

    # ------------------------------------------------------------------
    # Flat-parameter plumbing
    # ------------------------------------------------------------------

    def _get_flat_weights(self) -> torch.Tensor:
        """Flatten all trainable parameters into one vector."""
        views = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    views.append(p.detach().view(-1))
        return torch.cat(views)

    def _set_flat_weights(self, flat_weights: torch.Tensor) -> None:
        """Restore parameters from a flattened vector (in place)."""
        offset = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    numel = p.numel()
                    p.data.copy_(flat_weights[offset : offset + numel].view_as(p))
                    offset += numel

    # ------------------------------------------------------------------
    # Swarm metrics
    # ------------------------------------------------------------------

    def _compute_swarms(
        self, current_state: torch.Tensor
    ) -> tuple[float, float, float, dict[str, float]]:
        """Return ``(mu_bar, mu_macro, sigma_mu, per_window_mu)``.

        ``mu_bar`` is the median across *all* windows (macro included) --
        the "democratic truth" of the design document. ``sigma_mu`` is the
        sample standard deviation across the leaf windows only.
        """
        buffer_len = len(self.history_buffer)
        mu_values: list[float] = []
        window_metrics: dict[str, float] = {}

        for w in [self.macro_window, *self.leaf_windows]:
            if buffer_len > w:
                past_state = self.history_buffer[-w - 1]
                phi = float(torch.dot(past_state, current_state).clamp(-1.0, 1.0))
                mu = math.sqrt(max(0.0, 1.0 - phi * phi))
            else:
                mu = 0.0
            mu_values.append(mu)
            window_metrics[f"window_{w}"] = mu

        mu_bar = statistics.median(mu_values)
        mu_macro = mu_values[0]
        leaf_mus = mu_values[1:]
        sigma_mu = statistics.stdev(leaf_mus) if len(leaf_mus) > 1 else 0.0
        return mu_bar, mu_macro, sigma_mu, window_metrics

    def _apply_calibration(self) -> None:
        """Set governance thresholds from clean-phase metric quantiles."""
        q_flow, q_shock, q_dis = self._calibration_quantiles
        mus = sorted(self._calibration_mu)
        sigmas = sorted(self._calibration_sigma)
        if len(mus) >= 20:
            flow = _quantile(mus, q_flow)
            shock = _quantile(mus, q_shock)
            dis = _quantile(sigmas, q_dis)
            self.flow_threshold = flow
            self.shock_threshold = max(shock, flow + 0.05)
            self.dissonance_threshold = dis
        self._calibrated = True

    # ------------------------------------------------------------------
    # Governed step
    # ------------------------------------------------------------------

    def step(self, closure: Any | None = None) -> tuple[float | None, dict[str, Any]]:
        """Run one governed optimization step.

        Call after ``loss.backward()``. If ``closure`` is given it is
        forwarded to the inner optimizer and its loss value is returned.
        """
        self.step_counter += 1

        flat_pre = self._get_flat_weights()
        if self.last_clean_weights is None:
            self.last_clean_weights = flat_pre.clone()

        if closure is not None:
            loss = self.optimizer.step(closure)
        else:
            # Standard path: gradients were computed by the caller via
            # loss.backward(). The draft implementation skipped the inner
            # step entirely in this case, silently freezing training.
            self.optimizer.step()
            loss = None
        flat_candidate = self._get_flat_weights()

        if self.state_mode == "updates":
            delta = flat_candidate - flat_pre
            delta_norm = torch.norm(delta, p=2)
            if float(delta_norm) < 1e-12:
                # A zero-length step carries no directional information.
                current_state: torch.Tensor | None = None
            else:
                current_state = delta / delta_norm
        else:
            norm_val = torch.norm(flat_candidate, p=2) + 1e-12
            current_state = flat_candidate / norm_val

        if current_state is not None:
            self.history_buffer.append(current_state)

        if current_state is None:
            mu_bar = 0.0
            mu_macro = 0.0
            sigma_mu = 0.0
            window_metrics: dict[str, float] = {}
        else:
            mu_bar, mu_macro, sigma_mu, window_metrics = self._compute_swarms(
                current_state
            )

        reversal_applied = False
        alpha = 0.0
        buffer_full = len(self.history_buffer) == self.max_buffer_size

        if not buffer_full:
            # WARMUP: every window must be populated before the metrics
            # mean anything; refreshing the clean checkpoint avoids
            # reversals toward the raw initialization later.
            regime = "WARMUP"
            self.last_clean_weights = flat_candidate.clone()
        elif not self._calibrated or self._calibration_remaining > 0:
            # CALIBRATION: record clean-phase metric distribution.
            regime = "CALIBRATION"
            self._calibration_mu.append(mu_bar)
            self._calibration_sigma.append(sigma_mu)
            self.last_clean_weights = flat_candidate.clone()
            if self._calibration_remaining > 0:
                self._calibration_remaining -= 1
                if self._calibration_remaining == 0:
                    self._apply_calibration()
        elif mu_bar < self.flow_threshold:
            regime = "FLOW"
            if self.prune_in_flow:
                # Discard the redundant step entirely.
                self._set_flat_weights(flat_pre)
        elif mu_bar >= self.shock_threshold or sigma_mu >= self.dissonance_threshold:
            regime = "SHOCK"
            raw_alpha = self.gamma1 * mu_bar**2 + self.gamma2 * sigma_mu
            alpha = min(self.max_alpha, max(0.0, raw_alpha))
            w_rev = (1.0 - alpha) * flat_candidate + (alpha * self.last_clean_weights)
            self._set_flat_weights(w_rev)
            self.reversal_count += 1
            self.alpha_history.append(alpha)
        else:
            regime = "VISCOSITY"
            self.last_clean_weights = flat_candidate.clone()

        self.regime_counts[regime] = self.regime_counts.get(regime, 0) + 1

        telemetry: dict[str, Any] = {
            "step": self.step_counter,
            "regime": regime,
            "mu_bar": mu_bar,
            "mu_macro": mu_macro,
            "sigma_mu": sigma_mu,
            "reversal_alpha": alpha,
            "reversal_applied": reversal_applied,
            **window_metrics,
        }
        return loss, telemetry

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        """Delegate ``zero_grad`` to the wrapped optimizer."""
        self.optimizer.zero_grad(*args, **kwargs)

    @property
    def param_groups(self) -> Any:
        """Expose inner ``param_groups`` (schedulers, LR queries)."""
        return self.optimizer.param_groups
