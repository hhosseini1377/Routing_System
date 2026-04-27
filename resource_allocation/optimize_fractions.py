#!/usr/bin/env python3
"""
Optimize routing fractions w on the simplex using:

- Score gradient from dual prices (alpha) computed by score_under_fractions_dual.
- Latency gradient from per-model piecewise-linear latency-vs-load curves.

We solve the Lagrangian saddle problem (for fixed beta, lambda, Theta).
Constraint L ≤ τ is written as L/τ ≤ 1; we penalize (L/τ − 1) for numerics:

    max_{w in Δ_K}  S(w) - beta * (L(Theta, w) / tau - 1)

Gradient: ∇_w (L/τ) = (1/τ) ∇_w L, so we use (1/τ) * dL_dw in the update.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .dual_prices import score_under_fractions_dual

# Steep linear penalty above max profiled load (ms latency per RPS beyond grid).
DEFAULT_OVERLOAD_PENALTY_MS_PER_RPS = 5000.0


def project_to_simplex(w: np.ndarray) -> np.ndarray:
    """
    Project a vector w onto the probability simplex:
        { w : w_i >= 0, sum_i w_i = 1 }.
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size == 0:
        return w

    # Algorithm from "Efficient Projections onto the l1-Ball for Learning in High Dimensions" (Duchi et al.)
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, w.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        # All entries are non-positive; fall back to uniform
        return np.ones_like(w) / w.size
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w_proj = np.maximum(w - theta, 0.0)
    # Normalize for numerical safety
    s = w_proj.sum()
    if s <= 0:
        w_proj = np.ones_like(w_proj) / w_proj.size
    else:
        w_proj = w_proj / s
    return w_proj


@dataclass
class PiecewiseLinearLatency:
    """
    1D piecewise-linear latency curve for a single model:
        load_grid (in RPS) -> latency_ms.

    Assumes load_grid is strictly increasing. Between grid points we linearly
    interpolate.

    Below min(grid): extrapolate along the first segment (same as before).

    Above max(grid): by default we do *not* extrapolate the last profiled segment
    (unknown operating region). We attach latency at the last grid point plus a
    steep linear penalty in (load - max_grid) so latency is large *and* the slope
    w.r.t. load is positive, giving useful gradients for the routing optimizer.
    Set ``forbid_overload_extrapolation=False`` to restore legacy extrapolation
    past the last segment.
    """

    load_grid: np.ndarray  # shape (M,)
    latency_ms: np.ndarray  # shape (M,)
    forbid_overload_extrapolation: bool = True
    # Used when forbid_overload_extrapolation: latency = y(max) + rho * (load - max).
    overload_penalty_ms_per_rps: float = DEFAULT_OVERLOAD_PENALTY_MS_PER_RPS

    def __post_init__(self) -> None:
        self.load_grid = np.asarray(self.load_grid, dtype=float).reshape(-1)
        self.latency_ms = np.asarray(self.latency_ms, dtype=float).reshape(-1)
        if self.load_grid.shape != self.latency_ms.shape:
            raise ValueError("load_grid and latency_ms must have same shape")
        if self.load_grid.size < 2:
            raise ValueError("Need at least two points for piecewise-linear curve")
        if not np.all(np.diff(self.load_grid) > 0):
            raise ValueError("load_grid must be strictly increasing")
        if self.forbid_overload_extrapolation and self.overload_penalty_ms_per_rps <= 0:
            raise ValueError("overload_penalty_ms_per_rps must be > 0 when forbid_overload_extrapolation is True")

    def value_and_slope(self, load: float) -> Tuple[float, float]:
        """
        Return (latency_ms(load), d latency_ms / d load).

        - For load below min(grid): extrapolate along the first segment.
        - For load above max(grid): if ``forbid_overload_extrapolation``, return
          y(max) + rho * (load - max) with slope rho; else extrapolate the last segment.
        """
        x = float(load)
        xg = self.load_grid
        yg = self.latency_ms

        if self.forbid_overload_extrapolation and x > float(xg[-1]):
            rho = float(self.overload_penalty_ms_per_rps)
            dx = x - float(xg[-1])
            return float(yg[-1] + rho * dx), rho

        if x <= xg[0]:
            i = 0
        elif x >= xg[-1]:
            i = len(xg) - 2
        else:
            i = int(np.searchsorted(xg, x) - 1)
            i = max(0, min(i, len(xg) - 2))

        x0, x1 = xg[i], xg[i + 1]
        y0, y1 = yg[i], yg[i + 1]

        if x1 == x0:
            slope = 0.0
            val = float(y0)
        else:
            t = (x - x0) / (x1 - x0)
            val = float(y0 + t * (y1 - y0))
            slope = float((y1 - y0) / (x1 - x0))
        return val, slope


@dataclass
class OptimizationResult:
    w: np.ndarray
    history_L: list[float]
    history_S: list[float]
    history_obj: list[float]  # Lagrangian S - beta*(L/tau - 1)
    history_w: list[np.ndarray]
    tau: float  # latency requirement (SLO) in ms
    best_obj: float  # maximum objective achieved
    best_S: float  # score at best step
    best_L: float  # latency at best step
    best_w: np.ndarray  # w at best step
    # Dual prices α from score_under_fractions_dual at the best-objective step (same w as best_w).
    best_alpha: np.ndarray


def optimize_fractions(
    S: np.ndarray,
    latency_curves: Sequence[PiecewiseLinearLatency],
    lambda_global: float,
    beta: float,
    tau: float,
    w_init: np.ndarray | None = None,
    n_steps: int = 200,
    eta: float = 0.1,
    seed: int = 0,
    # Dual-prices (score_under_fractions_dual) hyperparameters.
    dual_max_iter: int = 300,
    dual_eta0: float = 1e-3,
    dual_tol: int = 1,
    dual_tie_noise: float = 1e-9,
    momentum: float = 0.0,
    w_ema_decay: float | None = None,
    verbose: bool = False,
    patience: int | None = None,
    obj_tol: float = 1e-4,
) -> OptimizationResult:
    """
    Optimize routing fractions w on the simplex for fixed lambda_global, beta, and latency requirement tau.

    Lagrangian: max_w  S(w) - beta * (L(Θ,w)/tau - 1).  Constraint L ≤ τ is equivalent
    to L/tau ≤ 1; scaling by tau improves gradient scale. tau (ms) is the SLO.

    Parameters
    ----------
    S : (N, K) array
        Per-prompt scores s_{ji}.
    latency_curves : list of PiecewiseLinearLatency, length K
        Per-model latency vs load curves.
    lambda_global : float
        Global request rate λ.
    beta : float
        Lagrange multiplier for latency constraint.
    tau : float
        Latency requirement (SLO) in ms. In the Lagrangian we penalize (L/tau - 1).
    w_init : (K,) array or None
        Initial routing fractions (will be projected to simplex). If None, use uniform.
    n_steps : int
        Number of projected gradient ascent steps.
    eta : float
        Step size for w updates.
    seed : int
        Seed passed to dual prices routine (for tie-breaking noise).
    dual_max_iter : int
        Number of dual/prices iterations inside `score_under_fractions_dual`.
    dual_eta0 : float
        Base step size for dual prices; actual eta_t is eta0 / sqrt(t+1).
    dual_tol : int
        Stop if max |count_i - c_i| <= dual_tol.
    dual_tie_noise : float
        Tiny noise added to break ties deterministically inside dual prices.
    momentum : float
        Momentum for gradient updates. velocity = momentum*velocity + grad; w = project(w + eta*velocity).
        Default 0 (no momentum). Use e.g. 0.9 for smoother updates.
    w_ema_decay : float or None
        If set (e.g. 0.99), maintain exponential moving average of w and return it as final w.
        Helps reduce fluctuation in the returned solution. Default None (no EMA).
    verbose : bool
        If True, print the values of score, latency, and normalized slack L/tau - 1 at each step.
    patience : int or None
        Early stopping: stop if objective has not improved by more than obj_tol for this many steps.
        Default None (no early stopping).
    obj_tol : float
        Minimum improvement to count as progress. Used only when patience is set.

    Returns
    -------
    OptimizationResult with final w, trajectories of L and S, and tau.
    """
    S = np.asarray(S, dtype=float)
    N, K = S.shape
    if len(latency_curves) != K:
        raise ValueError(f"latency_curves must have length K={K}")

    if w_init is None:
        w = np.ones(K, dtype=float) / K
    else:
        w = project_to_simplex(np.asarray(w_init, dtype=float))

    history_L: list[float] = []
    history_S: list[float] = []
    history_obj: list[float] = []
    history_w: list[np.ndarray] = []
    alpha_prev: np.ndarray | None = None
    velocity = np.zeros(K, dtype=float)
    w_ema = w.copy() if w_ema_decay is not None else None
    best_obj = float("-inf")
    best_S = 0.0
    best_L = 0.0
    best_w = w.copy()
    best_alpha = np.zeros(K, dtype=float)
    steps_without_improvement = 0

    for t in range(n_steps):
        # Score term: compute dual prices alpha(w), warm-start from previous alpha
        S_hat, _, _, _, alpha = score_under_fractions_dual(
            S,
            w,
            max_iter=dual_max_iter,
            eta0=dual_eta0,
            tol=dual_tol,
            tie_noise=dual_tie_noise,
            seed=seed,
            do_repair=True,
            alpha_init=alpha_prev,
        )
        alpha_prev = alpha

        # Latency term: L(Theta, w) = sum_i w_i * ℓ_i(λ_i), λ_i = λ * w_i
        lambda_i = lambda_global * w
        lat_vals = np.zeros(K, dtype=float)
        dlat_dlambda = np.zeros(K, dtype=float)
        for i in range(K):
            li, gi = latency_curves[i].value_and_slope(lambda_i[i])
            lat_vals[i] = li
            dlat_dlambda[i] = gi

        L_val = float(np.sum(w * lat_vals))
        obj = S_hat - beta * ((L_val / tau) - 1.0)
        # print(obj)
        history_L.append(L_val)
        history_S.append(S_hat)
        history_obj.append(obj)
        history_w.append(w.copy())

        normalized_slack = (L_val / tau) - 1.0
        if verbose:
            print(f"S_hat: {S_hat}, L_val: {L_val}, tau: {tau}, L/tau - 1: {normalized_slack:.4f}")

        prev_best = best_obj
        if obj > best_obj:
            best_obj = obj
            best_S = S_hat
            best_L = L_val
            best_w = w.copy()
            best_alpha = np.asarray(alpha, dtype=float).reshape(-1).copy()
        if patience is not None:
            if obj > prev_best + obj_tol:
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            if steps_without_improvement >= patience:
                break

        # Gradient of L wrt w_i: dL/dw_i = ℓ_i(λ_i) + λ * w_i * dℓ_i/dλ_i
        dL_dw = lat_vals + lambda_global * w * dlat_dlambda

        # Gradient of Lagrangian: ∇_w [S(w) − beta * (L/tau - 1)] = alpha − beta * (1/tau) * dL_dw
        grad_w = alpha - beta * (dL_dw / tau)

        # Momentum: velocity = momentum * velocity + grad_w
        if momentum > 0:
            velocity = momentum * velocity + grad_w
            delta = velocity
        else:
            delta = grad_w

        # Projected gradient ascent step on simplex
        w = project_to_simplex(w + eta * delta)

        # Exponential moving average of w (for smoother final solution)
        if w_ema_decay is not None:
            w_ema = w_ema_decay * w_ema + (1.0 - w_ema_decay) * w
            w_ema = project_to_simplex(w_ema)

    final_w = project_to_simplex(w_ema) if w_ema_decay is not None else w
    return OptimizationResult(
        w=final_w,
        history_L=history_L,
        history_S=history_S,
        history_obj=history_obj,
        history_w=history_w,
        tau=tau,
        best_obj=best_obj,
        best_S=best_S,
        best_L=best_L,
        best_w=best_w,
        best_alpha=best_alpha,
    )


@dataclass
class BetaOptimizationResult:
    """Result of searching for optimal β such that normalized slack ≈ 0."""

    best_beta: float
    result: OptimizationResult  # final run with best_beta
    history_beta: list[float]
    history_slack: list[float]


def optimize_beta(
    S: np.ndarray,
    latency_curves: Sequence[PiecewiseLinearLatency],
    lambda_global: float,
    tau: float,
    beta_init: float = 0.01,
    max_outer_steps: int = 50,
    eta_beta: float = 0.01,
    eta_beta_min: float = 1e-4,
    eta_beta_decay: float = 0.98,
    slack_tol: float = 0.01,
    beta_min: float = 0,
    beta_max: float = 5.0,
    show_progress: bool = True,
    **optimize_fractions_kwargs,
) -> BetaOptimizationResult:
    """
    Find β such that normalized slack (L/tau - 1) ≈ 0.

    Outer loop: β_new = β + eta_beta * (L/tau - 1). Warm-starts w from previous best_w.

    Adaptive eta_beta:
    - On sign flip (overshoot): reduce eta_beta by 50% to dampen oscillation.
    - Decay: eta_beta is multiplied by eta_beta_decay each iteration to take smaller
      steps as we approach convergence.
    """
    beta = beta_init
    eta_beta_current = eta_beta
    slack_prev: float | None = None
    history_beta: list[float] = []
    history_slack: list[float] = []
    w_init: np.ndarray | None = None

    for outer in tqdm(
        range(max_outer_steps),
        desc="Optimizing β",
        disable=not show_progress,
    ):
        result = optimize_fractions(
            S=S,
            latency_curves=latency_curves,
            lambda_global=lambda_global,
            beta=beta,
            tau=tau,
            w_init=w_init,
            **optimize_fractions_kwargs,
        )
        slack = (result.best_L / tau) - 1.0
        history_beta.append(beta)
        history_slack.append(slack)
        if show_progress:
            tqdm.write(f"slack: {slack:.4f}  eta_beta: {eta_beta_current:.6f} beta: {beta:.6f}")

        if abs(slack) < slack_tol:
            return BetaOptimizationResult(
                best_beta=beta,
                result=result,
                history_beta=history_beta,
                history_slack=history_slack,
            )

        # Adaptive eta_beta: reduce on oscillation (sign flip)
        if slack_prev is not None and (slack * slack_prev) < 0:
            eta_beta_current = max(eta_beta_current * 0.5, eta_beta_min)
        slack_prev = slack

        beta = np.clip(beta + eta_beta_current * slack, beta_min, beta_max)
        eta_beta_current = max(eta_beta_current * eta_beta_decay, eta_beta_min)
        w_init = result.best_w.copy()

    return BetaOptimizationResult(
        best_beta=beta,
        result=result,
        history_beta=history_beta,
        history_slack=history_slack,
    )


def optimize_beta_bisection(
    S: np.ndarray,
    latency_curves: Sequence[PiecewiseLinearLatency],
    lambda_global: float,
    tau: float,
    beta_init: float = 0.01,
    max_outer_steps: int = 50,
    slack_tol: float = 0.01,
    beta_min: float = 0.0,
    beta_max: float = 5.0,
    beta_interval_tol: float | None = None,
    show_progress: bool = False,
    **optimize_fractions_kwargs,
) -> BetaOptimizationResult:
    """
    Find β such that normalized slack (L/tau - 1) ≈ 0 via bracketing + bisection.

    Assumes slack is mostly decreasing in β (larger β penalizes latency more).
    If no sign change exists in [beta_min, beta_max], returns the endpoint with
    smaller |slack|.

    Each outer step is one full ``optimize_fractions`` call; ``max_outer_steps``
    limits the total number of such evaluations (bracket search + bisection).
    """
    history_beta: list[float] = []
    history_slack: list[float] = []

    if beta_interval_tol is None:
        beta_interval_tol = max(1e-9, (beta_max - beta_min) * 1e-7)

    n_eval = 0

    def eval_slack(
        beta_val: float,
        w_init: np.ndarray | None,
    ) -> tuple[float, OptimizationResult]:
        nonlocal n_eval
        if n_eval >= max_outer_steps:
            raise RuntimeError("max_outer_steps exhausted inside optimize_beta_bisection")
        n_eval += 1
        res = optimize_fractions(
            S=S,
            latency_curves=latency_curves,
            lambda_global=lambda_global,
            beta=float(beta_val),
            tau=tau,
            w_init=w_init,
            **optimize_fractions_kwargs,
        )
        slack = float((res.best_L / tau) - 1.0)
        history_beta.append(float(beta_val))
        history_slack.append(slack)
        if show_progress:
            tqdm.write(f"[bisection] slack: {slack:.4f}  beta: {beta_val:.6f}  (eval {n_eval}/{max_outer_steps})")
        return slack, res

    beta_init = float(np.clip(beta_init, beta_min, beta_max))

    s0, r0 = eval_slack(beta_init, None)
    if abs(s0) < slack_tol:
        return BetaOptimizationResult(
            best_beta=beta_init,
            result=r0,
            history_beta=history_beta,
            history_slack=history_slack,
        )

    # --- Bracket: find lo < hi with slack(lo) * slack(hi) <= 0 ---
    lo = beta_init
    slo = s0
    rlo = r0
    hi = beta_init
    shi = s0
    rhi = r0

    if slo > 0:
        # Need higher β → search toward beta_max
        while shi > 0 and hi < beta_max - 1e-15 and n_eval < max_outer_steps:
            lo, slo, rlo = hi, shi, rhi
            step = max(1e-4, min((beta_max - lo) * 0.25, (beta_max - lo) * 0.5))
            hi = min(beta_max, lo + step)
            if hi <= lo:
                hi = beta_max
            shi, rhi = eval_slack(hi, rlo.best_w.copy())
            if abs(shi) < slack_tol:
                return BetaOptimizationResult(
                    best_beta=hi,
                    result=rhi,
                    history_beta=history_beta,
                    history_slack=history_slack,
                )
        # Expand with doubling if still shi > 0 and room
        while shi > 0 and hi < beta_max - 1e-15 and n_eval < max_outer_steps:
            lo, slo, rlo = hi, shi, rhi
            span = max(hi - beta_init, 1e-4)
            hi = min(beta_max, lo + span)
            if hi <= lo:
                hi = beta_max
            shi, rhi = eval_slack(hi, rlo.best_w.copy())
            if abs(shi) < slack_tol:
                return BetaOptimizationResult(
                    best_beta=hi,
                    result=rhi,
                    history_beta=history_beta,
                    history_slack=history_slack,
                )
    else:
        # slo < 0: need lower β → search toward beta_min
        while slo < 0 and lo > beta_min + 1e-15 and n_eval < max_outer_steps:
            hi, shi, rhi = lo, slo, rlo
            step = max(1e-4, min((hi - beta_min) * 0.25, (hi - beta_min) * 0.5))
            lo = max(beta_min, hi - step)
            if lo >= hi:
                lo = beta_min
            slo, rlo = eval_slack(lo, rhi.best_w.copy())
            if abs(slo) < slack_tol:
                return BetaOptimizationResult(
                    best_beta=lo,
                    result=rlo,
                    history_beta=history_beta,
                    history_slack=history_slack,
                )
        while slo < 0 and lo > beta_min + 1e-15 and n_eval < max_outer_steps:
            hi, shi, rhi = lo, slo, rlo
            span = max(beta_init - lo, 1e-4)
            lo = max(beta_min, hi - span)
            if lo >= hi:
                lo = beta_min
            slo, rlo = eval_slack(lo, rhi.best_w.copy())
            if abs(slo) < slack_tol:
                return BetaOptimizationResult(
                    best_beta=lo,
                    result=rlo,
                    history_beta=history_beta,
                    history_slack=history_slack,
                )

    if slo * shi > 0:
        if abs(slo) <= abs(shi):
            return BetaOptimizationResult(
                best_beta=lo,
                result=rlo,
                history_beta=history_beta,
                history_slack=history_slack,
            )
        return BetaOptimizationResult(
            best_beta=hi,
            result=rhi,
            history_beta=history_beta,
            history_slack=history_slack,
        )

    # Ensure lo < hi for bisection
    if lo > hi:
        lo, hi = hi, lo
        slo, shi = shi, slo
        rlo, rhi = rhi, rlo

    # --- Bisection on β (slack bracket) ---
    result = rlo if abs(slo) < abs(shi) else rhi
    while (hi - lo) > beta_interval_tol and n_eval < max_outer_steps:
        mid = 0.5 * (lo + hi)
        smid, rmid = eval_slack(mid, rlo.best_w.copy())
        if abs(smid) < slack_tol:
            return BetaOptimizationResult(
                best_beta=mid,
                result=rmid,
                history_beta=history_beta,
                history_slack=history_slack,
            )
        # slack decreasing in β: same sign as lo → root is right of mid
        if smid * slo > 0:
            lo, slo, rlo = mid, smid, rmid
        else:
            hi, shi, rhi = mid, smid, rmid
        result = rmid

    mid = 0.5 * (lo + hi)
    if n_eval < max_outer_steps:
        _, result = eval_slack(mid, rlo.best_w.copy())
    else:
        result = rlo if abs(slo) <= abs(shi) else rhi
    return BetaOptimizationResult(
        best_beta=float(mid),
        result=result,
        history_beta=history_beta,
        history_slack=history_slack,
    )


