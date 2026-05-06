# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
"""BSV verifier-invocation gate (DEFINE_SPEC_v3 §1.2, §3.1, §3.2).

The gate decides, pair-by-pair, whether to spend a verifier call or to
rely on the self-judge. The decision is driven by the pairwise confidence
``c = 2 · |p_self − 0.5| ∈ [0, 1]`` thresholded at the adaptive
(1 − α)-quantile ``t(α)`` tracked online via P² (Jain & Chlamtac 1985;
see :mod:`bsv_p2_tracker`).

Gate rule (LOCKED):

    below = 1{c < t(α)}
    p_invoke = ε + (1 − ε) · below
    I_v      ~ Bernoulli(p_invoke)

With ``ε`` the exploration floor (REQUIRED per §3.1 to break MNAR
selective-label bias). At endpoints α=0 (t→∞) and α=1 (t→0) the gate
collapses to RLVR or URLVR respectively, modulo the ε floor.

The gate exposes the propensity ``p_invoke`` alongside each decision so
the training loop can log propensities (§3.2) and the evaluator can apply
IPS / DR corrections (§3.3, see :mod:`bsv_evaluators`).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .bsv_p2_tracker import P2QuantileTracker


class BSVGate:
    """Adaptive verifier-invocation gate with ε-exploration.

    Parameters
    ----------
    alpha : float
        Target autonomy fraction in [0, 1]. The expected fraction of pairs
        handled WITHOUT a verifier call (modulo ε). α=0 → RLVR (always
        call), α=1 → URLVR (never call, modulo ε).
    epsilon : float, default 0.05
        Exploration floor in [0, 1]. Forces a Bernoulli(ε) audit slice on
        high-confidence pairs so the verifier sees a random cross-section
        of the confident regime. Required per DEFINE_SPEC_v3 §3.1.
    rng_seed : int, default 0
        Seed for the internal Bernoulli sampler.
    """

    def __init__(
        self,
        alpha: float,
        epsilon: float = 0.05,
        rng_seed: int = 0,
        inverted: bool = False,
    ) -> None:
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not (0.0 <= float(epsilon) <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        # Inverted mode flips the gate direction: normally "below" = c < t(α)
        # routes a rollout to the verify set (low-confidence/self-judged-wrong);
        # with inverted=True the routing flips so HIGH-confidence (self-judged
        # correct) rollouts are the verify candidates. C13 ablation per
        # methodology §5c / Phase-1 E10.
        self.inverted = bool(inverted)
        # Target quantile of c such that E[c < t(α)] = 1 − α.
        # Endpoints pin to the two boundary semantics from §1.4:
        #   α=0 → t(α) = +∞ (below is always true → I_v = 1 always)
        #   α=1 → t(α) = 0  (below is never true → I_v = ε always)
        # We implement both as explicit branches; P² is used for the interior.
        self._tracker = self._make_tracker()
        self._rng = np.random.default_rng(rng_seed)

        # Running statistics for realized_call_rate().
        self._total_calls = 0
        self._total_decisions = 0

        # Last-batch tensors cached for the adaptive-α controller (Phase-1 E6).
        # Populated by decide_batch() and by the GRPO-BSV branch in ray_trainer.
        # Controller reads them via compute_calibration_signals(...).
        self.last_p_self: np.ndarray | None = None
        self.last_V: np.ndarray | None = None
        self.last_verify_mask: np.ndarray | None = None
        self.last_eps_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, p_self: float) -> Tuple[int, float]:
        """Decide whether to call the verifier on a single pair.

        Parameters
        ----------
        p_self : float
            Self-judge pairwise preference probability in [0, 1].

        Returns
        -------
        (I_v, p_invoke) : tuple[int, float]
            * ``I_v`` — invocation indicator (1 = call verifier).
            * ``p_invoke`` — propensity P(I_v = 1 | c) under this gate.

        Side effect: feeds ``c = 2·|p_self − 0.5|`` into the P² tracker and
        updates the running call-rate statistics.
        """
        c = self._confidence(p_self)
        # Read threshold from PRIOR sample history, draw I_v, THEN feed the
        # observation into the tracker. This ordering makes t_i a function
        # of c_{<i} only — matching the methodology doc's "prior history"
        # interpretation (bsv_methodology.md §6.1) and keeping the marginal
        # per-batch gate-call rate equal to α.
        p_invoke = self._p_invoke(c)
        I_v = int(self._rng.uniform() < p_invoke)
        self._tracker.update(c)
        self._total_decisions += 1
        self._total_calls += I_v
        return I_v, float(p_invoke)

    def decide_batch(self, p_self: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized variant of :meth:`decide`.

        Processes ``p_self`` elementwise, feeding each into the tracker in
        order. Shape-preserving: output I_v has dtype int64, p_invoke has
        dtype float64, both shape-matching the input.
        """
        arr = np.asarray(p_self, dtype=np.float64).ravel()
        I_v = np.empty(arr.shape, dtype=np.int64)
        p_inv = np.empty(arr.shape, dtype=np.float64)
        for i, x in enumerate(arr):
            iv, pi = self.decide(float(x))
            I_v[i] = iv
            p_inv[i] = pi
        target_shape = np.asarray(p_self).shape
        return I_v.reshape(target_shape), p_inv.reshape(target_shape)

    def current_threshold(self) -> float:
        """Current estimate of t(α) from the P² tracker.

        Endpoint semantics (§1.4):
          * α = 0 → +∞ (everything is "below" threshold).
          * α = 1 → 0  (nothing is "below" threshold).
        """
        if self.alpha <= 0.0:
            return float("inf")
        if self.alpha >= 1.0:
            return 0.0
        return self._tracker.quantile()

    def realized_call_rate(self) -> float:
        """Running mean of I_v across all decisions made so far."""
        if self._total_decisions == 0:
            return 0.0
        return self._total_calls / self._total_decisions

    def n_observations(self) -> int:
        """Number of pairs passed through the gate."""
        return self._total_decisions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence(p_self: float) -> float:
        """c = 2 · |p_self − 0.5|, clipped to [0, 1]."""
        c = 2.0 * abs(float(p_self) - 0.5)
        if c < 0.0:
            return 0.0
        if c > 1.0:
            return 1.0
        return c

    def _p_invoke(self, c: float) -> float:
        """Compute p_invoke = ε + (1 − ε) · 1{below t(α)}.

        In standard (non-inverted) mode ``below = 1{c < t(α)}`` so the
        low-confidence tail (rollouts the model thinks are wrong) are the
        verify candidates. In inverted mode the direction flips:
        ``below = 1{c ≥ t(α)}`` so the high-confidence rollouts (self-judged
        correct) are the verify candidates — matched-budget interventionist
        ablation (methodology §5c / C13).
        """
        t = self.current_threshold()
        if self.inverted:
            is_candidate = c >= t
        else:
            is_candidate = c < t
        below = 1.0 if is_candidate else 0.0
        return self.epsilon + (1.0 - self.epsilon) * below

    def set_alpha(self, new_alpha: float) -> None:
        """Mutate target α and recreate the P² tracker with the new quantile.

        Called by the adaptive-α controller (methodology §5c.4 /
        Phase-1 E2/E6). Plain assignment to ``self.alpha`` is NOT sufficient:
        the P² tracker's target quantile is fixed at its own ``__init__``
        time, so without a fresh tracker the gate continues to estimate the
        old quantile and ignores the new α.

        Parameters
        ----------
        new_alpha : float
            New target skip-fraction. Must be in (0, 1). Caller (the outer
            controller) is responsible for clipping to [α_min, α_max] before
            invoking this method.

        Raises
        ------
        ValueError
            If ``new_alpha`` is not strictly within (0, 1). The endpoint
            semantics (α=0 / α=1) are handled statically in
            :meth:`current_threshold` and do not touch the tracker — a
            controller should never push the gate to an endpoint anyway.
        """
        if not 0.0 < float(new_alpha) < 1.0:
            raise ValueError(f"set_alpha: new_alpha must be in (0, 1), got {new_alpha}")
        self.alpha = float(new_alpha)
        self._tracker = self._make_tracker()

    def _make_tracker(self) -> P2QuantileTracker:
        """Instantiate a P² tracker for the (1-α)-quantile.

        Endpoints are handled by :meth:`current_threshold` and do not touch
        the tracker, so we can always instantiate at the interior quantile.
        For the endpoint case where (1-α) is exactly 0 or 1 we clip to a
        safe interior value (the tracker itself is unused in that case).
        """
        target_q = max(min(1.0 - self.alpha, 1.0 - 1e-6), 1e-6)
        return P2QuantileTracker(p=target_q)
