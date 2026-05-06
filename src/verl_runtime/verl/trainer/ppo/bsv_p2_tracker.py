# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
"""P² (Jain & Chlamtac 1985) streaming quantile estimator.

Reference:
    Jain & Chlamtac, "The P² Algorithm for Dynamic Calculation of Quantiles
    and Histograms Without Storing Observations", CACM 28(10), 1985.

The estimator maintains five "markers" whose heights approximate a fixed
set of quantiles of the observed stream (0, p/2, p, (1+p)/2, 1). After each
observation, the markers are updated using a parabolic-interpolation rule
with a linear fallback so that the middle marker's height converges to the
p-th quantile of the stream.

Properties:
* O(1) memory per tracker (constant — 5 heights, 5 positions, 5 desired
  positions, 5 increments per observation).
* O(1) amortized update.
* Used by the BSV gate (DEFINE_SPEC_v3 §1.2) to track t(α), the
  (1-α)-quantile of pairwise-judge confidence on the training stream.

Design notes / deviations kept explicit:
* For n < 5 the canonical P² paper leaves the estimator undefined, since
  the parabolic step needs five sorted "markers". We provide a graceful
  fall-back: quantile() linearly interpolates from the sorted observations
  seen so far (matches np.quantile's behaviour on small samples). This
  keeps callers safe during training warmup without polluting the core
  P² machinery.
"""
from __future__ import annotations

from typing import List


class P2QuantileTracker:
    """Streaming estimator of the p-th quantile via Jain & Chlamtac's P².

    Parameters
    ----------
    p : float
        Target quantile in the open interval (0, 1). E.g., ``p=0.3`` tracks
        the 30th percentile.

    Raises
    ------
    ValueError
        If ``p`` is not strictly between 0 and 1.
    """

    # We deliberately annotate the instance attributes here to document the
    # fixed O(1) memory footprint: five-element marker arrays + four scalars.
    q: List[float]      # marker heights (sorted)
    n: List[float]      # marker positions (1-indexed in the paper; stored as floats)
    np_: List[float]    # desired marker positions
    dn: List[float]     # increments for desired positions
    _p: float
    _count: int
    _initial: List[float]

    def __init__(self, p: float) -> None:
        if not isinstance(p, (int, float)):
            raise ValueError(f"p must be a real number, got {type(p)!r}")
        if not (0.0 < float(p) < 1.0):
            raise ValueError(f"p must be in the open interval (0, 1), got {p}")
        self._p = float(p)
        # Marker heights, positions, desired positions, increments (Jain § §3).
        self.q = [0.0] * 5
        self.n = [0.0] * 5
        self.np_ = [0.0] * 5
        self.dn = [0.0] * 5
        self._count = 0
        self._initial = []  # buffer first five observations before initialization.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, x: float) -> None:
        """Feed a new observation into the estimator. O(1)."""
        x = float(x)
        self._count += 1
        if self._count <= 5:
            self._initial.append(x)
            if self._count == 5:
                self._initialize()
            return
        self._update_p2(x)

    def quantile(self) -> float:
        """Current best estimate of the p-th quantile.

        After initialization (n >= 5) this returns the middle-marker height
        (canonical P²). For n < 5 we linearly interpolate from the sorted
        observations to keep the output finite and monotone-ish. At n = 0
        we return 0.0 as a prior (the caller should treat this as a warmup
        value; all BSV callsites rely on ``n_observations()`` to gate
        behaviour during warmup).
        """
        if self._count == 0:
            return 0.0
        if self._count < 5:
            sorted_obs = sorted(self._initial)
            # Linear interpolation at position (n-1)*p  (numpy 'linear' rule).
            n = len(sorted_obs)
            if n == 1:
                return sorted_obs[0]
            idx_f = (n - 1) * self._p
            lo = int(idx_f)
            hi = min(lo + 1, n - 1)
            frac = idx_f - lo
            return sorted_obs[lo] * (1.0 - frac) + sorted_obs[hi] * frac
        return self.q[2]

    def n_observations(self) -> int:
        """Number of observations processed so far."""
        return self._count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Set up the 5 markers once we have 5 observations (Jain §3 step A)."""
        sorted_obs = sorted(self._initial)
        for i in range(5):
            self.q[i] = sorted_obs[i]
            # Marker positions: 1..5 (1-indexed in the paper).
            self.n[i] = float(i + 1)
        p = self._p
        self.np_ = [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0]
        self.dn = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]
        # Once initialised we no longer need the buffer.
        self._initial = []

    def _update_p2(self, x: float) -> None:
        """Standard P² update (Jain §3 steps B1-B4)."""
        q = self.q
        n = self.n
        np_ = self.np_
        dn = self.dn

        # Step B1 — find cell k such that q[k] <= x < q[k+1].
        if x < q[0]:
            q[0] = x
            k = 0
        elif x >= q[4]:
            q[4] = x
            k = 3
        else:
            k = 0
            for i in range(4):
                if q[i] <= x < q[i + 1]:
                    k = i
                    break

        # Step B2 — increment positions.
        for i in range(k + 1, 5):
            n[i] += 1.0
        for i in range(5):
            np_[i] += dn[i]

        # Step B3 — adjust heights of inner markers 2..4 (indices 1..3).
        for i in (1, 2, 3):
            d = np_[i] - n[i]
            cond_pos = d >= 1.0 and (n[i + 1] - n[i]) > 1.0
            cond_neg = d <= -1.0 and (n[i - 1] - n[i]) < -1.0
            if cond_pos or cond_neg:
                sign = 1.0 if d >= 0.0 else -1.0
                q_para = self._parabolic(i, sign)
                # Use parabolic if it keeps q[i-1] < q_para < q[i+1]; else linear.
                if q[i - 1] < q_para < q[i + 1]:
                    q[i] = q_para
                else:
                    q[i] = self._linear(i, sign)
                n[i] += sign

    def _parabolic(self, i: int, d: float) -> float:
        """Parabolic prediction (Jain §3 eq. after step B3)."""
        q = self.q
        n = self.n
        term1 = d / (n[i + 1] - n[i - 1])
        term2 = (n[i] - n[i - 1] + d) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
        term3 = (n[i + 1] - n[i] - d) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
        return q[i] + term1 * (term2 + term3)

    def _linear(self, i: int, d: float) -> float:
        """Linear fallback (Jain §3 eq. after the parabolic eq.)."""
        q = self.q
        n = self.n
        j = int(i + d)
        return q[i] + d * (q[j] - q[i]) / (n[j] - n[i])
