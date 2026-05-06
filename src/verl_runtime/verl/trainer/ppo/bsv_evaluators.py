# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
"""BSV off-policy evaluators (DEFINE_SPEC_v3 §3.3).

Three estimators, all CPU / NumPy-only:

1. :func:`ips_reward` — Horvitz-Thompson / inverse-propensity-scored mean
   reward. Unbiased under the true propensity; divides by the sum of the
   inverse propensities (Hájek / self-normalized variant).
2. :func:`dr_reward` — Doubly-robust estimator (Dudík, Langford, Li 2011,
   arXiv:1103.4601). Consistent if EITHER the propensity or the direct
   model of R is correct.
3. :func:`audit_regret` — Mean absolute error between the verifier and
   the self-judge on the ε-exploration audit slice. Measures gate quality
   INDEPENDENTLY of the gate's own choices (the headline "money metric"
   per §3.3).

All functions accept NumPy arrays and return a scalar ``float``. ``R`` is
expected to carry NaN at unobserved entries (I_v = 0); those entries are
never used (they are masked out by ``I_v``).
"""
from __future__ import annotations

import numpy as np


def ips_reward(
    R: np.ndarray,
    p_invoke: np.ndarray,
    I_v: np.ndarray,
    normalize: bool = True,
) -> float:
    """Inverse-Propensity-Scored reward estimator.

    Two variants, selected by ``normalize``:

    * ``normalize=False`` — **Horvitz-Thompson** (the estimator proved
      unbiased in `bsv_methodology.md §6.2`)::

          V̂_HT = (1/N) · Σ_i (I_v_i / p_invoke_i) · R_i

      ``E[V̂_HT] = E[R]`` exactly for any N, given positive propensities.
      Higher variance, not guaranteed to lie in the support of R.

    * ``normalize=True`` — **Self-normalized / Hájek** (lower-variance,
      consistent, finite-sample biased at O(1/N))::

          V̂_SNIPS = Σ_i (I_v_i / p_invoke_i) · R_i
                    -------------------------------
                       Σ_i (I_v_i / p_invoke_i)

      Ratio estimator of Swaminathan & Joachims 2015 (arXiv:1502.02362).
      Preferred in practice; the paper should report both.

    Parameters
    ----------
    R : np.ndarray, shape (n,)
        Verifier rewards. NaN where ``I_v = 0`` (not observed).
    p_invoke : np.ndarray, shape (n,)
        Propensity P(I_v = 1 | context) under the logging policy.
        Must be strictly positive wherever ``I_v = 1`` (otherwise the
        ratio is undefined).
    I_v : np.ndarray, shape (n,)
        Invocation indicator ∈ {0, 1}.
    normalize : bool, default True
        ``True`` → Hájek/SNIPS estimator (divide by sum of weights).
        ``False`` → Horvitz-Thompson estimator (divide by total N).

    Returns
    -------
    float
        IPS estimate. Returns 0.0 if ``normalize=True`` and no pair was
        observed (sum of weights is zero). Returns 0.0 if
        ``normalize=False`` and N == 0.

    Raises
    ------
    ValueError
        On shape mismatch or non-positive propensity at an observed pair.
    """
    R = np.asarray(R, dtype=np.float64)
    p = np.asarray(p_invoke, dtype=np.float64)
    iv = np.asarray(I_v, dtype=np.float64)
    if R.shape != p.shape or R.shape != iv.shape:
        raise ValueError(
            f"ips_reward: shape mismatch R={R.shape}, p_invoke={p.shape}, I_v={iv.shape}"
        )
    observed = iv > 0.5
    # Propensity must be positive on observed entries; silent zero would
    # otherwise yield Inf. NaN propensities are also rejected.
    if observed.any():
        p_obs = p[observed]
        if np.any(~np.isfinite(p_obs)) or np.any(p_obs <= 0.0):
            raise ValueError(
                "ips_reward: p_invoke must be positive and finite on observed entries"
            )
    # Mask out unobserved rows (NaN in R is allowed there).
    w = np.where(observed, 1.0 / np.where(observed, p, 1.0), 0.0)
    R_safe = np.where(observed, R, 0.0)
    num = float(np.sum(w * R_safe))
    if normalize:
        den = float(np.sum(w))
        if den == 0.0:
            return 0.0
        return num / den
    n = int(R.size)
    if n == 0:
        return 0.0
    return num / float(n)


def ips_reward_ht(
    R: np.ndarray,
    p_invoke: np.ndarray,
    I_v: np.ndarray,
) -> float:
    """Horvitz-Thompson IPS estimator (unbiased, higher variance).

    Thin convenience wrapper for ``ips_reward(..., normalize=False)`` that
    makes the estimator choice explicit at the call site and matches the
    unbiasedness proof in `bsv_methodology.md §6.2`.
    """
    return ips_reward(R, p_invoke, I_v, normalize=False)


def dr_reward(
    R: np.ndarray,
    p_invoke: np.ndarray,
    I_v: np.ndarray,
    direct_estimate: np.ndarray,
) -> float:
    """Doubly-robust reward estimator (Dudík-Langford-Li, arXiv:1103.4601).

    Formula::

        DR = (1/n) · sum_i [  direct_i  +  (R_i − direct_i) · 1{I_v=1} / p_invoke_i  ]

    The estimator is consistent if EITHER

      * the propensity ``p_invoke`` equals the true logging policy, OR
      * the direct model ``direct_estimate`` equals ``E[R | context]``

    hence "doubly robust".

    Parameters
    ----------
    R : np.ndarray, shape (n,)
        Verifier rewards; NaN where unobserved.
    p_invoke : np.ndarray, shape (n,)
        Propensity P(I_v = 1 | context). Must be positive where observed.
    I_v : np.ndarray, shape (n,)
        Invocation indicator.
    direct_estimate : np.ndarray, shape (n,)
        Model's ``R̂(context)`` prediction for every pair (observed or
        not).

    Returns
    -------
    float
        Doubly-robust reward estimate.
    """
    R = np.asarray(R, dtype=np.float64)
    p = np.asarray(p_invoke, dtype=np.float64)
    iv = np.asarray(I_v, dtype=np.float64)
    d = np.asarray(direct_estimate, dtype=np.float64)
    if not (R.shape == p.shape == iv.shape == d.shape):
        raise ValueError(
            f"dr_reward: shape mismatch R={R.shape}, p_invoke={p.shape}, "
            f"I_v={iv.shape}, direct_estimate={d.shape}"
        )
    n = R.shape[0]
    if n == 0:
        return 0.0
    observed = iv > 0.5
    if observed.any():
        p_obs = p[observed]
        if np.any(~np.isfinite(p_obs)) or np.any(p_obs <= 0.0):
            raise ValueError(
                "dr_reward: p_invoke must be positive and finite on observed entries"
            )
    R_safe = np.where(observed, R, 0.0)
    iv_f = observed.astype(np.float64)
    # (R - direct) is 0 by convention on unobserved rows (we set R_safe=0
    # there, and multiply by I_v anyway).
    correction = (R_safe - d) * iv_f / np.where(observed, p, 1.0)
    correction = np.where(observed, correction, 0.0)
    return float(np.mean(d + correction))


def audit_regret(
    R_audit: np.ndarray,
    p_self_audit: np.ndarray,
) -> float:
    """Mean absolute disagreement between verifier and self-judge on the audit slice.

    Formula::

        audit_regret = (1/|audit|) · sum_{i ∈ audit} |R_i − p_self_i|

    The caller is responsible for selecting the audit slice — per
    DEFINE_SPEC_v3 §3.3 the audit slice is the subset of pairs where
    ``c ≥ t(α)`` (high-confidence) BUT ``I_v = 1`` was nonetheless drawn
    by the ε-exploration floor. This estimator is completely agnostic to
    how the slice was formed; it just asks: "when the verifier DID answer
    on a pair the gate would have skipped, how wrong was the self-judge?"

    Lower is better. Zero means the self-judge was perfect on the audit
    sample. A rising audit_regret in α is the kill condition §5.2.3.

    Parameters
    ----------
    R_audit : np.ndarray, shape (m,)
        Verifier rewards on the audit slice.
    p_self_audit : np.ndarray, shape (m,)
        Self-judge preference probabilities on the audit slice.

    Returns
    -------
    float
        Mean absolute gap. Returns 0.0 on an empty slice (undefined but
        harmless — the caller should treat 0 on empty as "no audit data
        yet" and not as a perfect score).
    """
    R = np.asarray(R_audit, dtype=np.float64)
    ps = np.asarray(p_self_audit, dtype=np.float64)
    if R.shape != ps.shape:
        raise ValueError(
            f"audit_regret: shape mismatch R_audit={R.shape}, p_self_audit={ps.shape}"
        )
    if R.size == 0:
        return 0.0
    return float(np.mean(np.abs(R - ps)))
