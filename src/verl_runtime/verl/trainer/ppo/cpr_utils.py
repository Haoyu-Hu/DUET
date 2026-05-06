# Copyright 2026 CPR contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for Calibrated Preference RL (CPR).

This module is additive to the existing PPO/SIRL/SIRL-Pref stack. It provides
the CPR-specific data path:

- tie-aware self-judge probabilities ``p_self``;
- external-verifier mixing into ``p_pref``;
- pair construction from rollout batches;
- packaging paired responses into ``DataProto`` objects for log-prob evaluation.

Prior-art lineage (cite in any paper draft):

- GPO (Furuta et al., 2024, arXiv:2409.06691) — soft-label DPO via geometric
  aggregation of a *single* labeler. CPR shares the soft-label loss shape but
  treats the label itself as a mixture random variable indexed by ``alpha``.
- Self-Rewarding Language Models (Yuan et al., 2024) — iterative self-judge
  DPO. Recovered as ``alpha = 1`` (with prefix-branch pairs).
- Co-rewarding (Zhang et al., 2025, arXiv:2508.00410) — two-view agreement.
  Recovered as the ``alpha = 0.5`` special case.
- DICE / Implicit Reward as the Bridge (Chen et al., 2024, arXiv:2406.09760)
  — implicit-reward RL lineage; CPR's verifier-only end (``alpha = 0``)
  reduces to RLVR-style training under this framing.

CPR's contribution is the **continuous trust dial**: the phase diagram over
``alpha in [0, 1]`` is a measurement of the trust prior under which the
training distribution best matches the data-generating process, not a
hyperparameter sweep.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.sirl_pref_utils import select_branch_point
from verl.trainer.ppo.sirl_utils import _extract_question_from_prompt, _generate_follow_ups_with_tokens
from verl.utils.model import compute_position_id_with_mask


_log = logging.getLogger(__name__)


CPR_JUDGE_TEMPLATE = (
    "SYSTEM: You are an impartial grader. Given a problem and two candidate\n"
    "solutions labeled (A) and (B), output a single token: A, B, or T (tie),\n"
    "choosing the solution that is more likely correct (T if indistinguishable).\n"
    "Do not explain."
)


@dataclass
class CPRConfig:
    """Runtime configuration for CPR."""

    enable: bool = False
    alpha: float = 0.5
    beta: float = 0.15
    eta: float = 0.0
    pair_source: str = "independent"
    rollout_n: int = 4
    tau_ext: float = 1.0
    judge_model: str = "online"
    judge_ema_tau: float = 0.99
    judge_ema_refresh_steps: int = 100
    judge_max_new_tokens: int = 1
    judge_order_swap: bool = True
    label_noise_rho: float = 0.0
    arm_id: str = "cpr_alpha0.5"
    judge_prompt_template: str = CPR_JUDGE_TEMPLATE
    max_prompt_length: int = 1024
    max_response_length: int = 3072
    anchor_mode: str = "symmetric"
    alpha_gate_gamma: float = 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    denom = np.sum(e)
    if denom <= 0:
        return np.full_like(x, 1.0 / len(x), dtype=np.float32)
    return (e / denom).astype(np.float32)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Per-element binary entropy of a probability vector, in nats."""
    pc = np.clip(np.asarray(p, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return -(pc * np.log(pc) + (1.0 - pc) * np.log(1.0 - pc))


def _parse_judge_label(text: str) -> Optional[str]:
    """Parse a greedy judge reply into ``A`` / ``B`` / ``T``."""
    if not text:
        return None
    cleaned = text.strip().upper()
    if re.search(r"\bT(?:IE)?\b", cleaned):
        return "T"
    has_a = re.search(r"(?<![A-Z])A(?![A-Z])", cleaned) is not None
    has_b = re.search(r"(?<![A-Z])B(?![A-Z])", cleaned) is not None
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    return None


def _extract_abt_probs(raw: Any) -> np.ndarray:
    """Extract normalized {A, B, T} probabilities from judge output.

    Supported forms:
    - ``{"A": ..., "B": ..., "T": ...}``
    - ``{"token_probs": {...}}`` / ``{"first_token_probs": {...}}``
    - ``{"first_token_logits": {...}}``
    - hard string labels ``A`` / ``B`` / ``T``
    """
    if isinstance(raw, dict):
        for key in ("token_probs", "first_token_probs", "probs"):
            if key in raw and isinstance(raw[key], dict):
                rec = raw[key]
                probs = np.asarray([
                    float(rec.get("A", 0.0)),
                    float(rec.get("B", 0.0)),
                    float(rec.get("T", rec.get("tie", 0.0))),
                ], dtype=np.float32)
                denom = float(probs.sum())
                return probs / denom if denom > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "first_token_logits" in raw and isinstance(raw["first_token_logits"], dict):
            rec = raw["first_token_logits"]
            logits = np.asarray([
                float(rec.get("A", -1e9)),
                float(rec.get("B", -1e9)),
                float(rec.get("T", rec.get("tie", -1e9))),
            ], dtype=np.float32)
            return _softmax(logits)
        if {"A", "B", "T"}.issubset(set(raw.keys())):
            probs = np.asarray([float(raw["A"]), float(raw["B"]), float(raw["T"])], dtype=np.float32)
            denom = float(probs.sum())
            return probs / denom if denom > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    label = _parse_judge_label(str(raw))
    if label == "A":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if label == "B":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if label == "T":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)


def _build_cpr_judge_prompt(question: str, y_first: str, y_second: str, template: str) -> str:
    return (
        f"{template}\n\n"
        f"USER:\n"
        f"Problem:\n{question}\n\n"
        f"Solution (A):\n{y_first}\n\n"
        f"Solution (B):\n{y_second}\n\n"
        "Which solution is more likely correct? Answer with a single letter (A, B, or T):"
    )


def p_self_order_swap(
    *,
    questions: list[str],
    y_a_texts: list[str],
    y_b_texts: list[str],
    judge_generator: Callable[[list[str]], list[Any]],
    cfg: CPRConfig,
) -> np.ndarray:
    """Compute tie-aware, order-swapped ``p_self(y_a ≻ y_b | x)`` in [0, 1]."""

    batch_size = len(questions)
    assert len(y_a_texts) == batch_size and len(y_b_texts) == batch_size

    forward_prompts = [
        _build_cpr_judge_prompt(questions[i], y_a_texts[i], y_b_texts[i], cfg.judge_prompt_template)
        for i in range(batch_size)
    ]
    forward_outputs = judge_generator(forward_prompts) if batch_size > 0 else []
    forward_probs = [_extract_abt_probs(out) for out in forward_outputs]

    if cfg.judge_order_swap:
        swapped_prompts = [
            _build_cpr_judge_prompt(questions[i], y_b_texts[i], y_a_texts[i], cfg.judge_prompt_template)
            for i in range(batch_size)
        ]
        swapped_outputs = judge_generator(swapped_prompts) if batch_size > 0 else []
        swapped_probs = [_extract_abt_probs(out) for out in swapped_outputs]
    else:
        swapped_probs = [np.array([0.0, 0.0, 1.0], dtype=np.float32)] * batch_size

    out = np.zeros(batch_size, dtype=np.float32)
    for i in range(batch_size):
        q1 = forward_probs[i]
        if cfg.judge_order_swap:
            q2 = swapped_probs[i]
            out[i] = 0.5 * ((q1[0] + 0.5 * q1[2]) + (q2[1] + 0.5 * q2[2]))
        else:
            out[i] = q1[0] + 0.5 * q1[2]
    return np.clip(out, 0.0, 1.0)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pc = np.clip(p.astype(np.float32), eps, 1.0 - eps)
    return np.log(pc / (1.0 - pc))


# --------------------------------------------------------------------------
# E4 frozen-base judge helpers — DEFINE_SPEC_v2 §2.5
# --------------------------------------------------------------------------


def _order_swap_judge(
    judge_fn: Callable[[str], Any],
    prompt: str,
    cand_a: str,
    cand_b: str,
    template: str = CPR_JUDGE_TEMPLATE,
) -> float:
    """Tie-safe order-swap average of a single-item judge.

    ``judge_fn(prompt_AB)`` takes a single composed judge prompt (problem +
    candidates A, B inlined via :func:`_build_cpr_judge_prompt`) and returns
    either a 3-tuple ``(q_A, q_B, q_T)``, a dict parsable by
    :func:`_extract_abt_probs`, or a raw label string. The forward call scores
    ``(cand_a, cand_b)`` and the swapped call scores ``(cand_b, cand_a)``; we
    average both "cand_a wins" probabilities with tie mass split evenly,
    matching :func:`p_self_order_swap`.

    Returns a scalar probability ``p_self(cand_a ≻ cand_b | prompt)`` in [0, 1].
    """
    fwd_prompt = _build_cpr_judge_prompt(prompt, cand_a, cand_b, template)
    swp_prompt = _build_cpr_judge_prompt(prompt, cand_b, cand_a, template)
    raw_fwd = judge_fn(fwd_prompt)
    raw_swp = judge_fn(swp_prompt)

    def _to_abt(raw: Any) -> np.ndarray:
        if isinstance(raw, tuple) and len(raw) == 3:
            arr = np.asarray([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float32)
            s = float(arr.sum())
            return arr / s if s > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return _extract_abt_probs(raw)

    q1 = _to_abt(raw_fwd)  # (A, B, T) with A=cand_a
    q2 = _to_abt(raw_swp)  # (A, B, T) with A=cand_b, B=cand_a
    # Forward: prob(cand_a wins) = q1_A + 0.5·q1_T
    # Swapped: prob(cand_a wins) = prob(B wins) = q2_B + 0.5·q2_T
    p = 0.5 * ((q1[0] + 0.5 * q1[2]) + (q2[1] + 0.5 * q2[2]))
    return float(np.clip(p, 0.0, 1.0))


def compute_p_self(
    judge_live_fn: Callable[[str], Any],
    prompt: str,
    cand_a: str,
    cand_b: str,
    template: str = CPR_JUDGE_TEMPLATE,
) -> float:
    """Single-item ``p_self`` under the LIVE judge (thin wrapper around
    :func:`_order_swap_judge`). Mirrors the batched :func:`p_self_order_swap`
    one row at a time, taking the judge template as an override for tests."""
    return _order_swap_judge(judge_live_fn, prompt, cand_a, cand_b, template)


def compute_p_self_frozen(
    judge_frozen_fn: Callable[[str], Any],
    prompt: str,
    cand_a: str,
    cand_b: str,
    template: str = CPR_JUDGE_TEMPLATE,
) -> float:
    """Single-item ``p_self`` under a FROZEN-base judge.

    Takes a single-argument judge callable ``judge_frozen_fn(prompt_AB)`` that
    returns ``(q_A, q_B, q_T)`` (or an equivalent parseable object) from a
    frozen base model. Used for E4 post-hoc recovery where the policy's live
    ``p_self`` is confounded by policy drift.
    """
    return _order_swap_judge(judge_frozen_fn, prompt, cand_a, cand_b, template)


# --------------------------------------------------------------------------
# E2 coupling helpers — DEFINE_SPEC_v2 §2.3
# --------------------------------------------------------------------------
# HISTORY: these helpers were originally named with "MI" framing
# (``compute_coupling_kl`` / ``compute_mi_coupling_metrics``), but the
# per-pair Bernoulli KL is NOT the mutual information I(p_self; π_pref). The
# 2026-04-19 CPR develop review (BLOCKER-3) flagged this. The KL is now named
# explicitly as a "coupling calibration KL" (retaining its utility as a
# coupling proxy), and a separate KSG-KNN estimator ``compute_coupling_mi``
# provides a true MI on the empirical joint distribution.


def compute_coupling_calibration_kl(p_self: np.ndarray, pi_pref: np.ndarray) -> np.ndarray:
    """Per-pair Bernoulli KL ``KL(Bern(p_self_i) || Bern(π_pref_i))``.

    This is a **per-pair** Bernoulli KL divergence between two preference
    probabilities — **NOT the mutual information** ``I(p_self; π_pref)``
    between the random variables across pairs. See :func:`compute_coupling_mi`
    for a KSG-KNN estimator of the actual MI.

    Both inputs are pairwise preference probabilities in [0, 1]. Returns an
    array of shape matching the inputs, giving the Bernoulli KL per element:

        KL = p·log(p/q) + (1−p)·log((1−p)/(1−q))

    Used by the E2 logger as a **calibration coupling proxy**: large values
    indicate the live self-judge ``p_self`` disagrees with the policy's
    length-normalized preference ``π_pref`` on a per-pair basis. Zero does
    NOT imply MI is zero (two variables can be perfectly calibrated while
    carrying high mutual information); high does NOT imply MI is high
    (calibration mismatch can exist between independent variables with
    different means).
    """
    eps = 1e-6
    p = np.clip(np.asarray(p_self, dtype=np.float32), eps, 1.0 - eps)
    q = np.clip(np.asarray(pi_pref, dtype=np.float32), eps, 1.0 - eps)
    if p.shape != q.shape:
        raise ValueError(
            f"p_self and pi_pref must share shape; got {p.shape} vs {q.shape}"
        )
    kl = p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))
    return kl.astype(np.float32)


def compute_pi_pref_lengthnorm(
    logp_a: np.ndarray,
    logp_b: np.ndarray,
    len_a: np.ndarray,
    len_b: np.ndarray,
) -> np.ndarray:
    """Length-normalized sequence-log-prob pairwise preference.

    ``π_pref(a > b | x) = σ(log π(a|x)/len(a) − log π(b|x)/len(b))``.

    Accepts batched arrays of sequence log-probs and token counts. Returns a
    float32 array of shape ``logp_a.shape`` with entries in [0, 1].
    """
    logp_a = np.asarray(logp_a, dtype=np.float32)
    logp_b = np.asarray(logp_b, dtype=np.float32)
    len_a = np.asarray(len_a, dtype=np.float32)
    len_b = np.asarray(len_b, dtype=np.float32)
    # Guard against zero-length rows — avoids div-by-zero; such rows collapse
    # to 0 log-prob per token (a prior-free tie) rather than NaN.
    la = np.clip(len_a, 1.0, None)
    lb = np.clip(len_b, 1.0, None)
    diff = logp_a / la - logp_b / lb
    return _sigmoid(diff).astype(np.float32)


def compute_coupling_mi(
    p_self: np.ndarray,
    pi_pref: np.ndarray,
    k: int = 3,
) -> float:
    """Kraskov-Stoegbauer-Grassberger (KSG) KNN estimator of ``I(p_self; π_pref)``.

    Non-parametric estimator of the **mutual information** between two scalar
    samples on the empirical joint distribution
    ``{(p_self_i, π_pref_i)}_{i=1..N}``. Implements the KSG estimator from
    Kraskov, Stoegbauer & Grassberger (2004), "Estimating mutual information"
    (arXiv:cond-mat/0305641), algorithm 1 (Chebyshev / ℓ∞ distance).

    Formula (1D for each variable, Chebyshev joint distance):

        for each sample i:
            ε_i = distance to k-th nearest neighbour in the joint (max of
                  |Δp_self|, |Δπ_pref|)
            n_x_i = # other samples with |Δp_self| < ε_i / 2 + tiny
            n_y_i = # other samples with |Δπ_pref| < ε_i / 2 + tiny
        I ≈ ψ(k) + ψ(N) − (1/N) Σ_i [ψ(n_x_i + 1) + ψ(n_y_i + 1)]

    The result is clipped at 0 (KSG can go slightly negative under
    finite-sample bias for independent samples). Requires ``N ≥ 2k+1``; for
    smaller ``N`` returns 0.0 (uninformative).

    Parameters
    ----------
    p_self, pi_pref : arrays of shape (N,), entries in [0, 1].
    k : number of nearest neighbours for the local-density estimate. Default
        ``k=3`` matches the KSG paper recommendation.

    Returns
    -------
    float — MI estimate in nats, clipped to ``[0, ∞)``.
    """
    from scipy.special import digamma

    x = np.asarray(p_self, dtype=np.float64).ravel()
    y = np.asarray(pi_pref, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(
            f"p_self and pi_pref must share shape; got {x.shape} vs {y.shape}"
        )
    n = x.size
    if n < 2 * k + 1:
        return 0.0

    # Joint Chebyshev distance matrix between all pairs.
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    d_joint = np.maximum(dx, dy)

    # k-th nearest neighbour distance (excluding self at index 0). With
    # np.partition, position k gives the k-th smallest (we discard the
    # 0-distance self-match at position 0).
    d_sorted = np.partition(d_joint, kth=k, axis=1)
    eps = d_sorted[:, k]  # strictly positive in generic draws

    # Count marginal neighbours strictly inside ε_i (NOT including self).
    # KSG's algorithm 1 uses strict inequality |Δx_i| < ε_i; convention notes
    # below the formula in the paper. We exclude self by subtracting 1 from
    # the count (since dx[i, i] = 0 < ε_i trivially).
    n_x = (dx < eps[:, None]).sum(axis=1) - 1
    n_y = (dy < eps[:, None]).sum(axis=1) - 1
    # Guard against degenerate counts of 0 (e.g., zero ε from duplicates).
    n_x = np.clip(n_x, 0, None)
    n_y = np.clip(n_y, 0, None)

    mi = (
        digamma(k)
        + digamma(n)
        - float(np.mean(digamma(n_x + 1) + digamma(n_y + 1)))
    )
    return float(max(mi, 0.0))


def compute_coupling_metrics(
    p_self: np.ndarray,
    logp_a: np.ndarray,
    logp_b: np.ndarray,
    len_a: np.ndarray,
    len_b: np.ndarray,
    step: int | None = None,
    prior_step: int | None = None,
    prior_auc: float | None = None,
    prior_kl_mean: float | None = None,
    mi_k: int = 3,
) -> dict:
    """Aggregate E2 coupling metrics for a probe-pair batch.

    Computes ``π_pref`` from length-normalized sequence log-probs, pairs it
    with ``p_self``, and returns a metrics dict with:

      - ``cpr/coupling_calibration_kl_mean`` — mean per-pair Bernoulli KL
        (see :func:`compute_coupling_calibration_kl`; this is a coupling
        *calibration* proxy, NOT mutual information).
      - ``cpr/coupling_calibration_kl_auc`` — trapezoid-rule running integral
        across val steps; the caller threads ``prior_{auc,step,kl_mean}``
        forward.
      - ``cpr/pi_pref_mean`` — mean of ``π_pref`` over the batch.
      - ``cpr/coupling_mi`` — KSG-KNN estimate of ``I(p_self; π_pref)`` (only
        when the batch has ≥ ``2 · mi_k + 1`` samples; otherwise omitted).

    Trapezoid update (when all prior values + new ``step`` are supplied):

        AUC_new = AUC_prev + 0.5 · (kl_prev + kl_new) · Δstep

    When ``step`` is given without prior values, returns the AUC "seed"
    (kl_mean) as both the mean and the AUC — suitable for the first val step
    of a run. Callers that prefer a state-free logger should drop the AUC
    keys and rely on a CSV sidecar (see DEVELOP_REPORT open question).

    This was previously named ``compute_mi_coupling_metrics``; the name was
    changed in the BLOCKER-3 fix (2026-04-19 CPR develop review) because the
    per-pair KL was misidentified as mutual information. The new
    ``cpr/coupling_mi`` emits the actual KSG MI estimate.
    """
    pi_pref = compute_pi_pref_lengthnorm(logp_a, logp_b, len_a, len_b)
    kl_per = compute_coupling_calibration_kl(p_self, pi_pref)
    kl_mean = float(np.mean(kl_per)) if kl_per.size else 0.0

    out: dict = {
        "cpr/coupling_calibration_kl_mean": kl_mean,
        "cpr/pi_pref_mean": float(np.mean(pi_pref)) if pi_pref.size else 0.0,
    }
    # Emit KSG MI only when batch is large enough (≥ 2k+1 samples).
    if np.asarray(p_self).size >= 2 * mi_k + 1:
        out["cpr/coupling_mi"] = compute_coupling_mi(
            np.asarray(p_self, dtype=np.float64),
            np.asarray(pi_pref, dtype=np.float64),
            k=mi_k,
        )
    if step is not None:
        if prior_auc is not None and prior_step is not None and prior_kl_mean is not None \
                and step > prior_step:
            dx = float(step - prior_step)
            out["cpr/coupling_calibration_kl_auc"] = float(prior_auc) + 0.5 * (
                float(prior_kl_mean) + kl_mean
            ) * dx
        else:
            # Seed: no prior data point, contribute zero area at t=step.
            out["cpr/coupling_calibration_kl_auc"] = 0.0
    return out


def compute_alpha_eff(
    *,
    p_self: np.ndarray,
    p_verif: np.ndarray | None,
    alpha: float,
    alpha_gate_gamma: float,
) -> np.ndarray:
    """Per-example effective α for disagreement-gated CPR.

    ``α_eff(x) = α · exp(-γ · |logit(p_self) - logit(p_verif)|)`` when γ>0 and
    both labelers are present. γ=0 or missing verifier returns constant α.
    """
    p_self = np.asarray(p_self, dtype=np.float32)
    if p_verif is None or float(alpha_gate_gamma) <= 0.0 or float(alpha) <= 0.0:
        return np.full_like(p_self, float(alpha), dtype=np.float32)
    logit_gap = np.abs(_logit(p_self) - _logit(np.asarray(p_verif, dtype=np.float32)))
    gate = np.exp(-float(alpha_gate_gamma) * logit_gap).astype(np.float32)
    return (float(alpha) * gate).astype(np.float32)


def compute_p_pref(
    *,
    p_self: np.ndarray,
    r_ext_a: np.ndarray | None,
    r_ext_b: np.ndarray | None,
    alpha: float,
    tau_ext: float,
    label_noise_rho: float = 0.0,
    alpha_gate_gamma: float = 0.0,
) -> np.ndarray:
    """Compute the CPR soft preference label ``p_pref`` in [0, 1].

    When ``alpha_gate_gamma > 0`` and both labelers are present, α becomes a
    per-example ``α_eff(x)``; see :func:`compute_alpha_eff`. γ=0 (default) and
    the public signature are both unchanged from the committed API.
    """
    p_self = np.asarray(p_self, dtype=np.float32)
    has_verifier = r_ext_a is not None and r_ext_b is not None

    if not has_verifier:
        return np.clip(p_self.astype(np.float32), 0.0, 1.0)

    r_ext_a = np.asarray(r_ext_a, dtype=np.float32)
    r_ext_b = np.asarray(r_ext_b, dtype=np.float32)
    ext_term = _sigmoid((r_ext_a - r_ext_b) / max(float(tau_ext), 1e-6)).astype(np.float32)

    # Endpoint recovery short-circuits come FIRST so α=1 → pure self-rewarding
    # and α=0 → pure verifier hold regardless of alpha_gate_gamma. Otherwise
    # the gated branch would shrink α_eff at the α=1 endpoint and break the
    # "α=1 recovers Self-Rewarding LMs" methodology claim (CPR debate audit,
    # 2026-04-19).
    if float(alpha) >= 1.0:
        p_pref = p_self.copy()
    elif float(alpha) <= 0.0:
        p_pref = ext_term.copy()
    elif float(alpha_gate_gamma) > 0.0:
        alpha_eff = compute_alpha_eff(
            p_self=p_self, p_verif=ext_term, alpha=alpha, alpha_gate_gamma=alpha_gate_gamma
        )
        p_pref = (alpha_eff * p_self + (1.0 - alpha_eff) * ext_term).astype(np.float32)
    else:
        mix_alpha = float(alpha)
        p_pref = mix_alpha * p_self + (1.0 - mix_alpha) * ext_term

    if label_noise_rho > 0.0:
        rho = float(label_noise_rho)
        p_pref = (1.0 - rho) * p_pref + rho * (1.0 - p_pref)
    return np.clip(p_pref.astype(np.float32), 0.0, 1.0)


def _format_prompt(question: str, tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return question


def _build_text_response_dataproto(
    *,
    questions: list[str],
    response_texts: list[str],
    tokenizer,
    max_prompt_length: int,
    max_response_length: int,
    non_tensors: Optional[dict[str, np.ndarray]] = None,
) -> DataProto:
    """Build a standard prompt+response ``DataProto`` for log-prob or reward eval."""

    batch_size = len(questions)
    assert len(response_texts) == batch_size
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    prompt_ids_per: list[list[int]] = []
    response_ids_per: list[list[int]] = []
    for i in range(batch_size):
        prompt_ids = tokenizer.encode(_format_prompt(questions[i], tokenizer), add_special_tokens=False)
        response_ids = tokenizer.encode(response_texts[i], add_special_tokens=False)
        prompt_ids_per.append(prompt_ids[-max_prompt_length:])
        response_ids_per.append(response_ids[:max_response_length])

    max_prompt = max((len(x) for x in prompt_ids_per), default=0) or 1
    max_response = max((len(x) for x in response_ids_per), default=0) or 1

    prompts, responses, input_ids, attention_masks = [], [], [], []
    raw_prompt_ids = np.empty(batch_size, dtype=object)
    for i in range(batch_size):
        p_ids = prompt_ids_per[i]
        r_ids = response_ids_per[i]
        p_pad = [pad_id] * (max_prompt - len(p_ids)) + p_ids
        r_pad = r_ids + [pad_id] * (max_response - len(r_ids))
        p_mask = [0] * (max_prompt - len(p_ids)) + [1] * len(p_ids)
        r_mask = [1] * len(r_ids) + [0] * (max_response - len(r_ids))
        prompts.append(p_pad)
        responses.append(r_pad)
        input_ids.append(p_pad + r_pad)
        attention_masks.append(p_mask + r_mask)
        raw_prompt_ids[i] = np.asarray(p_ids, dtype=np.int64)

    prompts_t = torch.tensor(prompts, dtype=torch.long)
    responses_t = torch.tensor(responses, dtype=torch.long)
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    attention_mask_t = torch.tensor(attention_masks, dtype=torch.long)
    position_ids_t = compute_position_id_with_mask(attention_mask_t)

    merged_non_tensors = {"raw_prompt_ids": raw_prompt_ids}
    if non_tensors:
        merged_non_tensors.update(non_tensors)

    return DataProto.from_dict(
        tensors={
            "prompts": prompts_t,
            "responses": responses_t,
            "input_ids": input_ids_t,
            "attention_mask": attention_mask_t,
            "position_ids": position_ids_t,
        },
        non_tensors=merged_non_tensors,
        meta_info={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": pad_id,
            "do_sample": False,
            "recompute_log_prob": True,
            "n": 1,
        },
    )


def build_cpr_paired_dataproto(
    *,
    questions: list[str],
    y_a_texts: list[str],
    y_b_texts: list[str],
    tokenizer,
    max_prompt_length: int,
    max_response_length: int,
    label_key_suffix: str,
) -> DataProto:
    """Build a CPR log-prob ``DataProto`` for either response ``a`` or ``b``."""

    if label_key_suffix not in {"a", "b"}:
        raise ValueError(f"label_key_suffix must be 'a' or 'b', got {label_key_suffix!r}")
    texts = y_a_texts if label_key_suffix == "a" else y_b_texts
    return _build_text_response_dataproto(
        questions=questions,
        response_texts=texts,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
    )


def _extract_batch_questions_and_responses(batch: DataProto, tokenizer) -> tuple[list[str], list[str]]:
    prompt_ids = batch.batch["prompts"]
    response_ids = batch.batch["responses"]
    attention_mask = batch.batch["attention_mask"]
    prompt_len = prompt_ids.shape[1]

    questions, responses = [], []
    for i in range(len(batch)):
        questions.append(_extract_question_from_prompt(prompt_ids[i], attention_mask[i, :prompt_len], tokenizer))
        resp_len = int(attention_mask[i, prompt_len:].sum().item())
        responses.append(tokenizer.decode(response_ids[i, :resp_len], skip_special_tokens=True))
    return questions, responses


def _group_rows_by_prompt(batch: DataProto) -> tuple[list[list[int]], np.ndarray]:
    if "uid" in batch.non_tensor_batch:
        raw_ids = batch.non_tensor_batch["uid"]
    else:
        raw_ids = np.arange(len(batch), dtype=np.int64)

    ordered_ids: list[Any] = []
    grouped: dict[Any, list[int]] = {}
    for idx, key in enumerate(raw_ids.tolist()):
        if key not in grouped:
            grouped[key] = []
            ordered_ids.append(key)
        grouped[key].append(idx)
    return [grouped[key] for key in ordered_ids], np.asarray(ordered_ids, dtype=object)


def _unique_indices_by_text(indices: list[int], response_texts: list[str]) -> list[int]:
    seen: set[str] = set()
    unique = []
    for idx in indices:
        text = response_texts[idx]
        if text in seen:
            continue
        seen.add(text)
        unique.append(idx)
    return unique


def _make_cpr_judge_generator(
    actor_rollout_wg,
    tokenizer,
    max_prompt_length: int,
    judge_max_new_tokens: int,
) -> Callable[[list[str]], list[str]]:
    """Judge generator with a greedy-text fallback contract."""

    def _gen(prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        texts, _ = _generate_follow_ups_with_tokens(prompts, tokenizer, actor_rollout_wg, max_prompt_length)
        outputs = []
        for text in texts:
            trimmed = text.strip().split("\n", 1)[0][: max(1, 8 * judge_max_new_tokens)]
            outputs.append(trimmed)
        return outputs

    return _gen


def _build_reward_eval_dataproto(
    *,
    source_batch: DataProto,
    source_indices: list[int],
    questions: list[str],
    response_texts: list[str],
    tokenizer,
    max_prompt_length: int,
    max_response_length: int,
) -> DataProto:
    """Build a reward-evaluation ``DataProto`` preserving source non-tensor fields."""

    non_tensors: dict[str, np.ndarray] = {}
    for key in ("reward_model", "data_source", "extra_info", "__num_turns__"):
        if key in source_batch.non_tensor_batch:
            non_tensors[key] = np.asarray([source_batch.non_tensor_batch[key][idx] for idx in source_indices], dtype=object)

    return _build_text_response_dataproto(
        questions=questions,
        response_texts=response_texts,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        non_tensors=non_tensors,
    )


def _score_with_oracle(
    oracle_reward_fn: Optional[Callable[[DataProto], np.ndarray]],
    eval_dp: DataProto,
) -> np.ndarray | None:
    if oracle_reward_fn is None:
        return None
    scores = oracle_reward_fn(eval_dp)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1 or scores.shape[0] != len(eval_dp):
        raise ValueError(
            f"oracle_reward_fn must return shape [{len(eval_dp)}], got {scores.shape}"
        )
    return scores


def apply_cpr_rewards(
    *,
    batch: DataProto,
    tokenizer,
    actor_rollout_wg,
    cfg: CPRConfig,
    step: int,
    oracle_reward_fn: Callable | None = None,
) -> dict[str, Any]:
    """Construct CPR pairs, score them, and package paired ``DataProto`` objects."""
    if not cfg.enable:
        return {}

    questions, response_texts = _extract_batch_questions_and_responses(batch, tokenizer)
    prompt_groups, _ = _group_rows_by_prompt(batch)
    row_count = len(batch)
    rng = random.Random(step * 1000003 + 313)

    current_eval_scores = None
    if oracle_reward_fn is not None and (cfg.pair_source == "extremes" or cfg.alpha < 1.0):
        current_eval_dp = _build_reward_eval_dataproto(
            source_batch=batch,
            source_indices=list(range(row_count)),
            questions=questions,
            response_texts=response_texts,
            tokenizer=tokenizer,
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
        )
        current_eval_scores = _score_with_oracle(oracle_reward_fn, current_eval_dp)

    group_questions: list[str] = []
    group_y_a: list[str] = []
    group_y_b: list[str] = []
    group_a_source_idx: list[int] = []
    group_b_source_idx: list[int] = []

    if cfg.pair_source == "prefix_branch":
        revise_prompts = []
        prefix_texts = []
        original_texts = []
        source_indices = []
        improve_instr = (
            "\n\nPlease continue the reasoning above. If you see an error, correct it. "
            "Otherwise proceed to the next logical step and give the final answer. "
            "End with the final answer in \\boxed{}.\n"
        )
        for group in prompt_groups:
            base_idx = _unique_indices_by_text(group, response_texts)[0]
            question = questions[base_idx]
            response = response_texts[base_idx]
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            branch = select_branch_point(len(response_ids), rng=rng)
            prefix_text = tokenizer.decode(response_ids[:branch], skip_special_tokens=True)
            revise_prompts.append(question + "\n\n" + prefix_text + improve_instr)
            prefix_texts.append(prefix_text)
            original_texts.append(response)
            group_questions.append(question)
            source_indices.append(base_idx)

        revised_suffixes, _ = _generate_follow_ups_with_tokens(
            revise_prompts,
            tokenizer,
            actor_rollout_wg,
            cfg.max_prompt_length,
        )
        for i in range(len(prompt_groups)):
            y_a = prefix_texts[i] + revised_suffixes[i]
            y_b = original_texts[i]
            group_y_a.append(y_a)
            group_y_b.append(y_b)
            group_a_source_idx.append(source_indices[i])
            group_b_source_idx.append(source_indices[i])
    else:
        for group in prompt_groups:
            unique_indices = _unique_indices_by_text(group, response_texts)
            if not unique_indices:
                unique_indices = [group[0]]
            if cfg.pair_source == "extremes" and current_eval_scores is not None and len(unique_indices) >= 2:
                ordered = sorted(unique_indices, key=lambda idx: (float(current_eval_scores[idx]), idx))
                b_idx = ordered[0]
                a_idx = ordered[-1]
                if a_idx == b_idx and len(ordered) >= 2:
                    b_idx = ordered[1]
            else:
                if len(unique_indices) >= 2:
                    a_idx, b_idx = rng.sample(unique_indices, 2)
                else:
                    a_idx = unique_indices[0]
                    b_idx = unique_indices[0]
            group_questions.append(questions[a_idx])
            group_y_a.append(response_texts[a_idx])
            group_y_b.append(response_texts[b_idx])
            group_a_source_idx.append(a_idx)
            group_b_source_idx.append(b_idx)

    row_questions = [""] * row_count
    row_y_a = [""] * row_count
    row_y_b = [""] * row_count
    row_pair_ids = np.zeros(row_count, dtype=np.int64)
    row_a_src = np.zeros(row_count, dtype=np.int64)
    row_b_src = np.zeros(row_count, dtype=np.int64)
    for pair_id, group in enumerate(prompt_groups):
        for row_idx in group:
            row_questions[row_idx] = group_questions[pair_id]
            row_y_a[row_idx] = group_y_a[pair_id]
            row_y_b[row_idx] = group_y_b[pair_id]
            row_pair_ids[row_idx] = pair_id
            row_a_src[row_idx] = group_a_source_idx[pair_id]
            row_b_src[row_idx] = group_b_source_idx[pair_id]

    judge_generator = _make_cpr_judge_generator(
        actor_rollout_wg,
        tokenizer,
        cfg.max_prompt_length,
        cfg.judge_max_new_tokens,
    )
    p_self = p_self_order_swap(
        questions=row_questions,
        y_a_texts=row_y_a,
        y_b_texts=row_y_b,
        judge_generator=judge_generator,
        cfg=cfg,
    )

    r_ext_a = None
    r_ext_b = None
    if oracle_reward_fn is not None:
        if current_eval_scores is not None and cfg.pair_source != "prefix_branch":
            r_ext_a = current_eval_scores[row_a_src]
            r_ext_b = current_eval_scores[row_b_src]
        else:
            src_a = row_a_src.tolist()
            src_b = row_b_src.tolist()
            eval_dp_a = _build_reward_eval_dataproto(
                source_batch=batch,
                source_indices=src_a,
                questions=row_questions,
                response_texts=row_y_a,
                tokenizer=tokenizer,
                max_prompt_length=cfg.max_prompt_length,
                max_response_length=cfg.max_response_length,
            )
            eval_dp_b = _build_reward_eval_dataproto(
                source_batch=batch,
                source_indices=src_b,
                questions=row_questions,
                response_texts=row_y_b,
                tokenizer=tokenizer,
                max_prompt_length=cfg.max_prompt_length,
                max_response_length=cfg.max_response_length,
            )
            r_ext_a = _score_with_oracle(oracle_reward_fn, eval_dp_a)
            r_ext_b = _score_with_oracle(oracle_reward_fn, eval_dp_b)

    p_pref = compute_p_pref(
        p_self=p_self,
        r_ext_a=r_ext_a,
        r_ext_b=r_ext_b,
        alpha=cfg.alpha,
        tau_ext=cfg.tau_ext,
        label_noise_rho=cfg.label_noise_rho,
        alpha_gate_gamma=cfg.alpha_gate_gamma,
    )

    y_a_dp = build_cpr_paired_dataproto(
        questions=row_questions,
        y_a_texts=row_y_a,
        y_b_texts=row_y_b,
        tokenizer=tokenizer,
        max_prompt_length=cfg.max_prompt_length,
        max_response_length=cfg.max_response_length,
        label_key_suffix="a",
    )
    y_b_dp = build_cpr_paired_dataproto(
        questions=row_questions,
        y_a_texts=row_y_a,
        y_b_texts=row_y_b,
        tokenizer=tokenizer,
        max_prompt_length=cfg.max_prompt_length,
        max_response_length=cfg.max_response_length,
        label_key_suffix="b",
    )

    verifier_pref = None
    disagreement = 0.0
    agreement_mask: np.ndarray | None = None
    if r_ext_a is not None and r_ext_b is not None:
        verifier_pref = _sigmoid((r_ext_a - r_ext_b) / max(float(cfg.tau_ext), 1e-6)).astype(np.float32)
        agreement_mask = (p_self >= 0.5) == (verifier_pref >= 0.5)
        disagreement = float((~agreement_mask).mean())

    p_self_entropy = _binary_entropy(p_self)
    entropy_agree = float("nan")
    entropy_disagree = float("nan")
    if agreement_mask is not None and agreement_mask.any():
        entropy_agree = float(p_self_entropy[agreement_mask].mean())
    if agreement_mask is not None and (~agreement_mask).any():
        entropy_disagree = float(p_self_entropy[~agreement_mask].mean())

    unique_per_group = [
        len({response_texts[i] for i in group}) for group in prompt_groups
    ]
    unique_answer_count = float(np.mean(unique_per_group)) if unique_per_group else 0.0

    # Paired-response length (phase-surface axis): mean tokens in y_a / y_b per
    # row. Uses tokenizer.encode to stay consistent with build_cpr_paired_dataproto.
    def _mean_tok_len(texts: list[str]) -> float:
        if not texts:
            return 0.0
        lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
        return float(np.mean(lens))

    mean_length_a = _mean_tok_len(row_y_a)
    mean_length_b = _mean_tok_len(row_y_b)

    # Verifier exact-match mean (raw oracle score, pre-BT-sigmoid). For math
    # verifiers with 0/1 outputs this is the arithmetic EM rate on the paired
    # rollouts. Complements cpr/verifier_pref_mean which is the BT probability.
    verifier_em_a = float(r_ext_a.mean()) if r_ext_a is not None else float("nan")
    verifier_em_b = float(r_ext_b.mean()) if r_ext_b is not None else float("nan")

    alpha_eff_vec = compute_alpha_eff(
        p_self=p_self,
        p_verif=verifier_pref,
        alpha=cfg.alpha,
        alpha_gate_gamma=cfg.alpha_gate_gamma,
    )

    metrics = {
        "cpr/p_pref_mean": float(p_pref.mean()),
        "cpr/p_pref_std": float(p_pref.std()),
        "cpr/p_pref_frac_extreme": float(((p_pref <= 0.05) | (p_pref >= 0.95)).mean()),
        "cpr/p_pref_frac_middle": float(((p_pref >= 0.40) & (p_pref <= 0.60)).mean()),
        "cpr/p_self_mean": float(p_self.mean()),
        "cpr/p_self_std": float(p_self.std()),
        "cpr/p_self_entropy_mean": float(p_self_entropy.mean()) if len(p_self_entropy) else 0.0,
        "cpr/p_self_entropy_agree": entropy_agree,
        "cpr/p_self_entropy_disagree": entropy_disagree,
        "cpr/tie_rate": float((np.abs(p_self - 0.5) < 0.05).mean()),
        "cpr/self_verifier_disagreement": disagreement,
        "cpr/pair_agreement_rate": float(agreement_mask.mean()) if agreement_mask is not None else float("nan"),
        "cpr/unique_answer_count": unique_answer_count,
        "cpr/mean_response_length_a": mean_length_a,
        "cpr/mean_response_length_b": mean_length_b,
        "cpr/verifier_em_a_mean": verifier_em_a,
        "cpr/verifier_em_b_mean": verifier_em_b,
        "cpr/alpha_eff_mean": float(alpha_eff_vec.mean()),
        "cpr/alpha_eff_std": float(alpha_eff_vec.std()),
        "cpr/alpha_gate_gamma": float(cfg.alpha_gate_gamma),
        "cpr/label_noise_rho": float(cfg.label_noise_rho),
    }
    if verifier_pref is not None:
        metrics["cpr/verifier_pref_mean"] = float(verifier_pref.mean())

    return {
        "p_pref": p_pref.astype(np.float32),
        "p_self": p_self.astype(np.float32),
        "p_verif": None if verifier_pref is None else verifier_pref.astype(np.float32),
        "r_ext": None if r_ext_a is None or r_ext_b is None else {"a": r_ext_a, "b": r_ext_b},
        "pair_ids": row_pair_ids,
        "y_a_dp": y_a_dp,
        "y_b_dp": y_b_dp,
        "skip_mask": np.zeros(row_count, dtype=bool),
        "metrics": metrics,
    }
