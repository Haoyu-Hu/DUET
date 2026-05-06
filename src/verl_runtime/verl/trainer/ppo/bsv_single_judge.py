# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
"""Single-rollout self-judge for GRPO-BSV (DEFINE_SPEC_v3 §1.2b).

Produces a per-rollout self-confidence probability `p_self_i ∈ [0, 1]`
interpreting each rollout as its own candidate answer (no pairwise
comparison required, unlike the DPO/CPR frame).

**Role under Frame A (rev 2026-04-21, locked primary reward shape):**
p_self is GATE-ONLY. It NEVER enters the reward. Its sole purpose is to
feed the per-rollout BSV gate:
    c_i = 2 · |p_self_i − 0.5|                           # §1.2b confidence
    I_v_i ∼ Bernoulli(ε + (1 − ε) · 1{c_i < t(α)})      # gate

Under Frame A, the downstream reward uses ONLY R_ext on the gate-included
slice (I_v=1); unverified rollouts are masked from the GRPO gradient via
the group-mean fill trick in `compute_bsv_grpo_masked_reward`. See
`doc/bsv_frame_c_parking_lot.md` for the explored-but-parked mixture
variants that re-introduced p_self into reward (DO NOT import without
explicit approval).

Backends:
  * ``stub_noisy``  — synthetic Bernoulli(p = 0.55) with configurable noise;
                      used in unit tests + smoke wiring.
  * ``response_logprob`` — **DEFAULT backend for Frame A (rev 2026-04-21)**.
                      Uses the actor's per-token log-probs on its own
                      generated response (``old_log_probs`` already computed
                      by the trainer) as a per-rollout self-confidence signal:
                          conf_i = (Σ old_log_probs · response_mask) / Σ response_mask
                          p_self_i = sigmoid((conf_i − μ_batch) / σ_batch)
                      Per-rollout independent (different responses have
                      different content + length → different mean log-probs)
                      → solves the group-correlation problem that
                      ``self_consistency`` had (E1 showed ~34% degenerate groups).
                      Zero extra inference — reads tensors already in the batch.
                      Semantic caveat: this is "model's confidence in its own
                      generation" (anti-correlated with correctness on hard
                      problems; well-calibrated elsewhere) — NOT a 1-token
                      YES/NO/TIE judge. Under Frame A, calibration is not
                      required for T1/T2/T3 guarantees; routing just needs
                      to be informative. See §4.7 decision tree.
  * ``self_consistency`` — Within-group sibling-answer-agreement count:
                      ``p_self_i = (1/K) · Σ 1{answer(y_j) = answer(y_i)}``.
                      Granularity 1/K (0.25 for K=4). E1 (2026-04-21)
                      measured ~34% degenerate_group_rate at α=0.5 because
                      p_self is a GROUP STATISTIC → correlated c_i within
                      siblings → whole groups get gated OUT together. Still
                      available as a gate backend but no longer default.
  * ``actor_logits`` — YES/NO/TIE 1-token decode through the actor worker,
                      p_self = p_yes + 0.5 · p_tie. Requires a worker-side
                      hook that returns first-token logits. Not wired through
                      the trainer yet; kept as schema placeholder. Implementing
                      properly needs a second vllm pass with max_tokens=1 and
                      logprobs=k; deferred to a later commit.
  * ``openai``       — GPT-4o-mini backend; disabled by default since the
                      per-rollout call rate is K× higher than per-pair.
                      (see §4.6 cost budget.)

Shape contract: given a batch of N rollouts, returns an np.ndarray of
shape (N,) with dtype float32 in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------
# Public config + dispatch
# ---------------------------------------------------------------------


@dataclass
class SingleJudgeConfig:
    """Config for the single-rollout self-judge.

    Attributes
    ----------
    mode : str
        One of {"stub_noisy", "actor_logits", "openai"}.
    noise : float
        ``stub_noisy`` only: additive uniform noise range applied to the
        Bernoulli(0.55) reference value. Clipped to [0, 1] after noising.
    rng_seed : int
        Seed for the ``stub_noisy`` RNG; tests and smoke share.
    yes_token : str
        Token the actor judge emits when the response is judged correct.
    no_token : str
        Token emitted when judged incorrect.
    tie_token : str
        Token emitted when undecidable (contributes 0.5 · p_tie to p_self).
    """

    mode: str = "stub_noisy"
    noise: float = 0.15
    rng_seed: int = 0
    yes_token: str = "A"  # ttrl_math convention uses A/B/T
    no_token: str = "B"
    tie_token: str = "T"


def score_rollouts(
    prompts: Sequence[str],
    responses: Sequence[str],
    cfg: SingleJudgeConfig,
    *,
    actor_logits_fn: Optional[Callable[[Sequence[str], Sequence[str]], np.ndarray]] = None,
    openai_score_fn: Optional[Callable[[str, str], float]] = None,
    preds: Optional[Sequence[str]] = None,
    group_ids: Optional[Sequence] = None,
    old_log_probs: Optional[np.ndarray] = None,
    response_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score a batch of (prompt, response) pairs → p_self ∈ [0, 1]^N.

    This is the ONE public entrypoint. Backend selected by ``cfg.mode``.

    Parameters
    ----------
    prompts, responses : sequences of strings, length N.
    cfg : :class:`SingleJudgeConfig`.
    actor_logits_fn : callable, optional
        Required for ``mode="actor_logits"``. Receives
        ``(prompts, responses)`` and returns a ``(N, 3)`` float array of
        [p_yes, p_no, p_tie] probabilities.
    openai_score_fn : callable, optional
        Required for ``mode="openai"``. Receives one ``(prompt, response)``
        pair and returns ``p_self ∈ [0, 1]``.
    preds, group_ids : sequences, optional
        Required for ``mode="self_consistency"``. ``preds[i]`` is a
        string-typed extracted answer from rollout ``i``; ``group_ids[i]``
        is the prompt/group identifier (rollouts sharing a group id are
        siblings that share the same prompt). Length N each.
    old_log_probs, response_mask : np.ndarray, optional
        Required for ``mode="response_logprob"``. Both shape (N, L) where
        L is the response length. ``old_log_probs[i, t]`` is the actor's
        log-prob of response token t (already computed by the trainer via
        ``compute_log_prob``); ``response_mask[i, t] ∈ {0, 1}`` marks
        non-pad response positions.

    Returns
    -------
    np.ndarray, shape (N,), dtype float32, values in [0, 1].
    """
    if len(prompts) != len(responses):
        raise ValueError(
            f"prompt/response batch shapes mismatch: "
            f"{len(prompts)} vs {len(responses)}"
        )
    N = len(prompts)
    if N == 0:
        return np.zeros((0,), dtype=np.float32)

    if cfg.mode == "stub_noisy":
        return _score_stub_noisy(N, cfg)
    if cfg.mode == "response_logprob":
        if old_log_probs is None or response_mask is None:
            raise ValueError(
                "mode='response_logprob' requires old_log_probs and response_mask"
            )
        return _score_response_logprob(old_log_probs, response_mask)
    if cfg.mode == "self_consistency":
        if preds is None or group_ids is None:
            raise ValueError(
                "mode='self_consistency' requires preds and group_ids"
            )
        return _score_self_consistency(preds, group_ids)
    if cfg.mode == "actor_logits":
        if actor_logits_fn is None:
            raise ValueError("mode='actor_logits' requires actor_logits_fn")
        return _score_actor_logits(prompts, responses, actor_logits_fn)
    if cfg.mode == "shuffled":
        # C12 random-routing baseline (methodology §5c / Phase-1 E8). Compute
        # p_self via the actor-logits backend exactly as in actor_logits mode,
        # then permute within the batch to destroy the p_self↔rollout-content
        # correlation. The gate still sees a realistic p_self marginal
        # distribution and ε-exploration is unchanged, so realized_call_rate
        # matches BSV-actor_logits at the same α — this is a matched-budget
        # comparator that isolates "is the self-judge a non-trivial filter?".
        if actor_logits_fn is None:
            raise ValueError("mode='shuffled' requires actor_logits_fn")
        p_self = _score_actor_logits(prompts, responses, actor_logits_fn)
        # Seeded with cfg.rng_seed for reproducibility. Across training the
        # permutation differs whenever N or the seed nonce is perturbed; for
        # the baseline purpose (decorrelate p_self from rollout content) a
        # fixed-seed permutation still does the job.
        rng = np.random.default_rng(cfg.rng_seed)
        perm = rng.permutation(len(p_self))
        return p_self[perm].astype(np.float32)
    if cfg.mode == "openai":
        if openai_score_fn is None:
            raise ValueError("mode='openai' requires openai_score_fn")
        return _score_openai(prompts, responses, openai_score_fn)
    raise ValueError(f"Unknown single-judge mode: {cfg.mode!r}")


# ---------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------


def _score_stub_noisy(N: int, cfg: SingleJudgeConfig) -> np.ndarray:
    """Synthetic p_self ∼ clip(Bernoulli(0.55) + noise · U[-1,1], 0, 1).

    The base rate 0.55 encodes "the self-judge is weakly correct on
    average"; ``noise`` adds dispersion so c_i = 2|p_self - 0.5| is
    non-degenerate and the P² tracker has input to quantise.
    """
    rng = np.random.default_rng(cfg.rng_seed)
    base = (rng.uniform(size=N) < 0.55).astype(np.float32)
    perturbed = base + cfg.noise * (2.0 * rng.uniform(size=N) - 1.0)
    return np.clip(perturbed, 0.0, 1.0).astype(np.float32)


def _score_response_logprob(
    old_log_probs: np.ndarray,
    response_mask: np.ndarray,
) -> np.ndarray:
    """Self-confidence from actor log-prob on its own response (Frame A default).

    Per-rollout mean log-prob over non-pad response tokens, sigmoid-normalised
    within-batch so that the batch median maps to p_self = 0.5. This guarantees
    the BSV gate's confidence ``c_i = 2|p_self_i − 0.5|`` has meaningful spread
    across the batch regardless of the absolute log-prob scale (which varies
    by model, tokenizer, and training step).

    Shape contract
    --------------
    old_log_probs : ndarray, shape (N, L), float
        Actor's log-prob of each response token. Expected to come from
        verl's ``compute_log_prob`` (line 1519-1527 of ray_trainer.py).
    response_mask : ndarray, shape (N, L), {0, 1}
        1 on valid response positions, 0 on padding. Aligned with
        ``old_log_probs``.

    Returns
    -------
    ndarray, shape (N,), float32 in [0, 1].

    Mathematical note
    -----------------
    For each rollout i:
        n_i          = Σ_t response_mask[i, t]                    (length)
        s_i          = Σ_t old_log_probs[i, t] · response_mask[i, t]
        mean_logp_i  = s_i / max(n_i, 1)                           (per-token mean)

    Batch-level normalization:
        μ   = median_i mean_logp_i                                 (robust center)
        σ   = MAD_i (mean_logp_i) · 1.4826 + ε                     (robust scale)
        p_i = sigmoid((mean_logp_i − μ) / σ)

    The sigmoid saturates extreme outliers toward 0 or 1; c_i = 2|p_i − 0.5|
    is therefore in (0, 1] for non-degenerate batches.
    """
    lp = np.asarray(old_log_probs, dtype=np.float32)
    rm = np.asarray(response_mask, dtype=np.float32)
    if lp.shape != rm.shape:
        raise ValueError(
            f"old_log_probs shape {lp.shape} must match response_mask shape {rm.shape}"
        )
    if lp.ndim != 2:
        raise ValueError(
            f"old_log_probs expected (N, L); got shape {lp.shape}"
        )
    n = rm.sum(axis=1).clip(min=1.0)              # (N,)
    s = (lp * rm).sum(axis=1)                      # (N,)
    mean_lp = (s / n).astype(np.float32)           # (N,)
    # Robust center + scale (median / MAD); falls back to mean/std on tiny batches.
    if mean_lp.size >= 2:
        mu = float(np.median(mean_lp))
        mad = float(np.median(np.abs(mean_lp - mu))) * 1.4826
        sigma = max(mad, float(np.std(mean_lp)) * 0.5, 1e-3)
    else:
        mu = float(mean_lp.item()) if mean_lp.size == 1 else 0.0
        sigma = 1.0
    z = (mean_lp - mu) / sigma
    # Numerically stable sigmoid.
    p_self = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                       np.exp(z) / (1.0 + np.exp(z)))
    return np.clip(p_self.astype(np.float32), 0.0, 1.0)


def _score_self_consistency(
    preds: Sequence[str],
    group_ids: Sequence,
) -> np.ndarray:
    """Self-consistency p_self from within-group answer agreement.

    For each rollout i, ``p_self_i = (# group-siblings with pred == pred_i) / K_group``.
    An empty-string / None pred is treated as a distinct "parse-failed" bucket so
    reward_fn failures don't artificially inflate agreement.

    Granularity 1/K_group — for K=4 the possible values are
    {0.25, 0.50, 0.75, 1.00}, avoiding the {0, 0.5, 1} atoms of a
    3-label text judge. Sibling count K_group can vary per group if
    rollout_n differs per prompt, but typically all groups share K.
    """
    if len(preds) != len(group_ids):
        raise ValueError(
            f"preds ({len(preds)}) and group_ids ({len(group_ids)}) must match"
        )
    N = len(preds)
    # Normalise preds: strip whitespace, treat empty/None as a parse-fail bucket
    norm_preds = [
        (str(p).strip() if p is not None and str(p).strip() else "<parse_fail>")
        for p in preds
    ]
    # Index rollouts by group
    group_to_indices: dict = {}
    for i, gid in enumerate(group_ids):
        group_to_indices.setdefault(gid, []).append(i)
    out = np.zeros(N, dtype=np.float32)
    for gid, idxs in group_to_indices.items():
        K = len(idxs)
        if K == 0:
            continue
        # Count how many rollouts in this group share each pred value
        from collections import Counter

        counts = Counter(norm_preds[i] for i in idxs)
        for i in idxs:
            out[i] = counts[norm_preds[i]] / float(K)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _score_actor_logits(
    prompts: Sequence[str],
    responses: Sequence[str],
    actor_logits_fn: Callable[[Sequence[str], Sequence[str]], np.ndarray],
) -> np.ndarray:
    """Actor-logits backend: p_self = p_yes + 0.5 · p_tie.

    The caller provides ``actor_logits_fn`` which returns a (N, 3) array of
    [p_yes, p_no, p_tie] probabilities. Under Frame A the canonical caller
    is ``ray_trainer._build_actor_judge_probs`` which runs a 3-candidate
    ``compute_log_prob`` pass over [prompt + response + judge_prefix + A|B|T]
    and softmaxes the three log-probs per rollout. See DEFINE_SPEC_v3 §1.2b
    point (1).
    """
    probs = np.asarray(
        actor_logits_fn(list(prompts), list(responses)), dtype=np.float32
    )
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(
            f"actor_logits_fn must return (N, 3); got shape {probs.shape}"
        )
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    probs = probs / row_sums
    p_yes, _p_no, p_tie = probs[:, 0], probs[:, 1], probs[:, 2]
    return np.clip(p_yes + 0.5 * p_tie, 0.0, 1.0).astype(np.float32)


def make_judge_position_ids(
    existing_pos: np.ndarray,
    insert_start: int,
    k: int,
) -> np.ndarray:
    """Build position_ids for a newly-unmasked judge-tail region.

    Follows verl's position_ids convention for left-padded prompt + right-padded
    response (vllm_rollout_spmd.py:375-378):
      - Pad positions have position_id=0 (placeholder, stripped by unpad_input
        before RoPE applies).
      - Valid prompt tokens increment from 0.
      - Response tokens continue incrementally from the last valid prompt
        position (``pos[..., -1:] + delta`` pattern).

    For the actor_logits self-judge we overwrite the right-pad region
    immediately after the last valid response token with ``judge_prefix_ids``
    plus a candidate token (k = len(prefix) + 1 slots total). This helper
    returns the position_ids vector that should replace the existing
    values at ``[insert_start : insert_start + k]``.

    Parameters
    ----------
    existing_pos : np.ndarray, shape (seq_len,) or (N, seq_len), int
        Current position_ids for a single row OR a batch. The returned
        tensor has the SAME shape; the caller assigns it back in-place.
    insert_start : int
        Index of the first slot to overwrite. MUST satisfy
        ``insert_start >= 1`` (we read position at ``insert_start - 1``).
    k : int
        Number of consecutive slots to fill. Typically
        ``k = len(judge_prefix_ids) + 1`` (prefix + one candidate token).

    Returns
    -------
    np.ndarray, same shape as ``existing_pos``.
        Unchanged except at positions ``[insert_start : insert_start + k]``
        where ``new[..., insert_start + j] = existing_pos[..., insert_start - 1] + 1 + j``
        for ``j ∈ [0, k)``.

    Rationale
    ---------
    This matches the formula in vllm_rollout_spmd.py:377:
        response_position_ids = position_ids[..., -1:] + delta_position_id
    where delta_position_id = arange(1, response_length + 1). Our insert is
    a contiguous extension of existing valid-position indices, so the same
    +delta formula applies.
    """
    pos = np.asarray(existing_pos).copy()
    if insert_start < 1:
        raise ValueError(
            f"insert_start={insert_start} must be >= 1 (reading position at insert_start-1)"
        )
    if k < 1:
        raise ValueError(f"k={k} must be >= 1")
    if insert_start + k > pos.shape[-1]:
        raise ValueError(
            f"insert_start({insert_start}) + k({k}) = {insert_start + k} "
            f"exceeds seq_len {pos.shape[-1]}"
        )
    base = pos[..., insert_start - 1 : insert_start]   # (..., 1) — last valid pos
    delta = np.arange(1, k + 1, dtype=pos.dtype)       # (k,)
    pos[..., insert_start : insert_start + k] = base + delta
    return pos


def actor_logits_probs_from_log_abt(
    log_A: np.ndarray,
    log_B: np.ndarray,
    log_T: np.ndarray,
) -> np.ndarray:
    """Pure helper: softmax over {A, B, T} log-probs → (N, 3) probabilities.

    Used by the ray_trainer actor_logits wiring: after three compute_log_prob
    passes return log P(A|ctx), log P(B|ctx), log P(T|ctx) per rollout, this
    function converts them into a normalised simplex (N, 3).

    Note: the raw three logits were ONLY log-probs of A/B/T under the full
    vocab. Softmax here RESTRICTS to the 3-class simplex, which is what
    the gate needs. Mass assigned by the model to other tokens is
    redistributed proportionally — standard for constrained decoding.
    """
    log_A = np.asarray(log_A, dtype=np.float32).ravel()
    log_B = np.asarray(log_B, dtype=np.float32).ravel()
    log_T = np.asarray(log_T, dtype=np.float32).ravel()
    if not (log_A.shape == log_B.shape == log_T.shape):
        raise ValueError(
            f"log_A/B/T must share shape; got "
            f"{log_A.shape}, {log_B.shape}, {log_T.shape}"
        )
    # Numerically stable row-wise softmax over the 3-class simplex.
    stacked = np.stack([log_A, log_B, log_T], axis=1)  # (N, 3)
    row_max = stacked.max(axis=1, keepdims=True)
    exps = np.exp(stacked - row_max)
    probs = exps / exps.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def _score_openai(
    prompts: Sequence[str],
    responses: Sequence[str],
    openai_score_fn: Callable[[str, str], float],
) -> np.ndarray:
    """Per-rollout OpenAI judge. Each call costs money — gate on budget."""
    out: List[float] = []
    for p, r in zip(prompts, responses):
        try:
            val = float(openai_score_fn(p, r))
        except Exception:  # noqa: BLE001 — graceful fallback to tie on any error
            val = 0.5
        if not np.isfinite(val):
            val = 0.5
        out.append(max(0.0, min(1.0, val)))
    return np.asarray(out, dtype=np.float32)
