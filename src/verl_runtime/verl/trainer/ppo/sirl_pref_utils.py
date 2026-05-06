# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
"""SIRL-Pref: Preference-based Self-Improvement RL utilities.

This module is **additive** — it neither imports from nor modifies the existing
``sirl_utils.apply_sirl_rewards`` path, so current SIRL behavior is untouched.
SIRL-Pref introduces:

    - Prefix-branched dependent pairs (y0 and a counterfactual y_imp sharing
      a prefix of y0's reasoning).
    - A dual-mode signed pair-advantage ``A_pair`` in [-1, +1] that works
      with self-judge preferences, oracle preferences, or a hybrid.
    - A 2B-row batch layout (each pair unrolled into a ``revised`` row and an
      ``original`` row) routed via ``sirl_pref_pair_id`` + ``sirl_pref_channel``.

The advantage computation and DPO/shaped-GRPO loss are in ``core_algos.py``
(functions ``compute_sirl_pref_advantage`` and ``compute_sirl_pref_dpo_loss``).
This module supplies the *data path*: pair construction, judging, preference
merging, and batch packaging.

Dispatch entry point: ``apply_sirl_pref_rewards``.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

import numpy as np
import torch

from verl.utils.reward_score.ttrl_math import extract_answer

# Reuse existing helpers rather than forking.
from verl.trainer.ppo.sirl_utils import (
    _build_follow_up_dataproto,
    _generate_follow_ups_with_tokens,
    _extract_question_from_prompt,
)


_log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config dataclass (mirror of DEFINE_SPEC §2)
# -----------------------------------------------------------------------------


@dataclass
class SirlPrefConfig:
    """Runtime configuration for SIRL-Pref.

    Default values match DEFINE_SPEC §2. Overridden via Hydra by the launcher.
    """
    enable: bool = False
    arm_id: str = "dpo_prefix_self"
    loss: str = "dpo"  # "dpo" | "shaped_grpo"
    pair_construction: str = "prefix_branch"  # "prefix_branch" | "whole_response"
    preference_source: str = "self"  # "self" | "oracle" | "random" | "hybrid"

    # Branch selection
    branch_min_frac: float = 0.35
    branch_max_frac: float = 0.65
    branch_min_suffix_tokens: int = 32
    revise_temperature_delta: float = 0.20

    # Judge
    judge_prompt_template: str = (
        "You are a strict evaluator. Two candidates continue the same prompt "
        "and shared prefix. Choose the candidate with the more likely correct "
        "final answer and logically valid reasoning. If both are equally correct, "
        "prefer the shorter and cleaner candidate. Do not reward verbosity, "
        "paraphrase, copied text, or stylistic changes. If one candidate changes "
        "the final answer, prefer it only when the new answer is clearly better "
        "supported by the reasoning. If the difference is ambiguous or both are "
        "flawed to a similar degree, return TIE. Reply with exactly one token: A, B, or TIE."
    )
    judge_order_swap: bool = True
    judge_max_tokens: int = 8
    ema_judge_refresh_steps: int = 100

    # Guardrails
    tie_band_eps: float = 0.15
    min_divergence_threshold: float = 0.12
    length_ratio_cap: float = 1.50
    length_penalty_weight: float = 0.10
    length_penalty_warmup_steps: int = 100
    external_margin_floor: float = 0.50
    lambda_self: float = 1.0
    lambda_ext: float = 1.0

    # Loss
    dpo_beta: float = 0.15
    anchor_weight: float = 0.05
    grpo_reject_coef: float = 0.50
    kl_coef: float = 0.02
    kl_channel: str = "y0_only"  # "y0_only" | "both" | "off"

    # Rollout schedule
    rollout_n: int = 4
    max_prompts_per_step: int = 32
    max_response_length: int = 3072
    # Max prompt length for revise/judge/paired-logprob generations. Must be
    # tight enough that (prompt + response) fits within the vLLM max_model_len;
    # caller is responsible for setting this to at most
    # `max_model_len - max_response_length - slack`. 0 or None → use sensible
    # default of 1024 tokens.
    max_prompt_length: int = 1024


# -----------------------------------------------------------------------------
# Pair construction
# -----------------------------------------------------------------------------


def select_branch_point(
    response_len: int,
    *,
    branch_min_frac: float = 0.35,
    branch_max_frac: float = 0.65,
    min_suffix_tokens: int = 32,
    rng: Optional[random.Random] = None,
) -> int:
    """Pick a uniform-random branch token index in the middle span.

    Returns the token index ``b`` such that prefix = response[:b] and
    suffix = response[b:]. Falls back to ``response_len // 2`` when the
    response is too short to satisfy ``min_suffix_tokens``.
    """
    if rng is None:
        rng = random.Random()

    if response_len <= 2:
        return max(1, response_len // 2)

    lo = int(branch_min_frac * response_len)
    hi = int(branch_max_frac * response_len)
    # Enforce that the suffix keeps at least min_suffix_tokens.
    max_allowed = max(1, response_len - min_suffix_tokens)
    hi = min(hi, max_allowed)
    if lo >= hi:
        return max(1, response_len // 2)
    return rng.randint(lo, hi)


def _sequence_similarity(a: str, b: str) -> float:
    """Character-level ratio via difflib. Used *only* as a min-divergence gate."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# -----------------------------------------------------------------------------
# Judge
# -----------------------------------------------------------------------------


def _parse_judge_reply(text: str) -> Optional[str]:
    """Extract A / B / TIE from a judge reply using word-boundary matches.

    The old implementation cleaned to the first 3 non-letters and used
    ``startswith``, which mis-parsed common judge phrasings such as
    ``"ANSWER: A"`` → A (correct by luck), ``"BETTER: A"`` → B,
    ``"Choice: A"`` → TIE, ``"The answer is A"`` → TIE, ``"Both tied"`` → B.

    New implementation: prefer explicit ``TIE`` marker; require letter-
    word-boundary for A/B so ordinary English words don't match.
    """
    if not text:
        return None
    cleaned = text.strip().upper()

    # Prefer an explicit TIE marker, matching TIE / TIED / TIES / TIEING.
    # "C" remains a TIE synonym for templates that use A/B/C. Require word
    # boundary to avoid matching "Check" etc.
    if re.search(r"\bTIE\w*\b", cleaned):
        return "TIE"
    if re.search(r"\bC\b", cleaned):
        return "TIE"

    # Require letter-boundary for A/B (allow digit/punctuation/whitespace on both
    # sides, but NOT another letter). `(?<![A-Z])A(?![A-Z])` excludes "ANSWER",
    # "ABLE", "AROUND", etc.
    has_A = re.search(r"(?<![A-Z])A(?![A-Z])", cleaned) is not None
    has_B = re.search(r"(?<![A-Z])B(?![A-Z])", cleaned) is not None

    if has_A and not has_B:
        return "A"
    if has_B and not has_A:
        return "B"
    # Ambiguous ("both" or neither) → treat as tie rather than guessing.
    if has_A and has_B:
        return "TIE"
    return None


def _build_judge_prompt(
    question: str,
    prefix_text: str,
    candidate_a_text: str,
    candidate_b_text: str,
    template: str,
) -> str:
    """Assemble the judge prompt. Keeps the template as a leading instruction."""
    body = (
        f"{template}\n\n"
        f"Problem:\n{question}\n\n"
        f"Shared Prefix:\n{prefix_text}\n\n"
        f"Completion A:\n{candidate_a_text}\n\n"
        f"Completion B:\n{candidate_b_text}\n\n"
        f"Answer (A, B, or TIE):"
    )
    return body


def judge_self_preference(
    *,
    questions: list[str],
    prefix_texts: list[str],
    revised_texts: list[str],
    original_texts: list[str],
    judge_generator: Callable[[list[str]], list[str]],
    judge_prompt_template: str,
    order_swap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the self-judge over a batch of pairs.

    ``judge_generator(prompts)`` must return a list of short completions
    (typically ~8 tokens) parallel to ``prompts``. The caller is responsible for
    constructing a generator that uses the same (or EMA-lagged) actor.

    For each pair we build up to 2 queries (forward and swapped) and combine:
      - j_self = +1 if both agree revised wins (A beats B in forward AND
                       B beats A in swapped), shortened: "revised wins both".
      - j_self = -1 if both agree original wins.
      - j_self =  0 if they disagree, or if either is TIE, or if parsing fails.

    Returns
    -------
    j_self : np.ndarray[float32], shape [B]
        Signed preference in {-1.0, 0.0, +1.0}.
    disagree_mask : np.ndarray[bool], shape [B]
        True where the two swapped queries disagreed (used as a metric).
    """
    B = len(questions)
    assert len(prefix_texts) == B and len(revised_texts) == B and len(original_texts) == B

    # Forward: A=revised, B=original. Preference for A means revised wins.
    forward_prompts = [
        _build_judge_prompt(questions[i], prefix_texts[i], revised_texts[i], original_texts[i], judge_prompt_template)
        for i in range(B)
    ]
    forward_replies = judge_generator(forward_prompts) if B > 0 else []
    forward_parsed = [_parse_judge_reply(r) for r in forward_replies]

    if order_swap:
        # Swapped: A=original, B=revised. Preference for B means revised wins.
        swapped_prompts = [
            _build_judge_prompt(questions[i], prefix_texts[i], original_texts[i], revised_texts[i], judge_prompt_template)
            for i in range(B)
        ]
        swapped_replies = judge_generator(swapped_prompts) if B > 0 else []
        swapped_parsed = [_parse_judge_reply(r) for r in swapped_replies]
    else:
        swapped_parsed = [None] * B

    j_self = np.zeros(B, dtype=np.float32)
    disagree = np.zeros(B, dtype=bool)

    for i in range(B):
        fwd = forward_parsed[i]
        swp = swapped_parsed[i] if order_swap else None

        if fwd is None:
            j_self[i] = 0.0
            continue

        if fwd == "TIE":
            j_self[i] = 0.0
            continue

        # Map to "revised-wins / original-wins / tie" in a consistent frame.
        rev_wins_fwd = fwd == "A"
        orig_wins_fwd = fwd == "B"

        if not order_swap:
            j_self[i] = 1.0 if rev_wins_fwd else (-1.0 if orig_wins_fwd else 0.0)
            continue

        if swp is None or swp == "TIE":
            j_self[i] = 0.0
            continue

        rev_wins_swp = swp == "B"  # in swapped frame, B is revised
        orig_wins_swp = swp == "A"

        if rev_wins_fwd and rev_wins_swp:
            j_self[i] = 1.0
        elif orig_wins_fwd and orig_wins_swp:
            j_self[i] = -1.0
        else:
            j_self[i] = 0.0
            disagree[i] = True

    return j_self, disagree


# -----------------------------------------------------------------------------
# Preference merge + gates
# -----------------------------------------------------------------------------


def compute_external_preference(
    reward_revised: np.ndarray,
    reward_original: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Map (R_rev - R_orig) via tanh scaling to [-1, 1]. NaN-safe."""
    diff = np.asarray(reward_revised, dtype=np.float32) - np.asarray(reward_original, dtype=np.float32)
    scaled = np.tanh(diff / max(temperature, 1e-6))
    scaled = np.clip(scaled, -1.0, 1.0)
    return scaled.astype(np.float32)


def _detect_answer_flip(revised_texts: list[str], original_texts: list[str]) -> np.ndarray:
    """True where extracted final answers differ between revised and original."""
    B = len(revised_texts)
    out = np.zeros(B, dtype=bool)
    for i in range(B):
        a = extract_answer(original_texts[i]) or ""
        b = extract_answer(revised_texts[i]) or ""
        out[i] = (a.strip() != b.strip()) and bool(a.strip() or b.strip())
    return out


def merge_pair_preferences(
    *,
    j_self: np.ndarray,                # [B] in {-1,0,+1}
    j_ext: np.ndarray,                 # [B] in [-1,1]
    ext_available: np.ndarray,         # [B] bool
    len_ratio: np.ndarray,             # [B]
    answer_flip_mask: np.ndarray,      # [B] bool
    judge_disagree_mask: np.ndarray,   # [B] bool
    min_divergence: np.ndarray,        # [B]
    step: int,
    cfg: SirlPrefConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the merge rule + all gates to produce signed ``A_pair`` + weight + skip.

    Rule (DEFINE_SPEC §2):
        if not ext_available:
            a0 = j_self
        else:
            a0 = 0.5 * (j_self + j_ext)
            if abs(j_ext) >= δ:  a0 = j_ext         # oracle override
        len_pen = λ_len * max(0, len_ratio - τ_len)
        a1 = sign(a0) * max(0, abs(a0) - len_pen)
        a1 = 0 if judge_disagree or min_div < thresh or (answer_flip and not ext and j_self==0)
        skip if |a1| < ε
    """
    B = j_self.shape[0]
    a0 = np.where(ext_available, 0.5 * (j_self + j_ext), j_self).astype(np.float32)
    override = ext_available & (np.abs(j_ext) >= cfg.external_margin_floor)
    a0 = np.where(override, j_ext, a0)

    # Length penalty (warmup-gated)
    apply_len_pen = step >= cfg.length_penalty_warmup_steps
    len_pen = cfg.length_penalty_weight * np.maximum(0.0, len_ratio - cfg.length_ratio_cap)
    if not apply_len_pen:
        len_pen = np.zeros_like(len_pen)
    a1 = np.sign(a0) * np.maximum(0.0, np.abs(a0) - len_pen)

    # Gates
    divergence_gate = min_divergence < cfg.min_divergence_threshold
    flip_gate = answer_flip_mask & (~ext_available) & (j_self == 0)
    zero_mask = judge_disagree_mask | divergence_gate | flip_gate
    a1 = np.where(zero_mask, np.float32(0.0), a1)

    # Tie-band skip
    skip_mask = np.abs(a1) < cfg.tie_band_eps
    pair_weight = np.abs(a1).astype(np.float32)

    return a1.astype(np.float32), pair_weight, skip_mask


# -----------------------------------------------------------------------------
# Top-level orchestrator
# -----------------------------------------------------------------------------


def make_judge_generator(
    actor_rollout_wg,
    tokenizer,
    max_prompt_length: int,
    judge_max_tokens: int = 8,
) -> Callable[[list[str]], list[str]]:
    """Return a callable that runs short judge completions via the actor rollout.

    Reuses ``_generate_follow_ups_with_tokens``; judge_max_tokens is the caller's
    contract, enforced by post-hoc truncation of the decoded response.

    NOTE: max_tokens override at the vLLM engine level requires additional
    plumbing via sirl_config meta_info; in Week-1 we accept the overhead of
    full-length judge completions and truncate the decoded text. Tightening
    this is a Week-2 perf item.
    """
    def _gen(prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        texts, _ = _generate_follow_ups_with_tokens(
            prompts, tokenizer, actor_rollout_wg, max_prompt_length,
        )
        # Truncate on first newline or after ~16 chars to simulate the judge contract.
        out = []
        for t in texts:
            t = t.strip()
            # Keep only up to the first line break or 16 chars — judge is supposed
            # to emit a single token.
            t = t.split("\n", 1)[0][:16]
            out.append(t)
        return out
    return _gen


def compute_pair_len_ratio(revised_texts: list[str], original_texts: list[str]) -> np.ndarray:
    """Token-length-ish ratio len(revised) / len(original), clamped to [0, 10]."""
    B = len(revised_texts)
    out = np.ones(B, dtype=np.float32)
    for i in range(B):
        a = max(1, len(original_texts[i].split()))
        b = len(revised_texts[i].split())
        out[i] = min(10.0, max(0.0, b / a))
    return out


def compute_pair_min_divergence(revised_texts: list[str], original_texts: list[str]) -> np.ndarray:
    """``1 - SequenceMatcher(revised, original).ratio()``, per pair."""
    B = len(revised_texts)
    out = np.zeros(B, dtype=np.float32)
    for i in range(B):
        out[i] = 1.0 - _sequence_similarity(revised_texts[i], original_texts[i])
    return out


def build_paired_dataproto(
    *,
    questions: list[str],
    prefix_texts: list[str],
    suffix_texts: list[str],
    tokenizer,
    max_prompt_length: int = 4096,
    max_response_length: int = 3072,
):
    """Build a DataProto for log-probability computation over (x+prefix, suffix).

    Shares tokenization conventions with ``_build_follow_up_dataproto`` so that
    ``actor_rollout_wg.compute_log_prob(dp)`` returns per-token logps over
    ``suffix_texts`` under the current policy.

    Returns a DataProto with batch keys matching the standard verl training
    format: ``input_ids``, ``attention_mask``, ``position_ids``, ``prompts``,
    ``responses``.
    """
    from verl.protocol import DataProto
    from verl.utils.model import compute_position_id_with_mask

    B = len(questions)
    assert len(prefix_texts) == B and len(suffix_texts) == B
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    prompt_ids_per = []
    response_ids_per = []
    for i in range(B):
        # Build prompt: question + prefix. Follow the chat template if the
        # tokenizer has one; otherwise concatenate.
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [{"role": "user", "content": questions[i]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            full = formatted + prefix_texts[i]
        else:
            full = questions[i] + "\n" + prefix_texts[i]
        p_ids = tokenizer.encode(full, add_special_tokens=False)
        if len(p_ids) > max_prompt_length:
            p_ids = p_ids[-max_prompt_length:]
        prompt_ids_per.append(p_ids)

        r_ids = tokenizer.encode(suffix_texts[i], add_special_tokens=False)
        if len(r_ids) > max_response_length:
            r_ids = r_ids[:max_response_length]
        response_ids_per.append(r_ids)

    # Left-pad prompts, right-pad responses (standard verl layout).
    max_p = max((len(p) for p in prompt_ids_per), default=0) or 1
    max_p = min(max_p, max_prompt_length)
    max_r = max((len(r) for r in response_ids_per), default=0) or 1
    max_r = min(max_r, max_response_length)

    prompts = []
    responses = []
    input_ids = []
    attn = []
    for i in range(B):
        p = prompt_ids_per[i]
        r = response_ids_per[i]
        # Left-pad prompt
        p_pad = [pad_id] * (max_p - len(p)) + p
        p_mask = [0] * (max_p - len(p)) + [1] * len(p)
        # Right-pad response
        r_pad = r + [pad_id] * (max_r - len(r))
        r_mask = [1] * len(r) + [0] * (max_r - len(r))
        prompts.append(p_pad)
        responses.append(r_pad)
        input_ids.append(p_pad + r_pad)
        attn.append(p_mask + r_mask)

    prompts_t = torch.tensor(prompts, dtype=torch.long)
    responses_t = torch.tensor(responses, dtype=torch.long)
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    attn_t = torch.tensor(attn, dtype=torch.long)
    position_ids_t = compute_position_id_with_mask(attn_t)

    raw_prompt_ids = np.empty(B, dtype=object)
    for i in range(B):
        valid = torch.tensor(prompts[i], dtype=torch.long)[
            torch.tensor(attn[i][:max_p], dtype=torch.long).bool()
        ]
        raw_prompt_ids[i] = valid.cpu().numpy()

    dp = DataProto.from_dict(
        tensors={
            "prompts": prompts_t,
            "responses": responses_t,
            "input_ids": input_ids_t,
            "attention_mask": attn_t,
            "position_ids": position_ids_t,
        },
        non_tensors={"raw_prompt_ids": raw_prompt_ids},
        meta_info={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": pad_id,
            "do_sample": False,
            "recompute_log_prob": True,
            "n": 1,
        },
    )
    return dp


def apply_sirl_pref_rewards(
    *,
    batch,                         # DataProto with initial y0 rollout
    tokenizer,
    actor_rollout_wg,
    cfg: SirlPrefConfig,
    step: int = 0,
    oracle_reward_fn: Optional[Callable[[list[str], list[str]], tuple[np.ndarray, np.ndarray]]] = None,
) -> dict[str, Any]:
    """Top-level orchestrator. Given a batch of initial rollouts (``y0``):

    1. For each sample, pick a branch point ``b`` and form the prefix.
    2. Generate a revised suffix ``s+`` conditioned on (question, prefix, IMPROVE).
       The original suffix ``s-`` is read directly from y0's response past ``b``.
    3. Run the self-judge (or oracle) to produce a signed ``A_pair``.
    4. Apply gates (length, answer-flip, divergence, disagreement).
    5. Return a dict of tensors/arrays ready to be attached to the batch.

    Returns a dict with keys documented inline; the caller is expected to
    attach them to ``batch.non_tensor_batch`` (ragged) and ``batch.batch``
    (tensors) before dispatching to the trainer.
    """
    if not cfg.enable:
        return {}

    # 1) Decode y0 text + extract question text per sample.
    prompt_ids = batch.batch["prompts"]              # [B, Tp]
    response_ids = batch.batch["responses"]          # [B, Tr]
    attn_mask = batch.batch["attention_mask"]        # [B, Tp+Tr]
    Tp = prompt_ids.shape[1]
    B = prompt_ids.shape[0]

    y0_texts: list[str] = []
    prefix_texts: list[str] = []
    original_suffix_texts: list[str] = []
    branch_points: list[int] = []
    question_texts: list[str] = []
    rng = random.Random(step * 1000003 + 17)

    for i in range(B):
        resp_mask = attn_mask[i, Tp:]
        resp_len = int(resp_mask.sum().item())
        ids = response_ids[i, :resp_len]
        y0_texts.append(tokenizer.decode(ids, skip_special_tokens=True))

        if cfg.pair_construction == "prefix_branch":
            b = select_branch_point(
                resp_len,
                branch_min_frac=cfg.branch_min_frac,
                branch_max_frac=cfg.branch_max_frac,
                min_suffix_tokens=cfg.branch_min_suffix_tokens,
                rng=rng,
            )
        else:  # whole_response
            b = 0
        branch_points.append(b)
        pref_ids = ids[:b]
        suff_ids = ids[b:]
        prefix_texts.append(tokenizer.decode(pref_ids, skip_special_tokens=True))
        original_suffix_texts.append(tokenizer.decode(suff_ids, skip_special_tokens=True))

        q = _extract_question_from_prompt(prompt_ids[i], attn_mask[i, :Tp], tokenizer)
        question_texts.append(q)

    # 2) Build revise prompts: question + prefix + explicit IMPROVE instruction.
    revise_prompts: list[str] = []
    improve_instr = (
        "\n\nPlease continue the reasoning above. If you see an error, "
        "correct it. Otherwise, proceed with the next logical step "
        "and give the final answer. End your response with the final "
        "answer in \\boxed{}.\n"
    )
    for i in range(B):
        p = question_texts[i] + "\n\n" + prefix_texts[i] + improve_instr
        revise_prompts.append(p)

    revised_suffix_texts, revised_dp = _generate_follow_ups_with_tokens(
        revise_prompts, tokenizer, actor_rollout_wg, max_prompt_length=cfg.max_prompt_length,
    )

    # 3) Preferences
    j_self = np.zeros(B, dtype=np.float32)
    disagree_mask = np.zeros(B, dtype=bool)
    if cfg.preference_source in ("self", "hybrid"):
        judge_gen = make_judge_generator(
            actor_rollout_wg, tokenizer,
            max_prompt_length=cfg.max_prompt_length,
            judge_max_tokens=cfg.judge_max_tokens,
        )
        j_self, disagree_mask = judge_self_preference(
            questions=question_texts,
            prefix_texts=prefix_texts,
            revised_texts=revised_suffix_texts,
            original_texts=original_suffix_texts,
            judge_generator=judge_gen,
            judge_prompt_template=cfg.judge_prompt_template,
            order_swap=cfg.judge_order_swap,
        )
    elif cfg.preference_source == "random":
        j_self = np.random.choice([-1.0, 0.0, 1.0], size=B, p=[0.4, 0.2, 0.4]).astype(np.float32)

    # Oracle preference
    j_ext = np.zeros(B, dtype=np.float32)
    ext_available = np.zeros(B, dtype=bool)
    if cfg.preference_source in ("oracle", "hybrid") and oracle_reward_fn is not None:
        r_rev, r_orig = oracle_reward_fn(revised_suffix_texts, original_suffix_texts)
        j_ext = compute_external_preference(r_rev, r_orig)
        ext_available = np.ones(B, dtype=bool)

    # 4) Gates
    len_ratio = compute_pair_len_ratio(revised_suffix_texts, original_suffix_texts)
    answer_flip = _detect_answer_flip(revised_suffix_texts, original_suffix_texts)
    min_divergence = compute_pair_min_divergence(revised_suffix_texts, original_suffix_texts)

    pair_adv_raw, pair_weight, skip_mask = merge_pair_preferences(
        j_self=j_self,
        j_ext=j_ext,
        ext_available=ext_available,
        len_ratio=len_ratio,
        answer_flip_mask=answer_flip,
        judge_disagree_mask=disagree_mask,
        min_divergence=min_divergence,
        step=step,
        cfg=cfg,
    )

    label = (pair_adv_raw > 0).astype(np.int64)

    # 5a) Build matched DataProtos for logp computation on (x+prefix, s±).
    #     Only needed for the DPO arm (these feed compute_log_prob). Skip in
    #     shaped_grpo mode to reduce tensor churn and avoid any unintended
    #     interactions with the FSDP actor state between training steps.
    revised_paired_dp = None
    original_paired_dp = None
    if cfg.loss == "dpo":
        revised_paired_dp = build_paired_dataproto(
            questions=question_texts,
            prefix_texts=prefix_texts,
            suffix_texts=revised_suffix_texts,
            tokenizer=tokenizer,
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
        )
        original_paired_dp = build_paired_dataproto(
            questions=question_texts,
            prefix_texts=prefix_texts,
            suffix_texts=original_suffix_texts,
            tokenizer=tokenizer,
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
        )

    # 5b) Package tensors. The caller unrolls to 2B rows for the trainer.
    return {
        "revised_suffix_texts": revised_suffix_texts,
        "original_suffix_texts": original_suffix_texts,
        "prefix_texts": prefix_texts,
        "question_texts": question_texts,
        "branch_points": np.asarray(branch_points, dtype=np.int32),
        "revised_dataproto": revised_dp,
        "revised_paired_dataproto": revised_paired_dp,
        "original_paired_dataproto": original_paired_dp,
        "j_self": j_self,
        "j_ext": j_ext,
        "ext_available": ext_available,
        "len_ratio": len_ratio,
        "answer_flip_mask": answer_flip,
        "min_divergence": min_divergence,
        "disagree_mask": disagree_mask,
        "pair_advantage_raw": pair_adv_raw,
        "pair_weight": pair_weight,
        "skip_mask": skip_mask,
        "label": label,
        "tie_rate": float((pair_adv_raw == 0).mean()),
        "disagree_rate": float(disagree_mask.mean()),
        "len_ratio_mean": float(len_ratio.mean()),
        "answer_flip_rate": float(answer_flip.mean()),
    }
