# Copyright 2025 Self-Aware RL Project
#
# SIRL (Self-Improvement Reinforcement Learning) utilities for the verl trainer.
# Computes rewards based on the model's ability to improve/diversify/degrade its
# own solutions via follow-up generation, rather than majority voting alone.

from __future__ import annotations

import logging
import time
import numpy as np
import torch
from verl.utils.reward_score.ttrl_math import extract_answer, grade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text similarity (SequenceMatcher)
# ---------------------------------------------------------------------------

def _text_similarity(a: str, b: str) -> float:
    """Sequence-level similarity using difflib SequenceMatcher."""
    from difflib import SequenceMatcher
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _extract_reasoning(text: str) -> str:
    """Extract reasoning portion (everything before the boxed answer)."""
    answer = extract_answer(text)
    if answer and answer in text:
        idx = text.rfind(answer)
        return text[:idx].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------

def _improvement_reward(y0_text: str, y_imp_text: str, alpha: float) -> float:
    """High when model couldn't improve y0 (stability = good)."""
    y0_ans = extract_answer(y0_text)
    y_imp_ans = extract_answer(y_imp_text)
    ans_match = 1.0 if (y0_ans and y_imp_ans and grade(y0_ans, y_imp_ans)) else 0.0
    reas_sim = _text_similarity(_extract_reasoning(y0_text), _extract_reasoning(y_imp_text))
    return alpha * ans_match + (1.0 - alpha) * reas_sim


def _alternative_reward(y0_text: str, y_alt_text: str) -> float:
    """Binary self-consistency on answers."""
    y0_ans = extract_answer(y0_text)
    y_alt_ans = extract_answer(y_alt_text)
    return 1.0 if (y0_ans and y_alt_ans and grade(y0_ans, y_alt_ans)) else 0.0


def _deterioration_reward(y0_text: str, y_det_text: str, alpha: float) -> float:
    """Inverse of improvement — high when model successfully degraded."""
    return 1.0 - _improvement_reward(y0_text, y_det_text, alpha)


# ---------------------------------------------------------------------------
# Follow-up prompt builders
# ---------------------------------------------------------------------------

_IMP_TEMPLATE = (
    "{question}\n\nA previous attempt produced:\n{context}\n\n"
    "Please improve this solution. Reason step by step, and put your final answer within \\boxed{{}}."
)

_ALT_TEMPLATE = (
    "{question}\n\nA previous attempt produced:\n{context}\n\n"
    "Please give an alternative solution using a different approach. "
    "Reason step by step, and put your final answer within \\boxed{{}}."
)

_DET_TEMPLATE = (
    "{question}\n\nA previous attempt produced:\n{context}\n\n"
    "Please give a subtly worse version of this solution that is hard to tell apart from a correct one. "
    "Reason step by step, and put your final answer within \\boxed{{}}."
)


def _build_context(response_text: str, output_mode: str) -> str:
    """Build context string based on output_mode."""
    if output_mode == "answer":
        ans = extract_answer(response_text)
        return ans if ans else response_text
    return response_text  # "thinking" mode: full response


# ---------------------------------------------------------------------------
# Question extraction from prompt tokens
# ---------------------------------------------------------------------------

_SUFFIX_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def _extract_question_from_prompt(prompt_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> str:
    """Decode prompt tokens and extract the user question from chat-template formatting.

    Handles Qwen2.5 chat template: <|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n
    Strips the suffix prompt appended during data preparation.
    """
    # Skip left-padding: only decode tokens where attention_mask=1
    valid_mask = attention_mask[:prompt_ids.shape[-1]]
    valid_ids = prompt_ids[valid_mask.bool()]
    decoded = tokenizer.decode(valid_ids, skip_special_tokens=False)

    # Try to extract content between user markers
    user_start = "<|im_start|>user\n"
    user_end = "<|im_end|>"
    start_idx = decoded.find(user_start)
    if start_idx != -1:
        start_idx += len(user_start)
        end_idx = decoded.find(user_end, start_idx)
        if end_idx != -1:
            question = decoded[start_idx:end_idx].strip()
            # Strip suffix prompt if present
            if question.endswith(_SUFFIX_PROMPT):
                question = question[:-len(_SUFFIX_PROMPT)].strip()
            return question

    # Fallback: decode without special tokens and strip suffix
    text = tokenizer.decode(valid_ids, skip_special_tokens=True).strip()
    if text.endswith(_SUFFIX_PROMPT):
        text = text[:-len(_SUFFIX_PROMPT)].strip()
    return text


# ---------------------------------------------------------------------------
# Follow-up generation infrastructure
# ---------------------------------------------------------------------------

def _build_follow_up_dataproto(prompt_texts: list[str], tokenizer, max_prompt_length: int):
    """Build a generation-ready DataProto from follow-up prompt strings.

    Follows the same pattern as ray_trainer.py for constructing gen_batch:
    tokenize → per-sequence left-truncate → left-pad batch → compute
    position_ids → wrap in DataProto.

    Unlike original prompts, SIRL follow-ups embed the full response as
    context, so they can be much longer than max_prompt_length.  We truncate
    each sequence individually *before* batch padding so that short prompts
    are never drowned out by the padding of a longer neighbour (which caused
    vLLM "decoder prompt cannot be empty" errors).
    """
    from verl.protocol import DataProto
    from verl.utils.model import compute_position_id_with_mask

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ── 1. Tokenize each prompt individually and left-truncate ────────────
    per_seq_ids: list[list[int]] = []
    for text in prompt_texts:
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer.encode(formatted, add_special_tokens=False)
        # Left-truncate: keep the rightmost max_prompt_length tokens
        if len(ids) > max_prompt_length:
            ids = ids[-max_prompt_length:]
        per_seq_ids.append(ids)

    # ── 2. Left-pad to uniform length ─────────────────────────────────────
    max_len = max(len(ids) for ids in per_seq_ids) if per_seq_ids else 0
    # Ensure we don't exceed max_prompt_length (already truncated above,
    # but guard against edge cases)
    max_len = min(max_len, max_prompt_length) if max_len > 0 else max_prompt_length

    batch_ids = []
    batch_mask = []
    for ids in per_seq_ids:
        pad_len = max_len - len(ids)
        batch_ids.append([pad_id] * pad_len + ids)
        batch_mask.append([0] * pad_len + [1] * len(ids))

    input_ids = torch.tensor(batch_ids, dtype=torch.long)
    attention_mask = torch.tensor(batch_mask, dtype=torch.long)

    # ── 3. Compute position_ids & raw_prompt_ids ──────────────────────────
    position_ids = compute_position_id_with_mask(attention_mask)

    raw_prompt_ids = np.empty(len(prompt_texts), dtype=object)
    for i in range(len(prompt_texts)):
        valid = attention_mask[i].bool()
        raw_prompt_ids[i] = input_ids[i][valid].cpu().numpy()

    dp = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        non_tensors={"raw_prompt_ids": raw_prompt_ids},
        meta_info={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": pad_id,
            "do_sample": True,
            "recompute_log_prob": False,
            "n": 1,  # SIRL follow-ups: exactly 1 response per prompt
        },
    )
    return dp


def _generate_follow_ups_with_tokens(
    prompt_texts: list[str],
    tokenizer,
    actor_rollout_wg,
    max_prompt_length: int,
) -> tuple[list[str], "DataProto"]:
    """Generate follow-up responses and return both decoded text and raw DataProto.

    When the rollout worker is configured with ``n > 1`` (multiple completions
    per prompt), this function selects only the **first** completion for each
    prompt so that the output always has a 1-to-1 correspondence with
    ``prompt_texts``.

    Returns
    -------
    texts : list[str]
        Decoded response texts (one per prompt).
    output : DataProto
        Full generation output with batch keys ``input_ids``, ``attention_mask``,
        ``position_ids``, ``prompts``, ``responses``.  Useful for downstream
        log-prob computation on the improvement tokens.
    """
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    dp = _build_follow_up_dataproto(prompt_texts, tokenizer, max_prompt_length)
    num_prompts = len(prompt_texts)

    # Pad to be divisible by world_size
    world_size = actor_rollout_wg.world_size
    dp_padded, pad_size = pad_dataproto_to_divisor(dp, world_size)

    # Generate
    output = actor_rollout_wg.generate_sequences(dp_padded)

    # Unpad
    output = unpad_dataproto(output, pad_size)

    # Handle n > 1: the rollout may produce multiple completions per prompt.
    # Select the first completion for each prompt (stride = n).
    actual_n = len(output) // num_prompts if num_prompts > 0 else 1
    if actual_n > 1:
        keep_indices = list(range(0, len(output), actual_n))
        output = output[keep_indices]

    # Decode response tokens
    response_ids = output.batch["responses"]
    prompt_length = output.batch["prompts"].shape[1]
    attn_mask = output.batch["attention_mask"]

    texts = []
    for i in range(len(output)):
        valid_len = int(attn_mask[i, prompt_length:].sum().item())
        valid_resp = response_ids[i, :valid_len]
        texts.append(tokenizer.decode(valid_resp, skip_special_tokens=True))
    return texts, output


def _generate_follow_ups(
    prompt_texts: list[str],
    tokenizer,
    actor_rollout_wg,
    max_prompt_length: int,
) -> list[str]:
    """Generate follow-up responses for a batch of prompt strings.

    Returns decoded response texts (one per prompt).
    """
    texts, _ = _generate_follow_ups_with_tokens(
        prompt_texts, tokenizer, actor_rollout_wg, max_prompt_length,
    )
    return texts


# ---------------------------------------------------------------------------
# Legacy group-based reward (kept for reference, no longer called)
# ---------------------------------------------------------------------------

def _compute_group_rewards(
    group_texts: list[str],
    mode: str,
    alpha: float,
    lambda_alt: float,
    lambda_det: float,
) -> np.ndarray:
    """Legacy: Compute SIRL rewards via pairwise within-group comparison.

    Replaced by per-sample follow-up generation in apply_sirl_rewards.
    Kept for reference only.
    """
    n = len(group_texts)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    answers = [extract_answer(t) for t in group_texts]
    reasonings = [_extract_reasoning(t) for t in group_texts]

    grade_mat = np.zeros((n, n), dtype=np.float32)
    for j in range(n):
        if not answers[j]:
            continue
        for k in range(j + 1, n):
            if not answers[k]:
                continue
            match = 1.0 if grade(answers[j], answers[k]) else 0.0
            grade_mat[j, k] = match
            grade_mat[k, j] = match

    sim_mat = np.zeros((n, n), dtype=np.float32)
    for j in range(n):
        for k in range(j + 1, n):
            s = _text_similarity(reasonings[j], reasonings[k])
            sim_mat[j, k] = s
            sim_mat[k, j] = s

    rewards = np.zeros(n, dtype=np.float32)
    denom = max(n - 1, 1)

    for j in range(n):
        imp_sum = alpha * (grade_mat[j].sum() - grade_mat[j, j]) + \
                  (1.0 - alpha) * (sim_mat[j].sum() - sim_mat[j, j])
        reward = imp_sum / denom

        if mode in ("with-self-consistency", "with-full-recognition"):
            alt_sum = grade_mat[j].sum() - grade_mat[j, j]
            reward += lambda_alt * (alt_sum / denom)

        if mode == "with-full-recognition":
            det_sum = denom - (alpha * (grade_mat[j].sum() - grade_mat[j, j]) +
                               (1.0 - alpha) * (sim_mat[j].sum() - sim_mat[j, j]))
            reward += lambda_det * (det_sum / denom)

        rewards[j] = reward

    return rewards


# ---------------------------------------------------------------------------
# Main SIRL reward computation (follow-up generation based)
# ---------------------------------------------------------------------------

def apply_sirl_rewards(
    batch,
    gen_batch_output,
    n_samples: int,
    tokenizer,
    actor_rollout_wg,
    sirl_config,
):
    """Compute SIRL rewards via follow-up generation and write them as token_level_scores.

    For each response y₀, generates follow-up probes (improvement, alternative,
    deterioration) and scores y₀ based on whether the model can improve upon it.

    Parameters
    ----------
    batch : DataProto
        The batch with prompt info (after union with gen_batch_output).
    gen_batch_output : DataProto
        Generated responses, len = num_prompts * n_samples.
    n_samples : int
        Number of training samples per prompt.
    tokenizer : PreTrainedTokenizer
        For decoding token IDs to text.
    actor_rollout_wg : WorkerGroup
        Rollout worker group for generating follow-up probes.
    sirl_config : OmegaConf
        SIRL config with mode, output_mode, alpha, lambda_alt, lambda_det, etc.
    """
    mode = sirl_config.get("mode", "default")
    output_mode = sirl_config.get("output_mode", "thinking")
    alpha = sirl_config.get("alpha", 0.5)
    lambda_alt = sirl_config.get("lambda_alt", 0.5)
    lambda_det = sirl_config.get("lambda_det", 0.3)
    k = int(sirl_config.get("k", 1))
    last_step_only = str(sirl_config.get("last_step_only", "false")).lower() in ("true", "1")
    gamma = float(sirl_config.get("gamma", 0.95))
    max_prompt_length = int(sirl_config.get("max_prompt_length", 1024))

    logger.info(
        "SIRL config: mode=%s, output_mode=%s, alpha=%.2f, lambda_alt=%.2f, "
        "lambda_det=%.2f, k=%d, last_step_only=%s, gamma=%.2f",
        mode, output_mode, alpha, lambda_alt, lambda_det, k, last_step_only, gamma,
    )

    num_total = len(gen_batch_output)
    num_prompts = num_total // n_samples
    assert num_total == num_prompts * n_samples

    t0 = time.time()

    # Decode all original responses and extract questions
    response_texts = []
    questions = []
    for i in range(num_total):
        item = gen_batch_output[i]
        prompt_ids = item.batch["prompts"]
        response_ids = item.batch["responses"]
        prompt_length = prompt_ids.shape[-1]
        valid_len = item.batch["attention_mask"][prompt_length:].sum().item()
        valid_response_ids = response_ids[:int(valid_len)]
        response_texts.append(tokenizer.decode(valid_response_ids, skip_special_tokens=True))
        questions.append(_extract_question_from_prompt(
            prompt_ids, item.batch["attention_mask"], tokenizer,
        ))

    rewards = np.zeros(num_total, dtype=np.float32)
    imp_rewards = np.zeros(num_total, dtype=np.float32)
    alt_rewards = np.zeros(num_total, dtype=np.float32)
    det_rewards = np.zeros(num_total, dtype=np.float32)

    # ── Component 1: Improvement (always, k chained steps) ─────────────
    # Prompts chain (step t uses step t-1's output), but reward always
    # compares against the original y₀ (response_texts[j]).
    prevs = list(response_texts)  # copy — updated each step for prompt chaining
    for t in range(k):
        discount = gamma ** t
        imp_prompts = [
            _IMP_TEMPLATE.format(
                question=questions[j],
                context=_build_context(prevs[j], output_mode),
            )
            for j in range(num_total)
        ]
        logger.info("SIRL improvement step %d/%d: generating %d follow-ups", t + 1, k, num_total)
        y_imps = _generate_follow_ups(imp_prompts, tokenizer, actor_rollout_wg, max_prompt_length)
        if last_step_only:
            if t == k - 1:
                for j in range(num_total):
                    imp_rewards[j] = _improvement_reward(response_texts[j], y_imps[j], alpha)
        else:
            for j in range(num_total):
                r = _improvement_reward(response_texts[j], y_imps[j], alpha)
                imp_rewards[j] += discount * r
        prevs = y_imps  # chain for next step's PROMPT context

    rewards += imp_rewards

    # ── Component 2: Alternative (if mode includes it, k independent samples)
    if mode in ("with-self-consistency", "with-full-recognition"):
        for s in range(k):
            alt_prompts = [
                _ALT_TEMPLATE.format(
                    question=questions[j],
                    context=_build_context(response_texts[j], output_mode),
                )
                for j in range(num_total)
            ]
            logger.info("SIRL alternative sample %d/%d: generating %d follow-ups", s + 1, k, num_total)
            y_alts = _generate_follow_ups(alt_prompts, tokenizer, actor_rollout_wg, max_prompt_length)
            for j in range(num_total):
                alt_rewards[j] += _alternative_reward(response_texts[j], y_alts[j])
        if k > 0:
            alt_rewards /= k
        rewards += lambda_alt * alt_rewards

    # ── Component 3: Deterioration (if with-full-recognition, k independent samples)
    if mode == "with-full-recognition":
        for s in range(k):
            det_prompts = [
                _DET_TEMPLATE.format(
                    question=questions[j],
                    context=_build_context(response_texts[j], output_mode),
                )
                for j in range(num_total)
            ]
            logger.info("SIRL deterioration sample %d/%d: generating %d follow-ups", s + 1, k, num_total)
            y_dets = _generate_follow_ups(det_prompts, tokenizer, actor_rollout_wg, max_prompt_length)
            for j in range(num_total):
                det_rewards[j] += _deterioration_reward(response_texts[j], y_dets[j], alpha)
        if k > 0:
            det_rewards /= k
        rewards += lambda_det * det_rewards

    gen_time = time.time() - t0
    logger.info(
        "SIRL rewards computed: mode=%s, k=%d, last_step_only=%s, mean=%.4f, std=%.4f, gen_time=%.1fs",
        mode, k, last_step_only, rewards.mean(), rewards.std(), gen_time,
    )
    extra_info = {
        "mode": mode,
        "k": k,
        "last_step_only": last_step_only,
        "imp_rewards": imp_rewards,
        "gen_time_s": gen_time,
    }
    if mode in ("with-self-consistency", "with-full-recognition"):
        extra_info["alt_rewards"] = alt_rewards
    if mode == "with-full-recognition":
        extra_info["det_rewards"] = det_rewards
    return rewards, extra_info


# ---------------------------------------------------------------------------
# Distillation pathway: self-supervised quality gating + token targets
# ---------------------------------------------------------------------------

def _compute_mean_logp(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean log-probability over valid tokens. Shape: (batch,)."""
    lengths = mask.sum(dim=-1).clamp(min=1)
    return (log_probs * mask).sum(dim=-1) / lengths


def compute_sirl_distill_data(
    batch,
    gen_batch_output,
    n_samples: int,
    tokenizer,
    actor_rollout_wg,
    sirl_config,
):
    """Generate improvements and apply self-supervised quality gates.

    Returns the same scalar rewards as :func:`apply_sirl_rewards` (for fallback
    PPO on non-gated samples) plus a ``distill_data`` dict with the improvement
    tokens and per-sample gate decisions.

    Parameters
    ----------
    batch, gen_batch_output, n_samples, tokenizer, actor_rollout_wg, sirl_config :
        Same as :func:`apply_sirl_rewards`.

    Returns
    -------
    rewards : np.ndarray, shape ``(N,)``
        Fallback scalar rewards for non-gated samples (improvement reward).
    extra_info : dict
        Metrics including gate pass rates and perplexity drops.
    distill_data : dict
        Tensors for the actor update:

        - ``sirl_imp_input_ids`` : ``(N, imp_seq_len)``
        - ``sirl_imp_attention_mask`` : ``(N, imp_seq_len)``
        - ``sirl_imp_position_ids`` : ``(N, imp_seq_len)``
        - ``sirl_imp_responses`` : ``(N, imp_resp_len)``
        - ``sirl_gate_mask`` : ``(N,)`` bool — True ⟹ use distillation
        - ``sirl_gate_weight`` : ``(N,)`` float — perplexity-drop weight
    """
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from verl.utils.model import compute_position_id_with_mask

    mode = sirl_config.get("mode", "default")
    output_mode = sirl_config.get("output_mode", "thinking")
    alpha = sirl_config.get("alpha", 0.5)
    k = int(sirl_config.get("k", 1))
    last_step_only = str(sirl_config.get("last_step_only", "false")).lower() in ("true", "1")
    gamma = float(sirl_config.get("gamma", 0.95))
    max_prompt_length = int(sirl_config.get("max_prompt_length", 1024))

    distill_cfg = sirl_config.get("distill", {})
    use_perplexity_gate = distill_cfg.get("perplexity_gate", True)
    use_consistency_gate = distill_cfg.get("consistency_gate", True) and k >= 2
    min_weight = float(distill_cfg.get("min_weight", 0.1))

    num_total = len(gen_batch_output)
    t0 = time.time()

    logger.info(
        "SIRL distill: mode=%s, k=%d, perplexity_gate=%s, consistency_gate=%s",
        mode, k, use_perplexity_gate, use_consistency_gate,
    )

    # ── 1. Decode original responses and extract questions ────────────────
    response_texts = []
    questions = []
    for i in range(num_total):
        item = gen_batch_output[i]
        prompt_ids = item.batch["prompts"]
        response_ids = item.batch["responses"]
        prompt_length = prompt_ids.shape[-1]
        valid_len = item.batch["attention_mask"][prompt_length:].sum().item()
        valid_response_ids = response_ids[:int(valid_len)]
        response_texts.append(tokenizer.decode(valid_response_ids, skip_special_tokens=True))
        questions.append(_extract_question_from_prompt(
            prompt_ids, item.batch["attention_mask"], tokenizer,
        ))

    # ── 2. Generate k improvements (chained) ─────────────────────────────
    all_imp_texts: list[list[str]] = []  # [step][sample]
    last_imp_output = None  # DataProto from the last step's generation
    prevs = list(response_texts)

    for t in range(k):
        imp_prompts = [
            _IMP_TEMPLATE.format(
                question=questions[j],
                context=_build_context(prevs[j], output_mode),
            )
            for j in range(num_total)
        ]
        logger.info("SIRL distill improvement step %d/%d: generating %d follow-ups", t + 1, k, num_total)
        y_imps, imp_output = _generate_follow_ups_with_tokens(
            imp_prompts, tokenizer, actor_rollout_wg, max_prompt_length,
        )
        all_imp_texts.append(y_imps)
        last_imp_output = imp_output
        prevs = y_imps

    # Select the last step's improvement as the distillation target
    selected_texts = all_imp_texts[-1]

    # ── 3. Self-consistency gate (k >= 2) ─────────────────────────────────
    consistency_mask = np.ones(num_total, dtype=bool)  # default: all pass
    if use_consistency_gate and k >= 2:
        # For each sample, extract answers from all k improvements
        for j in range(num_total):
            answers_j = []
            for t in range(k):
                ans = extract_answer(all_imp_texts[t][j])
                answers_j.append(ans)
            # Cluster answers by mathematical equivalence using grade()
            # Each cluster is (representative_answer, count)
            valid_answers = [a for a in answers_j if a]
            clusters: list[tuple[str, int]] = []
            for ans in valid_answers:
                matched = False
                for idx, (rep, cnt) in enumerate(clusters):
                    if grade(ans, rep):
                        clusters[idx] = (rep, cnt + 1)
                        matched = True
                        break
                if not matched:
                    clusters.append((ans, 1))
            if clusters:
                # Find the largest cluster
                best_cluster = max(clusters, key=lambda x: x[1])
                majority_ans, majority_count = best_cluster
                # Check if selected improvement agrees with majority
                selected_ans = extract_answer(selected_texts[j])
                if not selected_ans or not grade(selected_ans, majority_ans):
                    consistency_mask[j] = False
                # Also fail if no majority (tie or all different)
                if majority_count <= k // 2:
                    consistency_mask[j] = False
            else:
                # No valid answers extracted at all
                consistency_mask[j] = False

        logger.info("SIRL distill consistency gate: %d/%d passed (%.1f%%)",
                     consistency_mask.sum(), num_total, 100 * consistency_mask.mean())

    # ── 4. Perplexity-drop gate ──────────────────────────────────────────
    perplexity_mask = np.ones(num_total, dtype=bool)
    perplexity_weights = np.zeros(num_total, dtype=np.float32)
    perplexity_drops = np.zeros(num_total, dtype=np.float32)

    if use_perplexity_gate:
        # 4a. Compute mean logp(y) from old_log_probs already in batch
        old_log_probs = batch.batch["old_log_probs"]  # (N, resp_len)
        response_mask = batch.batch["response_mask"]   # (N, resp_len)
        logp_y = _compute_mean_logp(old_log_probs, response_mask).cpu().numpy()

        # 4b. Compute mean logp(y_imp) via forward pass on improvement tokens
        # The last_imp_output DataProto has the right structure for compute_log_prob
        # but we need to reconstruct it with the *base prompt* (not the improvement prompt).
        # Actually, we want: p_θ(y_imp | original_prompt), not p_θ(y_imp | improvement_prompt).
        # So we need to build new input_ids = [original_prompt + y_imp_tokens].

        # Extract original prompt tokens and improvement response tokens
        orig_prompts = batch.batch["prompts"]  # (N, orig_prompt_len)
        orig_prompt_mask = batch.batch["attention_mask"][:, :orig_prompts.shape[1]]  # (N, orig_prompt_len)
        imp_responses = last_imp_output.batch["responses"]  # (N, imp_resp_len)
        imp_resp_prompt_len = last_imp_output.batch["prompts"].shape[1]
        imp_attn = last_imp_output.batch["attention_mask"]

        # Build [original_prompt | y_imp_response] sequences for log-prob computation
        # Valid imp response tokens per sample
        imp_resp_valid_lens = []
        for i in range(num_total):
            vl = int(imp_attn[i, imp_resp_prompt_len:].sum().item())
            imp_resp_valid_lens.append(vl)
        max_imp_resp_len = max(imp_resp_valid_lens) if imp_resp_valid_lens else 0

        # Right-truncate imp_responses to max_imp_resp_len
        imp_responses_trimmed = imp_responses[:, :max_imp_resp_len]

        # Build combined input: [orig_prompt | imp_response_trimmed]
        combined_ids = torch.cat([orig_prompts, imp_responses_trimmed], dim=1)
        imp_resp_mask = torch.zeros(num_total, max_imp_resp_len, dtype=torch.long,
                                     device=orig_prompts.device)
        for i in range(num_total):
            vl = min(imp_resp_valid_lens[i], max_imp_resp_len)
            imp_resp_mask[i, :vl] = 1
        combined_mask = torch.cat([orig_prompt_mask, imp_resp_mask], dim=1)
        combined_pos = compute_position_id_with_mask(combined_mask)

        from verl.protocol import DataProto
        imp_eval_dp = DataProto.from_dict(
            tensors={
                "input_ids": combined_ids,
                "attention_mask": combined_mask,
                "position_ids": combined_pos,
                "responses": imp_responses_trimmed,
            },
            meta_info={
                "micro_batch_size": batch.meta_info.get("micro_batch_size", 1),
                "temperature": 1.0,
                "use_dynamic_bsz": False,
                "max_token_len": combined_ids.shape[1],
            },
        )

        # Pad and compute log probs
        world_size = actor_rollout_wg.world_size
        imp_eval_padded, pad_size = pad_dataproto_to_divisor(imp_eval_dp, world_size)
        imp_log_prob_result = actor_rollout_wg.compute_log_prob(imp_eval_padded)
        imp_log_prob_result = unpad_dataproto(imp_log_prob_result, pad_size)

        imp_log_probs_tensor = imp_log_prob_result.batch["old_log_probs"]  # (N, max_imp_resp_len)
        logp_imp = _compute_mean_logp(imp_log_probs_tensor, imp_resp_mask.to(imp_log_probs_tensor.device)).cpu().numpy()

        # Gate and weight
        perplexity_drops = logp_imp - logp_y
        perplexity_mask = perplexity_drops > 0
        perplexity_weights = np.maximum(perplexity_drops, 0.0)

        logger.info(
            "SIRL distill perplexity gate: %d/%d passed (%.1f%%), mean_drop=%.4f",
            perplexity_mask.sum(), num_total, 100 * perplexity_mask.mean(),
            perplexity_drops.mean(),
        )

    # ── 5. Combine gates ─────────────────────────────────────────────────
    # When k>=2: consistency alone gates (perplexity used only for weight).
    # When k==1: perplexity alone gates (no consistency available).
    if k >= 2 and use_consistency_gate:
        gate_mask = consistency_mask
    else:
        gate_mask = perplexity_mask

    # Weight: max(Δℓ, min_weight) for gated samples, 0 for non-gated.
    if use_perplexity_gate:
        gate_weight = np.maximum(perplexity_drops, min_weight)
    else:
        gate_weight = np.full(num_total, min_weight, dtype=np.float32)
    gate_weight = gate_weight * gate_mask.astype(np.float32)

    gate_type = "consistency" if (k >= 2 and use_consistency_gate) else "perplexity"
    logger.info("SIRL distill gate (type=%s, k=%d, min_w=%.3f): %d/%d passed (%.1f%%)",
                 gate_type, k, min_weight, gate_mask.sum(), num_total, 100 * gate_mask.mean())

    # ── 6. Compute fallback scalar rewards for non-gated samples ─────────
    imp_rewards = np.zeros(num_total, dtype=np.float32)
    for j in range(num_total):
        imp_rewards[j] = _improvement_reward(response_texts[j], selected_texts[j], alpha)
    rewards = imp_rewards.copy()

    # ── 7. Package improvement tokens for distillation ────────────────────
    # Use last_imp_output but reconstruct with original prompt for the actor update
    # The actor needs [original_prompt + imp_response] as input_ids
    distill_data = {
        "sirl_imp_input_ids": combined_ids,
        "sirl_imp_attention_mask": combined_mask,
        "sirl_imp_position_ids": combined_pos,
        "sirl_imp_responses": imp_responses_trimmed,
        "sirl_gate_mask": torch.tensor(gate_mask, dtype=torch.bool),
        "sirl_gate_weight": torch.tensor(gate_weight, dtype=torch.float32),
    }

    gen_time = time.time() - t0
    extra_info = {
        "mode": mode,
        "k": k,
        "last_step_only": last_step_only,
        "imp_rewards": imp_rewards,
        "gen_time_s": gen_time,
        "distill_enabled": True,
        "gate_type": gate_type,
        "min_weight": min_weight,
        "gate_pass_rate": float(gate_mask.mean()),
        "mean_perplexity_drop": float(perplexity_drops.mean()),
        "mean_gate_weight": float(gate_weight[gate_mask].mean()) if gate_mask.any() else 0.0,
    }
    if use_consistency_gate:
        extra_info["consistency_gate_pass_rate"] = float(consistency_mask.mean())
    if use_perplexity_gate:
        extra_info["perplexity_gate_pass_rate"] = float(perplexity_mask.mean())

    logger.info(
        "SIRL distill complete: gate_pass=%.1f%%, mean_weight=%.4f, gen_time=%.1fs",
        100 * gate_mask.mean(), extra_info["mean_gate_weight"], gen_time,
    )
    return rewards, extra_info, distill_data


def compute_sirl_metrics(rewards: np.ndarray, n_samples: int, extra_info: dict | None = None) -> dict:
    """Compute SIRL-specific training metrics.

    Parameters
    ----------
    rewards : np.ndarray
        Per-sample reward array.
    n_samples : int
        Samples per prompt group.
    extra_info : dict, optional
        Extra info dict returned by apply_sirl_rewards (mode, multi_step,
        imp_rewards, alt_rewards, det_rewards, gen_time_s).
    """
    num_prompts = len(rewards) // n_samples
    per_prompt_means = [
        rewards[i * n_samples:(i + 1) * n_samples].mean()
        for i in range(num_prompts)
    ]
    metrics = {
        "sirl/reward_mean": float(np.mean(rewards)),
        "sirl/reward_std": float(np.std(rewards)),
        "sirl/reward_per_prompt_mean": float(np.mean(per_prompt_means)),
        "sirl/reward_per_prompt_std": float(np.std(per_prompt_means)),
    }
    if extra_info:
        # Config-level constants ("mode" is a string, k/last_step_only don't
        # change during training) are logged to experiment_config.json instead
        # of per-step metrics — tensorboard's add_scalar only accepts floats.
        metrics["sirl/k"] = int(extra_info.get("k", 1))
        metrics["sirl/last_step_only"] = float(bool(extra_info.get("last_step_only", False)))
        if "imp_rewards" in extra_info:
            metrics["sirl/imp_reward_mean"] = float(np.mean(extra_info["imp_rewards"]))
        if "alt_rewards" in extra_info:
            metrics["sirl/alt_reward_mean"] = float(np.mean(extra_info["alt_rewards"]))
        if "det_rewards" in extra_info:
            metrics["sirl/det_reward_mean"] = float(np.mean(extra_info["det_rewards"]))
        if "gen_time_s" in extra_info:
            metrics["sirl/gen_time_s"] = extra_info["gen_time_s"]
        # Distillation-specific metrics
        if extra_info.get("distill_enabled"):
            metrics["sirl/gate_pass_rate"] = float(extra_info.get("gate_pass_rate", 0.0))
            metrics["sirl/mean_perplexity_drop"] = float(extra_info.get("mean_perplexity_drop", 0.0))
            metrics["sirl/mean_gate_weight"] = float(extra_info.get("mean_gate_weight", 0.0))
            if "consistency_gate_pass_rate" in extra_info:
                metrics["sirl/consistency_gate_pass_rate"] = float(extra_info["consistency_gate_pass_rate"])
    return metrics
