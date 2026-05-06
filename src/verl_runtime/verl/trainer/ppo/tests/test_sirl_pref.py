# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
"""Unit tests for SIRL-Pref (DEFINE_SPEC §7).

Run with:  pytest -q src/verl_runtime/verl/trainer/ppo/tests/test_sirl_pref.py

The tests are intentionally narrow and do not exercise vLLM or FSDP; they
validate the core math/gating logic in isolation.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------- helpers --------------------------------------------------------


def _make_cfg(**overrides):
    from verl.trainer.ppo.sirl_pref_utils import SirlPrefConfig
    cfg = SirlPrefConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------- test 1: order-swap gating --------------------------------------


def test_pref_order_swap_gating():
    """Disagreement between forward and swapped judge queries → j_self = 0."""
    from verl.trainer.ppo.sirl_pref_utils import judge_self_preference

    # Canned judge that returns A on first query, A on second (inconsistent
    # under order-swap: first=revised wins, second=original wins).
    canned_replies = iter(["A", "A", "A", "A"])

    def judge_gen(prompts):
        return [next(canned_replies) for _ in prompts]

    questions = ["q1", "q2"]
    prefixes = ["p1", "p2"]
    revised = ["r1", "r2"]
    original = ["o1", "o2"]

    j_self, disagree = judge_self_preference(
        questions=questions,
        prefix_texts=prefixes,
        revised_texts=revised,
        original_texts=original,
        judge_generator=judge_gen,
        judge_prompt_template="TMPL",
        order_swap=True,
    )
    # Both forward answered A (revised wins). Both swapped answered A
    # (original wins in swapped frame). They disagree => j_self = 0, disagree=True.
    assert j_self.shape == (2,)
    assert np.all(j_self == 0.0)
    assert np.all(disagree == True)


def test_pref_order_swap_unanimous_agreement():
    """When forward says A (revised) and swapped says B (revised in swapped
    frame), both agree revised wins → j_self = +1."""
    from verl.trainer.ppo.sirl_pref_utils import judge_self_preference

    # Forward: A (revised wins); Swapped: B (revised wins in swapped frame)
    canned = ["A", "B"]
    it = iter(canned)

    def judge_gen(prompts):
        # two call rounds (forward, swapped), each len=1
        return [next(it) for _ in prompts]

    j_self, disagree = judge_self_preference(
        questions=["q"],
        prefix_texts=["p"],
        revised_texts=["r"],
        original_texts=["o"],
        judge_generator=judge_gen,
        judge_prompt_template="TMPL",
        order_swap=True,
    )
    assert j_self[0] == 1.0
    assert disagree[0] == False


# ---------- test 2: tie-band skip -----------------------------------------


def test_tie_band_skip():
    """|A_pair| < tie_band_eps must land in skip_mask."""
    from verl.trainer.ppo.sirl_pref_utils import merge_pair_preferences

    cfg = _make_cfg(tie_band_eps=0.15, length_penalty_warmup_steps=0)

    j_self = np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32)
    j_ext = np.zeros(4, dtype=np.float32)
    ext_avail = np.zeros(4, dtype=bool)
    len_ratio = np.ones(4, dtype=np.float32)
    flip = np.zeros(4, dtype=bool)
    disagree = np.zeros(4, dtype=bool)
    min_div = np.full(4, 0.5, dtype=np.float32)

    a_pair, weight, skip = merge_pair_preferences(
        j_self=j_self, j_ext=j_ext, ext_available=ext_avail,
        len_ratio=len_ratio, answer_flip_mask=flip,
        judge_disagree_mask=disagree, min_divergence=min_div,
        step=0, cfg=cfg,
    )
    # samples with |a| < 0.15 should be skipped (indices 1 with 0.0, index 3 with 0.5 is NOT skipped)
    # Actually j_self=0.5 is valid input only when hybrid mixing, but self-only should output 0.5.
    assert skip[0] == False and skip[2] == False  # ±1 not skipped
    assert skip[1] == True  # zero preference
    assert skip[3] == False  # 0.5 > 0.15


# ---------- test 3: sign-preserving group-norm at rollout_n=1 --------------


def test_groupnorm_rollout_n1_signed():
    """Singleton groups must return raw signed values unchanged."""
    from verl.trainer.ppo.core_algos import _grpo_group_normalize_signed

    # Each group has exactly one sample → singleton → return raw.
    scores = torch.tensor([1.0, -1.0, 0.3, -0.7])
    index = np.array([0, 1, 2, 3])
    out = _grpo_group_normalize_signed(scores, index)
    # Sign and magnitude preserved
    assert torch.allclose(out, scores, atol=1e-6), f"Expected {scores}, got {out}"


def test_groupnorm_nonsingleton_normalizes():
    """Non-singleton groups normalize as (x - mean) / std."""
    from verl.trainer.ppo.core_algos import _grpo_group_normalize_signed

    scores = torch.tensor([1.0, -1.0, 2.0, -2.0])
    index = np.array([0, 0, 1, 1])  # two groups of 2
    out = _grpo_group_normalize_signed(scores, index)
    # Group 0: mean=0, std=sqrt(2) → {1/sqrt(2), -1/sqrt(2)}
    # Group 1: mean=0, std=2*sqrt(2)  → {2/(2*sqrt(2)), -2/(2*sqrt(2))} = {1/sqrt(2), -1/sqrt(2)}
    expected_abs = 1.0 / np.sqrt(2)
    assert abs(abs(out[0].item()) - expected_abs) < 5e-2
    assert abs(abs(out[2].item()) - expected_abs) < 5e-2
    # Signs preserved
    assert out[0] > 0 and out[1] < 0
    assert out[2] > 0 and out[3] < 0


# ---------- test 4: DPO label swap ----------------------------------------


def test_dpo_label_swap():
    """Swapping the label should invert the sign of the loss gradient w.r.t. logprobs.

    Concretely: L_dpo(label=revised_wins) = L_dpo(label=original_wins) under
    sign-flip of (logp_rev - logp_orig). We validate the loss value changes
    sign predictably when we flip the label.
    """
    from verl.trainer.ppo.core_algos import compute_sirl_pref_dpo_loss

    B, T_rev, T_orig, T_y0 = 4, 6, 6, 6
    torch.manual_seed(0)

    rev_lp = torch.full((B, T_rev), -0.1)
    rev_ref_lp = torch.full((B, T_rev), -0.2)  # policy prefers revised slightly
    rev_mask = torch.ones(B, T_rev)

    orig_lp = torch.full((B, T_orig), -0.3)
    orig_ref_lp = torch.full((B, T_orig), -0.2)  # policy dis-prefers original
    orig_mask = torch.ones(B, T_orig)

    y0_lp = torch.full((B, T_y0), -0.2)
    y0_ref_lp = torch.full((B, T_y0), -0.2)
    y0_mask = torch.ones(B, T_y0)

    label_rev_wins = torch.ones(B, dtype=torch.int64)   # all pairs: revised wins
    label_orig_wins = torch.zeros(B, dtype=torch.int64)  # all pairs: original wins
    weights = torch.ones(B)
    skip = torch.zeros(B, dtype=torch.bool)

    loss_rev, m_rev = compute_sirl_pref_dpo_loss(
        revised_logprobs=rev_lp, revised_ref_logprobs=rev_ref_lp,
        revised_loss_mask=rev_mask,
        original_logprobs=orig_lp, original_ref_logprobs=orig_ref_lp,
        original_loss_mask=orig_mask,
        y0_logprobs=y0_lp, y0_ref_logprobs=y0_ref_lp, y0_loss_mask=y0_mask,
        label=label_rev_wins, pair_weight=weights, skip_mask=skip,
        beta=0.1, anchor_weight=0.0, kl_coef=0.0,
    )
    loss_orig, m_orig = compute_sirl_pref_dpo_loss(
        revised_logprobs=rev_lp, revised_ref_logprobs=rev_ref_lp,
        revised_loss_mask=rev_mask,
        original_logprobs=orig_lp, original_ref_logprobs=orig_ref_lp,
        original_loss_mask=orig_mask,
        y0_logprobs=y0_lp, y0_ref_logprobs=y0_ref_lp, y0_loss_mask=y0_mask,
        label=label_orig_wins, pair_weight=weights, skip_mask=skip,
        beta=0.1, anchor_weight=0.0, kl_coef=0.0,
    )
    # Policy favors the revised direction (logp higher than ref); when we say
    # revised wins, the margin is positive → L_dpo small. When we say original
    # wins, margin is negative → L_dpo large.
    assert loss_rev.item() < loss_orig.item(), f"rev_wins loss {loss_rev.item()} should be < orig_wins {loss_orig.item()}"


# ---------- test 5: oracle override margin --------------------------------


def test_oracle_override_margin():
    """When |j_ext| >= external_margin_floor, oracle overrides self-judge."""
    from verl.trainer.ppo.sirl_pref_utils import merge_pair_preferences

    cfg = _make_cfg(
        external_margin_floor=0.5,
        tie_band_eps=0.05,  # low so nothing is tie-skipped
        length_penalty_warmup_steps=0,
    )
    # self says original wins (-1); oracle says revised wins strongly (+0.9)
    j_self = np.array([-1.0], dtype=np.float32)
    j_ext = np.array([0.9], dtype=np.float32)
    ext_avail = np.array([True])
    len_ratio = np.array([1.0], dtype=np.float32)
    flip = np.array([False])
    disagree = np.array([False])
    min_div = np.array([0.5], dtype=np.float32)

    a_pair, weight, skip = merge_pair_preferences(
        j_self=j_self, j_ext=j_ext, ext_available=ext_avail,
        len_ratio=len_ratio, answer_flip_mask=flip,
        judge_disagree_mask=disagree, min_divergence=min_div,
        step=0, cfg=cfg,
    )
    # |j_ext|=0.9 >= 0.5 → oracle override; a_pair should be j_ext=0.9
    assert a_pair[0] == pytest.approx(0.9, abs=1e-5)
    assert not skip[0]

    # Now test that when |j_ext| < floor, 50/50 mix is used
    j_ext_weak = np.array([0.3], dtype=np.float32)
    a_pair2, _, _ = merge_pair_preferences(
        j_self=j_self, j_ext=j_ext_weak, ext_available=ext_avail,
        len_ratio=len_ratio, answer_flip_mask=flip,
        judge_disagree_mask=disagree, min_divergence=min_div,
        step=0, cfg=cfg,
    )
    # 0.5 * (-1 + 0.3) = -0.35
    assert a_pair2[0] == pytest.approx(-0.35, abs=1e-5)


# ---------- extra: shaped-GRPO advantage broadcasting ----------------------


def test_shaped_grpo_channel_sign():
    """shaped_grpo: revised rows get +A_norm, original rows get -c * A_norm."""
    from verl.trainer.ppo.core_algos import compute_sirl_pref_advantage

    # 2 pairs → 4 rows, organized as [pair0_revised, pair0_original, pair1_revised, pair1_original]
    pair_adv_raw = torch.tensor([0.8, 0.8, -0.6, -0.6])
    pair_id = np.array([0, 0, 1, 1], dtype=np.int64)
    channel = np.array(["revised", "original", "revised", "original"], dtype=object)
    skip = torch.zeros(4, dtype=torch.bool)
    token_level_rewards = torch.zeros(4, 5)
    response_mask = torch.ones(4, 5)

    adv, _ = compute_sirl_pref_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=pair_id,  # each pair as its own singleton group for groupnorm
        sirl_pref_pair_advantage_raw=pair_adv_raw,
        sirl_pref_pair_id=pair_id,
        sirl_pref_channel=channel,
        sirl_pref_skip_mask=skip,
        sirl_pref_loss="shaped_grpo",
        sirl_pref_grpo_reject_coef=0.5,
    )
    # Each pair is its own singleton group → A_norm = raw A_pair
    # pair0: revised row → +0.8 per token; original row → -0.4 per token
    # pair1: revised row → -0.6 per token; original row → +0.3 per token
    assert torch.allclose(adv[0], torch.full((5,), 0.8), atol=1e-6)
    assert torch.allclose(adv[1], torch.full((5,), -0.4), atol=1e-6)
    assert torch.allclose(adv[2], torch.full((5,), -0.6), atol=1e-6)
    assert torch.allclose(adv[3], torch.full((5,), 0.3), atol=1e-6)


def test_parse_judge_reply_tolerates_noise():
    """Parser must correctly extract A/B/TIE from realistic noisy judge replies."""
    from verl.trainer.ppo.sirl_pref_utils import _parse_judge_reply

    # Cases the old parser got wrong:
    assert _parse_judge_reply("ANSWER: A") == "A"
    assert _parse_judge_reply("BETTER: A") == "A"        # old: "B" (wrong)
    assert _parse_judge_reply("Choice: A") == "A"         # old: "TIE" (wrong)
    assert _parse_judge_reply("The answer is A") == "A"   # old: "TIE" (wrong)
    assert _parse_judge_reply("The answer is B") == "B"
    assert _parse_judge_reply("Both tied") == "TIE"       # old: "B" (wrong)
    assert _parse_judge_reply("Completion B") == "B"      # old: None (dropped)

    # Clean cases:
    assert _parse_judge_reply("A") == "A"
    assert _parse_judge_reply("B") == "B"
    assert _parse_judge_reply("TIE") == "TIE"
    assert _parse_judge_reply("C") == "TIE"               # A/B/C template
    assert _parse_judge_reply("") is None
    assert _parse_judge_reply(None) is None

    # Both A and B present → TIE (ambiguous)
    assert _parse_judge_reply("A is better than B") == "TIE"


def test_shaped_grpo_single_channel_b_row_layout():
    """B-row single-channel layout (all rows tagged 'original', one per sample).

    This is what the current trainer produces. Expected: advantage = -A_norm
    broadcast to y0 tokens (SCoRe-style push-y0-away-when-revision-wins).
    """
    from verl.trainer.ppo.core_algos import compute_sirl_pref_advantage

    # 3 samples, each its own pair_id, all tagged "original".
    pair_adv_raw = torch.tensor([0.8, -0.5, 0.2])
    pair_id = np.array([0, 1, 2], dtype=np.int64)
    channel = np.array(["original"] * 3, dtype=object)
    skip = torch.zeros(3, dtype=torch.bool)
    token_level_rewards = torch.zeros(3, 4)
    response_mask = torch.ones(3, 4)

    adv, _ = compute_sirl_pref_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=pair_id,
        sirl_pref_pair_advantage_raw=pair_adv_raw,
        sirl_pref_pair_id=pair_id,
        sirl_pref_channel=channel,
        sirl_pref_skip_mask=skip,
        sirl_pref_loss="shaped_grpo",
    )
    # Each pair is a singleton → A_norm = raw A_pair.
    # B-row single-channel: advantage = -A_norm per y0 sample.
    assert torch.allclose(adv[0], torch.full((4,), -0.8), atol=1e-6)
    assert torch.allclose(adv[1], torch.full((4,), 0.5), atol=1e-6)
    assert torch.allclose(adv[2], torch.full((4,), -0.2), atol=1e-6)


def test_dpo_advantage_returns_zero():
    """When loss='dpo', advantage path emits zero tensors (DPO loss handled elsewhere)."""
    from verl.trainer.ppo.core_algos import compute_sirl_pref_advantage

    token_level_rewards = torch.zeros(4, 5)
    response_mask = torch.ones(4, 5)
    adv, ret = compute_sirl_pref_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        sirl_pref_loss="dpo",
    )
    assert torch.all(adv == 0)
    assert torch.all(ret == 0)


def test_dpo_loss_has_nonzero_gradient_through_module():
    """Proves compute_sirl_pref_dpo_loss is differentiable end-to-end.

    Wires a tiny nn.Module, forwards twice (simulating revised and original
    log-prob computation with gradient flow), and verifies that backward
    produces non-zero gradients on parameters. This is the end-to-end
    correctness check for the DPO loss — the remaining "fresh-forward"
    plumbing in dp_actor will follow the same pattern.
    """
    from verl.trainer.ppo.core_algos import compute_sirl_pref_dpo_loss

    B, T = 4, 8
    torch.manual_seed(42)

    # Toy model: single linear projection that produces per-token logprobs.
    # Two forward passes simulate scoring revised vs original suffixes.
    model = torch.nn.Linear(4, 1)
    features_rev = torch.randn(B, T, 4)
    features_orig = torch.randn(B, T, 4)

    rev_logp = model(features_rev).squeeze(-1)   # [B, T] with grad
    orig_logp = model(features_orig).squeeze(-1)  # [B, T] with grad
    rev_ref = torch.randn(B, T).detach()
    orig_ref = torch.randn(B, T).detach()

    loss, metrics = compute_sirl_pref_dpo_loss(
        revised_logprobs=rev_logp,
        revised_ref_logprobs=rev_ref,
        revised_loss_mask=torch.ones(B, T),
        original_logprobs=orig_logp,
        original_ref_logprobs=orig_ref,
        original_loss_mask=torch.ones(B, T),
        label=torch.tensor([1, 0, 1, 0], dtype=torch.int64),
        pair_weight=torch.ones(B),
        skip_mask=torch.zeros(B, dtype=torch.bool),
        beta=0.15, anchor_weight=0.05, kl_coef=0.0,
    )
    # Loss is scalar and finite
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()

    # Backward produces non-zero gradient on the module
    loss.backward()
    assert model.weight.grad is not None
    assert model.weight.grad.abs().sum().item() > 0.0, "Expected non-zero gradient on model params"
    assert model.bias.grad is not None


def test_build_paired_dataproto_shapes_align():
    """build_paired_dataproto returns a DataProto whose prompt+response concatenates
    to input_ids at the right shape, and response tokens are right-padded."""
    from verl.trainer.ppo.sirl_pref_utils import build_paired_dataproto

    class _FakeTok:
        pad_token_id = 0
        eos_token_id = 1
        chat_template = None

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return messages[0]["content"] + " [GEN]"

        def encode(self, text, add_special_tokens=False):
            # Simple deterministic tokenization by word -> hash-ish id
            return [abs(hash(w)) % 200 + 2 for w in text.split()]

    tok = _FakeTok()
    dp = build_paired_dataproto(
        questions=["what is 2+2", "find x"],
        prefix_texts=["step 1: add", "let x be unknown"],
        suffix_texts=["the answer is 4", "x equals 10"],
        tokenizer=tok,
        max_prompt_length=64,
        max_response_length=16,
    )
    B = 2
    assert dp.batch["prompts"].shape[0] == B
    assert dp.batch["responses"].shape[0] == B
    # input_ids = concatenation of prompt + response
    assert dp.batch["input_ids"].shape[1] == (
        dp.batch["prompts"].shape[1] + dp.batch["responses"].shape[1]
    )
    # attention_mask has same shape as input_ids
    assert dp.batch["attention_mask"].shape == dp.batch["input_ids"].shape
    # Response portion of attn mask marks the valid response tokens
    Tp = dp.batch["prompts"].shape[1]
    resp_mask = dp.batch["attention_mask"][:, Tp:]
    assert resp_mask.sum(dim=-1).min() > 0
