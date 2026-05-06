# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
"""Unit tests for Calibrated Preference RL (DEFINE_SPEC §1-3).

Run with:
    PYTHONPATH=/workspace/src/verl_runtime python -m pytest -q \
        src/verl_runtime/verl/trainer/ppo/tests/test_cpr.py

These tests validate the core math of the CPR loss and p_pref mixture in
isolation. They do NOT exercise vLLM, FSDP, or the full ray_trainer dispatch.
Shape / gradient contracts follow DEFINE_SPEC §1.2 and PLAN.md.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch


# ---------- helpers --------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_logp_tensors(B: int = 4, T: int = 5, seed: int = 0, requires_grad: bool = False):
    """Build realistic (negative) logp tensors + full-ones masks."""
    torch.manual_seed(seed)
    logp_a = (-torch.rand(B, T) - 0.1).clone()
    logp_b = (-torch.rand(B, T) - 0.1).clone()
    if requires_grad:
        logp_a.requires_grad_(True)
        logp_b.requires_grad_(True)
    ref_logp_a = (-torch.rand(B, T) - 0.1).detach()
    ref_logp_b = (-torch.rand(B, T) - 0.1).detach()
    mask_a = torch.ones(B, T)
    mask_b = torch.ones(B, T)
    return logp_a, logp_b, ref_logp_a, ref_logp_b, mask_a, mask_b


# ---------- test 1: alpha=0 recovers verifier ------------------------------


def test_compute_p_pref_alpha0_recovers_verifier():
    """alpha=0 collapses to pure sigmoid((r_ext_a - r_ext_b)/tau_ext)."""
    from verl.trainer.ppo.cpr_utils import compute_p_pref

    np.random.seed(0)
    p_self = np.random.rand(4).astype(np.float32)
    r_ext_a = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    r_ext_b = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    p_pref = compute_p_pref(
        p_self=p_self, r_ext_a=r_ext_a, r_ext_b=r_ext_b,
        alpha=0.0, tau_ext=1.0,
    )
    expected = _sigmoid(r_ext_a - r_ext_b)
    assert np.allclose(p_pref, expected, atol=1e-6), f"got {p_pref}, expected {expected}"

    # Half tau_ext sharpens the signal.
    p_pref_sharp = compute_p_pref(
        p_self=p_self, r_ext_a=r_ext_a, r_ext_b=r_ext_b,
        alpha=0.0, tau_ext=0.5,
    )
    expected_sharp = _sigmoid((r_ext_a - r_ext_b) / 0.5)
    assert np.allclose(p_pref_sharp, expected_sharp, atol=1e-6)
    # Winner (idx 0) pulls further above 0.5 when tau shrinks.
    assert p_pref_sharp[0] > p_pref[0]


# ---------- test 2: alpha=1 recovers self-rewarding ------------------------


def test_compute_p_pref_alpha1_recovers_self_rewarding():
    """alpha=1 ignores verifier; missing verifier forces alpha=1."""
    from verl.trainer.ppo.cpr_utils import compute_p_pref

    np.random.seed(1)
    p_self = np.random.rand(4).astype(np.float32)
    r_ext_a = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    r_ext_b = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    p_pref = compute_p_pref(
        p_self=p_self, r_ext_a=r_ext_a, r_ext_b=r_ext_b,
        alpha=1.0, tau_ext=1.0,
    )
    assert np.allclose(p_pref, p_self, atol=1e-6)

    # Flipping r_ext does not change anything when alpha=1.
    p_pref_flip = compute_p_pref(
        p_self=p_self, r_ext_a=r_ext_b, r_ext_b=r_ext_a,
        alpha=1.0, tau_ext=1.0,
    )
    assert np.allclose(p_pref, p_pref_flip, atol=1e-6)

    # When r_ext is missing, alpha is forced to 1 even if caller passes 0.3.
    p_pref_none = compute_p_pref(
        p_self=p_self, r_ext_a=None, r_ext_b=None,
        alpha=0.3, tau_ext=1.0,
    )
    assert np.allclose(p_pref_none, p_self, atol=1e-6)


# ---------- test 3: CPR loss is flat at p_pref = 0.5 -----------------------


def test_cpr_loss_at_ptie_is_flat():
    """Soft-CE minimum is log(2) at margin=0 and p_pref=0.5; gradient vanishes.

    CPR's "no tie-band" claim: at a true tie, the pair loss is minimized at
    margin=0 (BT midpoint) with zero gradient — tie handling is intrinsic to
    the soft cross-entropy, not an engineering gate. We verify the minimum
    property by initializing logp_a == logp_b (margin=0 exactly) at p=0.5.

    Away from margin=0 the loss is still convex and differentiable, so the
    anchor term (eta > 0) contributes nonzero gradient even at p_pref=0.5 —
    exercised below.
    """
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 4, 5
    torch.manual_seed(0)
    base_logp = (-torch.rand(B, T) - 0.1).detach()
    base_ref = (-torch.rand(B, T) - 0.1).detach()
    # Identical logp → margin = 0 exactly. Tests the minimum-at-tie property.
    logp_a = base_logp.clone().requires_grad_(True)
    logp_b = base_logp.clone().requires_grad_(True)
    ref_a = base_ref.clone()
    ref_b = base_ref.clone()
    mask_a = torch.ones(B, T)
    mask_b = torch.ones(B, T)
    p_pref = torch.full((B,), 0.5)

    loss, metrics = compute_cpr_loss(
        logp_a=logp_a, logp_b=logp_b,
        ref_logp_a=ref_a, ref_logp_b=ref_b,
        mask_a=mask_a, mask_b=mask_b,
        p_pref=p_pref, beta=0.15, eta=0.0,
    )
    # At margin=0 and p=0.5: L_pair = -0.5·log(0.5) - 0.5·log(0.5) = log(2).
    pair_loss = metrics.get("cpr/pair_loss", metrics.get("cpr/dpo_loss", loss.detach()))
    if torch.is_tensor(pair_loss):
        pair_loss = pair_loss.item()
    assert abs(pair_loss - math.log(2.0)) < 1e-5, f"L_pair at tie = {pair_loss}, expected log(2)"

    loss.backward()
    # At margin=0 with p=0.5, dL/dmargin = σ(0) - 0.5 = 0 → grad on logp_a is zero.
    assert logp_a.grad is not None
    assert logp_a.grad.abs().max().item() < 1e-5, f"grad not flat: {logp_a.grad.abs().max()}"

    # Away from margin=0 with eta=0.05: anchor contributes nonzero gradient
    # (realistic scenario — tests that the anchor is load-bearing when on).
    torch.manual_seed(1)
    logp_a2 = (-torch.rand(B, T) - 0.1).requires_grad_(True)
    logp_b2 = (-torch.rand(B, T) - 0.1).requires_grad_(True)
    ref_a2 = (-torch.rand(B, T) - 0.1).detach()
    ref_b2 = (-torch.rand(B, T) - 0.1).detach()
    mask_a2 = torch.ones(B, T)
    mask_b2 = torch.ones(B, T)
    loss_anchor, _ = compute_cpr_loss(
        logp_a=logp_a2, logp_b=logp_b2,
        ref_logp_a=ref_a2, ref_logp_b=ref_b2,
        mask_a=mask_a2, mask_b=mask_b2,
        p_pref=torch.full((B,), 0.5), beta=0.15, eta=0.05,
    )
    loss_anchor.backward()
    assert logp_a2.grad.abs().max().item() > 1e-4, "anchor grad should be nonzero"


# ---------- test 4: p_self order-swap symmetry -----------------------------


def test_p_self_order_swap_symmetry():
    """Symmetrized judge must satisfy p_self(a,b) + p_self(b,a) == 1."""
    from verl.trainer.ppo.cpr_utils import CPRConfig, p_self_order_swap

    # Swap-symmetric judge: first call (AB order) sees (A=0.6, B=0.3, T=0.1).
    # Swapped call (BA order) sees (A=0.3, B=0.6, T=0.1) — i.e., the judge
    # picks the same physical candidate both times.
    responses = iter([
        {"A": 0.6, "B": 0.3, "T": 0.1},  # forward: y_a first
        {"A": 0.3, "B": 0.6, "T": 0.1},  # swapped: y_b first
    ])

    def judge_gen(prompts):
        # Returns list of {letter: prob} dicts, one per prompt.
        return [next(responses) for _ in prompts]

    cfg = CPRConfig()
    p_ab = p_self_order_swap(
        questions=["q"], y_a_texts=["ya"], y_b_texts=["yb"],
        judge_generator=judge_gen, cfg=cfg,
    )

    # Now compute reversed call for the same physical pair.
    responses2 = iter([
        {"A": 0.3, "B": 0.6, "T": 0.1},  # forward with (b first, a second)
        {"A": 0.6, "B": 0.3, "T": 0.1},  # swapped
    ])

    def judge_gen2(prompts):
        return [next(responses2) for _ in prompts]

    p_ba = p_self_order_swap(
        questions=["q"], y_a_texts=["yb"], y_b_texts=["ya"],
        judge_generator=judge_gen2, cfg=cfg,
    )
    assert p_ab.shape == (1,)
    assert abs(p_ab[0] + p_ba[0] - 1.0) < 1e-6, f"sym broken: {p_ab[0]} + {p_ba[0]} != 1"

    # Pure tie case: P_T = 1.0 → p_self = 0.5 exactly.
    tie_iter = iter([{"A": 0.0, "B": 0.0, "T": 1.0}, {"A": 0.0, "B": 0.0, "T": 1.0}])

    def judge_tie(prompts):
        return [next(tie_iter) for _ in prompts]

    p_tie = p_self_order_swap(
        questions=["q"], y_a_texts=["ya"], y_b_texts=["yb"],
        judge_generator=judge_tie, cfg=cfg,
    )
    assert abs(p_tie[0] - 0.5) < 1e-6


# ---------- test 5: gradient flows with correct sign -----------------------


def test_cpr_loss_gradient_flows_through_logp():
    """Grad on logp_a is negative (summed), logp_b positive, ref detached."""
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 3, 4
    torch.manual_seed(0)
    logp_a = (-torch.rand(B, T) - 0.1).requires_grad_(True)
    logp_b = (-torch.rand(B, T) - 0.1).requires_grad_(True)
    ref_a = (-torch.rand(B, T) - 0.1).detach()
    ref_b = (-torch.rand(B, T) - 0.1).detach()
    mask_a = torch.ones(B, T)
    mask_b = torch.ones(B, T)
    p_pref = torch.full((B,), 0.8)  # a preferred

    loss, _ = compute_cpr_loss(
        logp_a=logp_a, logp_b=logp_b,
        ref_logp_a=ref_a, ref_logp_b=ref_b,
        mask_a=mask_a, mask_b=mask_b,
        p_pref=p_pref, beta=0.15, eta=0.0,
    )
    loss.backward()
    assert logp_a.grad is not None and logp_b.grad is not None
    # Increasing logp_a should decrease loss → summed grad is negative.
    assert logp_a.grad.sum().item() < 0, f"logp_a summed grad {logp_a.grad.sum().item()} not < 0"
    assert logp_b.grad.sum().item() > 0, f"logp_b summed grad {logp_b.grad.sum().item()} not > 0"
    # Ref tensors are detached — they carry no grad.
    assert ref_a.grad is None
    assert ref_b.grad is None


# ---------- test 6: label-noise injection ----------------------------------


def test_compute_p_pref_label_noise_injection():
    """Symmetric flip shrinks |p-0.5|; rho=0.5 collapses all mass to 0.5."""
    from verl.trainer.ppo.cpr_utils import compute_p_pref

    np.random.seed(2)
    p_self = np.random.rand(64).astype(np.float32)
    r_ext_a = np.random.rand(64).astype(np.float32)
    r_ext_b = np.random.rand(64).astype(np.float32)

    p0 = compute_p_pref(p_self=p_self, r_ext_a=r_ext_a, r_ext_b=r_ext_b,
                       alpha=0.5, tau_ext=1.0, label_noise_rho=0.0)
    p1 = compute_p_pref(p_self=p_self, r_ext_a=r_ext_a, r_ext_b=r_ext_b,
                       alpha=0.5, tau_ext=1.0, label_noise_rho=0.1)
    p_half = compute_p_pref(p_self=p_self, r_ext_a=r_ext_a, r_ext_b=r_ext_b,
                           alpha=0.5, tau_ext=1.0, label_noise_rho=0.5)

    # rho=0.1 moves mean |p - 0.5| strictly toward 0 vs rho=0.
    sep0 = np.mean(np.abs(p0 - 0.5))
    sep1 = np.mean(np.abs(p1 - 0.5))
    assert sep1 < sep0, f"noise should reduce separation: {sep1} !< {sep0}"
    # rho=0.5 collapses to 0.5 everywhere.
    assert np.allclose(p_half, 0.5, atol=1e-6)


# ---------- test 7: compute_cpr_advantage returns zeros --------------------


def test_compute_cpr_advantage_returns_zeros():
    """Advantage path is a no-op; DPO loss is in compute_cpr_loss."""
    from verl.trainer.ppo.core_algos import compute_cpr_advantage

    B, T = 4, 6
    token_level_rewards = torch.randn(B, T)
    response_mask = torch.ones(B, T)
    adv, ret = compute_cpr_advantage(
        token_level_rewards=token_level_rewards, response_mask=response_mask,
    )
    assert adv.shape == (B, T) and ret.shape == (B, T)
    assert torch.all(adv == 0)
    assert torch.all(ret == 0)


# ---------- test 8: CPR config defaults ------------------------------------


def test_cpr_config_defaults():
    """Defaults match DEFINE_SPEC §2; forbidden legacy knobs are absent."""
    from verl.trainer.ppo.cpr_utils import CPRConfig

    cfg = CPRConfig()
    assert cfg.alpha == 0.5
    assert cfg.beta == 0.15
    assert cfg.eta == 0.0
    assert cfg.pair_source == "independent"
    assert cfg.rollout_n == 4
    assert cfg.tau_ext == 1.0
    assert cfg.judge_model == "online"
    assert cfg.label_noise_rho == 0.0
    assert cfg.anchor_mode == "symmetric"

    # Forbidden legacy knobs — CPR deliberately does NOT expose these.
    for banned in ("length_penalty", "tie_band_eps", "flip_gate",
                   "min_divergence_threshold", "disagree_gate"):
        assert not hasattr(cfg, banned), f"CPRConfig must not expose {banned}"


# ---------- test 9: anchor (eta) is active ---------------------------------


def test_cpr_eta_anchor_active():
    """With realistic negative logp, eta>0 strictly increases loss and logs anchor."""
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 4, 5
    torch.manual_seed(0)
    logp_a = (-torch.rand(B, T) - 0.5).detach()
    logp_b = (-torch.rand(B, T) - 0.5).detach()
    ref_a = (-torch.rand(B, T) - 0.5).detach()
    ref_b = (-torch.rand(B, T) - 0.5).detach()
    mask_a = torch.ones(B, T)
    mask_b = torch.ones(B, T)
    p_pref = torch.tensor([0.2, 0.4, 0.6, 0.8])

    loss0, m0 = compute_cpr_loss(
        logp_a=logp_a, logp_b=logp_b, ref_logp_a=ref_a, ref_logp_b=ref_b,
        mask_a=mask_a, mask_b=mask_b, p_pref=p_pref, beta=0.15, eta=0.0,
    )
    loss_eta, m_eta = compute_cpr_loss(
        logp_a=logp_a, logp_b=logp_b, ref_logp_a=ref_a, ref_logp_b=ref_b,
        mask_a=mask_a, mask_b=mask_b, p_pref=p_pref, beta=0.15, eta=0.05,
    )
    # L_anchor = -E[logp] is strictly positive for negative logp → loss grows.
    assert loss_eta.item() > loss0.item(), \
        f"eta=0.05 loss {loss_eta.item()} !> eta=0 loss {loss0.item()}"

    # Metrics dict advertises the anchor contribution when eta > 0.
    assert "cpr/anchor_loss" in m_eta
    anchor_val = m_eta["cpr/anchor_loss"]
    if torch.is_tensor(anchor_val):
        anchor_val = anchor_val.item()
    assert abs(anchor_val) > 1e-6, f"cpr/anchor_loss should be nonzero, got {anchor_val}"


# ---------- test 10 (T1): swap-consistency --------------------------------


def test_cpr_loss_swap_consistency():
    """Swapping (a, b) and replacing p with (1-p) must give identical loss.

    This is the soft-label DPO proper-scoring-rule property. Anchor-on-both
    preserves it; winner-only anchor would not. Catches future asymmetric
    refactors (e.g. someone "fixing" the anchor to weight only y_a).
    """
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 6, 5
    torch.manual_seed(7)
    logp_a = (-torch.rand(B, T) - 0.1).detach()
    logp_b = (-torch.rand(B, T) - 0.1).detach()
    ref_a = (-torch.rand(B, T) - 0.1).detach()
    ref_b = (-torch.rand(B, T) - 0.1).detach()
    mask_a = torch.ones(B, T)
    mask_b = torch.ones(B, T)
    p_pref = torch.tensor([0.1, 0.3, 0.5, 0.6, 0.8, 0.95])

    loss_ab, _ = compute_cpr_loss(
        logp_a=logp_a, logp_b=logp_b, ref_logp_a=ref_a, ref_logp_b=ref_b,
        mask_a=mask_a, mask_b=mask_b, p_pref=p_pref, beta=0.15, eta=0.05,
    )
    loss_ba, _ = compute_cpr_loss(
        logp_a=logp_b, logp_b=logp_a, ref_logp_a=ref_b, ref_logp_b=ref_a,
        mask_a=mask_b, mask_b=mask_a, p_pref=1.0 - p_pref, beta=0.15, eta=0.05,
    )
    assert abs(loss_ab.item() - loss_ba.item()) < 1e-5, (
        f"swap-consistency broken: {loss_ab.item()} vs {loss_ba.item()}"
    )


# ---------- test 11 (T5): order-swap disabled forward-only path -----------


def test_p_self_order_swap_disabled():
    """When judge_order_swap=False, only the forward call is issued.

    DEFINE_SPEC §2 locks order_swap=ON, but CPRConfig allows OFF for ablations.
    This test guards the OFF branch: only one call to judge_generator, p_self
    derived from the forward-only ABT distribution as p_self = P_A + 0.5*P_T.
    """
    from verl.trainer.ppo.cpr_utils import CPRConfig, p_self_order_swap

    call_count = {"n": 0}

    def judge_gen(prompts):
        call_count["n"] += 1
        return [{"A": 0.7, "B": 0.2, "T": 0.1}] * len(prompts)

    cfg = CPRConfig(judge_order_swap=False)
    p = p_self_order_swap(
        questions=["q1", "q2"], y_a_texts=["a1", "a2"], y_b_texts=["b1", "b2"],
        judge_generator=judge_gen, cfg=cfg,
    )

    assert call_count["n"] == 1, f"order-swap=OFF must issue 1 call, got {call_count['n']}"
    assert p.shape == (2,)
    # Forward-only: p_self = P_A + 0.5*P_T = 0.7 + 0.05 = 0.75.
    assert np.allclose(p, 0.75, atol=1e-6), f"forward-only p_self = {p}, expected 0.75"

    # Sanity check: enabling order-swap doubles the call count.
    call_count["n"] = 0
    cfg_on = CPRConfig(judge_order_swap=True)
    _ = p_self_order_swap(
        questions=["q1", "q2"], y_a_texts=["a1", "a2"], y_b_texts=["b1", "b2"],
        judge_generator=judge_gen, cfg=cfg_on,
    )
    assert call_count["n"] == 2, f"order-swap=ON must issue 2 calls, got {call_count['n']}"


# ---------- test 12 (T6): integration through p_self → p_pref → loss -----


def test_cpr_pipeline_integration_with_stubs():
    """End-to-end algebra path: p_self_order_swap → compute_p_pref → loss.

    Stubs the judge_generator (no vLLM) and the verifier rewards (no scoring
    function). Verifies that the metrics dict produced by compute_cpr_loss
    has the contracted keys and shapes that ray_trainer / dp_actor consume.
    Does not exercise apply_cpr_rewards, which needs a real tokenizer +
    actor_rollout_wg + DataProto stack (covered by GPU smoke instead).
    """
    from verl.trainer.ppo.cpr_utils import CPRConfig, compute_p_pref, p_self_order_swap
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 4, 6
    cfg = CPRConfig(alpha=0.5, beta=0.15, eta=0.05)

    abt = iter([
        {"A": 0.6, "B": 0.3, "T": 0.1}, {"A": 0.5, "B": 0.4, "T": 0.1},
        {"A": 0.7, "B": 0.2, "T": 0.1}, {"A": 0.4, "B": 0.5, "T": 0.1},
        {"A": 0.3, "B": 0.6, "T": 0.1}, {"A": 0.4, "B": 0.5, "T": 0.1},
        {"A": 0.2, "B": 0.7, "T": 0.1}, {"A": 0.5, "B": 0.4, "T": 0.1},
    ])

    def judge_gen(prompts):
        return [next(abt) for _ in prompts]

    p_self = p_self_order_swap(
        questions=[f"q{i}" for i in range(B)],
        y_a_texts=[f"ya{i}" for i in range(B)],
        y_b_texts=[f"yb{i}" for i in range(B)],
        judge_generator=judge_gen, cfg=cfg,
    )
    assert p_self.shape == (B,)
    assert ((0.0 <= p_self) & (p_self <= 1.0)).all()

    r_ext_a = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    r_ext_b = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    p_pref_np = compute_p_pref(
        p_self=p_self, r_ext_a=r_ext_a, r_ext_b=r_ext_b,
        alpha=cfg.alpha, tau_ext=cfg.tau_ext, label_noise_rho=cfg.label_noise_rho,
    )
    assert p_pref_np.shape == (B,)
    assert ((0.0 <= p_pref_np) & (p_pref_np <= 1.0)).all()

    torch.manual_seed(0)
    logp_a = (-torch.rand(B, T) - 0.1).requires_grad_(True)
    logp_b = (-torch.rand(B, T) - 0.1).requires_grad_(True)
    ref_a = (-torch.rand(B, T) - 0.1).detach()
    ref_b = (-torch.rand(B, T) - 0.1).detach()

    loss, metrics = compute_cpr_loss(
        logp_a=logp_a, logp_b=logp_b, ref_logp_a=ref_a, ref_logp_b=ref_b,
        mask_a=torch.ones(B, T), mask_b=torch.ones(B, T),
        p_pref=torch.from_numpy(p_pref_np), beta=cfg.beta, eta=cfg.eta,
    )
    loss.backward()

    expected_keys = {
        "cpr/dpo_loss", "cpr/anchor_loss", "cpr/margin_mean",
        "cpr/p_pref_mean", "cpr/p_pref_frac_extreme", "cpr/p_pref_frac_middle",
    }
    missing = expected_keys - set(metrics.keys())
    assert not missing, f"metrics missing keys: {missing}"
    for key, val in metrics.items():
        assert torch.is_tensor(val), f"metric {key} not a tensor: {type(val)}"
        assert val.numel() == 1, f"metric {key} not scalar: shape {val.shape}"
    assert torch.isfinite(loss)
    assert logp_a.grad is not None and logp_a.grad.abs().sum().item() > 0


# ---------- test 13 (T7): numerical stability on saturated bf16 margin ---


def test_cpr_loss_finite_on_saturated_margin():
    """Extreme margins in bf16 must produce finite loss + finite grad.

    Per review §P0 caveat: logsigmoid is stable up to |z|≈50 fp32; bf16 over a
    long response could saturate. We construct an explicitly extreme case and
    assert finite-ness rather than match the fp64 truth.
    """
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 2, 3072
    dtype = torch.bfloat16
    logp_a = torch.full((B, T), -0.001, dtype=dtype, requires_grad=True)
    logp_b = torch.full((B, T), -5.0, dtype=dtype, requires_grad=True)
    ref_a = torch.full((B, T), -0.001, dtype=dtype).detach()
    ref_b = torch.full((B, T), -0.001, dtype=dtype).detach()
    mask_a = torch.ones(B, T, dtype=dtype)
    mask_b = torch.ones(B, T, dtype=dtype)
    p_pref = torch.tensor([0.99, 0.01], dtype=dtype)

    loss, metrics = compute_cpr_loss(
        logp_a=logp_a, logp_b=logp_b, ref_logp_a=ref_a, ref_logp_b=ref_b,
        mask_a=mask_a, mask_b=mask_b, p_pref=p_pref, beta=0.15, eta=0.0,
    )
    assert torch.isfinite(loss), f"saturated bf16 loss = {loss}"
    loss.backward()
    assert torch.all(torch.isfinite(logp_a.grad)), "logp_a.grad has nan/inf"
    assert torch.all(torch.isfinite(logp_b.grad)), "logp_b.grad has nan/inf"
    margin_mean = metrics["cpr/margin_mean"]
    assert torch.isfinite(margin_mean), f"margin_mean = {margin_mean}"


# ---------- test 14 (T9): ref-detach is load-bearing ---------------------


def test_cpr_loss_ref_detach_paranoid():
    """Even if caller forgets to detach ref tensors, the in-loss .detach() wins.

    Belt-and-braces test: pass ref tensors with requires_grad=True. After
    backward, ref_a.grad and ref_b.grad must remain None — proving the
    .detach() inside compute_cpr_loss is load-bearing, not just decoration.
    """
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 3, 4
    torch.manual_seed(0)
    logp_a = (-torch.rand(B, T) - 0.1).requires_grad_(True)
    logp_b = (-torch.rand(B, T) - 0.1).requires_grad_(True)
    # Caller "forgets" to detach — these should still not get gradients.
    ref_a = (-torch.rand(B, T) - 0.1).requires_grad_(True)
    ref_b = (-torch.rand(B, T) - 0.1).requires_grad_(True)
    mask_a = torch.ones(B, T)
    mask_b = torch.ones(B, T)
    p_pref = torch.full((B,), 0.7)

    loss, _ = compute_cpr_loss(
        logp_a=logp_a, logp_b=logp_b, ref_logp_a=ref_a, ref_logp_b=ref_b,
        mask_a=mask_a, mask_b=mask_b, p_pref=p_pref, beta=0.15, eta=0.05,
    )
    loss.backward()

    assert ref_a.grad is None, f"in-loss detach broken: ref_a got grad {ref_a.grad}"
    assert ref_b.grad is None, f"in-loss detach broken: ref_b got grad {ref_b.grad}"
    assert logp_a.grad is not None and logp_a.grad.abs().sum().item() > 0
    assert logp_b.grad is not None and logp_b.grad.abs().sum().item() > 0


# ---------- P0 (post-URLVR-debate): binary entropy helper -----------------


def test_binary_entropy_bounds():
    """H(p) is maximized at p=0.5 (log 2 nats) and near zero at extremes."""
    import numpy as np
    from verl.trainer.ppo.cpr_utils import _binary_entropy

    pts = np.array([0.0, 0.05, 0.5, 0.95, 1.0], dtype=np.float32)
    h = _binary_entropy(pts)
    assert h[2] > h[1] > h[0] and h[2] > h[3] > h[4], \
        f"entropy not peaked at 0.5: {h}"
    # log(2) in nats ≈ 0.6931
    assert abs(h[2] - float(np.log(2))) < 1e-3, f"H(0.5) should be ln 2, got {h[2]}"
    # Clipped endpoints are near-zero, not NaN
    assert h[0] < 1e-3 and h[4] < 1e-3


# ---------- P0: anchor_mode winner_only is valid and differs from symmetric ---


def test_cpr_loss_anchor_mode_winner_only_differs_from_symmetric():
    """winner_only anchor must yield a different loss than symmetric when eta>0
    and p_pref is intermediate. At p_pref ∈ {0, 1} the two modes must agree.
    """
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 4, 5
    torch.manual_seed(0)
    logp_a = (-torch.rand(B, T) - 0.5).detach()
    logp_b = (-torch.rand(B, T) - 0.5).detach()
    ref_a = (-torch.rand(B, T) - 0.5).detach()
    ref_b = (-torch.rand(B, T) - 0.5).detach()
    mask = torch.ones(B, T)
    p_pref_mid = torch.tensor([0.2, 0.4, 0.6, 0.8])

    kw = dict(logp_a=logp_a, logp_b=logp_b, ref_logp_a=ref_a, ref_logp_b=ref_b,
              mask_a=mask, mask_b=mask, beta=0.15, eta=0.1)

    loss_sym, _ = compute_cpr_loss(p_pref=p_pref_mid, anchor_mode="symmetric", **kw)
    loss_win, _ = compute_cpr_loss(p_pref=p_pref_mid, anchor_mode="winner_only", **kw)

    assert abs(loss_sym.item() - loss_win.item()) > 1e-4, \
        "symmetric vs winner_only should differ at intermediate p_pref"

    # Degenerate: all p_pref ∈ {0, 1} ⇒ winner_only reduces to symmetric
    p_extreme = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss_sym_x, _ = compute_cpr_loss(p_pref=p_extreme, anchor_mode="symmetric", **kw)
    loss_win_x, _ = compute_cpr_loss(p_pref=p_extreme, anchor_mode="winner_only", **kw)
    assert abs(loss_sym_x.item() - loss_win_x.item()) < 1e-5, \
        f"modes should agree at extreme p_pref: sym={loss_sym_x.item()}, win={loss_win_x.item()}"


def test_cpr_loss_anchor_mode_rejects_bogus():
    """Unknown anchor_mode must raise ValueError — fast-fail, not silent."""
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 2, 3
    torch.manual_seed(0)
    t = torch.zeros(B, T)
    m = torch.ones(B, T)
    p = torch.full((B,), 0.5)

    with pytest.raises(ValueError, match="anchor_mode"):
        compute_cpr_loss(
            logp_a=t, logp_b=t, ref_logp_a=t, ref_logp_b=t,
            mask_a=m, mask_b=m, p_pref=p, beta=0.15, eta=0.0,
            anchor_mode="bogus",
        )


# ---------- P0: compute_p_pref + new metric plumbing smoke -----------------


def test_compute_p_pref_unchanged_under_p0_metrics():
    """The new metrics are additive — compute_p_pref output must still match
    the closed-form mixture, unchanged by P0 logging additions."""
    import numpy as np
    from verl.trainer.ppo.cpr_utils import compute_p_pref

    p_self = np.array([0.8, 0.3, 0.5], dtype=np.float32)
    r_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    r_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    p_pref = compute_p_pref(p_self=p_self, r_ext_a=r_a, r_ext_b=r_b,
                            alpha=0.5, tau_ext=1.0)
    # α=0.5 mix of p_self and sigmoid(Δ)
    ext = 1.0 / (1.0 + np.exp(-(r_a - r_b)))
    expected = 0.5 * p_self + 0.5 * ext
    assert np.allclose(p_pref, expected, atol=1e-6), f"{p_pref} vs {expected}"


# ---------- P1: disagreement-gated alpha_eff ------------------------------


def test_alpha_eff_gamma_zero_is_constant_alpha():
    """γ=0 must preserve the committed constant-α API exactly."""
    import numpy as np
    from verl.trainer.ppo.cpr_utils import compute_alpha_eff, compute_p_pref

    p_self = np.array([0.9, 0.1, 0.5, 0.7], dtype=np.float32)
    p_verif = np.array([0.1, 0.9, 0.5, 0.3], dtype=np.float32)  # sharp disagreement on 0, 1
    alpha_eff = compute_alpha_eff(p_self=p_self, p_verif=p_verif, alpha=0.5, alpha_gate_gamma=0.0)
    assert np.allclose(alpha_eff, 0.5), f"γ=0 should give constant α=0.5, got {alpha_eff}"

    # compute_p_pref at γ=0 must match the pre-patch mixture.
    r_a = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    r_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    p_pref = compute_p_pref(p_self=p_self, r_ext_a=r_a, r_ext_b=r_b, alpha=0.5, tau_ext=1.0)
    ext = 1.0 / (1.0 + np.exp(-(r_a - r_b)))
    expected = 0.5 * p_self + 0.5 * ext
    assert np.allclose(p_pref, expected, atol=1e-6), f"γ=0 p_pref must match baseline: {p_pref} vs {expected}"


def test_alpha_eff_shrinks_on_disagreement():
    """γ>0 must shrink α where self-judge and verifier disagree sharply."""
    import numpy as np
    from verl.trainer.ppo.cpr_utils import compute_alpha_eff

    # p_self=0.95 vs p_verif=0.05 is a maximal disagreement case
    # p_self=0.50 vs p_verif=0.50 is zero disagreement (identity gate)
    p_self = np.array([0.95, 0.50], dtype=np.float32)
    p_verif = np.array([0.05, 0.50], dtype=np.float32)
    alpha_eff = compute_alpha_eff(p_self=p_self, p_verif=p_verif, alpha=0.5, alpha_gate_gamma=1.0)
    assert alpha_eff[1] > alpha_eff[0], \
        f"α_eff should be larger at zero disagreement: {alpha_eff}"
    assert abs(alpha_eff[1] - 0.5) < 1e-3, f"zero-disagreement α_eff should ≈ α: {alpha_eff[1]}"
    assert alpha_eff[0] < 0.5 - 1e-3, f"disagreement α_eff should shrink: {alpha_eff[0]}"


def test_compute_p_pref_gated_at_full_disagreement_prefers_verifier():
    """At extreme disagreement and γ large, p_pref must collapse toward verifier."""
    import numpy as np
    from verl.trainer.ppo.cpr_utils import compute_p_pref

    p_self = np.array([0.98], dtype=np.float32)
    # Verifier says: A loses hard
    r_a = np.array([0.0], dtype=np.float32)
    r_b = np.array([1.0], dtype=np.float32)
    # Plain constant-α=0.9 would blend toward p_self=0.98
    p_plain = compute_p_pref(p_self=p_self, r_ext_a=r_a, r_ext_b=r_b,
                             alpha=0.9, tau_ext=1.0, alpha_gate_gamma=0.0)
    # Gated with large γ should collapse α_eff toward 0 → p_pref ≈ verifier
    p_gated = compute_p_pref(p_self=p_self, r_ext_a=r_a, r_ext_b=r_b,
                             alpha=0.9, tau_ext=1.0, alpha_gate_gamma=5.0)
    ext = 1.0 / (1.0 + np.exp(-(r_a - r_b)))[0]
    assert abs(p_gated[0] - ext) < abs(p_plain[0] - ext), \
        f"gated p_pref {p_gated[0]} should be closer to verifier {ext} than plain {p_plain[0]}"


def test_cpr_config_defaults_include_gate():
    """Regression: new fields preserve the existing default-API behavior."""
    from verl.trainer.ppo.cpr_utils import CPRConfig
    cfg = CPRConfig()
    assert cfg.alpha_gate_gamma == 0.0
    assert cfg.anchor_mode == "symmetric"


# ---------- P0 (code-audit debate 2026-04-19): shape, ties, endpoints ------


def test_compute_cpr_loss_rejects_p_pref_2d():
    """p_pref must be 1-D. [B,1] would silently broadcast to [B,B] and
    corrupt gradients — we catch it at the boundary instead."""
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 2, 3
    t = torch.zeros(B, T)
    m = torch.ones(B, T)
    # Pass [B,1] instead of [B]
    p_bad = torch.full((B, 1), 0.5)

    with pytest.raises(ValueError, match="p_pref must be 1-D"):
        compute_cpr_loss(
            logp_a=t, logp_b=t, ref_logp_a=t, ref_logp_b=t,
            mask_a=m, mask_b=m, p_pref=p_bad, beta=0.15, eta=0.0,
        )


def test_compute_cpr_loss_rejects_p_pref_length_mismatch():
    """p_pref length must match batch size."""
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 2, 3
    t = torch.zeros(B, T)
    m = torch.ones(B, T)
    p_wrong_len = torch.full((B + 1,), 0.5)

    with pytest.raises(ValueError, match="p_pref length"):
        compute_cpr_loss(
            logp_a=t, logp_b=t, ref_logp_a=t, ref_logp_b=t,
            mask_a=m, mask_b=m, p_pref=p_wrong_len, beta=0.15, eta=0.0,
        )


def test_cpr_loss_winner_only_tie_safe():
    """At p_pref exactly 0.5, winner_only must split anchor 0.5/0.5 to
    preserve swap-symmetry (not deterministically route to side a)."""
    from verl.trainer.ppo.core_algos import compute_cpr_loss

    B, T = 3, 4
    torch.manual_seed(0)
    logp_a = (-torch.rand(B, T) - 0.5).detach()
    logp_b = (-torch.rand(B, T) - 0.5).detach()
    ref_a = torch.zeros(B, T)
    ref_b = torch.zeros(B, T)
    mask = torch.ones(B, T)

    # All exact ties.
    p_tie = torch.full((B,), 0.5)
    kw = dict(logp_a=logp_a, logp_b=logp_b, ref_logp_a=ref_a, ref_logp_b=ref_b,
              mask_a=mask, mask_b=mask, p_pref=p_tie, beta=0.15, eta=0.1)
    loss_ab, _ = compute_cpr_loss(anchor_mode="winner_only", **kw)
    # Swap a and b — loss must be identical at exact ties.
    loss_ba, _ = compute_cpr_loss(
        anchor_mode="winner_only",
        logp_a=logp_b, logp_b=logp_a, ref_logp_a=ref_b, ref_logp_b=ref_a,
        mask_a=mask, mask_b=mask, p_pref=p_tie, beta=0.15, eta=0.1,
    )
    assert abs(loss_ab.item() - loss_ba.item()) < 1e-5, (
        f"winner_only at ties must be swap-symmetric: ab={loss_ab.item()} "
        f"ba={loss_ba.item()}"
    )


def test_compute_p_pref_alpha1_recovers_p_self_under_gamma():
    """α=1 endpoint must recover pure p_self REGARDLESS of alpha_gate_gamma
    (paper-claim correctness). Before the 2026-04-19 fix, γ>0 at α=1 fell
    into the gated branch and shrunk α_eff."""
    import numpy as np
    from verl.trainer.ppo.cpr_utils import compute_p_pref

    p_self = np.array([0.95, 0.10, 0.50], dtype=np.float32)
    r_a = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    r_b = np.array([1.0, 0.0, 0.5], dtype=np.float32)

    # γ = 0: traditional path — α=1 ⇒ p_pref == p_self
    p_g0 = compute_p_pref(p_self=p_self, r_ext_a=r_a, r_ext_b=r_b,
                          alpha=1.0, tau_ext=1.0, alpha_gate_gamma=0.0)
    assert np.allclose(p_g0, p_self), f"α=1 γ=0 must recover p_self: {p_g0}"

    # γ = 5 (strong gate): α=1 must STILL recover p_self after the short-circuit
    p_g5 = compute_p_pref(p_self=p_self, r_ext_a=r_a, r_ext_b=r_b,
                          alpha=1.0, tau_ext=1.0, alpha_gate_gamma=5.0)
    assert np.allclose(p_g5, p_self), (
        f"α=1 must recover p_self even with γ=5 after short-circuit fix: "
        f"got {p_g5}, expected {p_self}"
    )


def test_compute_p_pref_alpha0_recovers_verifier_under_gamma():
    """α=0 endpoint must recover pure verifier signal regardless of γ."""
    import numpy as np
    from verl.trainer.ppo.cpr_utils import compute_p_pref

    p_self = np.array([0.95, 0.10], dtype=np.float32)
    r_a = np.array([1.0, 0.0], dtype=np.float32)
    r_b = np.array([0.0, 1.0], dtype=np.float32)

    for gamma in (0.0, 1.0, 5.0):
        p = compute_p_pref(p_self=p_self, r_ext_a=r_a, r_ext_b=r_b,
                           alpha=0.0, tau_ext=1.0, alpha_gate_gamma=gamma)
        expected = 1.0 / (1.0 + np.exp(-(r_a - r_b)))
        assert np.allclose(p, expected, atol=1e-6), (
            f"α=0 γ={gamma} must recover verifier: got {p}, expected {expected}"
        )


# =========================================================================
# D1 — E1 α̂-fit module tests (DEFINE_SPEC_v2 §2.2)
# =========================================================================


def _simulate_probe_pairs(n: int, alpha_true: float, seed: int = 0):
    """Simulate n pairs with p_self, p_verif ~ Beta(2,2) and
    p_obs ∈ {0, 1} drawn from the true α-mixture.
    """
    rng = np.random.default_rng(seed)
    p_self = rng.beta(2.0, 2.0, size=n).astype(np.float32)
    p_verif = rng.beta(2.0, 2.0, size=n).astype(np.float32)
    p_true = alpha_true * p_self + (1.0 - alpha_true) * p_verif
    # Binary ground-truth labels drawn from p_true
    p_obs = (rng.uniform(size=n) < p_true).astype(np.float32)
    return p_self, p_verif, p_obs


def test_alpha_fit_recovers_synthetic_alpha_0p5():
    """With α=0.5 and 2000 pairs, α̂ should land within 0.1 and CI should contain 0.5."""
    from cpr_alpha_fit import fit_alpha_hat

    p_self, p_verif, p_obs = _simulate_probe_pairs(n=2000, alpha_true=0.5, seed=0)
    result = fit_alpha_hat(
        p_self, p_verif, p_obs,
        bootstrap=200,  # reduce for test speed; still > 100 for a usable CI
        restrict_disagreement=True,
        disagreement_threshold=0.2,
    )
    assert abs(result.alpha_hat - 0.5) < 0.1, (
        f"α̂ = {result.alpha_hat}, expected close to 0.5"
    )
    assert result.ci_lower <= 0.5 <= result.ci_upper, (
        f"CI [{result.ci_lower}, {result.ci_upper}] does not contain 0.5"
    )
    assert 0.0 <= result.alpha_hat <= 1.0
    assert result.n_used > 0
    assert result.profile_likelihood_width_95 > 0.0


def test_alpha_fit_recovers_synthetic_alpha_0p2():
    """With α=0.2, α̂ should again land within 0.1."""
    from cpr_alpha_fit import fit_alpha_hat

    p_self, p_verif, p_obs = _simulate_probe_pairs(n=2000, alpha_true=0.2, seed=1)
    result = fit_alpha_hat(
        p_self, p_verif, p_obs,
        bootstrap=200,
        restrict_disagreement=True,
        disagreement_threshold=0.2,
    )
    assert abs(result.alpha_hat - 0.2) < 0.1, (
        f"α̂ = {result.alpha_hat}, expected close to 0.2"
    )


def test_alpha_fit_degenerate_when_collinear():
    """p_self ≈ p_verif → α is non-identifiable → wide CI."""
    from cpr_alpha_fit import fit_alpha_hat

    rng = np.random.default_rng(42)
    n = 500
    p_self = rng.beta(2.0, 2.0, size=n).astype(np.float32)
    p_verif = np.clip(p_self + rng.normal(0.0, 0.01, size=n), 0.0, 1.0).astype(np.float32)
    # Ground-truth labels drawn from the mixture (which ≈ p_self either way)
    p_true = 0.5 * p_self + 0.5 * p_verif
    p_obs = (rng.uniform(size=n) < p_true).astype(np.float32)

    result = fit_alpha_hat(
        p_self, p_verif, p_obs,
        bootstrap=100,
        restrict_disagreement=False,  # collinear → disagreement filter would kill all pairs
        disagreement_threshold=0.2,
    )
    assert (result.ci_upper - result.ci_lower) > 0.5, (
        f"collinear case should give wide CI, got "
        f"[{result.ci_lower}, {result.ci_upper}]"
    )


def test_alpha_fit_rejects_bad_shapes():
    """Array-length mismatch must raise ValueError."""
    from cpr_alpha_fit import fit_alpha_hat

    p_self = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    p_verif = np.array([0.4, 0.5], dtype=np.float32)  # wrong length
    p_obs = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError):
        fit_alpha_hat(p_self, p_verif, p_obs, bootstrap=10)


def test_alpha_fit_filters_tie_pairs():
    """50% tie-labeled (p_obs=0.5) pairs must be filtered out; n_used should be ~ half."""
    from cpr_alpha_fit import fit_alpha_hat

    rng = np.random.default_rng(7)
    n = 400
    p_self = rng.beta(2.0, 2.0, size=n).astype(np.float32)
    p_verif = rng.beta(2.0, 2.0, size=n).astype(np.float32)
    p_obs = np.empty(n, dtype=np.float32)
    tie_mask = rng.uniform(size=n) < 0.5
    p_obs[tie_mask] = 0.5
    p_obs[~tie_mask] = (rng.uniform(size=(~tie_mask).sum()) < 0.5).astype(np.float32)

    result = fit_alpha_hat(
        p_self, p_verif, p_obs,
        bootstrap=50,
        restrict_disagreement=False,
    )
    # Allow wide tolerance — we are testing the filter, not binomial noise.
    n_non_tie = int((~tie_mask).sum())
    assert 0.3 * n_non_tie <= result.n_used <= n_non_tie, (
        f"n_used = {result.n_used}, expected roughly half of {n} (non-tie={n_non_tie})"
    )
    assert result.n_filtered_out >= tie_mask.sum() - 1  # ≥ tie count dropped


# =========================================================================
# D2 — S0 sanity gate evaluator (DEFINE_SPEC_v2 §2.2 gate)
# =========================================================================


def test_s0_gate_pass():
    """Spearman=0.3, disagreement=0.5 → gate_pass=True."""
    from cpr_alpha_fit import evaluate_gate

    passed, reason = evaluate_gate(spearman=0.3, disagreement_fraction=0.5)
    assert passed is True, f"gate should pass: {reason}"


def test_s0_gate_fail_collinear():
    """Spearman=0.95 → gate fails (too collinear)."""
    from cpr_alpha_fit import evaluate_gate

    passed, reason = evaluate_gate(spearman=0.95, disagreement_fraction=0.5)
    assert passed is False
    assert "spearman" in reason.lower(), f"reason should cite Spearman: {reason!r}"


def test_s0_gate_fail_low_disagreement():
    """Disagreement=0.1 → gate fails (insufficient disagreement mass)."""
    from cpr_alpha_fit import evaluate_gate

    passed, reason = evaluate_gate(spearman=0.3, disagreement_fraction=0.1)
    assert passed is False
    assert "disagreement" in reason.lower(), f"reason should cite disagreement: {reason!r}"


# =========================================================================
# D4 — E2 coupling logger (compute_coupling_calibration_kl, compute_pi_pref_lengthnorm)
# Note: renamed from compute_coupling_kl as part of B3a (see BLOCKER-3 docs).
# =========================================================================


def test_coupling_calibration_kl_zero_when_equal():
    """KL(p || p) = 0."""
    from verl.trainer.ppo.cpr_utils import compute_coupling_calibration_kl

    p_self = np.array([0.5, 0.3, 0.7, 0.9], dtype=np.float32)
    pi_pref = p_self.copy()
    kl = compute_coupling_calibration_kl(p_self, pi_pref)
    # mean over samples, should be near zero
    assert abs(float(np.mean(kl))) < 1e-5, f"KL(p||p) should be ~0, got {kl}"


def test_coupling_calibration_kl_positive_when_different():
    """KL between divergent p_self=0.9 and pi_pref=0.1 should be substantial."""
    from verl.trainer.ppo.cpr_utils import compute_coupling_calibration_kl

    p_self = np.array([0.9], dtype=np.float32)
    pi_pref = np.array([0.1], dtype=np.float32)
    kl = compute_coupling_calibration_kl(p_self, pi_pref)
    assert float(kl[0]) > 0.5, f"KL should be > 0.5 when p=0.9, q=0.1, got {kl[0]}"


def test_coupling_calibration_kl_symmetry_not_zero():
    """KL is NOT symmetric — KL(p||q) ≠ KL(q||p)."""
    from verl.trainer.ppo.cpr_utils import compute_coupling_calibration_kl

    p_self = np.array([0.9], dtype=np.float32)
    pi_pref = np.array([0.1], dtype=np.float32)
    kl_forward = float(compute_coupling_calibration_kl(p_self, pi_pref)[0])
    kl_reverse = float(compute_coupling_calibration_kl(pi_pref, p_self)[0])
    # For the binary-symmetric case KL(p||q) and KL(q||p) happen to coincide
    # when p, q = (0.9, 0.1) — use an asymmetric pair instead.
    p_self = np.array([0.9], dtype=np.float32)
    pi_pref = np.array([0.3], dtype=np.float32)
    kl_forward = float(compute_coupling_calibration_kl(p_self, pi_pref)[0])
    kl_reverse = float(compute_coupling_calibration_kl(pi_pref, p_self)[0])
    assert abs(kl_forward - kl_reverse) > 1e-3, (
        f"KL should not be symmetric: fwd={kl_forward}, rev={kl_reverse}"
    )


def test_pi_pref_lengthnorm():
    """Formula: σ(logp_a/len_a − logp_b/len_b)."""
    from verl.trainer.ppo.cpr_utils import compute_pi_pref_lengthnorm

    logp_a = np.array([-10.0], dtype=np.float32)
    logp_b = np.array([-20.0], dtype=np.float32)
    len_a = np.array([5], dtype=np.int64)
    len_b = np.array([10], dtype=np.int64)
    # normalized: -10/5 = -2.0 ; -20/10 = -2.0 → diff = 0 → σ(0) = 0.5
    pi_pref = compute_pi_pref_lengthnorm(logp_a, logp_b, len_a, len_b)
    assert abs(float(pi_pref[0]) - 0.5) < 1e-6, f"equal normalized logp → 0.5, got {pi_pref}"


def test_pi_pref_lengthnorm_equal_lengths():
    """When len_a == len_b, reduces to σ(logp_a − logp_b) (no normalization effect)."""
    from verl.trainer.ppo.cpr_utils import compute_pi_pref_lengthnorm

    logp_a = np.array([-2.0, -5.0], dtype=np.float32)
    logp_b = np.array([-3.0, -1.0], dtype=np.float32)
    L = np.array([7, 7], dtype=np.int64)
    pi_pref = compute_pi_pref_lengthnorm(logp_a, logp_b, L, L)
    # reduces to σ((logp_a - logp_b)/L), NOT σ(logp_a - logp_b)
    # Check the DEFINED formula: σ(logp_a/len_a − logp_b/len_b)
    diff = logp_a / L.astype(np.float32) - logp_b / L.astype(np.float32)
    expected = 1.0 / (1.0 + np.exp(-diff))
    assert np.allclose(pi_pref, expected, atol=1e-6), f"{pi_pref} vs {expected}"


def test_coupling_metrics_trapezoid_auc():
    """AUC threading: trapezoid contribution = 0.5·(y_prev + y_cur)·Δx.

    Two successive calls at step 0 and step 100 with kl_mean = 0.2 and 0.4
    respectively should give AUC = 0.5·(0.2 + 0.4)·100 = 30.0.
    """
    from verl.trainer.ppo.cpr_utils import compute_coupling_metrics

    # Stage 1: first val step. Identical p_self, pi_pref → kl_mean = 0.
    # We can't easily mint a specific kl_mean via the real inputs, so we pass
    # a small batch and just check the trapezoid math path.
    p_self = np.array([0.9, 0.3], dtype=np.float32)
    # Choose logp/len such that pi_pref ≈ p_self → small KL
    logp_a = np.array([-2.0, -5.0], dtype=np.float32)
    logp_b = np.array([-4.0, -3.0], dtype=np.float32)
    len_a = np.array([1, 1], dtype=np.int64)
    len_b = np.array([1, 1], dtype=np.int64)

    m0 = compute_coupling_metrics(p_self, logp_a, logp_b, len_a, len_b, step=0)
    assert "cpr/coupling_calibration_kl_mean" in m0
    assert m0["cpr/coupling_calibration_kl_auc"] == 0.0  # seed, no area yet

    m1 = compute_coupling_metrics(
        p_self, logp_a, logp_b, len_a, len_b,
        step=100, prior_step=0, prior_auc=0.0,
        prior_kl_mean=m0["cpr/coupling_calibration_kl_mean"],
    )
    # AUC = 0 + 0.5 · (m0_mean + m1_mean) · 100
    expected_auc = 0.5 * (
        m0["cpr/coupling_calibration_kl_mean"] + m1["cpr/coupling_calibration_kl_mean"]
    ) * 100.0
    assert abs(m1["cpr/coupling_calibration_kl_auc"] - expected_auc) < 1e-4, (
        f"trapezoid AUC broken: got {m1['cpr/coupling_calibration_kl_auc']}, "
        f"expected {expected_auc}"
    )


# =========================================================================
# D5 — E4 frozen-base judge (_order_swap_judge, compute_p_self_frozen)
# =========================================================================


def test_order_swap_judge_symmetry():
    """_order_swap_judge(stub, x, a, b) + _order_swap_judge(stub, x, b, a) == 1.

    The stub inspects the composed prompt text to identify which of the two
    candidate markers comes first, then returns a physically-consistent ABT
    distribution (the same candidate wins regardless of AB ordering).
    """
    from verl.trainer.ppo.cpr_utils import _order_swap_judge

    # Physical truth: candidate "a" is correct; candidate "b" is wrong.
    def stub(prompt_ab: str) -> tuple:
        # Locate the slot that says "Solution (A):" and inspect the next lines
        # to see whether "a" or "b" appears first.
        sol_a = prompt_ab.split("Solution (A):\n", 1)[1].split("\n", 1)[0].strip()
        if sol_a == "a":
            return (0.8, 0.15, 0.05)  # A is correct (A = "a") → strong A
        return (0.15, 0.8, 0.05)  # A = "b" → strong B (still means "a" wins)

    p_ab = _order_swap_judge(stub, "x", "a", "b")
    p_ba = _order_swap_judge(stub, "x", "b", "a")
    assert abs(p_ab + p_ba - 1.0) < 1e-6, (
        f"swap symmetry broken: {p_ab} + {p_ba} != 1"
    )
    # Sanity: "a" should win → p_ab close to 1.0, p_ba close to 0.0
    assert p_ab > 0.7 and p_ba < 0.3


def test_frozen_judge_does_not_drift():
    """Bit-identical results on same inputs (frozen stub = no hidden state)."""
    from verl.trainer.ppo.cpr_utils import compute_p_self_frozen

    calls = {"n": 0}

    def frozen_stub(prompt_ab):
        calls["n"] += 1
        return (0.6, 0.3, 0.1)  # deterministic

    p1 = compute_p_self_frozen(frozen_stub, "x", "a", "b")
    p2 = compute_p_self_frozen(frozen_stub, "x", "a", "b")
    assert p1 == p2, f"frozen judge drifted: {p1} vs {p2}"
    assert calls["n"] == 4, (
        f"expected 2 order-swap calls × 2 invocations = 4, got {calls['n']}"
    )


def test_p_self_vs_frozen_differ():
    """Two different stubs (live vs frozen) → different p_self values.

    Stubs must be order-consistent (swap-symmetric) to give non-0.5 output;
    the live judge "trusts a", the frozen judge "trusts b".
    """
    from verl.trainer.ppo.cpr_utils import compute_p_self, compute_p_self_frozen

    def live_stub(prompt_ab: str) -> tuple:
        sol_a = prompt_ab.split("Solution (A):\n", 1)[1].split("\n", 1)[0].strip()
        return (0.8, 0.15, 0.05) if sol_a == "a" else (0.15, 0.8, 0.05)

    def frozen_stub(prompt_ab: str) -> tuple:
        sol_a = prompt_ab.split("Solution (A):\n", 1)[1].split("\n", 1)[0].strip()
        return (0.15, 0.8, 0.05) if sol_a == "a" else (0.8, 0.15, 0.05)

    p_live = compute_p_self(live_stub, "x", "a", "b")
    p_frozen = compute_p_self_frozen(frozen_stub, "x", "a", "b")
    assert abs(p_live - p_frozen) > 0.3, (
        f"stubs should diverge: live={p_live}, frozen={p_frozen}"
    )


# =========================================================================
# BLOCKER-1 — prompt-level cluster bootstrap (cpr_alpha_fit.py)
# =========================================================================


def _simulate_clustered_probe_pairs(
    n_prompts: int,
    pairs_per_prompt: int,
    alpha_true: float,
    clustered: bool = True,
    seed: int = 0,
):
    """Simulate clustered pairs with strong per-prompt α-clustering.

    When ``clustered=True``, each prompt is assigned a per-prompt effective
    α_i ∈ {0, 1} via a Bernoulli(alpha_true) draw; ALL pairs from that prompt
    share that α_i. Aggregate α̂ across all pairs recovers ``alpha_true`` as
    the fraction of prompts assigned α_i=1, but the intra-prompt correlation
    is maximal: all pairs from a given prompt contribute information about
    the SAME α_i. This mirrors the real-world pattern where individual
    prompts systematically favor one labeler over the other (e.g., math vs
    prose prompts).

    When ``clustered=False``, each pair independently uses ``alpha_true`` —
    this is the IID case and cluster bootstrap should give CI widths
    comparable to the pair bootstrap.

    Returns ``(p_self, p_verif, p_obs, prompt_idx)``.
    """
    rng = np.random.default_rng(seed)
    p_selfs, p_verifs, p_obss, prompt_ids = [], [], [], []
    for p_idx in range(n_prompts):
        if clustered:
            alpha_i = 1.0 if rng.uniform() < alpha_true else 0.0
        else:
            alpha_i = alpha_true
        for _ in range(pairs_per_prompt):
            ps = float(rng.uniform(0.05, 0.95))
            pv = float(rng.uniform(0.05, 0.95))
            p_true = alpha_i * ps + (1.0 - alpha_i) * pv
            po = 1.0 if rng.uniform() < p_true else 0.0
            p_selfs.append(ps)
            p_verifs.append(pv)
            p_obss.append(po)
            prompt_ids.append(p_idx)
    return (
        np.asarray(p_selfs, dtype=np.float32),
        np.asarray(p_verifs, dtype=np.float32),
        np.asarray(p_obss, dtype=np.float32),
        np.asarray(prompt_ids, dtype=np.int64),
    )


def test_alpha_fit_cluster_bootstrap_wider_ci_than_pair_level():
    """Cluster bootstrap must widen CI vs IID pair-level on clustered data.

    With 50 prompts × 10 pairs and strong per-prompt α_i ∈ {0, 1} clustering
    (each prompt consistently favors self OR verif), the effective sample
    size is ~ n_prompts not n_pairs, so clustered CI should be ≥ 1.3× the
    pair-level CI width.
    """
    from cpr_alpha_fit import fit_alpha_hat

    p_self, p_verif, p_obs, prompt_idx = _simulate_clustered_probe_pairs(
        n_prompts=50, pairs_per_prompt=10, alpha_true=0.5, clustered=True, seed=0
    )

    # Pair-level (old / backwards-compat path).
    res_pair = fit_alpha_hat(
        p_self, p_verif, p_obs,
        bootstrap=300,
        restrict_disagreement=True,
        disagreement_threshold=0.2,
        rng_seed=0,
    )
    # Cluster bootstrap (pass prompt_idx).
    res_cluster = fit_alpha_hat(
        p_self, p_verif, p_obs,
        bootstrap=300,
        restrict_disagreement=True,
        disagreement_threshold=0.2,
        rng_seed=0,
        prompt_idx=prompt_idx,
    )
    width_pair = res_pair.ci_upper - res_pair.ci_lower
    width_cluster = res_cluster.ci_upper - res_cluster.ci_lower
    assert width_cluster >= 1.3 * width_pair, (
        f"cluster CI width ({width_cluster:.4f}) should be ≥ 1.3× pair CI width "
        f"({width_pair:.4f}) on clustered data"
    )


def test_alpha_fit_cluster_bootstrap_recovers_alpha():
    """α̂ recovery must still work with cluster bootstrap path.

    Uses the IID (clustered=False) variant so we can tighten the recovery
    tolerance to the same 0.1 used by test_alpha_fit_recovers_synthetic_alpha_0p5;
    cluster-bootstrap widening is tested separately in
    test_alpha_fit_cluster_bootstrap_wider_ci_than_pair_level.
    """
    from cpr_alpha_fit import fit_alpha_hat

    p_self, p_verif, p_obs, prompt_idx = _simulate_clustered_probe_pairs(
        n_prompts=200, pairs_per_prompt=6, alpha_true=0.5, clustered=False, seed=1
    )
    result = fit_alpha_hat(
        p_self, p_verif, p_obs,
        bootstrap=200,
        restrict_disagreement=True,
        disagreement_threshold=0.2,
        rng_seed=0,
        prompt_idx=prompt_idx,
    )
    assert abs(result.alpha_hat - 0.5) < 0.1, (
        f"α̂ = {result.alpha_hat}, expected close to 0.5 under cluster bootstrap"
    )
    # n_prompts_used should equal the number of distinct prompts that survived
    # filtering (must be > 0 and ≤ 200).
    assert 0 < result.n_prompts_used <= 200


def test_alpha_fit_prompt_idx_wrong_length():
    """Mismatched prompt_idx length must raise ValueError."""
    from cpr_alpha_fit import fit_alpha_hat

    p_self = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    p_verif = np.array([0.4, 0.5, 0.3], dtype=np.float32)
    p_obs = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    bad_prompt_idx = np.array([0, 1], dtype=np.int64)  # wrong length
    with pytest.raises(ValueError):
        fit_alpha_hat(p_self, p_verif, p_obs, bootstrap=10, prompt_idx=bad_prompt_idx)


def test_alpha_fit_prompt_idx_filtered_together():
    """When ties are dropped, prompt_idx must be filtered in lockstep."""
    from cpr_alpha_fit import fit_alpha_hat

    # Mix tie and non-tie labels; prompt_idx must survive the tie filter.
    rng = np.random.default_rng(0)
    n = 200
    p_self = rng.beta(2.0, 2.0, size=n).astype(np.float32)
    p_verif = rng.beta(2.0, 2.0, size=n).astype(np.float32)
    p_obs = np.empty(n, dtype=np.float32)
    tie_mask = rng.uniform(size=n) < 0.3
    p_obs[tie_mask] = 0.5  # ties → filtered out
    p_obs[~tie_mask] = (rng.uniform(size=(~tie_mask).sum()) < 0.5).astype(np.float32)
    prompt_idx = np.repeat(np.arange(n // 4, dtype=np.int64), 4)[:n]

    # Must not raise any shape mismatch after filtering.
    result = fit_alpha_hat(
        p_self, p_verif, p_obs,
        bootstrap=50,
        restrict_disagreement=False,
        rng_seed=0,
        prompt_idx=prompt_idx,
    )
    assert result.n_used > 0
    assert result.n_prompts_used > 0


# =========================================================================
# BLOCKER-2 — profile-likelihood width uses χ²₁(0.95)/2 ≈ 1.921
# =========================================================================


def test_pl_width_scales_with_sqrt_delta():
    """PL-width ~ √Δ near a quadratic MAP; width(Δ=4)/width(Δ=1) ≈ 2."""
    from cpr_alpha_fit import _profile_likelihood_width, _map_alpha

    p_self, p_verif, p_obs = _simulate_probe_pairs(n=2000, alpha_true=0.5, seed=0)
    # Drop ties is moot (sim uses binary draws) but fit the MAP.
    alpha_hat = _map_alpha(p_self.astype(np.float64), p_verif.astype(np.float64),
                           p_obs.astype(np.float64))
    w1 = _profile_likelihood_width(
        alpha_hat,
        p_self.astype(np.float64), p_verif.astype(np.float64), p_obs.astype(np.float64),
        delta=1.0,
    )
    w4 = _profile_likelihood_width(
        alpha_hat,
        p_self.astype(np.float64), p_verif.astype(np.float64), p_obs.astype(np.float64),
        delta=4.0,
    )
    ratio = w4 / max(w1, 1e-9)
    assert 1.7 <= ratio <= 2.3, (
        f"PL-width ratio (Δ=4)/(Δ=1) = {ratio:.3f}, expected ≈ 2 (range [1.7, 2.3])"
    )


def test_pl_width_95_default():
    """Default PL width field reflects Δ=1.921 ≥ Δ=1.0 computed explicitly."""
    from cpr_alpha_fit import fit_alpha_hat, _profile_likelihood_width, _map_alpha

    p_self, p_verif, p_obs = _simulate_probe_pairs(n=2000, alpha_true=0.5, seed=0)
    result = fit_alpha_hat(
        p_self, p_verif, p_obs,
        bootstrap=50,
        restrict_disagreement=True,
        disagreement_threshold=0.2,
    )
    assert hasattr(result, "profile_likelihood_width_95"), (
        "AlphaFitResult should have profile_likelihood_width_95 (renamed field)"
    )
    # Manually compute width at Δ=1.0 after filtering identical to fit_alpha_hat.
    from cpr_alpha_fit import _filter_pairs
    ps, pv, po, _, _ = _filter_pairs(
        np.asarray(p_self, dtype=np.float64),
        np.asarray(p_verif, dtype=np.float64),
        np.asarray(p_obs, dtype=np.float64),
        True, 0.2,
    )
    alpha_hat = _map_alpha(ps, pv, po)
    w1 = _profile_likelihood_width(alpha_hat, ps, pv, po, delta=1.0)
    assert result.profile_likelihood_width_95 >= w1 - 1e-9, (
        f"width_95 = {result.profile_likelihood_width_95} should be ≥ width(Δ=1.0) = {w1} "
        "(since 1.921 > 1.0)"
    )


# =========================================================================
# BLOCKER-3a — rename compute_coupling_kl → compute_coupling_calibration_kl
# =========================================================================
# (The D4 tests above have been renamed; confirm that the new names exist and
#  the old names remain absent.)


def test_coupling_calibration_kl_renamed():
    """New names exist; old names removed."""
    from verl.trainer.ppo import cpr_utils as m

    assert hasattr(m, "compute_coupling_calibration_kl"), (
        "compute_coupling_calibration_kl must exist (rename of compute_coupling_kl)"
    )
    assert hasattr(m, "compute_coupling_metrics"), (
        "compute_coupling_metrics must exist (rename of compute_mi_coupling_metrics)"
    )
    assert not hasattr(m, "compute_coupling_kl"), (
        "old name compute_coupling_kl must be removed"
    )
    assert not hasattr(m, "compute_mi_coupling_metrics"), (
        "old name compute_mi_coupling_metrics must be removed"
    )


# =========================================================================
# BLOCKER-3b — KSG-MI estimator compute_coupling_mi
# =========================================================================


def test_ksg_mi_zero_for_independent():
    """MI ≈ 0 for independent uniforms (modulo finite-sample bias)."""
    from verl.trainer.ppo.cpr_utils import compute_coupling_mi

    rng = np.random.default_rng(0)
    p_self = rng.uniform(size=500).astype(np.float32)
    pi_pref = rng.uniform(size=500).astype(np.float32)  # independent
    mi = compute_coupling_mi(p_self, pi_pref, k=3)
    assert mi < 0.1, f"MI for independent samples should be < 0.1, got {mi}"
    # Also clipped ≥ 0.
    assert mi >= 0.0, f"KSG MI must be clipped to ≥ 0, got {mi}"


def test_ksg_mi_positive_for_dependent():
    """Tight coupling pi_pref = p_self + N(0, 0.05) → MI > 0.5 nats."""
    from verl.trainer.ppo.cpr_utils import compute_coupling_mi

    rng = np.random.default_rng(1)
    p_self = rng.uniform(size=500).astype(np.float32)
    noise = rng.normal(0.0, 0.05, size=500).astype(np.float32)
    pi_pref = np.clip(p_self + noise, 0.0, 1.0)
    mi = compute_coupling_mi(p_self, pi_pref, k=3)
    assert mi > 0.5, f"MI for tightly coupled samples should be > 0.5, got {mi}"


def test_ksg_mi_monotonic_with_dependence():
    """As noise σ decreases, MI should monotonically increase."""
    from verl.trainer.ppo.cpr_utils import compute_coupling_mi

    rng = np.random.default_rng(2)
    p_self = rng.uniform(size=500).astype(np.float32)
    mis = []
    for sigma in (0.3, 0.1, 0.01):
        noise = rng.normal(0.0, sigma, size=500).astype(np.float32)
        pi_pref = np.clip(p_self + noise, 0.0, 1.0)
        mis.append(compute_coupling_mi(p_self, pi_pref, k=3))
    assert mis[0] < mis[1] < mis[2], (
        f"MI should monotonically increase as σ decreases; got {mis}"
    )


def test_coupling_metrics_emits_both():
    """compute_coupling_metrics emits both calibration KL and KSG MI keys."""
    from verl.trainer.ppo.cpr_utils import compute_coupling_metrics

    rng = np.random.default_rng(0)
    # Need N ≥ 2k+1 = 7 for MI emission (k=3 default).
    p_self = rng.uniform(size=50).astype(np.float32)
    logp_a = rng.normal(-3.0, 1.0, size=50).astype(np.float32)
    logp_b = rng.normal(-3.0, 1.0, size=50).astype(np.float32)
    len_a = np.full(50, 10, dtype=np.int64)
    len_b = np.full(50, 10, dtype=np.int64)

    m = compute_coupling_metrics(p_self, logp_a, logp_b, len_a, len_b, step=0)
    assert "cpr/coupling_calibration_kl_mean" in m
    assert "cpr/coupling_mi" in m
    assert "cpr/pi_pref_mean" in m
    assert m["cpr/coupling_mi"] >= 0.0


# --------------------------------------------------------------------------
# BSV (Budgeted Selective Verification) — DEFINE_SPEC_v3 §1-3
# --------------------------------------------------------------------------
# D1 — P² streaming quantile tracker (Jain & Chlamtac, CACM 28(10), 1985)


def test_p2_converges_uniform_p30():
    """P²(p=0.3) on 10000 Uniform(0,1) samples should converge to 0.3 ± 0.05."""
    from verl.trainer.ppo.bsv_p2_tracker import P2QuantileTracker

    rng = np.random.default_rng(0)
    tracker = P2QuantileTracker(p=0.3)
    for x in rng.uniform(size=10000):
        tracker.update(float(x))
    assert tracker.n_observations() == 10000
    assert abs(tracker.quantile() - 0.3) < 0.05, (
        f"P²(0.3) = {tracker.quantile():.4f}, expected 0.3 ± 0.05"
    )


def test_p2_converges_integer_median():
    """P²(p=0.5) on integers -500..500 shuffled should converge to 0.0 ± 10."""
    from verl.trainer.ppo.bsv_p2_tracker import P2QuantileTracker

    rng = np.random.default_rng(1)
    ints = np.arange(-500, 501, dtype=np.float64)
    rng.shuffle(ints)
    tracker = P2QuantileTracker(p=0.5)
    for x in ints:
        tracker.update(float(x))
    assert abs(tracker.quantile() - 0.0) < 10.0, (
        f"P²(0.5) median = {tracker.quantile():.4f}, expected 0 ± 10"
    )


def test_p2_constant_memory():
    """Internal state size must be bounded (O(1) memory)."""
    from verl.trainer.ppo.bsv_p2_tracker import P2QuantileTracker

    tracker = P2QuantileTracker(p=0.3)
    rng = np.random.default_rng(2)
    for x in rng.uniform(size=100):
        tracker.update(float(x))
    small_state = len(tracker.__dict__)
    for x in rng.uniform(size=10000):
        tracker.update(float(x))
    big_state = len(tracker.__dict__)
    # Dict keys should not grow with n observations.
    assert small_state == big_state
    # The P² marker array is fixed at 5 positions.
    for attr in ("q", "n", "np_", "dn"):
        if hasattr(tracker, attr):
            arr = getattr(tracker, attr)
            assert len(arr) == 5, f"{attr} length {len(arr)} != 5"


def test_p2_rejects_invalid_p():
    """p outside (0, 1) must raise ValueError."""
    from verl.trainer.ppo.bsv_p2_tracker import P2QuantileTracker

    with pytest.raises(ValueError):
        P2QuantileTracker(p=-0.1)
    with pytest.raises(ValueError):
        P2QuantileTracker(p=1.5)
    with pytest.raises(ValueError):
        P2QuantileTracker(p=0.0)
    with pytest.raises(ValueError):
        P2QuantileTracker(p=1.0)


def test_p2_fewer_than_5_observations():
    """With <5 samples the quantile() should still return a sensible (finite) value."""
    from verl.trainer.ppo.bsv_p2_tracker import P2QuantileTracker

    tracker = P2QuantileTracker(p=0.5)
    # Zero observations
    q0 = tracker.quantile()
    assert math.isfinite(q0), f"expected finite value at n=0, got {q0}"
    # Three observations
    for x in [1.0, 2.0, 3.0]:
        tracker.update(x)
    q3 = tracker.quantile()
    assert math.isfinite(q3) and not math.isnan(q3)
    assert 1.0 <= q3 <= 3.0, f"quantile {q3} outside observed range [1, 3]"


# --------------------------------------------------------------------------
# D2 — BSV gate + propensity logger (DEFINE_SPEC_v3 §1.2, §3.1)


def test_bsv_gate_endpoint_alpha_zero():
    """α=0, ε=0 → I_v = 1 always (all pairs call the verifier)."""
    from verl.trainer.ppo.bsv_gate import BSVGate

    gate = BSVGate(alpha=0.0, epsilon=0.0, rng_seed=0)
    rng = np.random.default_rng(0)
    calls = 0
    total = 2000
    for _ in range(total):
        p_self = float(rng.uniform())
        I_v, _ = gate.decide(p_self)
        calls += I_v
    # α=0 → expected call rate = 1.0 exactly (modulo P² warmup leniency)
    assert calls >= int(0.95 * total), f"calls={calls}/{total} — expected ≈ all"


def test_bsv_gate_endpoint_alpha_one():
    """α=1, ε=0 → I_v = 0 always after warmup."""
    from verl.trainer.ppo.bsv_gate import BSVGate

    gate = BSVGate(alpha=1.0, epsilon=0.0, rng_seed=0)
    rng = np.random.default_rng(1)
    # Warmup
    for _ in range(50):
        gate.decide(float(rng.uniform()))
    # Post-warmup: should never call
    calls = 0
    total = 2000
    for _ in range(total):
        I_v, _ = gate.decide(float(rng.uniform()))
        calls += I_v
    assert calls <= int(0.05 * total), f"calls={calls}/{total} — expected ≈ 0"


def test_bsv_gate_epsilon_floor():
    """α=1, ε=0.10 → realized_call_rate() in [0.05, 0.15] over 2000 draws."""
    from verl.trainer.ppo.bsv_gate import BSVGate

    gate = BSVGate(alpha=1.0, epsilon=0.10, rng_seed=7)
    rng = np.random.default_rng(2)
    for _ in range(2000):
        gate.decide(float(rng.uniform()))
    rate = gate.realized_call_rate()
    assert 0.05 <= rate <= 0.15, f"rate={rate:.3f} outside [0.05, 0.15]"


def test_bsv_gate_target_rate():
    """α=0.5, ε=0 on Uniform p_self → realized call rate ≈ 0.5 ± 0.05 after 5000 obs."""
    from verl.trainer.ppo.bsv_gate import BSVGate

    gate = BSVGate(alpha=0.5, epsilon=0.0, rng_seed=13)
    rng = np.random.default_rng(3)
    for _ in range(5000):
        gate.decide(float(rng.uniform()))
    rate = gate.realized_call_rate()
    assert abs(rate - 0.5) < 0.05, f"rate={rate:.3f} expected 0.5 ± 0.05"


def test_bsv_gate_propensity_matches_formula():
    """p_invoke must exactly equal ε + (1-ε)·1{c < t(α)}."""
    from verl.trainer.ppo.bsv_gate import BSVGate

    gate = BSVGate(alpha=0.5, epsilon=0.05, rng_seed=0)
    # Warm up the tracker so t(α) has a meaningful value
    rng = np.random.default_rng(5)
    for _ in range(200):
        gate.decide(float(rng.uniform()))
    t = gate.current_threshold()
    # Pick a p_self that gives a specific confidence
    # c = 2|p_self - 0.5|
    # Test two confidences — one definitely below t, one definitely above.
    # confidence ~ 0 → p_self near 0.5 → below threshold → p_invoke = 1
    I_v, p_inv_low = gate.decide(0.5)  # c = 0
    assert abs(p_inv_low - 1.0) < 1e-9, f"p_invoke(c=0) = {p_inv_low}, expected 1"
    # confidence ~ 1 → p_self = 1.0 → above threshold → p_invoke = ε
    I_v, p_inv_high = gate.decide(1.0)  # c = 1
    assert abs(p_inv_high - 0.05) < 1e-9, f"p_invoke(c=1) = {p_inv_high}, expected ε=0.05"


def test_bsv_gate_deterministic_with_seed():
    """Same inputs + same rng_seed → identical (I_v, p_invoke) sequence."""
    from verl.trainer.ppo.bsv_gate import BSVGate

    rng = np.random.default_rng(0)
    inputs = rng.uniform(size=1000).tolist()
    gate1 = BSVGate(alpha=0.3, epsilon=0.05, rng_seed=42)
    gate2 = BSVGate(alpha=0.3, epsilon=0.05, rng_seed=42)
    seq1, seq2 = [], []
    for x in inputs:
        seq1.append(gate1.decide(x))
        seq2.append(gate2.decide(x))
    for (iv1, p1), (iv2, p2) in zip(seq1, seq2):
        assert iv1 == iv2 and abs(p1 - p2) < 1e-12


def test_bsv_gate_confidence_symmetry():
    """p_self=0.3 and p_self=0.7 give same c=0.4, hence same propensity."""
    from verl.trainer.ppo.bsv_gate import BSVGate

    gate = BSVGate(alpha=0.5, epsilon=0.05, rng_seed=0)
    # Warm up
    rng = np.random.default_rng(4)
    for _ in range(200):
        gate.decide(float(rng.uniform()))
    _, p_inv_03 = gate.decide(0.3)
    _, p_inv_07 = gate.decide(0.7)
    assert abs(p_inv_03 - p_inv_07) < 1e-9, (
        f"p_invoke(0.3)={p_inv_03} vs p_invoke(0.7)={p_inv_07}"
    )


# --------------------------------------------------------------------------
# D3 — compute_bsv_loss: per-pair I_v dispatch (DEFINE_SPEC_v3 §1.3)


def test_bsv_loss_iv_one_uses_verifier():
    """I_v=1 everywhere → loss should match compute_cpr_loss with α=0 (verifier-only)."""
    from verl.trainer.ppo.core_algos import compute_bsv_loss, compute_cpr_loss
    from verl.trainer.ppo.cpr_utils import compute_p_pref

    torch.manual_seed(0)
    logp_a, logp_b, ref_logp_a, ref_logp_b, mask_a, mask_b = _make_logp_tensors(B=4, T=5, seed=0)
    np.random.seed(0)
    p_self = np.random.rand(4).astype(np.float32)
    r_a = np.array([1, 0, 1, 0], dtype=np.float32)
    r_b = np.array([0, 1, 1, 0], dtype=np.float32)
    p_verif = _sigmoid(r_a - r_b).astype(np.float32)
    I_v = np.ones(4, dtype=np.float32)

    # BSV with I_v=1 everywhere
    loss_bsv, _ = compute_bsv_loss(
        logp_a=logp_a, logp_b=logp_b,
        ref_logp_a=ref_logp_a, ref_logp_b=ref_logp_b,
        mask_a=mask_a, mask_b=mask_b,
        p_self=torch.from_numpy(p_self),
        p_verif=torch.from_numpy(p_verif),
        I_v=torch.from_numpy(I_v),
        beta=0.15, eta=0.0,
    )
    # CPR with α=0 → p_pref = p_verif
    p_pref_alpha0 = compute_p_pref(
        p_self=p_self, r_ext_a=r_a, r_ext_b=r_b, alpha=0.0, tau_ext=1.0
    )
    loss_cpr, _ = compute_cpr_loss(
        logp_a=logp_a, logp_b=logp_b,
        ref_logp_a=ref_logp_a, ref_logp_b=ref_logp_b,
        mask_a=mask_a, mask_b=mask_b,
        p_pref=torch.from_numpy(p_pref_alpha0),
        beta=0.15, eta=0.0,
    )
    assert torch.allclose(loss_bsv, loss_cpr, atol=1e-6), (
        f"BSV(I_v=1) = {loss_bsv.item()}, CPR(α=0) = {loss_cpr.item()}"
    )


def test_bsv_loss_iv_zero_uses_self():
    """I_v=0 everywhere → loss should match compute_cpr_loss with α=1 (self-only)."""
    from verl.trainer.ppo.core_algos import compute_bsv_loss, compute_cpr_loss
    from verl.trainer.ppo.cpr_utils import compute_p_pref

    torch.manual_seed(0)
    logp_a, logp_b, ref_logp_a, ref_logp_b, mask_a, mask_b = _make_logp_tensors(B=4, T=5, seed=0)
    np.random.seed(0)
    p_self = np.random.rand(4).astype(np.float32)
    r_a = np.array([1, 0, 1, 0], dtype=np.float32)
    r_b = np.array([0, 1, 1, 0], dtype=np.float32)
    p_verif = _sigmoid(r_a - r_b).astype(np.float32)
    I_v = np.zeros(4, dtype=np.float32)

    loss_bsv, _ = compute_bsv_loss(
        logp_a=logp_a, logp_b=logp_b,
        ref_logp_a=ref_logp_a, ref_logp_b=ref_logp_b,
        mask_a=mask_a, mask_b=mask_b,
        p_self=torch.from_numpy(p_self),
        p_verif=torch.from_numpy(p_verif),
        I_v=torch.from_numpy(I_v),
        beta=0.15, eta=0.0,
    )
    p_pref_alpha1 = compute_p_pref(
        p_self=p_self, r_ext_a=r_a, r_ext_b=r_b, alpha=1.0, tau_ext=1.0
    )
    loss_cpr, _ = compute_cpr_loss(
        logp_a=logp_a, logp_b=logp_b,
        ref_logp_a=ref_logp_a, ref_logp_b=ref_logp_b,
        mask_a=mask_a, mask_b=mask_b,
        p_pref=torch.from_numpy(p_pref_alpha1),
        beta=0.15, eta=0.0,
    )
    assert torch.allclose(loss_bsv, loss_cpr, atol=1e-6), (
        f"BSV(I_v=0) = {loss_bsv.item()}, CPR(α=1) = {loss_cpr.item()}"
    )


def test_bsv_loss_mixed_dispatch():
    """Mixed I_v=[1,0,1,0]: per-pair p_pref resolves to the right source per index."""
    from verl.trainer.ppo.core_algos import compute_bsv_loss, compute_cpr_loss

    torch.manual_seed(0)
    logp_a, logp_b, ref_logp_a, ref_logp_b, mask_a, mask_b = _make_logp_tensors(B=4, T=5, seed=0)
    p_self = np.array([0.8, 0.2, 0.6, 0.4], dtype=np.float32)
    p_verif = np.array([0.1, 0.9, 0.3, 0.7], dtype=np.float32)
    I_v = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    loss_bsv, _ = compute_bsv_loss(
        logp_a=logp_a, logp_b=logp_b,
        ref_logp_a=ref_logp_a, ref_logp_b=ref_logp_b,
        mask_a=mask_a, mask_b=mask_b,
        p_self=torch.from_numpy(p_self),
        p_verif=torch.from_numpy(p_verif),
        I_v=torch.from_numpy(I_v),
        beta=0.15, eta=0.0,
    )
    # Per-pair dispatch: I_v=1 → verif, I_v=0 → self
    expected_ppref = np.where(I_v > 0.5, p_verif, p_self).astype(np.float32)
    loss_ref, _ = compute_cpr_loss(
        logp_a=logp_a, logp_b=logp_b,
        ref_logp_a=ref_logp_a, ref_logp_b=ref_logp_b,
        mask_a=mask_a, mask_b=mask_b,
        p_pref=torch.from_numpy(expected_ppref),
        beta=0.15, eta=0.0,
    )
    assert torch.allclose(loss_bsv, loss_ref, atol=1e-6)


def test_bsv_loss_anchor_optional():
    """eta=0 vs eta=0.05: diff should equal eta·L_anchor."""
    from verl.trainer.ppo.core_algos import compute_bsv_loss

    torch.manual_seed(0)
    logp_a, logp_b, ref_logp_a, ref_logp_b, mask_a, mask_b = _make_logp_tensors(B=4, T=5, seed=0)
    p_self = np.array([0.7, 0.3, 0.5, 0.6], dtype=np.float32)
    p_verif = np.array([0.9, 0.1, 0.5, 0.4], dtype=np.float32)
    I_v = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    kw = dict(
        logp_a=logp_a, logp_b=logp_b,
        ref_logp_a=ref_logp_a, ref_logp_b=ref_logp_b,
        mask_a=mask_a, mask_b=mask_b,
        p_self=torch.from_numpy(p_self),
        p_verif=torch.from_numpy(p_verif),
        I_v=torch.from_numpy(I_v),
        beta=0.15,
    )
    L0, m0 = compute_bsv_loss(eta=0.0, **kw)
    L1, m1 = compute_bsv_loss(eta=0.05, **kw)
    # L1 - L0 should equal 0.05 * anchor_loss
    anchor = m1["cpr/anchor_loss"]
    assert torch.allclose(L1 - L0, 0.05 * anchor, atol=1e-6)


def test_bsv_loss_matches_cpr_loss_at_iv_extremes():
    """compute_bsv_loss with I_v∈{0,1} exactly reproduces compute_cpr_loss at α∈{0,1}."""
    from verl.trainer.ppo.core_algos import compute_bsv_loss, compute_cpr_loss

    torch.manual_seed(1)
    logp_a, logp_b, ref_logp_a, ref_logp_b, mask_a, mask_b = _make_logp_tensors(B=8, T=6, seed=1)
    rng = np.random.default_rng(1)
    p_self = rng.uniform(size=8).astype(np.float32)
    p_verif = rng.uniform(size=8).astype(np.float32)
    for iv_val in (0.0, 1.0):
        I_v = np.full(8, iv_val, dtype=np.float32)
        loss_bsv, _ = compute_bsv_loss(
            logp_a=logp_a, logp_b=logp_b,
            ref_logp_a=ref_logp_a, ref_logp_b=ref_logp_b,
            mask_a=mask_a, mask_b=mask_b,
            p_self=torch.from_numpy(p_self),
            p_verif=torch.from_numpy(p_verif),
            I_v=torch.from_numpy(I_v),
            beta=0.15, eta=0.0,
        )
        p_pref = p_verif if iv_val == 1.0 else p_self
        loss_cpr, _ = compute_cpr_loss(
            logp_a=logp_a, logp_b=logp_b,
            ref_logp_a=ref_logp_a, ref_logp_b=ref_logp_b,
            mask_a=mask_a, mask_b=mask_b,
            p_pref=torch.from_numpy(p_pref),
            beta=0.15, eta=0.0,
        )
        assert torch.allclose(loss_bsv, loss_cpr, atol=1e-6)


# --------------------------------------------------------------------------
# D4 — IPS / DR evaluators + audit regret (DEFINE_SPEC_v3 §3.3)


def test_ips_unbiased_when_propensity_correct():
    """With correct propensity, IPS estimator converges to E[R] within ±0.03 at n=10000."""
    from verl.trainer.ppo.bsv_evaluators import ips_reward

    rng = np.random.default_rng(0)
    n = 10000
    # Underlying R distribution: Bernoulli(0.6). True E[R] = 0.6.
    R_full = rng.binomial(1, 0.6, size=n).astype(np.float64)
    # MNAR selection: p_invoke depends on R.
    # P(I_v=1|R=1)=0.3, P(I_v=1|R=0)=0.8. Both > 0.
    p_invoke = np.where(R_full == 1, 0.3, 0.8)
    I_v = (rng.uniform(size=n) < p_invoke).astype(np.float64)
    R_obs = np.where(I_v > 0.5, R_full, np.nan)

    est = ips_reward(R=R_obs, p_invoke=p_invoke, I_v=I_v)
    assert abs(est - 0.6) < 0.03, f"IPS estimate {est}, expected 0.6 ± 0.03"


def test_ips_biased_when_propensity_wrong():
    """With wrong propensity, IPS is demonstrably biased (>0.1 off)."""
    from verl.trainer.ppo.bsv_evaluators import ips_reward

    rng = np.random.default_rng(1)
    n = 10000
    R_full = rng.binomial(1, 0.6, size=n).astype(np.float64)
    # True MNAR propensity
    p_true = np.where(R_full == 1, 0.3, 0.8)
    I_v = (rng.uniform(size=n) < p_true).astype(np.float64)
    R_obs = np.where(I_v > 0.5, R_full, np.nan)
    # Wrong propensity — uniform 0.5 for everyone
    p_wrong = np.full(n, 0.5)
    est = ips_reward(R=R_obs, p_invoke=p_wrong, I_v=I_v)
    # Under wrong uniform p, the IPS ratio collapses to the selected-sample mean,
    # which (under this MNAR) should be biased down from 0.6.
    assert abs(est - 0.6) > 0.1, (
        f"Wrong propensity should yield bias >0.1, got est={est} (diff {abs(est - 0.6):.3f})"
    )


def test_dr_matches_ips_when_direct_zero():
    """DR with direct_estimate=0 everywhere should equal IPS (Horvitz-Thompson)."""
    from verl.trainer.ppo.bsv_evaluators import dr_reward, ips_reward

    rng = np.random.default_rng(2)
    n = 500
    R_full = rng.uniform(size=n).astype(np.float64)
    p_invoke = rng.uniform(0.1, 0.9, size=n)
    I_v = (rng.uniform(size=n) < p_invoke).astype(np.float64)
    R_obs = np.where(I_v > 0.5, R_full, np.nan)

    direct = np.zeros(n)
    dr_est = dr_reward(R=R_obs, p_invoke=p_invoke, I_v=I_v, direct_estimate=direct)
    ips_est_mean = float(np.mean(np.where(I_v > 0.5, R_full / p_invoke, 0.0)))
    assert abs(dr_est - ips_est_mean) < 1e-9, f"DR(direct=0)={dr_est} vs IPS(unnorm)={ips_est_mean}"


def test_dr_still_unbiased_under_mnar_correct_direct_only():
    """Wrong propensity + correct per-context direct_estimate → DR still unbiased.

    "Doubly robust" = consistent if EITHER the propensity or the direct
    model is correct as a function of CONTEXT. A scalar E[R] is only the
    correct direct model when R has no context-dependence; to really
    exercise the DR guarantee we give the estimator the oracle
    context-conditional E[R | x] (equal to R itself in the degenerate
    deterministic-R regime).
    """
    from verl.trainer.ppo.bsv_evaluators import dr_reward

    rng = np.random.default_rng(3)
    n = 10000
    R_full = rng.binomial(1, 0.6, size=n).astype(np.float64)
    p_true = np.where(R_full == 1, 0.3, 0.8)
    I_v = (rng.uniform(size=n) < p_true).astype(np.float64)
    R_obs = np.where(I_v > 0.5, R_full, np.nan)
    # Oracle direct model: E[R | context] = R itself (deterministic per context).
    direct = R_full.copy()
    # Wrong propensity: uniform 0.5 everywhere (ignores R dependence).
    p_wrong = np.full(n, 0.5)
    est = dr_reward(R=R_obs, p_invoke=p_wrong, I_v=I_v, direct_estimate=direct)
    # With correct per-context direct model, the correction term is
    # (R - R) = 0, so DR = mean(direct) = E[R] = 0.6 exactly.
    assert abs(est - 0.6) < 0.03, f"DR with correct direct: {est}, expected 0.6 ± 0.03"


def test_audit_regret_zero_when_self_matches_R():
    """p_self exactly matches R on the audit slice → audit_regret = 0."""
    from verl.trainer.ppo.bsv_evaluators import audit_regret

    rng = np.random.default_rng(4)
    R_audit = rng.uniform(size=200).astype(np.float64)
    p_self_audit = R_audit.copy()
    reg = audit_regret(R_audit=R_audit, p_self_audit=p_self_audit)
    assert abs(reg) < 1e-9, f"expected 0, got {reg}"


def test_audit_regret_positive_when_self_wrong():
    """Systematically wrong p_self → audit_regret > 0.3."""
    from verl.trainer.ppo.bsv_evaluators import audit_regret

    rng = np.random.default_rng(5)
    n = 200
    R_audit = rng.binomial(1, 0.5, size=n).astype(np.float64)
    # p_self_audit always says "a wins" with prob 0 (so loss whenever R=1)
    p_self_audit = np.zeros(n)
    reg = audit_regret(R_audit=R_audit, p_self_audit=p_self_audit)
    # Mean of |R - 0| = mean(R) ≈ 0.5 > 0.3
    assert reg > 0.3, f"audit_regret={reg}, expected > 0.3"


def test_ips_ht_unbiased_on_synthetic():
    """HT IPS is unbiased: E[V̂_HT] = E[R] over the logging distribution.

    Regression test for the SNIPS-vs-HT distinction introduced in the
    H1 fix (bsv_methodology.md §6.2). With a known Bernoulli gate and
    a scalar Bernoulli reward, the HT average should hug the true mean
    across ~500 replicates.
    """
    from verl.trainer.ppo.bsv_evaluators import ips_reward_ht

    rng = np.random.default_rng(7)
    n = 400
    true_mean = 0.65
    R_true = rng.binomial(1, true_mean, size=n).astype(np.float64)
    p_invoke = np.full(n, 0.3)  # constant logging propensity

    estimates = []
    for _ in range(500):
        I_v = rng.binomial(1, p_invoke).astype(np.int64)
        R_obs = np.where(I_v == 1, R_true, np.nan)
        estimates.append(ips_reward_ht(R=R_obs, p_invoke=p_invoke, I_v=I_v))
    mean_est = float(np.mean(estimates))
    # HT is unbiased → |mean_est - true_mean| ≤ 2σ/√500. σ roughly
    # bounded by max(R)/min(p) = 1/0.3 ≈ 3.3. 2·3.3/sqrt(500) ≈ 0.3.
    assert abs(mean_est - true_mean) < 0.05, f"HT bias={mean_est - true_mean:.4f}"


def test_propensity_csv_round_trip_ips_dr_audit():
    """Full writer→DictReader→evaluator round trip (H4 integration test).

    Builds a synthetic propensity CSV using the exact fieldnames the
    trainer writes (`step, pair_idx, p_invoke, I_v, p_self, p_verif,
    R_a, R_b`), reads it back via csv.DictReader, feeds columns to
    ips_reward / dr_reward / audit_regret, and asserts each evaluator
    produces finite, in-range output consistent with a known construction.

    Protects against silent writer/reader drift — the paper's MNAR
    correction step depends on this round-trip working.
    """
    import csv
    import io
    from verl.trainer.ppo.bsv_evaluators import (
        audit_regret,
        dr_reward,
        ips_reward,
        ips_reward_ht,
    )

    # Writer-side fieldnames must match ray_trainer.py::_flush_bsv_propensity_rows.
    writer_fieldnames = ["step", "pair_idx", "p_invoke", "I_v", "p_self", "p_verif", "R_a", "R_b"]

    rng = np.random.default_rng(11)
    n_rows = 40
    # Synthetic pairs: half "below-threshold" (p_invoke=1.0 → I_v=1), half "above"
    # (p_invoke=0.05 → I_v=Bernoulli(0.05)). Matches how BSVGate emits.
    below = rng.integers(0, 2, size=n_rows).astype(bool)
    p_invoke = np.where(below, 1.0, 0.05)
    I_v = (rng.uniform(size=n_rows) < p_invoke).astype(np.int64)
    R_a_true = rng.binomial(1, 0.6, size=n_rows).astype(np.float64)

    rows = []
    for i in range(n_rows):
        rows.append({
            "step": 3,
            "pair_idx": int(i),
            "p_invoke": float(p_invoke[i]),
            "I_v": int(I_v[i]),
            "p_self": float(rng.uniform(0.3, 0.9)),
            "p_verif": float(rng.uniform(0.3, 0.9)) if I_v[i] else float("nan"),
            "R_a": float(R_a_true[i]) if I_v[i] else float("nan"),
            "R_b": float(rng.uniform()) if I_v[i] else float("nan"),
        })

    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=writer_fieldnames).writeheader()
    csv.DictWriter(buf, fieldnames=writer_fieldnames).writerows(rows)

    buf.seek(0)
    reader = csv.DictReader(buf)
    got_fieldnames = reader.fieldnames
    assert got_fieldnames == writer_fieldnames, (
        f"round-trip fieldname drift: got {got_fieldnames} expected {writer_fieldnames}"
    )

    parsed = list(reader)
    assert len(parsed) == n_rows, f"lost rows on round trip: {len(parsed)}/{n_rows}"

    # Extract columns — simulates what an offline IPS script would do.
    def _col(name, dtype):
        vals = []
        for r in parsed:
            raw = r[name]
            if raw == "" or raw == "nan" or raw == "NaN":
                vals.append(float("nan") if dtype is float else 0)
            else:
                vals.append(dtype(raw))
        return np.asarray(vals, dtype=np.float64 if dtype is float else np.int64)

    p_inv_rt = _col("p_invoke", float)
    I_v_rt = _col("I_v", int)
    R_a_rt = _col("R_a", float)
    p_self_rt = _col("p_self", float)

    np.testing.assert_allclose(p_inv_rt, p_invoke, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(I_v_rt, I_v)

    # All three evaluators must produce finite scalars.
    snips = ips_reward(R=R_a_rt, p_invoke=p_inv_rt, I_v=I_v_rt, normalize=True)
    ht = ips_reward_ht(R=R_a_rt, p_invoke=p_inv_rt, I_v=I_v_rt)
    dr = dr_reward(
        R=R_a_rt, p_invoke=p_inv_rt, I_v=I_v_rt,
        direct_estimate=p_self_rt,
    )
    audit_mask = (p_inv_rt < 0.5) & (I_v_rt == 1)
    if audit_mask.any():
        aud = audit_regret(R_audit=R_a_rt[audit_mask], p_self_audit=p_self_rt[audit_mask])
        assert np.isfinite(aud), f"audit_regret not finite: {aud}"

    for name, val in [("snips", snips), ("ht", ht), ("dr", dr)]:
        assert np.isfinite(val), f"{name} not finite: {val}"
        assert -0.01 <= val <= 1.01, f"{name}={val} out of [0,1] for a binary reward"
