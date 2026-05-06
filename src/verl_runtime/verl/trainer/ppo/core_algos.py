# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

POLICY_LOSS_REGISTRY = {}


def register_policy_loss(name):
    def decorator(func):
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


ADV_ESTIMATOR_REGISTRY = {}


def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    SIRL = "sirl"
    SIRL_RAW = "sirl_raw"
    SIRL_PREF = "sirl_pref"
    CPR = "cpr"


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_bsv_grpo_masked_reward(
    r_ext: np.ndarray,
    I_v: np.ndarray,
    group_ids,
    *,
    verifier_absent: bool = False,
) -> "tuple[np.ndarray, np.ndarray]":
    """GRPO-BSV per-rollout Frame A (mask-out) reward shaper (DEFINE_SPEC_v3 §1.3b, rev 2026-04-21).

    Under Frame A the reward uses only the external verifier signal on the
    labeled slice; rollouts without a verifier call are MASKED from the
    gradient via a fill-with-group-mean trick that yields zero advantage
    under downstream GRPO group-normalization.

    Specifically, for each GRPO group identified by ``group_ids``:
      - Included rollouts  (I_v == 1, verifier present): s_i = R_ext_i
      - Masked rollouts    (I_v == 0 or verifier absent): s_i = μ_incl
          where μ_incl is the mean R_ext over the group's included rollouts.
      - Degenerate groups  (|included| == 0): s_i = 0 for all members;
          caller's metric layer should log and count these.

    When downstream GRPO computes A_i = (s_i − μ_group) / (σ_group + ε):
      - Included i  : A_i = (R_ext_i − μ_incl) / (σ_incl · √(|incl|/K) + ε)
                      = true Frame-A advantage × √(K/|incl|).
                      Direction preserved; magnitude damped for masked-heavy groups
                      such that total gradient ∝ √(|incl|·K), which is the desired
                      down-weighting of groups with few labeled rollouts.
      - Masked i    : A_i = 0. Zero gradient contribution.
      - Degenerate  : all A_i = 0. Zero gradient contribution (skip-and-log).

    p_self does NOT appear anywhere in this function — it is gate-only under
    Frame A (§1.2b). See doc/bsv_frame_c_parking_lot.md for the explored-but-parked
    mixture variant that re-introduces p_self into reward (DO NOT import).

    Shape contract
    --------------
    r_ext : np.ndarray, shape (N,); dtype float
        External verifier outcome per rollout; value ignored on masked indices.
    I_v : np.ndarray, shape (N,); dtype int or float, values in {0, 1}
        Gate decisions from ``BSVGate.decide_batch``.
    group_ids : sequence of length N
        GRPO group identifier per rollout (e.g. prompt uid). Rollouts sharing
        a group_id are siblings.
    verifier_absent : bool, default False
        When True, the entire batch is treated as unobserved — include_mask is
        all zeros and s is all zeros regardless of I_v / R_ext. Used by the
        ray_trainer H3-style side-channel when the verifier HTTP path fails.

    Returns
    -------
    s : np.ndarray, shape (N,), dtype float32
        Scalar per-rollout reward to write to token_level_scores.
    include_mask : np.ndarray, shape (N,), dtype uint8
        1 iff (not verifier_absent) and I_v[i] == 1. Rollouts with include_mask=0
        contribute zero gradient under downstream GRPO group-normalization.

    Endpoint semantics (§1.4, rev 2026-04-21)
    -----------------------------------------
    - α=0 : I_v ≡ 1 → include_mask all 1 → s = r_ext → standard RLVR
    - α=1 : I_v ≡ 0 (modulo ε-audit) → mostly masked → DEGENERATE training
            (only ε-forced rollouts contribute). NOT URLVR.
    - α∈(0,1) : interior selective regime; gate decides per-rollout.
    """
    r_ext_arr = np.asarray(r_ext, dtype=np.float32).ravel()
    I_v_arr = np.asarray(I_v, dtype=np.float32).ravel()
    if r_ext_arr.shape != I_v_arr.shape:
        raise ValueError(
            f"r_ext {r_ext_arr.shape} and I_v {I_v_arr.shape} must match"
        )
    N = r_ext_arr.shape[0]
    if N == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.uint8),
        )
    group_arr = np.asarray(list(group_ids))
    if group_arr.shape[0] != N:
        raise ValueError(
            f"group_ids length {group_arr.shape[0]} != N {N}"
        )

    if verifier_absent:
        return (
            np.zeros(N, dtype=np.float32),
            np.zeros(N, dtype=np.uint8),
        )

    include_mask = (I_v_arr > 0.5).astype(np.uint8)
    s = np.zeros(N, dtype=np.float32)

    unique_groups = np.unique(group_arr)
    for g in unique_groups:
        mask_g = (group_arr == g)
        inc_g = mask_g & (include_mask > 0)
        if not inc_g.any():
            # Degenerate group (K'=0): leave s = 0 everywhere for this group.
            continue
        mu_incl = float(np.mean(r_ext_arr[inc_g]))
        s[inc_g] = r_ext_arr[inc_g]
        masked_g = mask_g & ~inc_g
        s[masked_g] = mu_incl

    return s, include_mask


def _grpo_group_normalize(
    scores: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """GRPO-style group-relative normalization for 1D scalar scores.

    Groups samples by prompt index, computes (score - group_mean) / (group_std + eps).
    """
    id2scores = defaultdict(list)
    id2mean = {}
    id2std = {}
    bsz = scores.shape[0]

    with torch.no_grad():
        for i in range(bsz):
            id2scores[index[i]].append(scores[i])
        for idx in id2scores:
            if len(id2scores[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                id2mean[idx] = torch.mean(torch.stack(id2scores[idx]))
                id2std[idx] = torch.std(torch.stack(id2scores[idx]))
        normalized = torch.zeros_like(scores)
        for i in range(bsz):
            normalized[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    return normalized


@register_adv_est("sirl")
def compute_sirl_direct_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    sirl_imp_rewards: torch.Tensor = None,
    sirl_alt_rewards: torch.Tensor = None,
    sirl_det_rewards: torch.Tensor = None,
    sirl_mode: str = "default",
    sirl_adv_w_imp: float = 0.6,
    sirl_lambda_alt: float = 0.2,
    sirl_lambda_det: float = 0.1,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SIRL per-component group-relative advantages.

    For S3 (with-self-consistency) and S4 (with-full-recognition), computes
    GRPO-style group normalization *within* each action mode (improvement,
    alternative, deterioration) separately, then combines with explicit weights.

    The combined advantage is:
        A = adv_w_imp * GRPO(r_imp) + lambda_alt * GRPO(r_alt) + lambda_det * GRPO(r_det)

    Default ratio 6:2:1 is achieved by adv_w_imp=0.6, lambda_alt=0.2, lambda_det=0.1.

    For default mode (S1/S2), falls back to standard GRPO on the combined reward.
    """
    use_per_component = (
        sirl_mode in ("with-self-consistency", "with-full-recognition")
        and sirl_imp_rewards is not None
        and index is not None
    )

    if use_per_component:
        with torch.no_grad():
            # Per-component group-relative advantages
            adv_imp = _grpo_group_normalize(sirl_imp_rewards, index, epsilon)
            combined = sirl_adv_w_imp * adv_imp

            if sirl_alt_rewards is not None:
                adv_alt = _grpo_group_normalize(sirl_alt_rewards, index, epsilon)
                combined = combined + sirl_lambda_alt * adv_alt

            if sirl_det_rewards is not None and sirl_mode == "with-full-recognition":
                adv_det = _grpo_group_normalize(sirl_det_rewards, index, epsilon)
                combined = combined + sirl_lambda_det * adv_det

            advantages = combined.unsqueeze(-1) * response_mask
    else:
        # Default mode: standard GRPO on the combined reward
        scores = token_level_rewards.sum(dim=-1)
        with torch.no_grad():
            if index is not None:
                id2score = defaultdict(list)
                id2mean = {}
                id2std = {}
                bsz = scores.shape[0]
                for i in range(bsz):
                    id2score[index[i]].append(scores[i])
                for idx in id2score:
                    if len(id2score[idx]) == 1:
                        id2mean[idx] = torch.tensor(0.0)
                        id2std[idx] = torch.tensor(1.0)
                    else:
                        id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                        id2std[idx] = torch.std(torch.stack(id2score[idx]))
                for i in range(bsz):
                    if norm_adv_by_std_in_grpo:
                        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                    else:
                        scores[i] = scores[i] - id2mean[index[i]]
            advantages = scores.unsqueeze(-1) * response_mask

    return advantages, advantages


@register_adv_est("sirl_raw")
def compute_sirl_raw_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw SIRL advantage: A_j = R_j (no group normalization).

    Uses the SIRL reward directly as the advantage signal without subtracting
    the group mean or dividing by group std.  This tests whether the raw
    self-improvement reward is sufficient to drive learning (S10 ablation).
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        advantages = scores.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (dict) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config=None,
    **kwargs,
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, torch.tensor(0.0), ppo_kl_abs, torch.tensor(0.0)


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


# =============================================================================
# SIRL-Pref: Preference-Based Self-Improvement RL (new, additive)
# =============================================================================
# Design in DEFINE_SPEC (sirl-define-spec-20260418.md). Uses prefix-branched
# dependent-pair preferences with a signed pair-advantage A_pair in [-1, +1].
#
# The batch for SIRL-Pref is organized as 2B rows (B pairs, each unrolled into
# a revised row and an original row sharing a prefix). Routing between
# rows of the same pair uses batch.non_tensor_batch["sirl_pref_pair_id"] and
# batch.non_tensor_batch["sirl_pref_channel"] in {"revised", "original"}.
# =============================================================================


def _grpo_group_normalize_signed(
    scores: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Group-relative normalization with sign preservation for singleton groups.

    This is a SIRL-Pref-specific variant of ``_grpo_group_normalize``.

    For non-singleton groups the behavior is identical:
        normalized[i] = (score[i] - group_mean) / (group_std + eps)

    For singleton groups, the reference implementation returns
    (score[i] - 0) / (1 + eps) which happens to preserve sign for scalar
    rewards but depends on the zero-mean trick. Here we document and
    enforce sign preservation explicitly by returning the raw score for
    singletons, so that preference signals in {-1, 0, +1} round-trip
    cleanly when ``rollout_n == 1``.
    """
    id2scores = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2singleton = {}
    bsz = scores.shape[0]

    with torch.no_grad():
        for i in range(bsz):
            id2scores[index[i]].append(scores[i])
        for idx in id2scores:
            if len(id2scores[idx]) == 1:
                id2singleton[idx] = True
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                id2singleton[idx] = False
                id2mean[idx] = torch.mean(torch.stack(id2scores[idx]))
                id2std[idx] = torch.std(torch.stack(id2scores[idx]))
        normalized = torch.zeros_like(scores)
        for i in range(bsz):
            if id2singleton[index[i]]:
                # Singleton group: preserve raw signed scalar directly.
                normalized[i] = scores[i]
            else:
                normalized[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    return normalized


@register_adv_est("sirl_pref")
def compute_sirl_pref_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray = None,
    epsilon: float = 1e-6,
    sirl_pref_pair_advantage_raw: torch.Tensor = None,
    sirl_pref_pair_id: np.ndarray = None,
    sirl_pref_channel: np.ndarray = None,  # per-row tag: "revised" or "original"
    sirl_pref_skip_mask: torch.Tensor = None,  # [2B] bool
    sirl_pref_loss: str = "shaped_grpo",
    sirl_pref_grpo_reject_coef: float = 0.5,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatcher for SIRL-Pref advantages.

    Auto-detects the input layout from ``sirl_pref_channel``:

    **2B-row layout (full DPO-like mode):** rows alternate or interleave
    ``"revised"`` and ``"original"`` tags. Shaped-GRPO broadcasts
        +A_norm           to revised-channel tokens
        -c * A_norm       to original-channel tokens  (c = grpo_reject_coef)

    **B-row layout (SCoRe-style single-channel mode):** all rows tagged
    ``"original"`` — this is what the current trainer produces because y0
    is the only response in the batch. Shaped-GRPO applies
        -A_norm           to y0 tokens
    so that when ``A_pair > 0`` (revision wins) y0's probability is pushed
    DOWN, and when ``A_pair < 0`` (original wins) y0's probability is pushed
    UP. This is a valid SCoRe-style single-channel update — the revised
    suffix doesn't appear in the training batch, but its implicit reward
    differential drives the y0 gradient.

    ``sirl_pref_pair_advantage_raw`` is a per-row tensor of shape ``[2B]``
    or ``[B]`` containing the signed ``A_pair`` (duplicated on both rows of
    a pair in the 2B case; one value per sample in the B case).

    - ``loss == "dpo"``: return zeros; DPO loss computed separately.
    """
    bsz_2b = token_level_rewards.shape[0]
    device = token_level_rewards.device

    if sirl_pref_loss == "dpo":
        # DPO path: no token-level advantages; the DPO loss is computed
        # externally from chosen/rejected logprobs.
        zeros = torch.zeros_like(token_level_rewards)
        return zeros, zeros

    # ---- shaped_grpo path ----
    assert sirl_pref_pair_advantage_raw is not None, (
        "sirl_pref_pair_advantage_raw must be provided for sirl_pref/shaped_grpo"
    )
    assert sirl_pref_pair_id is not None and sirl_pref_channel is not None, (
        "sirl_pref_pair_id and sirl_pref_channel are required for shaped_grpo"
    )

    pair_adv_raw_rows = sirl_pref_pair_advantage_raw.to(device=device, dtype=torch.float32)

    # Dedupe row-level pair_adv_raw (same value on both rows of a pair) to
    # per-pair values, keyed by pair_id. Group-norm at the PAIR level using
    # the per-pair prompt grouping (``index``) — this avoids the degenerate
    # within-pair groupnorm collapse where the two row duplicates produce std=0.
    pair_ids_np = np.asarray(sirl_pref_pair_id)
    # Extract first-occurrence A_pair per pair_id, preserving ordering by pair_id.
    unique_pair_ids, first_idx = np.unique(pair_ids_np, return_index=True)
    # Sort back by first-occurrence order (not by pair_id value) so grouping
    # key semantics match whatever the caller supplied.
    order = np.argsort(first_idx)
    unique_pair_ids = unique_pair_ids[order]
    first_idx_sorted = first_idx[order]
    per_pair_raw = pair_adv_raw_rows[first_idx_sorted]  # [num_pairs]
    # Build a per-pair grouping: use ``index`` if provided (prompt-level key for
    # multi-pair-per-prompt cases), else each pair is its own singleton group.
    if index is not None:
        index_np = np.asarray(index)
        per_pair_group = index_np[first_idx_sorted]
    else:
        per_pair_group = unique_pair_ids

    per_pair_norm = _grpo_group_normalize_signed(per_pair_raw, per_pair_group, epsilon)

    # Broadcast per-pair normalized value back to each row via pair_id lookup.
    pair_id_to_norm = {int(unique_pair_ids[i]): per_pair_norm[i] for i in range(len(unique_pair_ids))}
    adv_norm = torch.stack([pair_id_to_norm[int(pid)] for pid in pair_ids_np]).to(device=device, dtype=torch.float32)

    # Channel sign: auto-detect 2B-row (dual-channel) vs B-row (single-channel).
    channel_arr = np.asarray(sirl_pref_channel)
    tags = []
    for i in range(bsz_2b):
        raw = channel_arr[i]
        if isinstance(raw, str):
            tags.append(raw)
        elif hasattr(raw, "decode"):
            tags.append(raw.decode())
        else:
            tags.append(str(raw))
    has_revised = any(t == "revised" for t in tags)
    has_original = any(t == "original" for t in tags)
    is_dual_channel = has_revised and has_original  # 2B layout

    channel_sign = torch.zeros(bsz_2b, device=device, dtype=torch.float32)
    if is_dual_channel:
        # 2B-row layout: +A_norm on revised, -c · A_norm on original.
        for i in range(bsz_2b):
            if tags[i] == "revised":
                channel_sign[i] = 1.0
            elif tags[i] == "original":
                channel_sign[i] = -float(sirl_pref_grpo_reject_coef)
    else:
        # B-row single-channel layout (SCoRe-style): rows are y0 only.
        # Apply -A_norm so a positive A_pair (revision wins) pushes y0 DOWN
        # and a negative A_pair (original wins) pushes y0 UP.
        for i in range(bsz_2b):
            if tags[i] == "original":
                channel_sign[i] = -1.0
            # Unknown tags default to 0 (no-op); maintained for safety.

    scalar_adv = adv_norm * channel_sign  # [B] or [2B]

    if sirl_pref_skip_mask is not None:
        keep = (~sirl_pref_skip_mask.to(device=device, dtype=torch.bool)).to(torch.float32)
        scalar_adv = scalar_adv * keep

    advantages = scalar_adv.unsqueeze(-1) * response_mask.to(torch.float32)
    return advantages, advantages


def compute_sirl_pref_dpo_loss(
    *,
    revised_logprobs: torch.Tensor,       # [B, T_rev] per-token logprobs under policy
    revised_ref_logprobs: torch.Tensor,   # [B, T_rev]
    revised_loss_mask: torch.Tensor,      # [B, T_rev]
    original_logprobs: torch.Tensor,      # [B, T_orig]
    original_ref_logprobs: torch.Tensor,  # [B, T_orig]
    original_loss_mask: torch.Tensor,     # [B, T_orig]
    label: torch.Tensor,                  # [B] int64 (1=revised preferred, 0=original preferred)
    pair_weight: torch.Tensor,            # [B] |A_pair|
    skip_mask: torch.Tensor,              # [B] bool
    y0_logprobs: torch.Tensor = None,     # [B, T_y0] optional KL-anchor logprobs
    y0_ref_logprobs: torch.Tensor = None,
    y0_loss_mask: torch.Tensor = None,
    beta: float = 0.15,
    anchor_weight: float = 0.05,
    kl_coef: float = 0.02,
) -> tuple[torch.Tensor, dict]:
    """Weighted pairwise DPO loss with optional anchor imitation and y0 KL.

    L_dpo    = -|A_pair| * log σ(β * [(logπθ(y_ch) - logπθ(y_rj))
                                      - (logπref(y_ch) - logπref(y_rj))])
    L_anchor = -1[label=revised] * |A_pair| * mean_t logπθ(y_rev_t)
    L_total  = L_dpo + η·L_anchor + kl_coef·KL(y0 || ref)

    Per-sample log-probabilities are first aggregated as masked sums over the
    response tokens (standard DPO aggregation).

    Returns (scalar_loss, metrics_dict).
    """
    device = revised_logprobs.device
    keep = (~skip_mask.to(torch.bool)).to(torch.float32).to(device)  # [B]

    # Masked sum-logprob per sample.
    def _masked_sum(lp, mask):
        return (lp * mask.to(lp.dtype)).sum(dim=-1)

    rev_logp_sum = _masked_sum(revised_logprobs, revised_loss_mask)
    rev_ref_logp_sum = _masked_sum(revised_ref_logprobs, revised_loss_mask)
    orig_logp_sum = _masked_sum(original_logprobs, original_loss_mask)
    orig_ref_logp_sum = _masked_sum(original_ref_logprobs, original_loss_mask)

    # Determine chosen/rejected from label.
    label_f = label.to(device=device, dtype=torch.float32)  # 1 if revised preferred
    rev_is_chosen = label_f  # shape [B]
    orig_is_chosen = 1.0 - label_f

    # π_θ(y_ch) − π_θ(y_rj) under policy:
    policy_delta = (rev_is_chosen * rev_logp_sum + orig_is_chosen * orig_logp_sum) - (
        orig_is_chosen * rev_logp_sum + rev_is_chosen * orig_logp_sum
    )
    # Under reference:
    ref_delta = (rev_is_chosen * rev_ref_logp_sum + orig_is_chosen * orig_ref_logp_sum) - (
        orig_is_chosen * rev_ref_logp_sum + rev_is_chosen * orig_ref_logp_sum
    )

    margin = beta * (policy_delta - ref_delta)
    # L_dpo per sample (negative because we want to MAXIMIZE log-σ).
    per_sample_dpo = -torch.nn.functional.logsigmoid(margin)  # [B]
    weights = pair_weight.to(device=device, dtype=torch.float32) * keep
    # Weighted average; avoid div-by-zero when all samples skipped.
    weight_sum = weights.sum().clamp_min(1e-8)
    L_dpo = (per_sample_dpo * weights).sum() / weight_sum

    # Anchor loss (imitation of revised suffix when it won).
    rev_mean_logp = _masked_sum(revised_logprobs, revised_loss_mask) / revised_loss_mask.to(revised_logprobs.dtype).sum(dim=-1).clamp_min(1.0)
    anchor_active = (label_f > 0.5).to(torch.float32)
    L_anchor = -(anchor_active * weights * rev_mean_logp).sum() / weight_sum

    # y0 KL (optional, only if y0 logprobs provided).
    L_y0_kl = torch.tensor(0.0, device=device)
    if y0_logprobs is not None and y0_ref_logprobs is not None and y0_loss_mask is not None:
        diff = (y0_logprobs - y0_ref_logprobs) * y0_loss_mask.to(y0_logprobs.dtype)
        denom = y0_loss_mask.to(y0_logprobs.dtype).sum(dim=-1).clamp_min(1.0)
        per_sample_kl = diff.sum(dim=-1) / denom
        L_y0_kl = per_sample_kl.mean()

    L_total = L_dpo + anchor_weight * L_anchor + kl_coef * L_y0_kl

    metrics = {
        "sirl_pref/dpo_loss": L_dpo.detach(),
        "sirl_pref/anchor_loss": L_anchor.detach(),
        "sirl_pref/y0_kl_mean": L_y0_kl.detach(),
        "sirl_pref/margin_mean": margin.detach().mean(),
        "sirl_pref/active_pairs": weights.sum().detach(),
        "sirl_pref/skip_fraction": 1.0 - keep.mean().detach(),
        "sirl_pref/label_revised_win_rate": label_f.mean().detach(),
    }
    return L_total, metrics


# =============================================================================
# CPR — Calibrated Preference RL (additive; does not modify sirl_pref code)
# =============================================================================
#
# CPR is the NeurIPS-2026-oral candidate successor to SIRL-Pref. It uses a
# single soft-label DPO loss over a trust-weighted mixture of label sources:
#
#   p_pref = alpha * p_self + (1 - alpha) * sigmoid((R_ext_a - R_ext_b) / tau)
#   margin = beta * [ (logpi_theta(y_a) - logpi_theta(y_b))
#                   - (logpi_ref  (y_a) - logpi_ref  (y_b)) ]
#   L_pair   = - p_pref * log σ(margin) - (1 - p_pref) * log σ(-margin)
#   L_anchor = - p_pref * mean_t logpi(y_a_t) - (1 - p_pref) * mean_t logpi(y_b_t)
#   L_total  = L_pair + eta * L_anchor
#
# The primary knobs are exactly three: alpha (trust dial), beta (DPO inverse
# temperature), eta (optional anchor weight; default 0). Everything else is
# either fixed or a categorical ablation.
#
# The advantage-estimator dispatch returns zeros because CPR's loss path runs
# inside the actor (fresh-forward DPO); it does not consume token-level
# advantages.
#
# Prior-art lineage (cite in any paper draft):
#
# - GPO (Furuta et al., 2024, arXiv:2409.06691). Soft-label DPO via
#   geometric aggregation of a single labeler. CPR shares the loss shape
#   but treats the label itself as a mixture random variable indexed by
#   alpha; the headline empirical claim is that an interior alpha* beats
#   both endpoints.
# - Self-Rewarding Language Models (Yuan et al., 2024). Iterative
#   self-judge DPO. Recovered exactly as alpha = 1 (with prefix-branch
#   pairs).
# - Co-rewarding (Zhang et al., 2025, arXiv:2508.00410). Two-view label
#   agreement. Recovered as the alpha = 0.5 special case.
# - DICE / Implicit Reward as the Bridge (Chen et al., 2024,
#   arXiv:2406.09760). Implicit-reward RL lineage; CPR's verifier-only
#   end (alpha = 0) reduces to RLVR-style training under this framing.
#
# Anchor symmetry note (paper §1.6 should make this explicit): the anchor
# weights BOTH candidates by p_pref / (1 - p_pref) rather than only the
# winner. This preserves swap-consistency under (a, b) -> (b, a) /
# (p, 1 - p), which is required for the soft-label loss to be a proper
# scoring rule. A winner-only anchor would break this and would also
# collapse to standard SFT-on-self-judge at alpha = 1.


@register_adv_est("cpr")
def compute_cpr_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPR uses a fresh-forward DPO loss, so token advantages are zero."""
    zeros = torch.zeros_like(token_level_rewards)
    return zeros, zeros


def compute_cpr_loss(
    *,
    logp_a: torch.Tensor,
    logp_b: torch.Tensor,
    ref_logp_a: torch.Tensor,
    ref_logp_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    p_pref: torch.Tensor,
    beta: float = 0.15,
    eta: float = 0.0,
    anchor_mode: str = "symmetric",
) -> tuple[torch.Tensor, dict]:
    """Compute the locked CPR soft-label DPO loss.

    ``anchor_mode`` controls how the anchor term weights the two candidates:
      * ``"symmetric"`` (default) — weight_a = p_pref, weight_b = 1 - p_pref.
        Pulls the reference gradient for BOTH candidates according to p_pref.
      * ``"winner_only"`` — weight_a = 1 where p_pref >= 0.5 else 0, and mirror
        for b. Only the predicted winner's log-prob is pushed up; loser is
        untouched by the anchor. Ablation knob for anchor-symmetry × α study.
    """
    if logp_a.shape[0] != logp_b.shape[0]:
        raise ValueError(
            f"compute_cpr_loss: batch size mismatch logp_a={logp_a.shape}, logp_b={logp_b.shape}"
        )
    if anchor_mode not in {"symmetric", "winner_only"}:
        raise ValueError(
            f"compute_cpr_loss: anchor_mode must be 'symmetric' or 'winner_only', got {anchor_mode!r}"
        )
    # Guard against silent [B,1] broadcast: if p_pref is not rank-1, PyTorch
    # broadcasts it against the rank-1 margin into a [B,B] cross-pair matrix,
    # and .mean() averages cross-sample interactions that do not exist. Catch
    # this at the boundary.
    if p_pref.dim() != 1:
        raise ValueError(
            f"compute_cpr_loss: p_pref must be 1-D [B], got shape {tuple(p_pref.shape)}"
        )
    if p_pref.shape[0] != logp_a.shape[0]:
        raise ValueError(
            f"compute_cpr_loss: p_pref length {p_pref.shape[0]} != batch size {logp_a.shape[0]}"
        )

    device = logp_a.device
    mask_a = mask_a.to(device=device, dtype=logp_a.dtype)
    mask_b = mask_b.to(device=device, dtype=logp_b.dtype)
    p_pref = p_pref.to(device=device, dtype=logp_a.dtype).clamp(0.0, 1.0)

    def _masked_sum(lp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (lp * mask).sum(dim=-1)

    def _masked_mean(lp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=-1).clamp_min(1.0)
        return _masked_sum(lp, mask) / denom

    policy_delta = _masked_sum(logp_a, mask_a) - _masked_sum(logp_b, mask_b)
    ref_delta = _masked_sum(ref_logp_a.detach().to(device=device, dtype=logp_a.dtype), mask_a) - _masked_sum(
        ref_logp_b.detach().to(device=device, dtype=logp_b.dtype), mask_b
    )
    margin = float(beta) * (policy_delta - ref_delta)

    per_sample_pair = -p_pref * torch.nn.functional.logsigmoid(margin) - (1.0 - p_pref) * torch.nn.functional.logsigmoid(
        -margin
    )
    L_pair = per_sample_pair.mean()

    if anchor_mode == "winner_only":
        # Tie-safe: strict > routes anchor to predicted winner; ties split 0.5/0.5
        # so swap-consistency holds at p_pref == 0.5 (e.g. verifier ties).
        winner_a = (p_pref > 0.5).to(dtype=logp_a.dtype)
        tie_mask = (p_pref == 0.5).to(dtype=logp_a.dtype)
        weight_a = winner_a + 0.5 * tie_mask
        weight_b = 1.0 - weight_a
    else:
        weight_a = p_pref
        weight_b = 1.0 - p_pref

    per_sample_anchor = -(weight_a * _masked_mean(logp_a, mask_a) + weight_b * _masked_mean(logp_b, mask_b))
    L_anchor = per_sample_anchor.mean()

    L_total = L_pair + float(eta) * L_anchor
    metrics = {
        "cpr/dpo_loss": L_pair.detach(),
        "cpr/anchor_loss": L_anchor.detach(),
        "cpr/margin_mean": margin.detach().mean(),
        "cpr/p_pref_mean": p_pref.detach().mean(),
        "cpr/p_pref_frac_extreme": (((p_pref <= 0.05) | (p_pref >= 0.95)).to(torch.float32).mean().detach()),
        "cpr/p_pref_frac_middle": (((p_pref >= 0.40) & (p_pref <= 0.60)).to(torch.float32).mean().detach()),
    }
    return L_total, metrics


def compute_bsv_loss(
    *,
    logp_a: torch.Tensor,
    logp_b: torch.Tensor,
    ref_logp_a: torch.Tensor,
    ref_logp_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    p_self: torch.Tensor,
    p_verif: torch.Tensor,
    I_v: torch.Tensor,
    beta: float = 0.15,
    eta: float = 0.0,
    anchor_mode: str = "symmetric",
) -> tuple[torch.Tensor, dict]:
    """BSV loss (DEFINE_SPEC_v3 §1.3): per-pair dispatch on the invocation indicator.

    Per-pair target preference::

        p_pref_i = I_v_i · p_verif_i + (1 − I_v_i) · p_self_i

    The rest of the objective — the DPO / Bradley-Terry soft-label pair
    loss, optional symmetric anchor, margin convention — is identical to
    :func:`compute_cpr_loss`. The reference is detached inside the margin;
    there is no length normalization in the margin (sum-logp convention,
    v2 §1 / v3 §1.3). This function is a thin per-pair wrapper around
    :func:`compute_cpr_loss` and exists so the training loop can separate
    the BSV dispatch concern from the locked CPR loss kernel.

    Parameters
    ----------
    logp_a, logp_b : torch.Tensor
        Per-token policy log-probs, shape [B, T]. Masks select the
        response span.
    ref_logp_a, ref_logp_b : torch.Tensor
        Per-token frozen-reference log-probs. Detached inside the margin.
    mask_a, mask_b : torch.Tensor
        Per-token response masks (1 on response tokens, 0 elsewhere).
    p_self : torch.Tensor
        Pairwise self-judge probability, shape [B], values in [0, 1].
    p_verif : torch.Tensor
        Verifier BT probability ``σ((R_a - R_b) / τ_ext)``, shape [B].
    I_v : torch.Tensor
        Invocation indicator ∈ {0, 1}, shape [B]. Accepts floating-point
        tensors (for gradient-free Bernoulli draws downstream); the
        threshold for dispatch is 0.5.
    beta : float, default 0.15
        DPO temperature on the margin.
    eta : float, default 0.0
        Anchor weight on the SFT-on-self term. η=0 disables the anchor.
    anchor_mode : str, default "symmetric"
        Forwarded to :func:`compute_cpr_loss`. See that function for the
        semantics of ``"symmetric"`` vs ``"winner_only"``.

    Returns
    -------
    (L_total, metrics) : (torch.Tensor, dict)
        Scalar loss and a metrics dict. ``metrics`` includes the keys
        emitted by :func:`compute_cpr_loss` plus BSV-specific:

          * ``"bsv/p_self_mean"`` — mean self-judge preference.
          * ``"bsv/p_verif_mean"`` — mean verifier BT probability.
          * ``"bsv/iv_mean"``     — empirical call rate I_v on this batch.
    """
    if p_self.dim() != 1:
        raise ValueError(f"compute_bsv_loss: p_self must be 1-D, got {tuple(p_self.shape)}")
    if p_verif.dim() != 1:
        raise ValueError(f"compute_bsv_loss: p_verif must be 1-D, got {tuple(p_verif.shape)}")
    if I_v.dim() != 1:
        raise ValueError(f"compute_bsv_loss: I_v must be 1-D, got {tuple(I_v.shape)}")
    B = logp_a.shape[0]
    if not (p_self.shape[0] == p_verif.shape[0] == I_v.shape[0] == B):
        raise ValueError(
            f"compute_bsv_loss: shape mismatch — logp_a batch {B}, "
            f"p_self {p_self.shape[0]}, p_verif {p_verif.shape[0]}, I_v {I_v.shape[0]}"
        )

    device = logp_a.device
    dtype = logp_a.dtype
    p_self_t = p_self.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    p_verif_t = p_verif.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    # Dispatch on I_v; accept floating-point indicators (>=0.5 → 1).
    iv_bool = (I_v.to(device=device, dtype=dtype) >= 0.5)
    iv_f = iv_bool.to(dtype=dtype)
    p_pref = iv_f * p_verif_t + (1.0 - iv_f) * p_self_t

    L_total, metrics = compute_cpr_loss(
        logp_a=logp_a,
        logp_b=logp_b,
        ref_logp_a=ref_logp_a,
        ref_logp_b=ref_logp_b,
        mask_a=mask_a,
        mask_b=mask_b,
        p_pref=p_pref,
        beta=beta,
        eta=eta,
        anchor_mode=anchor_mode,
    )
    metrics = dict(metrics)
    metrics["bsv/p_self_mean"] = p_self_t.detach().mean()
    metrics["bsv/p_verif_mean"] = p_verif_t.detach().mean()
    metrics["bsv/iv_mean"] = iv_f.detach().mean()
    return L_total, metrics
