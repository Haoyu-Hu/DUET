# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    config=None,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.SIRL:
        # SIRL per-component GRPO advantages (S3/S4) or fallback GRPO (S1/S2)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "norm_adv_by_std_in_grpo": norm_adv_by_std_in_grpo,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        # Pass SIRL component rewards and config
        for key in ("sirl_imp_rewards", "sirl_alt_rewards", "sirl_det_rewards"):
            if key in data.batch:
                adv_kwargs[key] = data.batch[key]
        adv_kwargs["sirl_mode"] = data.meta_info.get("sirl_mode", "default")
        adv_kwargs["sirl_adv_w_imp"] = data.meta_info.get("sirl_adv_w_imp", 0.6)
        adv_kwargs["sirl_lambda_alt"] = data.meta_info.get("sirl_lambda_alt", 0.2)
        adv_kwargs["sirl_lambda_det"] = data.meta_info.get("sirl_lambda_det", 0.1)
        advantages, returns = core_algos.compute_sirl_direct_advantage(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.SIRL_PREF:
        # SIRL-Pref: signed pair-advantage dispatcher.
        # For loss=="shaped_grpo" we emit shaped token-level advantages.
        # For loss=="dpo" we emit zero advantages (actual DPO loss is computed
        # via compute_sirl_pref_dpo_loss, hooked into the policy update step).
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        for key in (
            "sirl_pref_pair_advantage_raw",
            "sirl_pref_skip_mask",
        ):
            if key in data.batch:
                adv_kwargs[key] = data.batch[key]
        for nt_key in ("sirl_pref_pair_id", "sirl_pref_channel"):
            if nt_key in data.non_tensor_batch:
                adv_kwargs[nt_key] = data.non_tensor_batch[nt_key]
        adv_kwargs["sirl_pref_loss"] = data.meta_info.get("sirl_pref_loss", "shaped_grpo")
        adv_kwargs["sirl_pref_grpo_reject_coef"] = data.meta_info.get(
            "sirl_pref_grpo_reject_coef", 0.5,
        )
        advantages, returns = core_algos.compute_sirl_pref_advantage(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.CPR:
        # CPR: soft-label DPO loss replaces the PPO policy loss inside the actor.
        # The advantage path returns zeros of the right shape so the existing
        # PPO machinery stays inert; gradients flow via compute_cpr_loss
        # (invoked from dp_actor.py when meta_info["cpr_enabled"] is True).
        advantages, returns = core_algos.compute_cpr_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


# ----------------------------------------------------------------------
# BSV actor_logits self-judge — 1-token YES/NO/TIE via compute_log_prob
# (DEFINE_SPEC_v3 §1.2b point 1, rev 2026-04-21).
#
# Implementation uses the in-place right-pad extension pattern from the
# multi-LLM review synthesis at
# /root/.claude-octopus/review/20260421-actor-logits-risks/synthesis.md
# Each row of the input batch has its right-pad region overwritten with
# [judge_prefix_ids + candidate_id], attention_mask flipped 0→1 on those
# slots, position_ids continued from the last valid position via
# make_judge_position_ids. Candidate ∈ {A, B, T} produces 3 replicas per
# row → 3N sequences through a single compute_log_prob pass.
#
# Risk mitigations (from the review):
#   N1 tail-slice:        candidate always the last non-pad token (by construction).
#   N2 metric collision:  do NOT batch.union(judge_out) — consume scalars only.
#   N3 temperature:       set meta_info["bsv_judge_pass"]=True so fsdp_workers.py
#                         skips its rollout-temperature override (line ~780).
#   N4 memory:            verl's rollout log_prob config handles micro-batching.
#   R1 tokenization:      candidate_ids derived via tokenize(prefix+letter)-diff.
#   R4 position_ids:      via make_judge_position_ids (verl convention, unit-tested).
# ----------------------------------------------------------------------
_BSV_ACTOR_JUDGE_PREFIX = (
    "\n\nIs the response above correct? Answer with a single letter: "
    "A=yes, B=no, T=unclear.\nAnswer:\n"
)
# The trailing "\n" is LOAD-BEARING: on Qwen3 tokenizer, "Answer:" + "A"
# tokenizes to a merged ":A" single token (55648), which defeats the
# tokenize-diff candidate-id extraction. Only a newline-bounded prefix
# produces clean [A=32, B=33, T=51] single-token continuations.
# Verified empirically 2026-04-21.


def _build_and_score_actor_judge(
    batch: "DataProto",
    tokenizer,
    actor_rollout_wg,
    judge_temperature: float = 1.0,
    judge_prefix_text: str = _BSV_ACTOR_JUDGE_PREFIX,
) -> "tuple[np.ndarray, dict]":
    """Run the 1-token YES/NO/TIE self-judge and return p_self.

    Parameters
    ----------
    batch : DataProto
        Active training batch after compute_log_prob (line ~1527 of main loop).
        Must contain input_ids, attention_mask, position_ids, response_mask,
        prompts, responses.
    tokenizer :
        HuggingFace tokenizer (self.tokenizer).
    actor_rollout_wg :
        WorkerGroup with a compute_log_prob(DataProto) method.
    judge_temperature : float
        Softmax temperature for the judge logits. Default 1.0 = native. The
        worker's default-override is bypassed via meta_info["bsv_judge_pass"].
    judge_prefix_text : str
        Natural-language judge question appended after the response.

    Returns
    -------
    (p_self, diag) where:
      p_self : np.ndarray (N,) float32 in [0, 1]
      diag   : {"top3_mass_median": float,
                "fallback_rate": float,
                "degraded": bool}

    If a row's response is too long to fit the judge extension (resp_len + len(prefix)
    + 1 > L_r), that row falls back to p_self = 0.5 (neutral). If >30% of rows fall
    back, the entire batch is marked degraded and p_self is set uniformly to 0.5.
    """
    import torch
    import numpy as np
    from tensordict import TensorDict
    from verl import DataProto
    from verl.trainer.ppo.bsv_single_judge import (
        actor_logits_probs_from_log_abt,
        make_judge_position_ids,
    )

    ids = batch.batch["input_ids"]                # [N, seq]
    attn = batch.batch["attention_mask"]
    pos = batch.batch["position_ids"]
    resp_mask = batch.batch["response_mask"]      # [N, L_r]
    prompts = batch.batch["prompts"]              # [N, L_p]
    device = ids.device
    N, seq_len = ids.shape
    L_p = prompts.shape[1]

    # --- R1: derive candidate IDs by tokenize-diff (handles merges) ---
    prefix_ids_list = tokenizer.encode(
        judge_prefix_text, add_special_tokens=False
    )
    k = len(prefix_ids_list) + 1
    candidate_ids = []  # [aid, bid, tid]
    for letter in ("A", "B", "T"):
        full = tokenizer.encode(
            judge_prefix_text + letter, add_special_tokens=False
        )
        assert full[: len(prefix_ids_list)] == prefix_ids_list, (
            f"tokenizer merges prefix+{letter} boundary — cannot recover single "
            f"candidate id. prefix={prefix_ids_list}, full={full}"
        )
        tail = full[len(prefix_ids_list):]
        assert len(tail) == 1, (
            f"candidate {letter!r} is not a single token: tail={tail}. "
            f"Try a different judge_prefix_text or candidate letters."
        )
        candidate_ids.append(int(tail[0]))
    aid, bid, tid = candidate_ids

    prefix_t = torch.tensor(
        prefix_ids_list, device=device, dtype=ids.dtype
    )

    # --- Build 3N augmented batch via repeat_interleave(3) + in-place writes ---
    # Layout: row 3i → rollout i with candidate A; 3i+1 → B; 3i+2 → T.
    # CRITICAL (2026-04-21 fix): responses must keep its ORIGINAL length (L_r) so
    # that compute_log_prob's internal slice `log_probs[..., -response_length:]`
    # covers ALL response positions. A response_length=1 slice reads the LAST
    # position of input_ids, which is PAD (past the candidate), giving
    # log P(pad|everything) ≈ 0 for every candidate and a nonsense top3_mass=3.
    # We extract the candidate log-prob at position resp_len_i + k - 1 WITHIN
    # the L_r-wide response log-probs tensor.
    L_r = seq_len - L_p
    judge_ids = ids.repeat_interleave(3, dim=0).clone()      # [3N, seq]
    judge_attn = attn.repeat_interleave(3, dim=0).clone()
    judge_pos = pos.repeat_interleave(3, dim=0).clone()
    # Keep original response shape; content doesn't affect log_prob output,
    # only the .size(-1) matters for slicing.
    judge_responses = (
        batch.batch["responses"].repeat_interleave(3, dim=0).clone()
    )                                                        # [3N, L_r]

    fallback_rows = []
    resp_lens = resp_mask.sum(dim=1).to(torch.int64)          # [N]
    # Per-row target index into the response log-probs tensor (L_r-wide).
    candidate_resp_idx = np.zeros(N, dtype=np.int64)
    for i in range(N):
        resp_len_i = int(resp_lens[i].item())
        insert_start = L_p + resp_len_i
        insert_end = insert_start + k
        if insert_end > seq_len:
            fallback_rows.append(i)
            continue
        # Base position from the last valid position before insert_start
        base_pos_i = int(pos[i, insert_start - 1].item())
        new_pos_slice = base_pos_i + torch.arange(
            1, k + 1, device=device, dtype=judge_pos.dtype
        )
        for j, cid in enumerate(candidate_ids):
            row = 3 * i + j
            judge_ids[row, insert_start : insert_end - 1] = prefix_t
            judge_ids[row, insert_end - 1] = cid
            judge_attn[row, insert_start:insert_end] = 1
            judge_pos[row, insert_start:insert_end] = new_pos_slice
            # Mirror the candidate into the responses tensor so responses and
            # input_ids[:, L_p:] stay aligned (diagnostic; not used by log_prob).
            judge_responses[row, resp_len_i : resp_len_i + k - 1] = prefix_t
            judge_responses[row, resp_len_i + k - 1] = cid
        candidate_resp_idx[i] = resp_len_i + k - 1

    fallback_rate = float(len(fallback_rows)) / max(N, 1)
    if fallback_rate > 0.30:
        return (
            np.full(N, 0.5, dtype=np.float32),
            {
                "top3_mass_median": float("nan"),
                "fallback_rate": fallback_rate,
                "degraded": True,
            },
        )

    # --- Wrap as DataProto + call compute_log_prob with temperature bypass ---
    judge_td = TensorDict(
        {
            "input_ids": judge_ids,
            "attention_mask": judge_attn,
            "position_ids": judge_pos,
            "responses": judge_responses,
        },
        batch_size=3 * N,
    )
    judge_meta = {
        "temperature": float(judge_temperature),
        "bsv_judge_pass": True,
    }
    judge_batch = DataProto(
        batch=judge_td,
        non_tensor_batch={},
        meta_info=judge_meta,
    )

    # N2 mitigation: DO NOT batch.union(judge_out). Consume scalars only.
    judge_out = actor_rollout_wg.compute_log_prob(judge_batch)
    # old_log_probs has shape [3N, L_r] — log P(response[t] | prompt + response[<t])
    log_probs_full = judge_out.batch["old_log_probs"].detach().cpu().numpy().astype(
        np.float32
    )  # [3N, L_r]

    # Extract candidate log-prob per row at its target position.
    log_candidates = np.zeros(3 * N, dtype=np.float32)
    for i in range(N):
        if i in fallback_rows:
            # Leave log_candidates at 0 → probs uniform → p_self 0.5 (caller handles)
            continue
        target_idx = int(candidate_resp_idx[i])
        for j in range(3):
            row = 3 * i + j
            log_candidates[row] = log_probs_full[row, target_idx]

    # [3N] → (N, 3) where row i = [log_A, log_B, log_T]
    log_ABT = log_candidates.reshape(N, 3)

    probs_N3 = actor_logits_probs_from_log_abt(
        log_ABT[:, 0], log_ABT[:, 1], log_ABT[:, 2]
    )                                                         # (N, 3)
    p_self = (probs_N3[:, 0] + 0.5 * probs_N3[:, 2]).astype(np.float32)

    # Fallback rows get neutral p_self = 0.5
    for i in fallback_rows:
        p_self[i] = 0.5

    # Sanity metric: how much mass did the model put on {A, B, T} jointly?
    # Low value → judge signal is near-noise; kill the pilot.
    top3_mass = np.exp(log_ABT).sum(axis=1)                   # [N]
    top3_mass_median = float(np.median(top3_mass))

    return p_self, {
        "top3_mass_median": top3_mass_median,
        "fallback_rate": fallback_rate,
        "degraded": False,
    }


class RayPPOTrainer:
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.SIRL,
            AdvantageEstimator.SIRL_RAW,
            AdvantageEstimator.SIRL_PREF,
            AdvantageEstimator.CPR,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # BSV gate singleton (DEFINE_SPEC_v3 §1.2). Persists P² state across
        # training steps; rebuilt per run. Only instantiated when BSV is
        # enabled — the gate overlays the CPR kernel, so +cpr.enable=True is
        # also required at the CLI level.
        self.bsv_gate = None
        self.adaptive_alpha_ctrl = None
        self._bsv_propensity_rows: list[dict] = []
        self._bsv_propensity_csv_path: Optional[Path] = None
        _bsv_cfg = self.config.get("bsv", {}) if hasattr(self.config, "get") else {}
        if _bsv_cfg and bool(_bsv_cfg.get("enable", False)):
            from verl.trainer.ppo.bsv_gate import BSVGate

            _bsv_eps = float(_bsv_cfg.get("epsilon", 0.05))
            # Defense in depth: the CLI rejects ε≤0 already; re-check here so a
            # direct hydra override (+bsv.epsilon=0) can't silently disable MNAR.
            if _bsv_eps <= 0.0:
                raise ValueError(
                    f"BSV requires epsilon > 0 (got {_bsv_eps}); DEFINE_SPEC_v3 §3.1 "
                    "makes the ε-exploration floor non-optional for MNAR correction."
                )
            self.bsv_gate = BSVGate(
                alpha=float(_bsv_cfg.get("alpha", 0.5)),
                epsilon=_bsv_eps,
                rng_seed=int(_bsv_cfg.get("rng_seed", 0)),
                inverted=bool(_bsv_cfg.get("inverted", False)),
            )
            # Flush-on-exit safety net (DEFINE_SPEC_v3 §3.2): register an
            # atexit handler so KeyboardInterrupt / uncaught exceptions /
            # ordinary run completion always drain any buffered propensity
            # rows. The in-step flush inside fit() handles the common path;
            # this is the crash-safety fallback. Safe to call multiple times
            # (clears the buffer on each call).
            import atexit
            atexit.register(self._safe_flush_bsv_propensity)

            # Adaptive-α slow-SA controller (Phase-1 E5/E6 / methodology §5c.4).
            # Constructed only when the hydra override +bsv.adaptive_alpha=True
            # is present. Drives the gate via self.bsv_gate.set_alpha() once
            # per update_cadence steps post-warmup. Gate lives on the driver
            # only (no Ray-worker gate replicas), so no worker broadcast is
            # needed — set_alpha() on the driver singleton is sufficient.
            if bool(_bsv_cfg.get("adaptive_alpha", False)):
                from duet.adaptive_alpha import AdaptiveAlphaController

                self.adaptive_alpha_ctrl = AdaptiveAlphaController(
                    alpha_init=float(_bsv_cfg.get("alpha", 0.5)),
                    alpha_min=float(_bsv_cfg.get("adaptive_alpha_min", 0.1)),
                    alpha_max=float(_bsv_cfg.get("adaptive_alpha_max", 0.9)),
                    beta=float(_bsv_cfg.get("adaptive_alpha_beta", 4.0)),
                    rho=float(_bsv_cfg.get("adaptive_alpha_rho", 0.05)),
                    update_cadence=int(
                        _bsv_cfg.get("adaptive_alpha_cadence", 20)
                    ),
                    warmup_steps=int(
                        _bsv_cfg.get("adaptive_alpha_warmup", 40)
                    ),
                )
                print(
                    f"[ray_trainer] adaptive-α controller armed: "
                    f"α_init={self.adaptive_alpha_ctrl.alpha:.3f} "
                    f"α_range=[{self.adaptive_alpha_ctrl.alpha_min}, "
                    f"{self.adaptive_alpha_ctrl.alpha_max}] "
                    f"β={self.adaptive_alpha_ctrl.beta} "
                    f"ρ={self.adaptive_alpha_ctrl.rho} "
                    f"cadence={self.adaptive_alpha_ctrl.update_cadence} "
                    f"warmup={self.adaptive_alpha_ctrl.warmup_steps}"
                )

        # Ensure the `duet` package (Phase-1 baseline modules) is importable
        # inside Ray workers. The generated launcher sets PYTHONPATH to
        # include /workspace/src, but some Ray-worker paths (notably
        # TaskRunner constructed via runtime_env) can miss it. Insert
        # unconditionally so subsequent `from duet.* import *` calls work.
        import sys as _sys_bsl
        import os as _os_bsl
        _duet_src_root = _os_bsl.path.join("/workspace", "src")
        if _os_bsl.path.isdir(_duet_src_root) and _duet_src_root not in _sys_bsl.path:
            _sys_bsl.path.insert(0, _duet_src_root)

        # ARRoL quality-head filter (Phase-1 InPo baseline).
        # Per-rollout learned MLP head predicts pass/fail from log-prob
        # summary features over the first L_detect response tokens. Post
        # warmup, rollouts predicted-fail are MASKED in token_level_rewards
        # (Frame-A-style), preserving batch shape under vLLM. The head is
        # trained online each step with true verifier labels.
        #
        # **Phase-1 feature choice.** ARRoL paper uses hidden states at
        # position L_detect; vLLM cannot stop mid-decode and a new FSDP
        # forward pass is costly, so Phase-1 uses log-prob summary features
        # (mean/std/min/max over first L_detect tokens) as a stand-in for
        # the partial-trajectory-signal head input. The head mechanism
        # (learned MLP, online BCE, prune-with-ε) is preserved.
        self.arrol_head = None
        self._arrol_l_detect = None
        self._arrol_feature_dim = 4  # [mean_lp, std_lp, min_lp, max_lp]
        _arrol_cfg = (
            self.config.get("arrol", {}) if hasattr(self.config, "get") else {}
        )
        if _arrol_cfg and bool(_arrol_cfg.get("enable", False)):
            from duet.arrol_head import ARRoLHead, ARRoLHeadConfig

            # Paper-aligned defaults (Nguyen et al. arXiv:2603.24840):
            #   L_detect=512, warmup=20, lr=1e-6, κ=0.5, ρ=0.5, B=20 bins.
            # Our experimental-level params (rollout=8, max_response=1024,
            # actor LR) are unchanged — only ARRoL-specific algorithmic
            # knobs are aligned.
            _arrol_head_cfg = ARRoLHeadConfig(
                hidden_dim=self._arrol_feature_dim,
                l_detect=int(_arrol_cfg.get("l_detect", 512)),
                mlp_hidden=int(_arrol_cfg.get("mlp_hidden", 128)),
                warmup_steps=int(_arrol_cfg.get("warmup_steps", 20)),
                lr=float(_arrol_cfg.get("lr", 1e-6)),
                kappa=float(_arrol_cfg.get("kappa", 0.5)),
                rho=float(_arrol_cfg.get("rho", 0.5)),
                n_bins=int(_arrol_cfg.get("n_bins", 20)),
                calibrator_history_len=int(
                    _arrol_cfg.get("calibrator_history_len", 2048)
                ),
                eps_exploration=float(
                    _arrol_cfg.get("eps_exploration", 0.05)
                ),
            )
            self.arrol_head = ARRoLHead(
                _arrol_head_cfg,
                device=str(_arrol_cfg.get("device", "cpu")),
                rng_seed=int(_arrol_cfg.get("rng_seed", 0)),
            )
            self._arrol_l_detect = _arrol_head_cfg.l_detect
            print(
                f"[ray_trainer] ARRoL head armed (paper-aligned Tier B): "
                f"feature_dim={self._arrol_feature_dim} "
                f"l_detect={_arrol_head_cfg.l_detect} "
                f"warmup={_arrol_head_cfg.warmup_steps} "
                f"lr={_arrol_head_cfg.lr} "
                f"κ={_arrol_head_cfg.kappa} ρ={_arrol_head_cfg.rho} "
                f"n_bins={_arrol_head_cfg.n_bins} "
                f"eps={_arrol_head_cfg.eps_exploration}"
            )

        # VIP per-prompt rollout-budget allocator (paper arXiv:2602.01601).
        # Replaces uniform rollout_n with variable n_q per prompt (allocated
        # by SLSQP minimizing Σp(1-p)/n on GP-predicted accuracies).
        # Requires precomputed embeddings cache at vip_cache_path. Predictor
        # is fit each step on the previous step's observed (uid, acc) pairs
        # (matches official vip_ray_trainer.py: most-recent-batch only).
        self.vip_allocator = None
        self.vip_predictor = None
        self._vip_prompt_to_idx = None
        self._vip_last_observations = None  # list[(prompt_text, observed_acc)]
        _vip_cfg = (
            self.config.get("vip", {}) if hasattr(self.config, "get") else {}
        )
        if _vip_cfg and bool(_vip_cfg.get("enable", False)):
            import numpy as _np

            from duet.vip_allocator import Allocator as _VipAllocator
            from duet.vip_predictor import GPR as _VipGPR

            _cache_path = str(_vip_cfg.get("cache_path", ""))
            if not _cache_path:
                raise ValueError(
                    "[ray_trainer] vip.enable=True but vip.cache_path unset; "
                    "build via setup/cache_vip_embeddings.py first."
                )
            _cache = _np.load(_cache_path, allow_pickle=True)
            _vip_distances = _cache["distances"].astype(_np.float32)
            _vip_index_keys = [str(k) for k in _cache["index_keys"]]
            # Runtime lookup: prompts arrive in non_tensor_batch with
            # extra_info.index = task_id (stable across runs). The cache
            # was built using the same convention, so a string-key map
            # joins them directly.
            self._vip_prompt_to_idx = {k: i for i, k in enumerate(_vip_index_keys)}
            self.vip_predictor = _VipGPR(
                distance_matrix=_vip_distances,
                qid_to_idx=self._vip_prompt_to_idx,
                length_scale=float(_vip_cfg.get("length_scale", 0.5)),
                prior_value=float(_vip_cfg.get("prior_value", -5.0)),
                return_std=False,
                reuse_mean=False,
            )
            self.vip_allocator = _VipAllocator(
                allocation_rule=str(_vip_cfg.get("rule", "vip")),
                lower=int(_vip_cfg.get("lower", 4)),
                upper=int(_vip_cfg.get("upper", 16)),
                budget_per_question=int(_vip_cfg.get("budget", 8)),
                difficult_bias=float(_vip_cfg.get("bias", 0.00004)),
                verbose=False,
            )
            self._vip_last_observations = []
            print(
                f"[ray_trainer] VIP armed: rule={self.vip_allocator.allocation_rule} "
                f"budget={self.vip_allocator.budget_per_question} "
                f"L={self.vip_allocator.lower} U={self.vip_allocator.upper} "
                f"length_scale={self.vip_predictor.length_scale} "
                f"prior={self.vip_predictor.prior_value} "
                f"|cache|={len(_vip_index_keys)}"
            )

        # GRESO pre-rollout prompt-skip filter (Phase-1 InPo baseline).
        # Skips prompts whose recent reward variance is below threshold
        # (saturated → all-pass, or hopeless → all-fail). Saves BOTH rollout
        # and verifier cost on skipped prompts. History is per-prompt across
        # training steps; state_dict round-trips for checkpoint resume.
        self.greso_filter = None
        _greso_cfg = (
            self.config.get("greso", {}) if hasattr(self.config, "get") else {}
        )
        if _greso_cfg and bool(_greso_cfg.get("enable", False)):
            from duet.greso_filter import GRESOFilter

            self.greso_filter = GRESOFilter(
                history_window=int(_greso_cfg.get("history_window", 4)),
                variance_threshold=float(
                    _greso_cfg.get("variance_threshold", 0.01)
                ),
                warmup_seen=int(_greso_cfg.get("warmup_seen", 4)),
                max_tracked_prompts=int(
                    _greso_cfg.get("max_tracked_prompts", 50_000)
                ),
            )
            print(
                f"[ray_trainer] GRESO filter armed: "
                f"window={self.greso_filter.window} "
                f"threshold={self.greso_filter.threshold} "
                f"warmup={self.greso_filter.warmup} "
                f"max_tracked={self.greso_filter.max_tracked}"
            )

        # ---- ARRoL faithful (paper §4.3, isolated vLLM patch) ----
        # SEPARATE from self.arrol_head (Phase-1 log-prob path) so both can
        # coexist in the codebase. Active only when +arrol_faithful.enable=True.
        self.arrol_faithful_head = None
        self.arrol_faithful_cfg = None
        _arrol_faithful_cfg = (
            self.config.get("arrol_faithful", {}) if hasattr(self.config, "get") else {}
        )
        if _arrol_faithful_cfg and bool(_arrol_faithful_cfg.get("enable", False)):
            from duet.arrol_faithful_head import ArrolFaithfulConfig, ArrolFaithfulHead

            # We need the model's hidden_dim. Read from model config; default to
            # 2048 (Qwen3-1.7B). Larger models override via Hydra
            # +arrol_faithful.hidden_dim=4096 etc.
            try:
                from transformers import AutoConfig
                hf_cfg = AutoConfig.from_pretrained(self.config.actor_rollout_ref.model.path)
                hidden_dim = int(getattr(hf_cfg, "hidden_size", 2048))
            except Exception:
                hidden_dim = int(_arrol_faithful_cfg.get("hidden_dim", 2048))

            self.arrol_faithful_cfg = ArrolFaithfulConfig(
                hidden_dim=hidden_dim,
                head_hidden_dim=int(_arrol_faithful_cfg.get("head_hidden_dim", 128)),
                l_detect=int(_arrol_faithful_cfg.get("l_detect", 512)),
                warmup_steps=int(_arrol_faithful_cfg.get("warmup_steps", 20)),
                eps_exploration=float(_arrol_faithful_cfg.get("eps_exploration", 0.05)),
                target_keep_rate=float(_arrol_faithful_cfg.get("target_keep_rate", 0.5)),
                history_window=int(_arrol_faithful_cfg.get("history_window", 2048)),
                learning_rate=float(_arrol_faithful_cfg.get("head_lr", 3e-4)),
                rng_seed=int(_arrol_faithful_cfg.get("rng_seed", 0)),
            )
            self.arrol_faithful_head = ArrolFaithfulHead(self.arrol_faithful_cfg)
            print(
                f"[ray_trainer] ARRoL FAITHFUL armed: L_detect={self.arrol_faithful_cfg.l_detect} "
                f"hidden_dim={self.arrol_faithful_cfg.hidden_dim} "
                f"warmup={self.arrol_faithful_cfg.warmup_steps} "
                f"εexp={self.arrol_faithful_cfg.eps_exploration} "
                f"κ={self.arrol_faithful_cfg.target_keep_rate}"
            )
            print(
                "[ray_trainer] ARRoL FAITHFUL NOTICE: vLLM monkey-patch scaffolding "
                "is wired; v1 of the implementation requires (a) cross-process bridge "
                "to replicate head weights into the rollout worker each step, "
                "(b) hidden-state row-to-request alignment in compute_logits, and "
                "(c) post-step label collection for online BCE training. See "
                "doc/paper_agent_work/ARRoL_Faithful_Implementation.md §6 'Open "
                "items for v1.f.5 smoke validation'."
            )

        # ---- DUET joint controller (paper §5) ----
        # v1.a stub: read +duet.* config, instantiate the allocator + load
        # the ridge surrogate. The trainer-side rollout/SamplingParams
        # override and the per-token Δ̂ stop callback land in v1.b/v1.c
        # (see doc/paper_agent_work/DUET_Implementation_v1.md). For now the
        # presence of +duet.enable=True is benign — the cell trains as
        # uniform GRPO until the rollout-side hooks land.
        self.duet_allocator = None
        self.duet_surrogate = None
        self.duet_feature_cache: dict[str, float] | None = None
        self.duet_feature_median: float | None = None
        # DUET per-prompt state (paper §4.1, §4.2). σ̂_q^obs running mean and
        # L̂_q running mean keyed by extra_info.index. Mirrors VIP's per-prompt
        # key convention (line ~842) and the inline _duet_k_estimator pattern.
        self._duet_s_hat_mode = os.environ.get(
            "DUET_S_HAT_MODE", "per_prompt_observed"
        )
        self._duet_l_hat_mode = os.environ.get("DUET_L_HAT_MODE", "per_prompt")
        self._duet_s_hat_cold = os.environ.get("DUET_S_HAT_COLD", "ridge")
        self._duet_prompt_state = None  # initialized below if DUET enabled
        _duet_cfg = (
            self.config.get("duet", {}) if hasattr(self.config, "get") else {}
        )
        if _duet_cfg and bool(_duet_cfg.get("enable", False)):
            from duet.duet_allocator import DuetAllocator
            from duet.duet_prompt_state import new_state as _new_duet_state
            from duet.duet_surrogate import DuetSurrogate
            self._duet_prompt_state = _new_duet_state(
                k_warmup=int(os.environ.get("DUET_S_HAT_K_WARMUP", "1")),
                max_tracked=int(os.environ.get("DUET_PROMPT_STATE_MAX", "50000")),
            )

            duet_budget = float(_duet_cfg.get("budget", 0.5))
            duet_eps_pre = float(_duet_cfg.get("eps_pre", 0.05))
            duet_eps_len = float(_duet_cfg.get("eps_len", 0.05))
            duet_bisection_iters = int(_duet_cfg.get("bisection_iters", 10))
            duet_n_min = int(_duet_cfg.get("n_min", 1))
            duet_n_max = int(_duet_cfg.get("n_max", 32))
            surrogate_path = str(_duet_cfg.get(
                "surrogate_path", "outputs/duet/ridge_weights.json"
            ))
            features_path = str(_duet_cfg.get(
                "features_path",
                "outputs/duet/duet_features_math_train_clean.parquet",
            ))

            self.duet_allocator = DuetAllocator(
                budget=duet_budget,
                eps_pre=duet_eps_pre,
                eps_len=duet_eps_len,
                bisection_iters=duet_bisection_iters,
                n_min=duet_n_min,
                n_max=duet_n_max,
            )
            try:
                self.duet_surrogate = DuetSurrogate.load(surrogate_path)
                surrogate_status = (
                    f"α0={self.duet_surrogate.intercept:.4f} "
                    f"α1={self.duet_surrogate.slope:.4f} "
                    f"σ_min={self.duet_surrogate.sigma_min:.4f}"
                )
            except FileNotFoundError:
                print(
                    f"[ray_trainer] WARNING: DUET surrogate not found at "
                    f"{surrogate_path} — falling back to constant ŝ "
                    f"(uniform allocation). Run scripts/duet_fit_surrogate.py "
                    f"to ship a calibrated ridge."
                )
                surrogate_status = "MISSING — uniform fallback"

            # Per-prompt feature cache (prompt_hash → prompt_logprob_mean).
            # Populated by scripts/inpo_prompt_logprob_modal.py running on the
            # SAME dataset's calibration dumps so blake2b(int64-token-ids)
            # hashes match. Cache misses fall back to the median feature.
            try:
                import pandas as _pd
                from pathlib import Path as _Path
                if _Path(features_path).exists():
                    _plp = _pd.read_parquet(features_path)
                    if {"prompt_hash", "prompt_logprob_mean"}.issubset(_plp.columns):
                        self.duet_feature_cache = dict(zip(
                            _plp["prompt_hash"].astype(str).tolist(),
                            _plp["prompt_logprob_mean"].astype(float).tolist(),
                        ))
                        self.duet_feature_median = float(
                            _plp["prompt_logprob_mean"].median()
                        )
                        feature_status = (
                            f"{len(self.duet_feature_cache)} prompts cached, "
                            f"median={self.duet_feature_median:.3f}"
                        )
                    else:
                        feature_status = (
                            f"FILE PRESENT BUT MISSING COLUMNS at {features_path}"
                        )
                else:
                    feature_status = (
                        f"NOT FOUND at {features_path} — every prompt will use "
                        f"a constant feature (uniform allocation)"
                    )
            except Exception as _e:  # noqa: BLE001
                feature_status = f"LOAD FAILED: {_e}"

            print(
                f"[ray_trainer] DUET armed: budget={duet_budget} "
                f"eps_pre={duet_eps_pre} eps_len={duet_eps_len} "
                f"bisection_iters={duet_bisection_iters} "
                f"surrogate={surrogate_status} features={feature_status}"
            )

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
                or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None
            ), (
                "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, "
                "due to no role-playing support"
            )
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], (
                "only GRPO is tested for multi-turn with tool"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_inpo_calibration(self, batch, dump_dir):
        """Per-rollout dump for InPo Day-3 calibration analysis.

        Writes two parquet files per training step under dump_dir:
          rollouts_step_NNNNN.parquet — one row per rollout: uid, lengths,
            R_ext, advantages, summed and per-token old-policy logprobs.
          prompts_step_NNNNN.parquet  — one row per unique uid in the step:
            prompt token IDs trimmed to actual length.
        UIDs are per-step UUIDs (not stable across steps); recover stable
        prompt identity offline by hashing prompt_token_ids.
        """
        import os
        import numpy as np
        import pandas as pd

        os.makedirs(dump_dir, exist_ok=True)

        response_mask = batch.batch["response_mask"].cpu().numpy()
        token_scores = batch.batch["token_level_scores"].cpu().numpy()
        old_lp = batch.batch["old_log_probs"].cpu().numpy().astype(np.float32)
        advantages = batch.batch["advantages"].cpu().numpy().astype(np.float32)
        uids = np.asarray(batch.non_tensor_batch["uid"]).astype(str)

        N = response_mask.shape[0]
        response_lengths = response_mask.sum(axis=-1).astype(np.int32)

        if "attention_mask" in batch.batch:
            am = batch.batch["attention_mask"].cpu().numpy()
            prompt_lengths = (am.sum(axis=-1).astype(np.int32) - response_lengths)
        else:
            prompt_lengths = np.zeros(N, dtype=np.int32)

        R_ext = (token_scores * response_mask).sum(axis=-1).astype(np.float32)
        adv_sum = (advantages * response_mask).sum(axis=-1).astype(np.float32)
        adv_mean = np.where(
            response_lengths > 0, adv_sum / np.maximum(response_lengths, 1), 0.0
        ).astype(np.float32)
        old_lp_sum = (old_lp * response_mask).sum(axis=-1).astype(np.float32)
        old_lp_mean = np.where(
            response_lengths > 0, old_lp_sum / np.maximum(response_lengths, 1), 0.0
        ).astype(np.float32)

        rollout_idx = np.zeros(N, dtype=np.int32)
        seen: dict = {}
        for i, u in enumerate(uids):
            rollout_idx[i] = seen.get(u, 0)
            seen[u] = int(rollout_idx[i]) + 1

        old_lp_per_token = [
            old_lp[i, : int(response_lengths[i])].tolist() for i in range(N)
        ]

        has_ref = "ref_log_prob" in batch.batch
        if has_ref:
            ref_lp = batch.batch["ref_log_prob"].cpu().numpy().astype(np.float32)
            ref_lp_sum = (ref_lp * response_mask).sum(axis=-1).astype(np.float32)
        else:
            ref_lp_sum = np.zeros(N, dtype=np.float32)

        # Optional: per-rollout response token IDs (for offline decoding to
        # text — used in figures showing how DUET truncates rollouts). Off
        # by default for legacy run compatibility; enabled via env var
        # INPO_CALIBRATION_DUMP_RESPONSES=1. Adds ~1MB/step at 1.7B+b=1.0.
        dump_responses = os.environ.get(
            "INPO_CALIBRATION_DUMP_RESPONSES", "1"
        ) == "1"
        response_token_ids = None
        if dump_responses and "responses" in batch.batch:
            responses_t = batch.batch["responses"].cpu().numpy().astype(np.int64)
            response_token_ids = [
                responses_t[i, : int(response_lengths[i])].tolist()
                for i in range(N)
            ]

        rollouts_dict = {
            "step": np.full(N, self.global_steps, dtype=np.int32),
            "uid": uids,
            "rollout_idx_in_prompt": rollout_idx,
            "response_length": response_lengths,
            "R_ext": R_ext,
            "advantage_sum": adv_sum,
            "advantage_mean": adv_mean,
            "old_logprob_sum": old_lp_sum,
            "old_logprob_mean": old_lp_mean,
            "old_logprob_per_token": old_lp_per_token,
            "ref_logprob_sum": ref_lp_sum,
            "ref_logprob_available": np.full(N, has_ref, dtype=bool),
        }
        if response_token_ids is not None:
            rollouts_dict["response_token_ids"] = response_token_ids
        rollouts_df = pd.DataFrame(rollouts_dict)
        rollouts_path = os.path.join(
            dump_dir, f"rollouts_step_{int(self.global_steps):05d}.parquet"
        )
        rollouts_df.to_parquet(rollouts_path, index=False, compression="snappy")

        if "prompts" in batch.batch:
            prompts_t = batch.batch["prompts"].cpu().numpy()
            first_idx_per_uid: dict = {}
            for i, u in enumerate(uids):
                if u not in first_idx_per_uid:
                    first_idx_per_uid[u] = i
            prompt_rows = []
            for u, i in first_idx_per_uid.items():
                L_p = int(prompt_lengths[i])
                tok_ids = (
                    prompts_t[i, -L_p:].astype(np.int64).tolist() if L_p > 0 else []
                )
                prompt_rows.append({
                    "step": np.int32(self.global_steps),
                    "uid": u,
                    "prompt_length": L_p,
                    "prompt_token_ids": tok_ids,
                })
            pd.DataFrame(prompt_rows).to_parquet(
                os.path.join(
                    dump_dir, f"prompts_step_{int(self.global_steps):05d}.parquet"
                ),
                index=False,
                compression="snappy",
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _filter_validation_batch_by_data_source(
        self, test_batch: DataProto, data_source_regex: Optional[str]
    ) -> Optional[DataProto]:
        if not data_source_regex:
            return test_batch

        data_sources = test_batch.non_tensor_batch.get("data_source")
        if data_sources is None:
            return test_batch

        matches = np.array(
            [re.search(data_source_regex, str(data_source)) is not None for data_source in data_sources],
            dtype=bool,
        )
        if not matches.any():
            return None
        if matches.all():
            return test_batch
        return test_batch.select_idxs(matches)

    def _validate(self, data_source_regex: Optional[str] = None):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = self._filter_validation_batch_by_data_source(test_batch, data_source_regex)
            if test_batch is None or len(test_batch) == 0:
                continue

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        if not data_source_lst:
            if data_source_regex:
                print(f"No validation rows matched data_source regex {data_source_regex!r}; skipping validation.")
            return {}

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # Optionally drop bootstrap side metrics for faster validation
        if not self.config.trainer.get("compute_val_aux", False):
            metric_dict = {k: v for k, v in metric_dict.items() if not k.startswith("val-aux/")}

        # ── Greedy pass@1 evaluation ──────────────────────────────────────────
        greedy_data_source_lst = []
        greedy_infos_dict: dict[str, list] = defaultdict(list)
        greedy_inputs = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = self._filter_validation_batch_by_data_source(test_batch, data_source_regex)
            if test_batch is None or len(test_batch) == 0:
                continue
            # No repeat — greedy pass@1 uses exactly one response per prompt

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                break

            greedy_input_ids = test_batch.batch["input_ids"]
            greedy_input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in greedy_input_ids]
            greedy_inputs.extend(greedy_input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            for optional_key in ("multi_modal_data", "raw_prompt", "tools_kwargs", "interaction_kwargs", "agent_name"):
                if optional_key in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append(optional_key)
            greedy_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            greedy_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            greedy_gen_batch_padded, pad_size = pad_dataproto_to_divisor(greedy_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                greedy_output_padded = self.actor_rollout_wg.generate_sequences(greedy_gen_batch_padded)
            else:
                greedy_output_padded = self.async_rollout_manager.generate_sequences(greedy_gen_batch_padded)
            greedy_output = unpad_dataproto(greedy_output_padded, pad_size=pad_size)

            test_batch = test_batch.union(greedy_output)
            test_batch.meta_info["validate"] = True

            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()

            greedy_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    greedy_infos_dict[key].extend(lst)

            greedy_data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        if greedy_inputs:
            greedy_data_sources = np.concatenate(greedy_data_source_lst, axis=0)
            greedy_metrics = process_validation_metrics(greedy_data_sources, greedy_inputs, greedy_infos_dict)
            for data_source, var2metric2val in greedy_metrics.items():
                for var_name, metric2val in var2metric2val.items():
                    for metric_name, metric_val in metric2val.items():
                        metric_dict[f"val-greedy/{data_source}/{var_name}/{metric_name}"] = metric_val

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        # Persist BSVGate state so resume preserves P² markers + RNG + counters.
        # Without this, resume rebuilds a fresh gate and post-resume propensities
        # are drawn from a different policy than pre-resume CSV rows — biasing
        # IPS/DR estimators. Pickle is fine here (no cross-process restore).
        if self.bsv_gate is not None:
            import pickle
            bsv_gate_path = os.path.join(local_global_step_folder, "bsv_gate.pkl")
            with open(bsv_gate_path, "wb") as f:
                pickle.dump(self.bsv_gate, f)

        # Persist GRESO filter history so resume picks up per-prompt
        # reward windows and warmup counts from where the previous run
        # stopped. Uses state_dict() (JSON-round-trippable) rather than
        # pickle to remain schema-stable across module refactors.
        if self.greso_filter is not None:
            import json
            greso_path = os.path.join(
                local_global_step_folder, "greso_filter.json"
            )
            with open(greso_path, "w") as f:
                json.dump(self.greso_filter.state_dict(), f)

        # Persist ARRoL head (MLP weights + optimizer + step counters) so
        # resume preserves the learned quality-head state. torch.save since
        # the state_dict contains tensors.
        if self.arrol_head is not None:
            arrol_path = os.path.join(
                local_global_step_folder, "arrol_head.pt"
            )
            torch.save(self.arrol_head.state_dict(), arrol_path)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # Restore BSVGate state so P² markers + RNG + call counters pick up
        # where the previous run left off. If the checkpoint pre-dates BSV
        # wiring (no pkl on disk) we keep the freshly-constructed gate —
        # emit a warning so the user knows IPS aggregation across the break
        # may be biased.
        if self.bsv_gate is not None:
            import pickle
            bsv_gate_path = os.path.join(global_step_folder, "bsv_gate.pkl")
            if os.path.exists(bsv_gate_path):
                with open(bsv_gate_path, "rb") as f:
                    self.bsv_gate = pickle.load(f)
                print(f"Restored BSVGate state from {bsv_gate_path} "
                      f"(realized_call_rate={self.bsv_gate.realized_call_rate():.3f}, "
                      f"n_observations={self.bsv_gate.n_observations()})")
            else:
                print(f"Warning: No BSVGate state at {bsv_gate_path}; "
                      "resuming with fresh gate. Post-resume propensities will "
                      "be drawn from a different policy than pre-resume rows — "
                      "IPS/DR estimates spanning the break may be biased.")

        # Restore GRESO filter history from JSON sidecar. If the
        # checkpoint pre-dates GRESO wiring we keep the fresh filter —
        # warmup simply re-initializes per-prompt windows.
        if self.greso_filter is not None:
            import json
            greso_path = os.path.join(global_step_folder, "greso_filter.json")
            if os.path.exists(greso_path):
                with open(greso_path, "r") as f:
                    self.greso_filter.load_state_dict(json.load(f))
                _stats = self.greso_filter.stats()
                print(
                    f"Restored GRESOFilter state from {greso_path} "
                    f"(n_prompts_tracked={int(_stats['greso/n_prompts_tracked'])}, "
                    f"skip_rate={_stats['greso/skip_rate']:.3f})"
                )
            else:
                print(
                    f"Warning: No GRESOFilter state at {greso_path}; "
                    "resuming with fresh filter history (warmup restarts)."
                )

        # Restore ARRoL head (MLP + optimizer + step counters) so resume
        # preserves learned quality-head state. Fresh head is kept if the
        # checkpoint pre-dates ARRoL wiring (warmup restarts from 0).
        if self.arrol_head is not None:
            arrol_path = os.path.join(global_step_folder, "arrol_head.pt")
            if os.path.exists(arrol_path):
                self.arrol_head.load_state_dict(
                    torch.load(arrol_path, weights_only=False)
                )
                _stats_a = self.arrol_head.stats()
                print(
                    f"Restored ARRoL head from {arrol_path} "
                    f"(step_count={int(_stats_a['arrol/step_count'])}, "
                    f"predict_count={int(_stats_a['arrol/predict_count'])}, "
                    f"prune_rate={_stats_a['arrol/prune_rate']:.3f})"
                )
            else:
                print(
                    f"Warning: No ARRoL head state at {arrol_path}; "
                    "resuming with fresh head (warmup restarts)."
                )

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _safe_flush_bsv_propensity(self) -> None:
        """Best-effort flush of buffered BSV propensity rows.

        Called from the atexit hook and from the fit() finally block.
        Swallows all exceptions so a broken CSV path never masks the
        original training exception. Safe to call multiple times.
        """
        if self.bsv_gate is None:
            return
        try:
            if getattr(self, "_bsv_propensity_rows", None):
                self._flush_bsv_propensity_rows()
        except Exception:
            # Intentionally swallow — do not mask the original training
            # exception that triggered this atexit/finally path.
            pass

    def _flush_bsv_propensity_rows(self) -> None:
        """Append buffered BSV propensity rows to the per-run CSV sidecar.

        Path: ``outputs/bsv_propensity/<experiment_name>__<run_id>.csv`` where
        ``run_id`` uniquely identifies this run (start timestamp + bsv_alpha +
        bsv_epsilon + rng_seed). Two runs with the same ``experiment_name``
        but different BSV hyperparameters will NOT collide because run_id
        differs. Header is written once per file; subsequent flushes append.
        """
        import csv

        if self._bsv_propensity_csv_path is None:
            run_name = getattr(self.config.trainer, "experiment_name", None) or "bsv_run"
            run_name = str(run_name).replace("/", "_")
            out_dir = Path("outputs") / "bsv_propensity"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Disambiguate across concurrent or repeated runs with the same
            # experiment_name: include a start-time token + the BSV config.
            # Silently appending rows from different (α, ε, seed) cells would
            # poison downstream IPS/DR aggregation.
            ts = getattr(self, "_bsv_run_start_ts", None)
            if ts is None:
                import datetime
                ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                self._bsv_run_start_ts = ts
            _alpha = float(self.bsv_gate.alpha) if self.bsv_gate is not None else -1.0
            _eps = float(self.bsv_gate.epsilon) if self.bsv_gate is not None else -1.0
            _seed = int(self.bsv_gate._rng.integers(0, 2**31 - 1, size=0).size and 0 or 0)  # noqa
            # Prefer the configured rng_seed if we can fish it out of the config.
            try:
                _seed = int(self.config.bsv.get("rng_seed", 0))
            except Exception:
                _seed = 0
            run_id = f"{ts}__a{_alpha:.2f}_e{_eps:.3f}_s{_seed}"
            self._bsv_propensity_csv_path = out_dir / f"{run_name}__{run_id}.csv"

        write_header = not self._bsv_propensity_csv_path.exists()
        fieldnames = ["step", "pair_idx", "p_invoke", "I_v", "p_self", "p_verif", "R_a", "R_b"]
        with self._bsv_propensity_csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(self._bsv_propensity_rows)
        self._bsv_propensity_rows.clear()

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        # Cumulative efficiency metrics for post-hoc efficiency plots
        # (train_time vs val-score, rollouts vs val-score). "Train-only"
        # time excludes testing + save_checkpoint phases so GRPO/DAPO/ARRoL
        # can be compared on training-proper wall time, independent of
        # val suite cost.
        self._custom_train_time_cum_s = 0.0
        self._custom_rollouts_produced_cum = 0
        self._custom_rollouts_used_cum = 0

        repeat_sampling_sglang_grpo = (
            self.config.actor_rollout_ref.rollout.name == "sglang"
            and self.config.actor_rollout_ref.rollout.multi_turn.enable
        )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]

                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # ---- GRESO pre-rollout prompt filter (Phase-1 baseline) ----
                # Decide which prompts in this batch to skip based on per-
                # prompt reward history. If too few survive world-size
                # divisibility, skip the entire batch (continue without
                # advancing global_steps). Skipped prompts consume a
                # dataloader slot but zero rollout / verifier budget.
                if self.greso_filter is not None:
                    try:
                        _gb_input_ids = gen_batch.batch["input_ids"]
                        _prompts = [
                            self.tokenizer.decode(
                                _gb_input_ids[i], skip_special_tokens=True
                            )
                            for i in range(_gb_input_ids.shape[0])
                        ]
                        _keep = self.greso_filter.decide_batch(_prompts)
                        _kept_count = int(_keep.sum())
                        _total = int(_keep.shape[0])
                        _skip_rate_step = (
                            1.0 - (_kept_count / max(1, _total))
                        )
                        # Divisibility: kept * rollout.n % n_gpus == 0
                        _n_gpus_cfg = int(
                            self.config.trainer.n_gpus_per_node
                            * self.config.trainer.nnodes
                        )
                        _roll_n = int(
                            self.config.actor_rollout_ref.rollout.n
                        )
                        from math import gcd as _gcd
                        _divisor = max(
                            1, _n_gpus_cfg // _gcd(_n_gpus_cfg, _roll_n)
                        )
                        _safe_kept = (
                            (_kept_count // _divisor) * _divisor
                        )
                        metrics.setdefault(
                            "greso/skip_rate_step", _skip_rate_step
                        )
                        metrics.setdefault(
                            "greso/kept_count", float(_kept_count)
                        )
                        metrics.setdefault(
                            "greso/safe_kept", float(_safe_kept)
                        )
                        if _safe_kept < _divisor:
                            metrics.setdefault("greso/batch_skipped", 1.0)
                            metrics.update(self.greso_filter.stats())
                            logger.log(
                                data=metrics, step=self.global_steps
                            )
                            continue
                        if _safe_kept < _total:
                            _keep_idx = np.where(_keep)[0][:_safe_kept]
                            _mask_safe = np.zeros(_total, dtype=bool)
                            _mask_safe[_keep_idx] = True
                            batch = batch.select_idxs(_mask_safe)
                            gen_batch = gen_batch.select_idxs(_mask_safe)
                    except Exception as _greso_err:  # noqa: BLE001
                        # GRESO is a baseline comparator; a crash must not
                        # kill training. Flag and continue with full batch.
                        metrics.setdefault("greso/filter_failed", 1.0)
                        print(
                            f"[ray_trainer] GRESO pre-rollout filter error: "
                            f"{_greso_err}"
                        )

                if repeat_sampling_sglang_grpo:
                    uids_for_prompts = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch.non_tensor_batch["uid"] = uids_for_prompts
                    gen_batch.non_tensor_batch["uid"] = uids_for_prompts
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    assert np.array_equal(batch.non_tensor_batch["uid"], gen_batch.non_tensor_batch["uid"]), (
                        "UIDs must be identical for SGLang rollout"
                    )

                # VIP variable-rate prompt replication BEFORE generation.
                # Replaces uniform rollout_n duplication: each original prompt
                # is replicated n_q times where n_q is the SLSQP allocator's
                # output (paper arXiv:2602.01601). vLLM is called with
                # SamplingParams.n=1 (--rollout-n 1 in CLI) so the inflated
                # batch yields exactly Σ n_q responses. Sets `_vip_did_replicate`
                # so the standard post-generate replicate (line ~2146) is skipped.
                _vip_did_replicate = False
                self._vip_step_keys_per_uid = None  # uid → cache key map for post-reward hook
                if self.vip_allocator is not None and not repeat_sampling_sglang_grpo:
                    try:
                        # 1. Fit GP on previous step's observations (most-recent-batch only,
                        #    per official vip_ray_trainer.py).
                        if self._vip_last_observations:
                            _prev_keys = [k for k, _ in self._vip_last_observations]
                            _prev_accs = [a for _, a in self._vip_last_observations]
                            self.vip_predictor.fit_qids(_prev_keys, _prev_accs)
                            self._vip_last_observations = []
                        # 2. Extract task_id keys from extra_info (stable string IDs
                        #    matching the precomputed embedding cache row ordering).
                        _vip_keys = []
                        _ei = batch.non_tensor_batch.get("extra_info", None)
                        if _ei is not None:
                            for _info in _ei:
                                _vip_keys.append(str(_info.get("index", "") if isinstance(_info, dict) else ""))
                        if not _vip_keys or len(_vip_keys) != len(batch.batch):
                            raise RuntimeError(
                                f"VIP: extra_info.index missing or shape mismatch "
                                f"(got {len(_vip_keys)} keys for {len(batch.batch)} prompts)"
                            )
                        # 3. Predict per-prompt accuracy via GP (unknown keys → prior).
                        _vip_accs, _ = self.vip_predictor.predict_qids(_vip_keys)
                        # 4. Allocate budgets via SLSQP.
                        _n_per_prompt = self.vip_allocator.allocate(_vip_accs.tolist())
                        # 5. Generate one uid per ORIGINAL prompt; variable-replicate
                        #    both batch and gen_batch by per-prompt count.
                        _uids = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                            dtype=object,
                        )
                        batch.non_tensor_batch["uid"] = _uids
                        gen_batch.non_tensor_batch["uid"] = _uids
                        from verl.protocol import DataProto as _DataProto
                        _sub_b, _sub_g = [], []
                        for _i, _n in enumerate(_n_per_prompt):
                            _ni = int(_n)
                            if _ni <= 0:
                                continue
                            _sub_b.append(
                                batch[_i:_i + 1].repeat(repeat_times=_ni, interleave=True)
                            )
                            _sub_g.append(
                                gen_batch[_i:_i + 1].repeat(repeat_times=_ni, interleave=True)
                            )
                        if not _sub_b:
                            raise RuntimeError("VIP allocator returned all-zero budget")
                        batch = _DataProto.concat(_sub_b)
                        gen_batch = _DataProto.concat(_sub_g)
                        # Cache uid→key mapping for post-reward observation logging.
                        # Each original prompt's uid maps to its cache key.
                        self._vip_step_keys_per_uid = {
                            str(_uids[_i]): _vip_keys[_i]
                            for _i in range(len(_uids))
                            if int(_n_per_prompt[_i]) > 0
                        }
                        _vip_did_replicate = True
                        metrics["vip/budget_mean"] = float(np.mean(_n_per_prompt))
                        metrics["vip/budget_std"] = float(np.std(_n_per_prompt))
                        metrics["vip/budget_min"] = float(np.min(_n_per_prompt))
                        metrics["vip/budget_max"] = float(np.max(_n_per_prompt))
                        metrics["vip/n_zero_budget"] = float(
                            np.sum(np.array(_n_per_prompt) == 0)
                        )
                        metrics["vip/n_predicted_known"] = float(
                            np.sum([k in self._vip_prompt_to_idx for k in _vip_keys])
                        )
                    except Exception as _vip_err:  # noqa: BLE001
                        # VIP is a baseline comparator; a failure here must
                        # not kill training. Fall back to uniform replication.
                        metrics.setdefault("vip/replicate_failed", 1.0)
                        print(f"[ray_trainer] VIP pre-rollout failed: {_vip_err}")
                        _vip_did_replicate = False

                # DUET joint controller (paper §5): cost-weighted Neyman
                # allocation of (n_q, max_tokens_q) under a single dual λ★.
                # Mirrors VIP's variable-n_q pattern: pre-replicate batch by
                # n_q, stash per-replica max_tokens into non_tensor_batch for
                # vllm_rollout_spmd to consume. Mutually exclusive with VIP
                # (both can't replicate the same batch).
                _duet_did_replicate = False
                if (self.duet_allocator is not None
                        and not _vip_did_replicate
                        and not repeat_sampling_sglang_grpo):
                    if os.environ.get("DUET_TRACE", "0") == "1":
                        print(f"[DUET-TRACE] entering allocation block, |batch|={len(batch.batch)}", flush=True)
                    try:
                        import hashlib as _hashlib
                        # 1. Hash each prompt's unpadded token_ids and look
                        #    up its prompt_logprob_mean feature. Cache misses
                        #    fall back to the median feature. raw_prompt_ids
                        #    lives in gen_batch.non_tensor_batch as a numpy
                        #    object array of lists — no padding to strip.
                        _raw = gen_batch.non_tensor_batch.get("raw_prompt_ids", None)
                        if _raw is None:
                            raise RuntimeError(
                                "raw_prompt_ids missing from gen_batch.non_tensor_batch — "
                                "DUET cannot compute features"
                            )
                        _M_orig = len(_raw)
                        feature_default = (
                            self.duet_feature_median
                            if self.duet_feature_median is not None
                            else 0.0
                        )
                        # Per-prompt state lookups (paper §4.1, §4.2). When
                        # _duet_s_hat_mode == "per_prompt_observed" we override
                        # the ridge prediction with σ̂_q^obs once warm; cold
                        # prompts fall through to the strategy in DUET_S_HAT_COLD.
                        # Same shape for L̂_q: per-prompt running mean once warm,
                        # else cold-fallback to mean(K-window) → max_response_len.
                        from duet.duet_prompt_state import (
                            get_l as _duet_get_l, get_s as _duet_get_s,
                        )
                        _ei_alloc = batch.non_tensor_batch.get("extra_info", None)
                        _q_indices = [
                            (str(_ei_alloc[_i].get("index", ""))
                             if _ei_alloc is not None
                             and _i < len(_ei_alloc)
                             and isinstance(_ei_alloc[_i], dict) else "")
                            for _i in range(_M_orig)
                        ]
                        _sigma_min = (
                            float(self.duet_surrogate.sigma_min)
                            if self.duet_surrogate is not None else 1e-6
                        )
                        _default_L = float(self.config.data.max_response_length)
                        # Cold L̂ fallback: mean of K-estimator window if non-empty,
                        # else max_response_length (per Phase 0 finding: realized
                        # kept-rollout length, not the hard ceiling).
                        _est_alloc = getattr(self, "_duet_k_estimator", None)
                        _cold_L = (
                            float(np.mean(_est_alloc["lengths"]))
                            if _est_alloc is not None
                            and len(_est_alloc.get("lengths", [])) > 0
                            else _default_L
                        )
                        s_hat_list: list[float] = []
                        L_hat: list[float] = []
                        n_hits = n_misses = 0
                        n_s_observed = n_s_cold = 0
                        n_l_observed = n_l_cold = 0
                        for _i in range(_M_orig):
                            tok_ids = _raw[_i]
                            tok_bytes = (
                                np.asarray(list(tok_ids), dtype=np.int64).tobytes()
                                if len(tok_ids) > 0 else b""
                            )
                            _h = _hashlib.blake2b(tok_bytes, digest_size=8).hexdigest()
                            feat = (
                                self.duet_feature_cache.get(_h, feature_default)
                                if self.duet_feature_cache is not None
                                else feature_default
                            )
                            if self.duet_feature_cache is not None and _h in self.duet_feature_cache:
                                n_hits += 1
                            else:
                                n_misses += 1
                            # ŝ_q resolution: per-prompt observed ⟹ override with
                            # Welford running mean once warm; otherwise cold-start.
                            _q = _q_indices[_i]
                            _s_obs = (
                                _duet_get_s(self._duet_prompt_state, _q)
                                if (self._duet_s_hat_mode == "per_prompt_observed"
                                    and self._duet_prompt_state is not None and _q)
                                else None
                            )
                            if _s_obs is not None:
                                s_q = max(_sigma_min, float(_s_obs))
                                n_s_observed += 1
                            else:
                                # Cold-start: env-var toggle. "ridge" is production-
                                # faithful (≈ ridge.predict(median_feature) ≈ const
                                # under feature_cache_hit_rate=0). "sigma_min" is
                                # paper-literal (≈ σ_min directly).
                                if (self._duet_s_hat_cold == "sigma_min"
                                        and self._duet_s_hat_mode == "per_prompt_observed"):
                                    s_q = _sigma_min
                                elif self.duet_surrogate is not None:
                                    s_q = self.duet_surrogate.predict(feat)
                                else:
                                    s_q = 1.0
                                n_s_cold += 1
                            s_hat_list.append(s_q)
                            # L̂_q resolution: per-prompt running mean over kept
                            # lengths once warm; else cold-fallback (K-window mean).
                            _L_obs = (
                                _duet_get_l(self._duet_prompt_state, _q)
                                if (self._duet_l_hat_mode == "per_prompt"
                                    and self._duet_prompt_state is not None and _q)
                                else None
                            )
                            if _L_obs is not None:
                                L_hat.append(float(_L_obs))
                                n_l_observed += 1
                            else:
                                L_hat.append(_cold_L)
                                n_l_cold += 1
                        # 3. B_full = uniform-allocation budget reference.
                        # Use the EXPLICIT --duet-ref-rollout-n flag (default 8,
                        # matching GRPO baseline's rollout.n) rather than vLLM's
                        # actual SamplingParams.n=1. The latter is a pre-replication
                        # implementation detail; the budget reference must match
                        # the GRPO baseline's per-prompt rollout count for the
                        # "B = budget × B_GRPO_full" semantic to hold.
                        ref_rollout_n = int(
                            self.config.duet.get("ref_rollout_n", 8)
                            if hasattr(self.config, "duet")
                            else 8
                        )
                        # B_full = M × ref_rollout_n × L̂_global (paper-faithful,
                        # post 2026-04-28 fix). The pre-fix formula used
                        # max_response_length — a "fictitious upper bound" that
                        # over-states the GRPO budget by ~2× and inflates n_q
                        # at low budgets. Use the realized per-step token cost
                        # via the L̂_q vector mean (matches the allocator's
                        # docstring "mean_resp" semantic).
                        B_full = float(
                            _M_orig * ref_rollout_n * float(np.mean(L_hat))
                        )
                        # 4. Solve the joint controller's bisection.
                        _duet_result = self.duet_allocator.allocate(
                            s_hat_list, L_hat, B_full
                        )
                        # 4b. Post-bisection downward budget trim
                        # (post 2026-04-28 fix): bisection rounding + clipping
                        # to [n_min, n_max] can leave Σ n_q · L̂_q over target_B
                        # by 25-100% at low budgets. Enforce ≤ as hard upper
                        # bound by decrementing the largest n_q (above n_min)
                        # whose token-cost contribution is heaviest, until ≤.
                        _target_B = self.duet_allocator.budget * B_full
                        _n_q_post = list(_duet_result.n_q)
                        _n_min_alloc = int(self.duet_allocator.n_min)
                        _trim_iters = 0
                        while sum(
                            _n_q_post[_i] * float(L_hat[_i])
                            for _i in range(_M_orig)
                        ) > _target_B and _trim_iters < 4 * _M_orig:
                            # Pick the prompt whose decrement reduces budget
                            # the most among entries above n_min — equivalent
                            # to argmax L̂_q over indices with n_q > n_min.
                            _best, _best_L = -1, -1.0
                            for _i in range(_M_orig):
                                if _n_q_post[_i] > _n_min_alloc and float(L_hat[_i]) > _best_L:
                                    _best_L = float(L_hat[_i])
                                    _best = _i
                            if _best < 0:
                                break  # all at floor; cannot trim further
                            _n_q_post[_best] -= 1
                            _trim_iters += 1
                        from dataclasses import replace as _dc_replace
                        _duet_result = _dc_replace(
                            _duet_result,
                            n_q=_n_q_post,
                            budget_used=float(sum(
                                _n_q_post[_i] * float(L_hat[_i])
                                for _i in range(_M_orig)
                            )),
                        )
                        metrics["duet/budget_trim_iters"] = float(_trim_iters)
                        # 5. Make Σn_q divisible by n_gpus. DataProto.chunk()
                        # requires batch_size % n_gpus == 0 (FSDP shards must
                        # be equal). The pre-fix single-pass trim crashed
                        # D3 σ_obs at step 222 (sum=178 % 8 ≠ 0); the
                        # multi-pass helper trims first, falls back to pad-up
                        # when all entries are at floor.
                        n_gpus = int(self.config.trainer.n_gpus_per_node) * int(
                            self.config.trainer.nnodes
                        )
                        from duet.duet_allocator import (
                            make_n_q_divisible as _duet_make_div,
                        )
                        _n_q_list, _div_mode = _duet_make_div(
                            list(_duet_result.n_q),
                            n_gpus,
                            self.duet_allocator.n_min,
                        )
                        metrics["duet/divisibility_mode"] = float(
                            {"noop": 0, "trim": 1, "pad": 2, "trim+pad": 3}[_div_mode]
                        )
                        metrics["duet/n_q_total"] = float(sum(_n_q_list))
                        assert sum(_n_q_list) % n_gpus == 0, (
                            f"DUET divisibility guard: sum(n_q)={sum(_n_q_list)} "
                            f"n_gpus={n_gpus} mode={_div_mode}"
                        )

                        # 6. Build per-prompt UIDs and replicate by adjusted n_q.
                        _uids = np.array(
                            [str(uuid.uuid4()) for _ in range(_M_orig)],
                            dtype=object,
                        )
                        batch.non_tensor_batch["uid"] = _uids
                        gen_batch.non_tensor_batch["uid"] = _uids
                        from verl.protocol import DataProto as _DataProto
                        _sub_b, _sub_g = [], []
                        max_tokens_per_replica: list[int] = []
                        p_pre_per_replica: list[float] = []
                        for _i, _n in enumerate(_n_q_list):
                            _ni = int(_n)
                            if _ni <= 0:
                                continue
                            _sub_b.append(
                                batch[_i:_i + 1].repeat(repeat_times=_ni, interleave=True)
                            )
                            _sub_g.append(
                                gen_batch[_i:_i + 1].repeat(repeat_times=_ni, interleave=True)
                            )
                            max_tokens_per_replica.extend(
                                [int(_duet_result.max_tokens_q[_i])] * _ni
                            )
                            p_pre_per_replica.extend(
                                [float(_duet_result.p_pre_q[_i])] * _ni
                            )
                        if not _sub_b:
                            raise RuntimeError(
                                "DUET allocator returned all-zero n_q "
                                "(λ★ bisection collapsed)"
                            )
                        batch = _DataProto.concat(_sub_b)
                        gen_batch = _DataProto.concat(_sub_g)
                        # DUET stopping is now ENTIRELY via the LogitsProcessor
                        # (paper §5.2 / Theorem 2's adaptive per-token rule).
                        # We no longer override per-prompt max_tokens — that
                        # was an M3.b3 first-order approximation that caused
                        # cross-DP-worker tensor shape mismatches and was
                        # already a paper-divergence (T2 specifies online
                        # first-crossing, not deterministic length cap).
                        # vLLM uses the shared config.response_length, which
                        # gives uniform response shape across workers.
                        from duet.duet_logits_processor import (
                            SIGNAL_MAX_PROB, THRESHOLD_LINEAR,
                            assign_thresholds_batched,
                        )
                        signal_mode = str(self.config.duet.get(
                            "stop_signal", SIGNAL_MAX_PROB
                        )) if hasattr(self.config, "duet") else SIGNAL_MAX_PROB
                        # Floor knob (v10 calibration). Default 0.5 lands max_prob
                        # at ~0.5 and fires LP at min_tokens on every rollout —
                        # use 0.85–0.95 with linear mode, or batch_percentile to
                        # guarantee per-batch dispersion regardless of floor.
                        _stop_floor = float(self.config.duet.get("stop_floor", 0.5)
                                            if hasattr(self.config, "duet") else 0.5)
                        # V2 threshold mapping mode: linear (paper) | log_scale |
                        # batch_percentile. The latter rescues per-prompt
                        # discrimination when the linear mapping saturates.
                        _threshold_mode = str(self.config.duet.get(
                            "threshold_mode", THRESHOLD_LINEAR
                        )) if hasattr(self.config, "duet") else THRESHOLD_LINEAR
                        # Build per-prompt thresholds for the kept prompts (n_q > 0).
                        _kept_idx = [_i for _i, _n in enumerate(_n_q_list) if int(_n) > 0]
                        _per_prompt_thr = assign_thresholds_batched(
                            [float(s_hat_list[_i]) for _i in _kept_idx],
                            [float(L_hat[_i]) for _i in _kept_idx],
                            signal_mode=signal_mode,
                            threshold_mode=_threshold_mode,
                            floor=_stop_floor,
                        )
                        stop_threshold_per_replica = []
                        for _ki, _i in enumerate(_kept_idx):
                            _ni = int(_n_q_list[_i])
                            stop_threshold_per_replica.extend([_per_prompt_thr[_ki]] * _ni)
                        # Stash per-replica thresholds + signal mode for the LP.
                        gen_batch.non_tensor_batch["duet_stop_threshold"] = np.array(
                            stop_threshold_per_replica, dtype=np.float32,
                        )
                        gen_batch.meta_info["duet_stop_signal_mode"] = signal_mode
                        gen_batch.meta_info["duet_stop_threshold_mode"] = _threshold_mode
                        gen_batch.meta_info["duet_stop_eps_len"] = float(
                            self.config.duet.get("eps_len", 0.05)
                            if hasattr(self.config, "duet") else 0.05
                        )
                        gen_batch.meta_info["duet_stop_min_tokens"] = int(
                            self.config.duet.get("min_tokens", 32)
                            if hasattr(self.config, "duet") else 32
                        )
                        gen_batch.meta_info["duet_stop_top_k"] = int(
                            self.config.duet.get("top_k", 3)
                            if hasattr(self.config, "duet") else 3
                        )
                        gen_batch.meta_info["duet_stop_hysteresis_k"] = int(
                            self.config.duet.get("hysteresis_k", 1)
                            if hasattr(self.config, "duet") else 1
                        )
                        # DUET v3 marker-gated knobs (None disables → v2 LP).
                        # K1/K2 are dynamically estimated from observed
                        # natural-EOS lengths via _duet_k_estimator. Initial
                        # defaults: K1 = 0.3 × max_response, K2 = 0.7 × max_response.
                        _duet_marker_domain = (
                            str(self.config.duet.get("marker_domain", "math"))
                            if hasattr(self.config, "duet") else None
                        )
                        if _duet_marker_domain and _duet_marker_domain != "none":
                            _max_resp = int(self.config.data.max_response_length)
                            _est = getattr(self, "_duet_k_estimator", None)
                            if _est is None:
                                _est = {
                                    "lengths": [],          # observed natural-EOS lengths
                                    "current_k1": int(0.3 * _max_resp),
                                    "current_k2": int(0.7 * _max_resp),
                                    "max_history": int(self.config.duet.get(
                                        "k_history", 1024
                                    )) if hasattr(self.config, "duet") else 1024,
                                    "update_every": int(self.config.duet.get(
                                        "k_update_every", 10
                                    )) if hasattr(self.config, "duet") else 10,
                                    "last_update_step": -1,
                                }
                                self._duet_k_estimator = _est
                            gen_batch.meta_info["duet_stop_marker_domain"] = _duet_marker_domain
                            gen_batch.meta_info["duet_stop_k1"] = int(_est["current_k1"])
                            gen_batch.meta_info["duet_stop_k2"] = int(_est["current_k2"])
                            gen_batch.meta_info["duet_stop_grace"] = int(
                                self.config.duet.get("grace_window", 150)
                                if hasattr(self.config, "duet") else 150
                            )
                            gen_batch.meta_info["duet_stop_abort_eps"] = float(
                                self.config.duet.get("abort_eps", 0.05)
                                if hasattr(self.config, "duet") else 0.05
                            )
                            # Hash-based ε-keep RNG: pass base + step so the
                            # rollout worker can derive a per-row seed
                            # hash((base, step, _i)) — independent across
                            # batches even when row indices repeat.
                            gen_batch.meta_info["duet_stop_rng_seed_base"] = int(
                                self.config.duet.get("rng_seed", 0)
                                if hasattr(self.config, "duet") else 0
                            )
                            gen_batch.meta_info["duet_stop_global_step"] = int(
                                self.global_steps
                            )
                            metrics["duet/k1"] = float(_est["current_k1"])
                            metrics["duet/k2"] = float(_est["current_k2"])
                            metrics["duet/k_history_size"] = float(len(_est["lengths"]))
                        # V1+V2 metric: per-batch threshold dispersion. If max-min
                        # < 0.05 after V2 mapping, threshold isn't discriminating.
                        if _per_prompt_thr:
                            metrics["duet/threshold_max"] = float(max(_per_prompt_thr))
                            metrics["duet/threshold_min"] = float(min(_per_prompt_thr))
                            metrics["duet/threshold_mean"] = float(
                                sum(_per_prompt_thr) / len(_per_prompt_thr)
                            )
                            metrics["duet/threshold_range"] = float(
                                max(_per_prompt_thr) - min(_per_prompt_thr)
                            )
                        # Stash SNIPS weight 1/p_q^pre as a tensor in batch.batch
                        # so it survives the post-rollout `batch.union(gen_batch_output)`
                        # and is available for the post-advantage SNIPS correction
                        # in compute_advantage's caller. p_q^len = 1 for v1.b
                        # (no per-token Δ̂ stop yet); when M3.c lands the weight
                        # becomes 1/(p_q^pre · p_{q,i}^len).
                        _snips_w = torch.tensor(
                            [1.0 / max(p, self.duet_allocator.eps_pre)
                             for p in p_pre_per_replica],
                            dtype=torch.float32,
                        )
                        batch.batch["duet_snips_weight"] = _snips_w
                        # Also retain p_pre as non-tensor for offline SNIPS audit.
                        batch.non_tensor_batch["duet_p_pre"] = np.array(
                            p_pre_per_replica, dtype=np.float32,
                        )
                        _duet_did_replicate = True
                        if os.environ.get("DUET_TRACE", "0") == "1":
                            print(f"[DUET-TRACE] replicated: orig_M={_M_orig}, sum(n_q)={sum(_duet_result.n_q)}, "
                                  f"|batch.batch|={len(batch.batch)}, |gen_batch.batch|={len(gen_batch.batch)}, "
                                  f"max_tokens range=[{min(max_tokens_per_replica)}, {max(max_tokens_per_replica)}]",
                                  flush=True)
                        # Per-step metrics for the DUET dashboard.
                        _n_arr = np.asarray(_duet_result.n_q)
                        _t_arr = np.asarray(_duet_result.max_tokens_q)
                        metrics["duet/n_q_mean"] = float(_n_arr.mean())
                        metrics["duet/n_q_min"] = float(_n_arr.min())
                        metrics["duet/n_q_max"] = float(_n_arr.max())
                        metrics["duet/n_q_std"] = float(_n_arr.std())
                        metrics["duet/max_tokens_mean"] = float(_t_arr.mean())
                        metrics["duet/max_tokens_min"] = float(_t_arr.min())
                        metrics["duet/max_tokens_max"] = float(_t_arr.max())
                        metrics["duet/budget_used"] = float(_duet_result.budget_used)
                        metrics["duet/budget_target"] = float(self.config.duet.budget * B_full)
                        metrics["duet/lambda_star"] = float(_duet_result.lambda_star)
                        metrics["duet/saturation_pre"] = float(_duet_result.saturation_pre)
                        metrics["duet/saturation_len"] = float(_duet_result.saturation_len)
                        metrics["duet/feature_cache_hit_rate"] = (
                            float(n_hits) / max(1, n_hits + n_misses)
                        )
                        # Per-prompt σ_obs / L̂_q distribution metrics. Match the
                        # exact key names used by the production train.log
                        # (recovered in Phase 0): duet/s_hat_per_prompt_*,
                        # duet/L_hat_per_prompt_*, duet/n_prompts_seen, etc.
                        _s_arr = np.asarray(s_hat_list, dtype=np.float64)
                        _L_arr = np.asarray(L_hat, dtype=np.float64)
                        metrics["duet/s_hat_n_observed"] = float(n_s_observed)
                        metrics["duet/s_hat_n_cold"] = float(n_s_cold)
                        metrics["duet/s_hat_per_prompt_mean"] = float(_s_arr.mean())
                        metrics["duet/s_hat_per_prompt_std"] = float(_s_arr.std())
                        metrics["duet/s_hat_per_prompt_min"] = float(_s_arr.min())
                        metrics["duet/s_hat_per_prompt_max"] = float(_s_arr.max())
                        metrics["duet/L_hat_per_prompt_mean"] = float(_L_arr.mean())
                        metrics["duet/L_hat_per_prompt_std"] = float(_L_arr.std())
                        metrics["duet/L_hat_per_prompt_min"] = float(_L_arr.min())
                        metrics["duet/L_hat_per_prompt_max"] = float(_L_arr.max())
                        metrics["duet/n_prompts_seen"] = float(n_s_observed)
                        metrics["duet/n_prompts_cold"] = float(n_s_cold)
                        if self._duet_prompt_state is not None:
                            from duet.duet_prompt_state import (
                                stats as _duet_state_stats,
                            )
                            for _k, _v in _duet_state_stats(
                                self._duet_prompt_state).items():
                                metrics[f"duet/{_k}"] = _v
                    except Exception as _duet_err:  # noqa: BLE001
                        import traceback as _tb
                        metrics.setdefault("duet/replicate_failed", 1.0)
                        print(f"[ray_trainer] DUET pre-rollout failed: {_duet_err}", flush=True)
                        print(f"[DUET-TRACE-ERR] {_tb.format_exc()}", flush=True)
                        _duet_did_replicate = False

                # ARRoL faithful (paper §4.3) cross-process bridge: pickle head
                # state into meta_info so the rollout-worker can reconstruct it
                # and use it as the patch callback. Decision tuples come back
                # via non_tensor_batch["arrol_faithful_decisions"] for label
                # collection + online BCE training after verifier rewards land.
                if self.arrol_faithful_head is not None:
                    import pickle as _pkl
                    head_blob = _pkl.dumps(self.arrol_faithful_head.state_dict())
                    cfg_blob = _pkl.dumps(vars(self.arrol_faithful_cfg))
                    gen_batch.meta_info["arrol_faithful_active"] = True
                    gen_batch.meta_info["arrol_faithful_head_blob"] = head_blob
                    gen_batch.meta_info["arrol_faithful_cfg_blob"] = cfg_blob
                    gen_batch.meta_info["arrol_faithful_l_detect"] = (
                        self.arrol_faithful_cfg.l_detect
                    )
                    # Cross-process: ship _step_count too so the rollout-worker
                    # head doesn't get reconstructed in eternal warmup. Without
                    # this, decide_keep always returns True regardless of head
                    # output → n_pruned=0 even post-warmup. Confirmed in
                    # arrol_tiny v1 where head_loss trained but n_pruned stayed 0.
                    gen_batch.meta_info["arrol_faithful_step_count"] = int(
                        self.arrol_faithful_head._step_count
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if self.config.get("ttrl", {}).get("enable", False):
                            sirl_enabled = self.config.get("sirl", {}).get("enable", False)
                            if sirl_enabled:
                                # SIRL does not need majority voting or oversampling —
                                # generate exactly rollout.n responses directly.
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                from verl.trainer.ppo.ttrl_utils import select_top_k_per_prompt, apply_ttrl_gt

                                gen_batch.meta_info["kwargs"] = {"n": self.config.ttrl.n_votes_per_prompt}
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                                assert len(gen_batch_output) == len(batch) * self.config.ttrl.n_votes_per_prompt

                                batch = apply_ttrl_gt(batch, gen_batch_output, self.config.ttrl.n_votes_per_prompt, self.tokenizer)
                                gen_batch_output = select_top_k_per_prompt(gen_batch_output, self.config.ttrl.n_votes_per_prompt, self.config.ttrl.n_samples_per_prompt)

                                assert len(gen_batch_output) == len(batch) * self.config.ttrl.n_samples_per_prompt
                        else:
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                # vllm should set async_rollout_mode to enable async rollout
                                # sglang turns on async_rollout_mode by default
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    if (not repeat_sampling_sglang_grpo
                            and not _vip_did_replicate
                            and not _duet_did_replicate):
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )
                        # repeat to align with repeated responses in rollout.
                        # When TTRL is enabled, gen_batch_output has n_samples_per_prompt
                        # responses per prompt (after select_top_k), which may differ
                        # from rollout.n.
                        ttrl_cfg = self.config.get("ttrl", {})
                        if ttrl_cfg.get("enable", False) and not self.config.get("sirl", {}).get("enable", False):
                            repeat_n = ttrl_cfg.get("n_samples_per_prompt", self.config.actor_rollout_ref.rollout.n)
                        else:
                            repeat_n = self.config.actor_rollout_ref.rollout.n
                        batch = batch.repeat(repeat_times=repeat_n, interleave=True)

                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ============================================================
                    # Reward fast-path: swap old_log_prob BEFORE reward, then gate
                    # on p_self (InPo) or a quality-head (ARRoL) BEFORE the reward
                    # call. Compute_reward runs only on KEPT rollouts, saving
                    # verifier budget proportional to the skip rate.
                    #
                    # Dispatch precedence:
                    #   InPo fast-path — BSV enabled in GRPO frame, single-judge
                    #     backend = response_logprob (derived from old_log_prob,
                    #     no extra forward pass).
                    #   ARRoL fast-path — arrol_head configured AND post-warmup.
                    #     During warmup, fall back to original order so the head
                    #     trains on all rollouts before pruning fires.
                    #   Fallback — original order (reward → old_log_prob).
                    _bsv_cfg_fp = (
                        self.config.get("bsv", {}) if hasattr(self.config, "get") else {}
                    )
                    _bsv_grpo_fp_eligible = (
                        self.bsv_gate is not None
                        and str(self.config.algorithm.adv_estimator).lower().endswith("grpo")
                        and not (
                            hasattr(self.config, "cpr")
                            and bool(getattr(self.config.cpr, "enable", False))
                        )
                        and str(_bsv_cfg_fp.get("single_judge_mode", "response_logprob"))
                        == "response_logprob"
                    )
                    _arrol_fp_eligible = (
                        self.arrol_head is not None
                        and not self.arrol_head.in_warmup
                        and not _bsv_grpo_fp_eligible  # BSV takes precedence
                    )
                    # State set by InPo fast-path when taken. Consumed by the
                    # post-reward Frame A block to avoid a duplicate
                    # bsv_gate.decide_batch (which would double-count state).
                    _bsv_grpo_fastpath_used = False
                    _bsv_fp_I_v = None
                    _bsv_fp_p_inv = None
                    _bsv_fp_p_self = None
                    _bsv_fp_reward_calls_saved = 0
                    _bsv_fp_reward_calls_made = 0
                    # State set by ARRoL fast-path when taken. Consumed by the
                    # post-reward ARRoL block to avoid re-predicting and to feed
                    # labels back to head training on the kept subset.
                    _arrol_fp_used = False
                    _arrol_fp_feat_t = None
                    _arrol_fp_keep_a = None
                    _arrol_fp_probs_a = None

                    if _bsv_grpo_fp_eligible:
                        # --- Step 1: compute old_log_prob FIRST so we can derive p_self ---
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]
                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                metrics.update({
                                    "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
                                    "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
                                    "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
                                })

                        # --- Step 2: p_self from old_log_probs → gate → I_v ---
                        from verl.trainer.ppo.bsv_single_judge import (
                            SingleJudgeConfig,
                            score_rollouts,
                        )
                        _olp_np_fp = batch.batch["old_log_probs"].detach().cpu().numpy().astype(np.float32)
                        _rm_np_fp = batch.batch["response_mask"].detach().cpu().numpy().astype(np.float32)
                        _step_fp = getattr(self, "global_steps", 0)
                        _sj_cfg_fp = SingleJudgeConfig(
                            mode="response_logprob",
                            noise=float(_bsv_cfg_fp.get("single_judge_noise", 0.15)),
                            rng_seed=int(_bsv_cfg_fp.get("rng_seed", 0)) + int(_step_fp),
                        )
                        _N_fp = _olp_np_fp.shape[0]
                        _bsv_fp_p_self = score_rollouts(
                            [""] * _N_fp, [""] * _N_fp, _sj_cfg_fp,
                            old_log_probs=_olp_np_fp, response_mask=_rm_np_fp,
                        )
                        _bsv_fp_I_v, _bsv_fp_p_inv = self.bsv_gate.decide_batch(_bsv_fp_p_self)
                        _bsv_grpo_fastpath_used = True

                        # --- Step 3: compute_reward ONLY on I_v=1 subset ---
                        with marked_timer("reward", timing_raw, color="yellow"):
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)
                            _keep_fp = np.where(_bsv_fp_I_v == 1)[0]
                            _T_resp_fp = batch.batch["responses"].shape[1]
                            _dev_fp = batch.batch["responses"].device
                            reward_tensor = torch.zeros(
                                (_N_fp, _T_resp_fp), dtype=torch.float32, device=_dev_fp,
                            )
                            if _keep_fp.size > 0:
                                _sub_batch = batch.select_idxs(_keep_fp.tolist())
                                if self.config.reward_model.launch_reward_fn_async:
                                    future_reward = compute_reward_async.remote(
                                        _sub_batch, self.config, self.tokenizer,
                                    )
                                    reward_extra_infos_dict = {}
                                else:
                                    _sub_reward, reward_extra_infos_dict = compute_reward(
                                        _sub_batch, self.reward_fn,
                                    )
                                    reward_tensor[_keep_fp] = _sub_reward.to(_dev_fp)
                            else:
                                reward_extra_infos_dict = {}
                            _bsv_fp_reward_calls_made = int(_keep_fp.size)
                            _bsv_fp_reward_calls_saved = int(_N_fp - _keep_fp.size)
                            metrics["bsv_grpo/reward_calls_made"] = float(_bsv_fp_reward_calls_made)
                            metrics["bsv_grpo/reward_calls_saved"] = float(_bsv_fp_reward_calls_saved)
                            metrics["bsv_grpo/reward_calls_saved_frac"] = (
                                float(_bsv_fp_reward_calls_saved) / float(max(1, _N_fp))
                            )
                    elif _arrol_fp_eligible:
                        # --- ARRoL fast-path: Step 1: old_log_prob BEFORE reward ---
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]
                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                metrics.update({
                                    "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
                                    "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
                                    "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
                                })

                        # --- ARRoL fast-path: Step 2: extract features + predict ---
                        _arrol_L = int(self._arrol_l_detect)
                        _arrol_olp = batch.batch["old_log_probs"].detach().float().cpu().numpy()
                        _arrol_rmk = batch.batch["response_mask"].detach().float().cpu().numpy()
                        _arrol_T_resp = _arrol_olp.shape[-1]
                        _arrol_L_eff = min(_arrol_L, _arrol_T_resp)
                        _arrol_B = _arrol_olp.shape[0]
                        _arrol_feat = np.zeros(
                            (_arrol_B, self._arrol_feature_dim), dtype=np.float32,
                        )
                        for _bi in range(_arrol_B):
                            _mask_bi = _arrol_rmk[_bi, :_arrol_L_eff] > 0
                            if _mask_bi.sum() < 1:
                                continue
                            _lp_valid = _arrol_olp[_bi, :_arrol_L_eff][_mask_bi]
                            _arrol_feat[_bi, 0] = float(_lp_valid.mean())
                            _arrol_feat[_bi, 1] = float(_lp_valid.std())
                            _arrol_feat[_bi, 2] = float(_lp_valid.min())
                            _arrol_feat[_bi, 3] = float(_lp_valid.max())
                        _arrol_fp_feat_t = torch.from_numpy(_arrol_feat)
                        # Paper-aligned protocol: forward_raw → calibrator → sample_survival
                        _arrol_raw_np = self.arrol_head.forward_raw(_arrol_fp_feat_t)
                        if self.arrol_head.calibrator.n_observed() >= 16:
                            _arrol_fp_probs_a = self.arrol_head.calibrator.calibrate(_arrol_raw_np)
                        else:
                            # Cold-start: not enough history for binned estimator
                            _arrol_fp_probs_a = _arrol_raw_np
                        _arrol_fp_keep_a = self.arrol_head.sample_survival(_arrol_fp_probs_a)
                        _arrol_fp_used = True

                        # --- ARRoL fast-path: Step 3: compute_reward on KEEP subset ---
                        with marked_timer("reward", timing_raw, color="yellow"):
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)
                            _arrol_keep_idx = np.where(_arrol_fp_keep_a)[0]
                            _dev_arrol = batch.batch["responses"].device
                            reward_tensor = torch.zeros(
                                (_arrol_B, _arrol_T_resp), dtype=torch.float32, device=_dev_arrol,
                            )
                            if _arrol_keep_idx.size > 0:
                                _sub_batch_a = batch.select_idxs(_arrol_keep_idx.tolist())
                                if self.config.reward_model.launch_reward_fn_async:
                                    future_reward = compute_reward_async.remote(
                                        _sub_batch_a, self.config, self.tokenizer,
                                    )
                                    reward_extra_infos_dict = {}
                                else:
                                    _sub_reward_a, reward_extra_infos_dict = compute_reward(
                                        _sub_batch_a, self.reward_fn,
                                    )
                                    reward_tensor[_arrol_keep_idx] = _sub_reward_a.to(_dev_arrol)
                            else:
                                reward_extra_infos_dict = {}
                            metrics["arrol/reward_calls_made"] = float(_arrol_keep_idx.size)
                            metrics["arrol/reward_calls_saved"] = float(_arrol_B - _arrol_keep_idx.size)
                            metrics["arrol/reward_calls_saved_frac"] = (
                                float(_arrol_B - _arrol_keep_idx.size) / float(max(1, _arrol_B))
                            )
                    else:
                        # Original path: reward first, then old_log_prob.
                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]
                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                metrics.update({
                                    "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
                                    "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
                                    "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
                                })

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        # SIRL: override rewards with self-improvement reward computation
                        if self.config.get("sirl", {}).get("enable", False):
                            from verl.trainer.ppo.sirl_utils import apply_sirl_rewards, compute_sirl_metrics
                            from omegaconf import open_dict
                            sirl_cfg = self.config.sirl
                            with open_dict(sirl_cfg):
                                # Follow-up prompts embed the original response as context,
                                # so they can be much longer than the original prompt.
                                # Use max_prompt_length + max_response_length as the budget,
                                # capped at max_model_len.
                                max_model_len = self.config.actor_rollout_ref.rollout.get(
                                    "max_model_len", 4096)
                                sirl_cfg.max_prompt_length = min(
                                    self.config.data.max_prompt_length + self.config.data.max_response_length,
                                    max_model_len,
                                )
                                sirl_cfg.max_response_length = self.config.data.max_response_length
                            n_sirl = self.config.get("ttrl", {}).get("n_samples_per_prompt", self.config.actor_rollout_ref.rollout.n)

                            sirl_distill_enabled = sirl_cfg.get("distill", {}).get("enable", False)

                            if sirl_distill_enabled:
                                # ── Distillation pathway: self-supervised gating + token targets ──
                                from verl.trainer.ppo.sirl_utils import compute_sirl_distill_data
                                sirl_rewards, sirl_extra, distill_data = compute_sirl_distill_data(
                                    batch=batch,
                                    gen_batch_output=batch,
                                    n_samples=n_sirl,
                                    tokenizer=self.tokenizer,
                                    actor_rollout_wg=self.actor_rollout_wg,
                                    sirl_config=sirl_cfg,
                                )
                                # Add distillation tensors to batch for the actor update
                                for dkey, dtensor in distill_data.items():
                                    batch.batch[dkey] = dtensor
                                batch.meta_info["sirl_distill_enabled"] = True
                                batch.meta_info["sirl_distill_loss_coeff"] = float(
                                    sirl_cfg.get("distill", {}).get("loss_coeff", 1.0))
                            else:
                                # ── Original scalar-reward pathway ───────────────────────────────
                                sirl_rewards, sirl_extra = apply_sirl_rewards(
                                    batch=batch,
                                    gen_batch_output=batch,
                                    n_samples=n_sirl,
                                    tokenizer=self.tokenizer,
                                    actor_rollout_wg=self.actor_rollout_wg,
                                    sirl_config=sirl_cfg,
                                )

                            # Place SIRL rewards at last valid response token (same as base reward)
                            sirl_reward_tensor = torch.zeros_like(reward_tensor)
                            response_length = batch.batch["responses"].shape[1]
                            prompt_length = batch.batch["prompts"].shape[1]
                            attn_mask = batch.batch["attention_mask"]
                            for si in range(len(sirl_rewards)):
                                valid_resp_len = int(attn_mask[si, prompt_length:].sum().item())
                                if valid_resp_len > 0:
                                    sirl_reward_tensor[si, valid_resp_len - 1] = float(sirl_rewards[si])
                            batch.batch["token_level_scores"] = sirl_reward_tensor
                            reward_tensor = sirl_reward_tensor
                            sirl_metrics = compute_sirl_metrics(sirl_rewards, n_sirl, sirl_extra)
                            metrics.update(sirl_metrics)

                            # Store per-component rewards for SIRL per-component advantage (S3/S4)
                            batch.batch["sirl_imp_rewards"] = torch.tensor(
                                sirl_extra["imp_rewards"], dtype=torch.float32,
                            )
                            if "alt_rewards" in sirl_extra:
                                batch.batch["sirl_alt_rewards"] = torch.tensor(
                                    sirl_extra["alt_rewards"], dtype=torch.float32,
                                )
                            if "det_rewards" in sirl_extra:
                                batch.batch["sirl_det_rewards"] = torch.tensor(
                                    sirl_extra["det_rewards"], dtype=torch.float32,
                                )
                            batch.meta_info["sirl_mode"] = sirl_cfg.get("mode", "default")
                            batch.meta_info["sirl_adv_w_imp"] = float(sirl_cfg.get("adv_w_imp", 0.6))
                            batch.meta_info["sirl_lambda_alt"] = float(sirl_cfg.get("lambda_alt", 0.2))
                            batch.meta_info["sirl_lambda_det"] = float(sirl_cfg.get("lambda_det", 0.1))

                        # =====================================================
                        # SIRL-Pref (additive, runs in parallel to SIRL).
                        # Requires: `+sirl_pref.enable=True` in the config.
                        # Does NOT modify the existing SIRL reward pathway above.
                        # =====================================================
                        if self.config.get("sirl_pref", {}).get("enable", False):
                            from verl.trainer.ppo.sirl_pref_utils import (
                                SirlPrefConfig,
                                apply_sirl_pref_rewards,
                            )
                            _spc = self.config.sirl_pref
                            # Compute max_prompt_length the same way existing SIRL
                            # does: prompt_length + response_length, capped at
                            # max_model_len minus a small slack for the
                            # revise/judge instructions.
                            _max_model_len = self.config.actor_rollout_ref.rollout.get(
                                "max_model_len", 4096,
                            )
                            _sp_max_resp = int(self.config.data.get("max_response_length", 1024))
                            _sp_max_prompt = int(self.config.data.get("max_prompt_length", 1024))
                            _sp_slack = 128  # room for IMPROVE instruction / chat template
                            _sp_budget = max(
                                256,
                                min(_sp_max_prompt + _sp_max_resp, _max_model_len - _sp_max_resp - _sp_slack),
                            )
                            sirl_pref_cfg = SirlPrefConfig(
                                enable=True,
                                max_prompt_length=_sp_budget,
                                max_response_length=_sp_max_resp,
                                arm_id=_spc.get("arm_id", "dpo_prefix_self"),
                                loss=_spc.get("loss", "dpo"),
                                pair_construction=_spc.get("pair_construction", "prefix_branch"),
                                preference_source=_spc.get("preference_source", "self"),
                                branch_min_frac=float(_spc.get("branch_min_frac", 0.35)),
                                branch_max_frac=float(_spc.get("branch_max_frac", 0.65)),
                                branch_min_suffix_tokens=int(_spc.get("branch_min_suffix_tokens", 32)),
                                revise_temperature_delta=float(_spc.get("revise_temperature_delta", 0.20)),
                                judge_order_swap=bool(_spc.get("judge_order_swap", True)),
                                judge_max_tokens=int(_spc.get("judge_max_tokens", 8)),
                                ema_judge_refresh_steps=int(_spc.get("ema_judge_refresh_steps", 100)),
                                tie_band_eps=float(_spc.get("tie_band_eps", 0.15)),
                                min_divergence_threshold=float(_spc.get("min_divergence_threshold", 0.12)),
                                length_ratio_cap=float(_spc.get("length_ratio_cap", 1.50)),
                                length_penalty_weight=float(_spc.get("length_penalty_weight", 0.10)),
                                length_penalty_warmup_steps=int(_spc.get("length_penalty_warmup_steps", 100)),
                                external_margin_floor=float(_spc.get("external_margin_floor", 0.50)),
                                dpo_beta=float(_spc.get("dpo_beta", 0.15)),
                                anchor_weight=float(_spc.get("anchor_weight", 0.05)),
                                grpo_reject_coef=float(_spc.get("grpo_reject_coef", 0.50)),
                                kl_coef=float(_spc.get("kl_coef", 0.02)),
                                kl_channel=_spc.get("kl_channel", "y0_only"),
                                rollout_n=int(self.config.actor_rollout_ref.rollout.get("n", 1)),
                            )
                            sirl_pref_out = apply_sirl_pref_rewards(
                                batch=batch,
                                tokenizer=self.tokenizer,
                                actor_rollout_wg=self.actor_rollout_wg,
                                cfg=sirl_pref_cfg,
                                step=self.global_steps if hasattr(self, "global_steps") else 0,
                                oracle_reward_fn=None,  # wired by ablation arms in a follow-up
                            )
                            # Package into batch for the advantage estimator and any
                            # custom DPO loss hook.
                            if sirl_pref_out:
                                B = batch.batch["prompts"].shape[0]
                                batch.batch["sirl_pref_pair_advantage_raw"] = torch.tensor(
                                    sirl_pref_out["pair_advantage_raw"], dtype=torch.float32,
                                )
                                batch.batch["sirl_pref_pair_weight"] = torch.tensor(
                                    sirl_pref_out["pair_weight"], dtype=torch.float32,
                                )
                                batch.batch["sirl_pref_label"] = torch.tensor(
                                    sirl_pref_out["label"], dtype=torch.int64,
                                )
                                batch.batch["sirl_pref_skip_mask"] = torch.tensor(
                                    sirl_pref_out["skip_mask"], dtype=torch.bool,
                                )
                                batch.non_tensor_batch["sirl_pref_pair_id"] = np.arange(B, dtype=np.int64)
                                batch.non_tensor_batch["sirl_pref_channel"] = np.array(
                                    ["original"] * B, dtype=object,
                                )
                                batch.meta_info["sirl_pref_enabled"] = True
                                batch.meta_info["sirl_pref_loss"] = sirl_pref_cfg.loss
                                batch.meta_info["sirl_pref_grpo_reject_coef"] = float(
                                    sirl_pref_cfg.grpo_reject_coef,
                                )
                                batch.meta_info["sirl_pref_arm_id"] = sirl_pref_cfg.arm_id
                                batch.meta_info["sirl_pref_dpo_beta"] = float(sirl_pref_cfg.dpo_beta)
                                batch.meta_info["sirl_pref_anchor_weight"] = float(sirl_pref_cfg.anchor_weight)
                                batch.meta_info["sirl_pref_kl_coef"] = float(sirl_pref_cfg.kl_coef)
                                # Emit watchdog metrics
                                metrics.update({
                                    "sirl_pref/tie_rate": sirl_pref_out["tie_rate"],
                                    "sirl_pref/disagree_rate": sirl_pref_out["disagree_rate"],
                                    "sirl_pref/len_ratio_mean": sirl_pref_out["len_ratio_mean"],
                                    "sirl_pref/answer_flip_rate": sirl_pref_out["answer_flip_rate"],
                                })

                                # --- DPO logprob plumbing --------------------
                                # For the DPO arm, run log-prob computation on
                                # the paired (x+prefix, s±) DataProtos and stash
                                # per-token logps onto batch.batch so dp_actor
                                # can consume them in its DPO loss path.
                                if sirl_pref_cfg.loss == "dpo":
                                    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
                                    rev_dp = sirl_pref_out["revised_paired_dataproto"]
                                    orig_dp = sirl_pref_out["original_paired_dataproto"]

                                    def _lp(dp, wg, method_name):
                                        # Pad → dispatch → unpad, matching the
                                        # pattern in _generate_follow_ups_with_tokens.
                                        dp_padded, pad_size = pad_dataproto_to_divisor(
                                            dp, wg.world_size,
                                        )
                                        out = getattr(wg, method_name)(dp_padded)
                                        return unpad_dataproto(out, pad_size)

                                    rev_lp_dp = _lp(rev_dp, self.actor_rollout_wg, "compute_log_prob")
                                    orig_lp_dp = _lp(orig_dp, self.actor_rollout_wg, "compute_log_prob")
                                    # Reference log-probs from ref policy if available.
                                    rev_ref_lp_dp = None
                                    orig_ref_lp_dp = None
                                    if self.use_reference_policy:
                                        ref_wg = self.actor_rollout_wg if self.ref_in_actor else self.ref_policy_wg
                                        rev_ref_lp_dp = _lp(rev_dp, ref_wg, "compute_ref_log_prob")
                                        orig_ref_lp_dp = _lp(orig_dp, ref_wg, "compute_ref_log_prob")
                                    # Stash per-token logps. Shapes are
                                    # [B, T_rev] and [B, T_orig]; loss masks are
                                    # the response-portion of attention_mask.
                                    batch.batch["sirl_pref_revised_logprobs"] = rev_lp_dp.batch["old_log_probs"]
                                    batch.batch["sirl_pref_original_logprobs"] = orig_lp_dp.batch["old_log_probs"]
                                    batch.batch["sirl_pref_revised_loss_mask"] = (
                                        rev_dp.batch["attention_mask"][:, rev_dp.batch["prompts"].shape[1]:].to(torch.float32)
                                    )
                                    batch.batch["sirl_pref_original_loss_mask"] = (
                                        orig_dp.batch["attention_mask"][:, orig_dp.batch["prompts"].shape[1]:].to(torch.float32)
                                    )
                                    if rev_ref_lp_dp is not None:
                                        batch.batch["sirl_pref_revised_ref_logprobs"] = rev_ref_lp_dp.batch["ref_log_prob"]
                                        batch.batch["sirl_pref_original_ref_logprobs"] = orig_ref_lp_dp.batch["ref_log_prob"]
                                    else:
                                        # Degenerate case: DPO without reference is just the margin
                                        # without the -logπref term. Use zero ref logps.
                                        batch.batch["sirl_pref_revised_ref_logprobs"] = torch.zeros_like(
                                            rev_lp_dp.batch["old_log_probs"]
                                        )
                                        batch.batch["sirl_pref_original_ref_logprobs"] = torch.zeros_like(
                                            orig_lp_dp.batch["old_log_probs"]
                                        )
                                    batch.meta_info["sirl_pref_dpo_logprobs_ready"] = True

                        # =====================================================
                        # CPR — Calibrated Preference RL (additive; mutually
                        # exclusive with SIRL-Pref — enforced at CLI level).
                        # Requires: `+cpr.enable=True` in the config.
                        # =====================================================
                        if self.config.get("cpr", {}).get("enable", False):
                            from verl.trainer.ppo.cpr_utils import CPRConfig, apply_cpr_rewards

                            _cpc = self.config.cpr
                            _max_model_len = self.config.actor_rollout_ref.rollout.get("max_model_len", 4096)
                            _cpr_max_resp = int(self.config.data.get("max_response_length", 1024))
                            _cpr_max_prompt = int(self.config.data.get("max_prompt_length", 1024))
                            _cpr_slack = 128
                            _cpr_budget = max(
                                256,
                                min(_cpr_max_prompt + _cpr_max_resp, _max_model_len - _cpr_max_resp - _cpr_slack),
                            )
                            cpr_cfg = CPRConfig(
                                enable=True,
                                alpha=float(_cpc.get("alpha", 0.5)),
                                beta=float(_cpc.get("beta", 0.15)),
                                eta=float(_cpc.get("eta", 0.0)),
                                pair_source=_cpc.get("pair_source", "independent"),
                                rollout_n=int(_cpc.get("rollout_n", self.config.actor_rollout_ref.rollout.get("n", 4))),
                                tau_ext=float(_cpc.get("tau_ext", 1.0)),
                                judge_model=_cpc.get("judge_model", "online"),
                                judge_ema_tau=float(_cpc.get("judge_ema_tau", 0.99)),
                                judge_ema_refresh_steps=int(_cpc.get("judge_ema_refresh_steps", 100)),
                                judge_max_new_tokens=int(_cpc.get("judge_max_new_tokens", 1)),
                                judge_order_swap=bool(_cpc.get("judge_order_swap", True)),
                                label_noise_rho=float(_cpc.get("label_noise_rho", 0.0)),
                                arm_id=_cpc.get("arm_id", "cpr_alpha0.5"),
                                max_prompt_length=_cpr_budget,
                                max_response_length=_cpr_max_resp,
                                anchor_mode=_cpc.get("anchor_mode", "symmetric"),
                                alpha_gate_gamma=float(_cpc.get("alpha_gate_gamma", 0.0)),
                            )

                            # H3 side-channel: any response whose verifier failed
                                # (e.g. openai 401/429) flips this flag, and the BSV
                                # overlay then treats the batch as verifier-absent
                                # rather than emitting propensity rows against
                                # fabricated p_verif derived from a 0.0 reward.
                            self._cpr_oracle_had_failure = False

                            def _cpr_oracle_fn(eval_dp):
                                try:
                                    reward_out = self.reward_fn(eval_dp, return_dict=True)
                                    reward_tensor = reward_out.get("reward_tensor", None)
                                    rei = reward_out.get("reward_extra_info", None)
                                    if rei is not None:
                                        failed_flags = rei.get("verifier_failed", None)
                                        if failed_flags and any(bool(f) for f in failed_flags):
                                            self._cpr_oracle_had_failure = True
                                except TypeError:
                                    reward_tensor = self.reward_fn(eval_dp)
                                if reward_tensor is None:
                                    return np.zeros(len(eval_dp), dtype=np.float32)
                                if reward_tensor.ndim == 2:
                                    scores = reward_tensor.sum(dim=-1)
                                else:
                                    scores = reward_tensor
                                return scores.detach().cpu().numpy().astype(np.float32)

                            oracle_reward_fn = None
                            if cpr_cfg.alpha < 1.0 and self.reward_fn is not None:
                                oracle_reward_fn = _cpr_oracle_fn

                            cpr_out = apply_cpr_rewards(
                                batch=batch,
                                tokenizer=self.tokenizer,
                                actor_rollout_wg=self.actor_rollout_wg,
                                cfg=cpr_cfg,
                                step=self.global_steps if hasattr(self, "global_steps") else 0,
                                oracle_reward_fn=oracle_reward_fn,
                            )

                            if cpr_out:
                                # Re-import locally: the SIRL-Pref branch above also imports
                                # these names inside its `if` body, which causes Python's
                                # compiler to bind them as locals of the enclosing function
                                # and shadow the module-level imports. When SIRL-Pref's
                                # branch is skipped (as it is here for CPR), the locals are
                                # unbound and the closure below would NameError.
                                from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

                                y_a_dp = cpr_out["y_a_dp"]
                                y_b_dp = cpr_out["y_b_dp"]

                                def _cpr_lp(dp, wg, method_name):
                                    dp_padded, pad_size = pad_dataproto_to_divisor(dp, wg.world_size)
                                    out = getattr(wg, method_name)(dp_padded)
                                    return unpad_dataproto(out, pad_size)

                                # NOTE on stashed log-probs: dp_actor.update_policy runs a
                                # fresh forward over (cpr_a_input_ids, cpr_b_input_ids) for
                                # the gradient-bearing log-probs, so the precomputed
                                # `cpr_{a,b}_logprobs` are never read on the loss path. We
                                # also skip the precomputed ref log-probs under LoRA: with
                                # FSDP + use_orig_params=True, calling compute_log_prob /
                                # compute_ref_log_prob from inside the training step on a
                                # paired DataProto (different shape than the rollout batch)
                                # leaves the actor's orig_params in their 2D eager view, and
                                # the next FSDP `summon_full_params(writeback=False)` (e.g.
                                # the post-step validation generate) crashes with
                                # "Cannot writeback when the parameter shape changes".
                                # Until that LoRA+FSDP interaction is fixed upstream, do
                                # NOT call _cpr_lp here at all under LoRA — drop the
                                # ref/anchor term and run CPR as no-ref soft-label DPO
                                # (still a valid training signal; KL is implicit through
                                # the policy update only). When LoRA is off, the standard
                                # ref + actor compute_log_prob path is safe.
                                _lora_rank = int(
                                    self.config.actor_rollout_ref.get("model", {}).get("lora_rank", 0)
                                    or 0
                                )
                                _skip_cpr_aux_lp_under_lora = _lora_rank > 0

                                if not _skip_cpr_aux_lp_under_lora:
                                    a_lp_dp = _cpr_lp(y_a_dp, self.actor_rollout_wg, "compute_log_prob")
                                    b_lp_dp = _cpr_lp(y_b_dp, self.actor_rollout_wg, "compute_log_prob")
                                    cpr_a_logprobs = a_lp_dp.batch["old_log_probs"]
                                    cpr_b_logprobs = b_lp_dp.batch["old_log_probs"]
                                else:
                                    # Build zero stand-ins shaped like the response slice so
                                    # downstream stash code (and any future consumer) sees a
                                    # well-formed tensor.
                                    _resp_len_a = y_a_dp.batch["responses"].shape[1]
                                    _resp_len_b = y_b_dp.batch["responses"].shape[1]
                                    cpr_a_logprobs = torch.zeros(
                                        y_a_dp.batch["responses"].shape[0], _resp_len_a, dtype=torch.float32
                                    )
                                    cpr_b_logprobs = torch.zeros(
                                        y_b_dp.batch["responses"].shape[0], _resp_len_b, dtype=torch.float32
                                    )
                                    metrics["cpr/aux_lp_skipped_under_lora"] = 1.0

                                if self.use_reference_policy and not _skip_cpr_aux_lp_under_lora:
                                    ref_wg = self.actor_rollout_wg if self.ref_in_actor else self.ref_policy_wg
                                    a_ref_lp_dp = _cpr_lp(y_a_dp, ref_wg, "compute_ref_log_prob")
                                    b_ref_lp_dp = _cpr_lp(y_b_dp, ref_wg, "compute_ref_log_prob")
                                    cpr_a_ref_logprobs = a_ref_lp_dp.batch["ref_log_prob"]
                                    cpr_b_ref_logprobs = b_ref_lp_dp.batch["ref_log_prob"]
                                else:
                                    cpr_a_ref_logprobs = torch.zeros_like(cpr_a_logprobs)
                                    cpr_b_ref_logprobs = torch.zeros_like(cpr_b_logprobs)
                                    if _skip_cpr_aux_lp_under_lora:
                                        metrics["cpr/ref_skipped_under_lora"] = 1.0
                                    else:
                                        metrics["cpr/ref_policy_absent"] = 1.0

                                cpr_a_loss_mask = y_a_dp.batch["attention_mask"][:, y_a_dp.batch["prompts"].shape[1]:].to(torch.float32)
                                cpr_b_loss_mask = y_b_dp.batch["attention_mask"][:, y_b_dp.batch["prompts"].shape[1]:].to(torch.float32)

                                # ---- BSV overlay (DEFINE_SPEC_v3 §1.2/§3) ----
                                # When the BSV gate is active, α stops being the
                                # soft blend coefficient and becomes the autonomy
                                # fraction: gate per pair on self-judge confidence
                                # c = 2|p_self-0.5|, thresholded at the adaptive
                                # (1-α)-quantile, with ε exploration floor. The
                                # kernel then collapses p_pref to the per-pair
                                # I_v·p_verif + (1-I_v)·p_self formula (compute_bsv_loss).
                                _bsv_on = self.bsv_gate is not None
                                if _bsv_on:
                                    _p_self_vec = np.asarray(cpr_out["p_self"], dtype=np.float32)
                                    _raw_p_verif = cpr_out.get("p_verif", None)
                                    # Verifier-absent handling (DEFINE_SPEC_v3 §3.1): when
                                    # r_ext is unavailable, the gate's I_v·p_verif term
                                    # cannot be populated without fabricating a signal. We
                                    # raise a loud metric, set p_pref := p_self, and SKIP
                                    # propensity row emission — writing rows here would
                                    # silently bias the IPS/DR evaluators because the
                                    # "verified reward" column would be missing.
                                    # H3: also treat oracle-level verifier failures
                                    # (e.g. OpenAI 401/429) as verifier-absent. Without
                                    # this, failed samples would contribute score=0.0
                                    # scores whose σ((R_a−R_b)/τ) becomes a non-None
                                    # `p_verif` — silently biasing the propensity CSV
                                    # that the IPS/DR evaluators consume.
                                    _verifier_failed_batch = bool(getattr(self, "_cpr_oracle_had_failure", False))
                                    _verifier_absent = (_raw_p_verif is None) or _verifier_failed_batch
                                    if _verifier_absent:
                                        _p_verif_vec = _p_self_vec.copy()
                                    else:
                                        _p_verif_vec = np.asarray(_raw_p_verif, dtype=np.float32)
                                    _I_v_vec, _p_inv_vec = self.bsv_gate.decide_batch(_p_self_vec)
                                    _I_v_f = _I_v_vec.astype(np.float32)
                                    if _verifier_absent:
                                        # Force p_pref = p_self — the gate's decision is recorded
                                        # in the tracker (for quantile adaptation) but NOT used to
                                        # route a non-existent verifier signal.
                                        _bsv_p_pref = _p_self_vec.astype(np.float32)
                                    else:
                                        _bsv_p_pref = (
                                            _I_v_f * _p_verif_vec + (1.0 - _I_v_f) * _p_self_vec
                                        ).astype(np.float32)
                                    batch.batch["cpr_p_pref"] = torch.tensor(_bsv_p_pref, dtype=torch.float32)
                                    batch.batch["bsv_p_self"] = torch.tensor(_p_self_vec, dtype=torch.float32)
                                    batch.batch["bsv_p_verif"] = torch.tensor(_p_verif_vec, dtype=torch.float32)
                                    batch.batch["bsv_I_v"] = torch.tensor(_I_v_f, dtype=torch.float32)
                                    batch.batch["bsv_p_invoke"] = torch.tensor(
                                        _p_inv_vec.astype(np.float32), dtype=torch.float32
                                    )
                                    if not _verifier_absent:
                                        _step_now = self.global_steps if hasattr(self, "global_steps") else 0
                                        _pair_ids_arr = np.asarray(cpr_out["pair_ids"], dtype=np.int64).ravel()
                                        _r_ext = cpr_out.get("r_ext", None)
                                        _r_a = None if _r_ext is None else np.asarray(_r_ext["a"], dtype=np.float32)
                                        _r_b = None if _r_ext is None else np.asarray(_r_ext["b"], dtype=np.float32)
                                        for _i in range(_p_self_vec.shape[0]):
                                            self._bsv_propensity_rows.append({
                                                "step": int(_step_now),
                                                "pair_idx": int(_pair_ids_arr[_i]) if _i < _pair_ids_arr.size else _i,
                                                "p_invoke": float(_p_inv_vec[_i]),
                                                "I_v": int(_I_v_vec[_i]),
                                                "p_self": float(_p_self_vec[_i]),
                                                "p_verif": float(_p_verif_vec[_i]),
                                                "R_a": float(_r_a[_i]) if _r_a is not None else float("nan"),
                                                "R_b": float(_r_b[_i]) if _r_b is not None else float("nan"),
                                            })
                                    metrics["bsv/iv_mean"] = float(_I_v_f.mean())
                                    metrics["bsv/p_invoke_mean"] = float(_p_inv_vec.mean())
                                    # Emit threshold at native + log10 scale: on a tie-heavy
                                    # discrete judge (e.g. 1-token math judge) the 0.5-quantile
                                    # collapses to ~1e-4 and renders as 0.000 in 3-decimal logs,
                                    # hiding real gate dynamics. log10 keeps it readable.
                                    _thr = float(self.bsv_gate.current_threshold())
                                    metrics["bsv/threshold"] = _thr
                                    metrics["bsv/log10_threshold"] = float(np.log10(max(_thr, 1e-12)))
                                    metrics["bsv/realized_call_rate"] = float(self.bsv_gate.realized_call_rate())
                                    metrics["bsv/verifier_absent"] = 1.0 if _verifier_absent else 0.0
                                    metrics["bsv/verifier_failed"] = 1.0 if _verifier_failed_batch else 0.0
                                else:
                                    batch.batch["cpr_p_pref"] = torch.tensor(cpr_out["p_pref"], dtype=torch.float32)
                                batch.batch["cpr_a_input_ids"] = y_a_dp.batch["input_ids"]
                                batch.batch["cpr_a_attention_mask"] = y_a_dp.batch["attention_mask"]
                                batch.batch["cpr_a_position_ids"] = y_a_dp.batch["position_ids"]
                                batch.batch["cpr_a_responses"] = y_a_dp.batch["responses"]
                                batch.batch["cpr_a_loss_mask"] = cpr_a_loss_mask
                                batch.batch["cpr_b_input_ids"] = y_b_dp.batch["input_ids"]
                                batch.batch["cpr_b_attention_mask"] = y_b_dp.batch["attention_mask"]
                                batch.batch["cpr_b_position_ids"] = y_b_dp.batch["position_ids"]
                                batch.batch["cpr_b_responses"] = y_b_dp.batch["responses"]
                                batch.batch["cpr_b_loss_mask"] = cpr_b_loss_mask
                                batch.batch["cpr_a_logprobs"] = cpr_a_logprobs
                                batch.batch["cpr_b_logprobs"] = cpr_b_logprobs
                                batch.batch["cpr_a_ref_logprobs"] = cpr_a_ref_logprobs
                                batch.batch["cpr_b_ref_logprobs"] = cpr_b_ref_logprobs
                                batch.non_tensor_batch["cpr_pair_id"] = np.asarray(cpr_out["pair_ids"], dtype=np.int64)
                                batch.meta_info["cpr_enabled"] = True
                                batch.meta_info["cpr_alpha"] = float(cpr_cfg.alpha)
                                batch.meta_info["cpr_beta"] = float(cpr_cfg.beta)
                                batch.meta_info["cpr_eta"] = float(cpr_cfg.eta)
                                batch.meta_info["cpr_anchor_mode"] = str(cpr_cfg.anchor_mode)
                                batch.meta_info["cpr_arm_id"] = cpr_cfg.arm_id
                                batch.meta_info["cpr_pair_source"] = cpr_cfg.pair_source
                                if self.bsv_gate is not None:
                                    batch.meta_info["bsv_enabled"] = True
                                    batch.meta_info["bsv_alpha"] = float(self.bsv_gate.alpha)
                                    batch.meta_info["bsv_epsilon"] = float(self.bsv_gate.epsilon)
                                metrics.update(cpr_out.get("metrics", {}))

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # ---- GRPO-BSV overlay (DEFINE_SPEC_v3 §1.2b/§1.3b) ----
                        # When BSV is active in GRPO frame, gate per-rollout on
                        # unary self-judge confidence c_i = 2|p_self_i - 0.5|,
                        # then replace token_level_scores with the gated
                        # scalar reward r_i = I_v · R_ext + (1 - I_v) · p_self,
                        # broadcast onto the last response token (mirrors
                        # verl's standard outcome-reward convention). GRPO
                        # advantage computation consumes the new scores
                        # unchanged via compute_grpo_outcome_advantage's
                        # `scores = token_level_rewards.sum(dim=-1)` path.
                        #
                        # Mutual exclusion: this branch runs only when CPR is
                        # disabled (the DPO-BSV overlay above already handles
                        # the pair-construction path).
                        _bsv_grpo_on = (
                            self.bsv_gate is not None
                            and str(self.config.algorithm.adv_estimator).lower().endswith("grpo")
                            and not (
                                hasattr(self.config, "cpr")
                                and bool(getattr(self.config.cpr, "enable", False))
                            )
                        )
                        if _bsv_grpo_on:
                            # GRPO-BSV Frame A overlay (DEFINE_SPEC_v3 §1.3b, rev 2026-04-21).
                            # - p_self drives ONLY the gate; it NEVER enters the reward.
                            # - Reward on included (I_v=1) rollouts: R_ext.
                            # - Masked (I_v=0) rollouts: filled with group-μ_incl so
                            #   downstream GRPO group normalization yields A_i=0
                            #   (zero gradient). Degenerate groups (K'=0) filled
                            #   with 0 and counted via bsv_grpo/degenerate_group_rate.
                            # See doc/bsv_frame_c_parking_lot.md for the parked mixture.
                            from verl.trainer.ppo.bsv_single_judge import (
                                SingleJudgeConfig,
                                score_rollouts,
                            )
                            from verl.trainer.ppo.core_algos import (
                                compute_bsv_grpo_masked_reward,
                            )

                            _bsv_cfg_local = (
                                self.config.get("bsv", {})
                                if hasattr(self.config, "get")
                                else {}
                            )
                            # Per-rollout external reward: sum token-level scores
                            # to one scalar per sample (standard GRPO outcome
                            # convention).
                            R_ext_t = batch.batch["token_level_scores"].sum(dim=-1)
                            R_ext_np = R_ext_t.detach().cpu().numpy().astype(np.float32)
                            _N_roll = R_ext_np.shape[0]

                            # p_self_i drives the BSV gate (§1.2b). Under Frame A,
                            # p_self is gate-only — it does not appear in r_i.
                            # Default backend (rev 2026-04-21): response_logprob
                            # — per-rollout independent, content-based, zero extra
                            # inference. self_consistency retained for ablation;
                            # E1 measured ~34% degenerate_group_rate with it due
                            # to group-statistic coupling.
                            _sj_mode = str(
                                _bsv_cfg_local.get("single_judge_mode", "response_logprob")
                            )
                            _sj_noise = float(
                                _bsv_cfg_local.get("single_judge_noise", 0.15)
                            )
                            _step_now = (
                                self.global_steps if hasattr(self, "global_steps") else 0
                            )
                            _sj_cfg = SingleJudgeConfig(
                                mode=_sj_mode,
                                noise=_sj_noise,
                                rng_seed=int(
                                    _bsv_cfg_local.get("rng_seed", 0)
                                )
                                + int(_step_now),
                            )
                            _uids_arr = batch.non_tensor_batch.get("uid", None)
                            # Fast-path reuse: if the pre-reward hook already ran
                            # bsv_gate.decide_batch, reuse its p_self + I_v so we
                            # don't double-advance gate state. Only response_logprob
                            # mode takes the fast path.
                            if _bsv_grpo_fastpath_used:
                                p_self_np = _bsv_fp_p_self
                                metrics["bsv_grpo/p_self_backend_degraded"] = 0.0
                                metrics["bsv_grpo/fastpath_used"] = 1.0
                            elif _sj_mode == "response_logprob":
                                _olp_t = batch.batch.get("old_log_probs", None)
                                _rm_t = batch.batch.get("response_mask", None)
                                if _olp_t is None or _rm_t is None:
                                    # Trainer may not have computed old_log_probs
                                    # yet, or on a no-critic fast path. Degrade.
                                    metrics["bsv_grpo/p_self_backend_degraded"] = 1.0
                                    _fallback_cfg = SingleJudgeConfig(
                                        mode="stub_noisy",
                                        noise=_sj_noise,
                                        rng_seed=int(
                                            _bsv_cfg_local.get("rng_seed", 0)
                                        )
                                        + int(_step_now),
                                    )
                                    p_self_np = score_rollouts(
                                        [""] * _N_roll,
                                        [""] * _N_roll,
                                        _fallback_cfg,
                                    )
                                else:
                                    _olp_np = _olp_t.detach().cpu().numpy().astype(np.float32)
                                    _rm_np = _rm_t.detach().cpu().numpy().astype(np.float32)
                                    p_self_np = score_rollouts(
                                        [""] * _N_roll,
                                        [""] * _N_roll,
                                        _sj_cfg,
                                        old_log_probs=_olp_np,
                                        response_mask=_rm_np,
                                    )
                                    metrics["bsv_grpo/p_self_backend_degraded"] = 0.0
                            elif _sj_mode == "self_consistency":
                                _preds_arr = batch.non_tensor_batch.get("pred", None)
                                if _preds_arr is None or _uids_arr is None:
                                    metrics["bsv_grpo/p_self_backend_degraded"] = 1.0
                                    _fallback_cfg = SingleJudgeConfig(
                                        mode="stub_noisy",
                                        noise=_sj_noise,
                                        rng_seed=int(
                                            _bsv_cfg_local.get("rng_seed", 0)
                                        )
                                        + int(_step_now),
                                    )
                                    p_self_np = score_rollouts(
                                        [""] * _N_roll,
                                        [""] * _N_roll,
                                        _fallback_cfg,
                                    )
                                else:
                                    p_self_np = score_rollouts(
                                        [""] * _N_roll,
                                        [""] * _N_roll,
                                        _sj_cfg,
                                        preds=[str(p) for p in _preds_arr],
                                        group_ids=list(_uids_arr),
                                    )
                                    metrics["bsv_grpo/p_self_backend_degraded"] = 0.0
                            elif _sj_mode == "actor_logits":
                                # 1-token YES/NO/TIE self-judge via in-place
                                # right-pad extension + compute_log_prob
                                # (DEFINE_SPEC_v3 §1.2b point 1).
                                p_self_np, _judge_diag = _build_and_score_actor_judge(
                                    batch=batch,
                                    tokenizer=self.tokenizer,
                                    actor_rollout_wg=self.actor_rollout_wg,
                                    judge_temperature=float(
                                        _bsv_cfg_local.get("judge_temperature", 1.0)
                                    ),
                                )
                                metrics["bsv_grpo/p_self_backend_degraded"] = (
                                    1.0 if _judge_diag.get("degraded", False) else 0.0
                                )
                                metrics["bsv_grpo/actor_logits_top3_mass"] = float(
                                    _judge_diag.get("top3_mass_median", float("nan"))
                                )
                                metrics["bsv_grpo/actor_logits_fallback_rate"] = float(
                                    _judge_diag.get("fallback_rate", 0.0)
                                )
                            elif _sj_mode == "stub_noisy":
                                p_self_np = score_rollouts(
                                    [""] * _N_roll, [""] * _N_roll, _sj_cfg
                                )
                                metrics["bsv_grpo/p_self_backend_degraded"] = 0.0
                            else:
                                raise NotImplementedError(
                                    f"single_judge_mode={_sj_mode!r} not yet wired "
                                    "in ray_trainer; supported in this cut: "
                                    "{'response_logprob', 'self_consistency', "
                                    "'actor_logits', 'stub_noisy'}"
                                )

                            # Gate decisions (per-rollout). Reuse cached values
                            # if fast-path already ran decide_batch (pre-reward).
                            if _bsv_grpo_fastpath_used:
                                I_v_np = _bsv_fp_I_v
                                p_inv_np = _bsv_fp_p_inv
                            else:
                                I_v_np, p_inv_np = self.bsv_gate.decide_batch(p_self_np)

                            # Cache last-batch tensors on the gate so the
                            # adaptive-α controller (Phase-1 E6) can read
                            # p_self / V / subset masks without re-plumbing
                            # the fields through batch.meta_info.
                            #
                            # verify_mask vs eps_mask split (methodology §5c.2):
                            # the gate's p_invoke encodes provenance —
                            #   p_invoke == 1     → gate-routed (below=1, c<t)
                            #   p_invoke == ε     → ε-exploration draw (below=0)
                            # A rollout with I_v=1 AND p_invoke≈1 is a verify
                            # candidate; I_v=1 AND p_invoke<1 came from the
                            # ε-audit slice. Use a 0.5 cutoff against ε (which
                            # is 0.05 in practice) — robust to both edges.
                            _verify_mask_np = (p_inv_np >= 0.5) & (I_v_np == 1)
                            _eps_mask_np = (p_inv_np < 0.5) & (I_v_np == 1)
                            # V := R_ext where observed, sentinel -1 elsewhere.
                            _V_np = np.where(I_v_np == 1, R_ext_np, -1.0)
                            self.bsv_gate.last_p_self = p_self_np.astype(np.float32)
                            self.bsv_gate.last_V = _V_np.astype(np.float32)
                            self.bsv_gate.last_verify_mask = _verify_mask_np
                            self.bsv_gate.last_eps_mask = _eps_mask_np

                            # Adaptive-α step (Phase-1 E6).  The controller
                            # updates internal EMAs every step but only moves
                            # α on cadence ticks post-warmup. set_alpha()
                            # rebuilds the P² tracker, so fire it only when α
                            # actually moves (otherwise we'd drop the tracker
                            # state every step).
                            if self.adaptive_alpha_ctrl is not None:
                                from duet.adaptive_alpha import (
                                    compute_c_ref,
                                    compute_calibration_signals,
                                )

                                _signals = compute_calibration_signals(
                                    p_self=p_self_np,
                                    V=_V_np,
                                    is_verify_candidate=_verify_mask_np,
                                    is_eps_sample=_eps_mask_np,
                                )
                                _c_ref_t = compute_c_ref(p_self_np)
                                _prev_alpha = float(self.bsv_gate.alpha)
                                _new_alpha = self.adaptive_alpha_ctrl.step(
                                    step_idx=int(_step_now),
                                    c_verify=_signals["c_verify"],
                                    c_ref=_c_ref_t,
                                )
                                if abs(_new_alpha - _prev_alpha) > 1e-9:
                                    self.bsv_gate.set_alpha(_new_alpha)
                                # Log every step so the trajectory plot stays dense.
                                metrics["bsv_adaptive/alpha"] = float(_new_alpha)
                                metrics["bsv_adaptive/alpha_prev"] = _prev_alpha
                                metrics["bsv_adaptive/c_verify"] = float(
                                    _signals["c_verify"]
                                )
                                metrics["bsv_adaptive/c_eps"] = float(
                                    _signals["c_eps"]
                                )
                                metrics["bsv_adaptive/c_ref"] = float(_c_ref_t)
                                metrics["bsv_adaptive/n_verify"] = int(
                                    _signals["n_verify"]
                                )
                                metrics["bsv_adaptive/n_eps"] = int(
                                    _signals["n_eps"]
                                )
                                if self.adaptive_alpha_ctrl.c_verify_ema is not None:
                                    metrics["bsv_adaptive/c_verify_ema"] = float(
                                        self.adaptive_alpha_ctrl.c_verify_ema
                                    )
                                    metrics["bsv_adaptive/c_ref_ema"] = float(
                                        self.adaptive_alpha_ctrl.c_ref_ema
                                    )
                                    metrics["bsv_adaptive/gap"] = float(
                                        self.adaptive_alpha_ctrl.c_verify_ema
                                        - self.adaptive_alpha_ctrl.c_ref_ema
                                    )

                            # Group identity for the mask fill-mean (§1.3b Frame A).
                            # Must come from the same uid the downstream GRPO uses.
                            # If uid unavailable, fall back to a single-group id so
                            # every rollout shares one group (still correct; just
                            # less granular than per-prompt groups).
                            if _uids_arr is not None:
                                _group_ids = list(_uids_arr)
                            else:
                                _group_ids = ["_nogroup_"] * _N_roll

                            # Verifier-absent side-channel (H3 pattern): batch-level
                            # oracle failure masks the ENTIRE batch.
                            _verifier_absent = bool(
                                getattr(self, "_cpr_oracle_had_failure", False)
                            )

                            # Frame A masked reward (§1.3b): s_i = R_ext on included,
                            # μ_incl on masked (zero advantage), 0 on degenerate.
                            s_np, include_mask_np = compute_bsv_grpo_masked_reward(
                                R_ext_np,
                                I_v_np,
                                _group_ids,
                                verifier_absent=_verifier_absent,
                            )

                            # Broadcast s_i to token_level_scores: last valid
                            # response token carries the scalar, rest are zero.
                            _resp_slice = batch.batch["attention_mask"][
                                :, batch.batch["prompts"].shape[1]:
                            ]
                            _last_idx = (
                                _resp_slice.to(torch.int64).sum(dim=-1) - 1
                            )
                            _last_idx = torch.clamp(_last_idx, min=0)
                            _new_scores = torch.zeros_like(
                                batch.batch["token_level_scores"]
                            )
                            _s_t = torch.tensor(
                                s_np,
                                dtype=batch.batch["token_level_scores"].dtype,
                                device=batch.batch["token_level_scores"].device,
                            )
                            _new_scores[
                                torch.arange(_N_roll, device=_new_scores.device),
                                _last_idx,
                            ] = _s_t
                            batch.batch["token_level_scores"] = _new_scores

                            # Compute degenerate-group stats from group-by-uid view.
                            _unique = list(set(_group_ids))
                            _group_id_arr = np.asarray(_group_ids)
                            _Kp_list = [
                                int(include_mask_np[_group_id_arr == g].sum())
                                for g in _unique
                            ]
                            _n_groups = max(len(_unique), 1)
                            _degen = sum(1 for k in _Kp_list if k == 0)
                            _degen_rate = float(_degen) / float(_n_groups)
                            _mean_Kp = (
                                float(sum(_Kp_list)) / float(_n_groups)
                                if _n_groups > 0
                                else 0.0
                            )

                            # Diagnostic metrics (§1.3b, §3.4).
                            metrics["bsv_grpo/iv_mean"] = float(I_v_np.mean())
                            metrics["bsv_grpo/p_invoke_mean"] = float(p_inv_np.mean())
                            metrics["bsv_grpo/threshold"] = float(
                                self.bsv_gate.current_threshold()
                            )
                            metrics["bsv_grpo/log10_threshold"] = float(
                                np.log10(
                                    max(float(self.bsv_gate.current_threshold()), 1e-12)
                                )
                            )
                            metrics["bsv_grpo/realized_call_rate"] = float(
                                self.bsv_gate.realized_call_rate()
                            )
                            # Retired under Frame A: bsv_grpo/r_mean (meaningless
                            # with fill-mean semantics). Replace with conditional
                            # R_ext_mean on the labeled slice.
                            _incl_bool = include_mask_np.astype(bool)
                            if _incl_bool.any():
                                metrics["bsv_grpo/R_ext_mean"] = float(
                                    R_ext_np[_incl_bool].mean()
                                )
                            else:
                                metrics["bsv_grpo/R_ext_mean"] = float("nan")
                            metrics["bsv_grpo/p_self_mean"] = float(p_self_np.mean())
                            metrics["bsv_grpo/degenerate_group_rate"] = _degen_rate
                            metrics["bsv_grpo/effective_group_size_mean"] = _mean_Kp
                            metrics["bsv_grpo/verifier_absent_batch"] = float(
                                1.0 if _verifier_absent else 0.0
                            )

                            # Propensity CSV emission: §3.4 schema v2.
                            # {step, uid, group_id, p_invoke, I_v, R_ext, p_self}
                            # R_ext is NaN when I_v=0 OR verifier_absent.
                            try:
                                _csv_path = getattr(self, "_bsv_rollout_csv_path", None)
                                if _csv_path is None:
                                    _exp_name = getattr(
                                        self.config.trainer, "experiment_name", "unknown"
                                    )
                                    _csv_dir = os.path.join(
                                        "outputs", "bsv_propensity_rollout"
                                    )
                                    os.makedirs(_csv_dir, exist_ok=True)
                                    _run_tag = getattr(
                                        self.config.trainer, "run_tag", ""
                                    ) or str(int(_step_now))
                                    _csv_path = os.path.join(
                                        _csv_dir,
                                        f"{_exp_name}__{_run_tag}.csv",
                                    )
                                    with open(_csv_path, "w") as _fh:
                                        _fh.write(
                                            "step,uid,group_id,p_invoke,I_v,R_ext,p_self\n"
                                        )
                                    self._bsv_rollout_csv_path = _csv_path
                                with open(self._bsv_rollout_csv_path, "a") as _fh:
                                    for _j in range(_N_roll):
                                        _uid_val = (
                                            str(_uids_arr[_j])
                                            if _uids_arr is not None
                                            else f"row{_step_now}_{_j}"
                                        )
                                        _gid_val = str(_group_ids[_j])
                                        _Iv_j = int(I_v_np[_j])
                                        if _verifier_absent or _Iv_j == 0:
                                            _Rext_str = ""
                                        else:
                                            _Rext_str = f"{float(R_ext_np[_j]):.6f}"
                                        _fh.write(
                                            f"{int(_step_now)},"
                                            f"{_uid_val},"
                                            f"{_gid_val},"
                                            f"{float(p_inv_np[_j]):.6f},"
                                            f"{_Iv_j},"
                                            f"{_Rext_str},"
                                            f"{float(p_self_np[_j]):.6f}\n"
                                        )
                            except Exception as _e:  # noqa: BLE001
                                # Propensity CSV is diagnostic; failure must not
                                # kill training. Log and continue.
                                metrics.setdefault(
                                    "bsv_grpo/propensity_csv_failed", 1.0
                                )

                            if "bsv_enabled" not in batch.meta_info:
                                batch.meta_info["bsv_enabled"] = True
                                batch.meta_info["bsv_alpha"] = float(
                                    self.bsv_gate.alpha
                                )
                                batch.meta_info["bsv_epsilon"] = float(
                                    self.bsv_gate.epsilon
                                )
                                batch.meta_info["bsv_frame"] = "grpo_frame_a"

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # ---- VIP post-reward observation logging ----
                        # Group rewards by uid (one uid per ORIGINAL prompt;
                        # all replicated rollouts of the same prompt share a
                        # uid by construction in the pre-rollout VIP block).
                        # Mean reward per uid → observed pass rate, mapped
                        # back to the cache key → stored for next step's
                        # GP fit. Per official vip_ray_trainer.py semantics
                        # (most-recent-batch only).
                        if (
                            self.vip_allocator is not None
                            and self._vip_step_keys_per_uid is not None
                        ):
                            try:
                                _vip_uids = batch.non_tensor_batch.get("uid", None)
                                if _vip_uids is not None:
                                    _vip_tr = batch.batch["token_level_rewards"]
                                    _vip_rew_per_rollout = (
                                        _vip_tr.sum(dim=-1).detach().cpu().numpy()
                                    )
                                    _vip_uids_str = np.asarray(_vip_uids).astype(str)
                                    _vip_obs = []
                                    for _uid_str, _key in self._vip_step_keys_per_uid.items():
                                        _mask = _vip_uids_str == _uid_str
                                        if _mask.any():
                                            _mean_acc = float(_vip_rew_per_rollout[_mask].mean())
                                            _vip_obs.append((_key, _mean_acc))
                                    self._vip_last_observations = _vip_obs
                                    if _vip_obs:
                                        _accs_arr = np.array([a for _, a in _vip_obs])
                                        metrics["vip/observed_acc_mean"] = float(_accs_arr.mean())
                                        metrics["vip/observed_acc_std"] = float(_accs_arr.std())
                                        metrics["vip/n_observations"] = float(len(_vip_obs))
                            except Exception as _vip_obs_err:  # noqa: BLE001
                                metrics.setdefault("vip/observation_failed", 1.0)
                                print(f"[ray_trainer] VIP observation logging failed: {_vip_obs_err}")
                            self._vip_step_keys_per_uid = None  # consumed

                        # ---- DAPO Overlong Reward Shaping (paper §4.3) ----
                        # Soft length bias: responses that run close to
                        # max_response_length without EOS get penalized.
                        # Penalty ramps linearly from 0 at (max - buffer) to
                        # full penalty at max. Applied BEFORE Dynamic Sampling
                        # filter so length-hacked responses can also be
                        # treated as degenerate.
                        _dapo_cfg_pre = (
                            self.config.get("dapo", {})
                            if hasattr(self.config, "get")
                            else {}
                        )
                        if _dapo_cfg_pre and bool(_dapo_cfg_pre.get("enable", False)):
                            try:
                                _ob = float(_dapo_cfg_pre.get("overlong_buffer", 512))
                                _op = float(_dapo_cfg_pre.get("overlong_penalty", 1.0))
                                _max_resp = int(
                                    self.config.data.get("max_response_length", 3072)
                                )
                                _threshold = float(_max_resp - _ob)
                                # response length per rollout = sum of response_mask
                                _resp_mask = batch.batch["response_mask"]
                                _resp_len = _resp_mask.sum(dim=-1).float()  # (B,)
                                _excess = torch.clamp(_resp_len - _threshold, min=0.0, max=_ob)
                                _penalty = (_excess / _ob) * _op  # in [0, _op]
                                # subtract penalty from trajectory reward by
                                # zeroing it evenly across response tokens on
                                # the last non-masked position (single-token subtract):
                                _tr_pre = batch.batch["token_level_rewards"]
                                _B, _T = _tr_pre.shape
                                _last_idx = _resp_mask.long().sum(dim=-1) - 1
                                _last_idx = torch.clamp(_last_idx, min=0)
                                _row_idx = torch.arange(_B, device=_tr_pre.device)
                                _tr_pre[_row_idx, _last_idx] = (
                                    _tr_pre[_row_idx, _last_idx] - _penalty.to(_tr_pre.device)
                                )
                                metrics["dapo/overlong_penalty_mean"] = float(_penalty.mean())
                                metrics["dapo/overlong_penalty_nonzero_frac"] = float(
                                    (_penalty > 0).float().mean()
                                )
                            except Exception as _e:  # noqa: BLE001
                                metrics.setdefault("dapo/overlong_failed", 1.0)

                        # ---- DAPO dynamic-sampling filter (Phase-1 E9 / C11) ----
                        # Drops rollouts in groups whose pass-rate ∈ {0, 1}
                        # by zeroing their token_level_rewards. Post mean-
                        # normalization in compute_advantage this yields
                        # zero advantage → zero gradient contribution, while
                        # keeping the tensor shape (vLLM / FSDP invariants).
                        _dapo_cfg = (
                            self.config.get("dapo", {})
                            if hasattr(self.config, "get")
                            else {}
                        )
                        if _dapo_cfg and bool(_dapo_cfg.get("enable", False)):
                            try:
                                from duet.dapo_filter import apply_dapo_filter

                                _tr = batch.batch["token_level_rewards"]
                                _rew_per_rollout = (
                                    _tr.sum(dim=-1).detach().cpu().numpy()
                                )
                                _uids = batch.non_tensor_batch.get("uid", None)
                                if _uids is not None:
                                    _gids = np.asarray(
                                        [str(u) for u in _uids]
                                    )
                                    _keep = apply_dapo_filter(
                                        _gids, _rew_per_rollout
                                    )
                                    _drop = ~_keep
                                    _drop_rate = float(_drop.mean())
                                    metrics["dapo/drop_rate"] = _drop_rate
                                    metrics["dapo/kept_rollouts"] = int(
                                        _keep.sum()
                                    )
                                    if _drop.any():
                                        _drop_t = torch.tensor(
                                            _drop,
                                            dtype=torch.bool,
                                            device=_tr.device,
                                        )
                                        batch.batch["token_level_rewards"][
                                            _drop_t
                                        ] = 0.0
                                else:
                                    metrics.setdefault(
                                        "dapo/missing_uid", 1.0
                                    )
                            except Exception as _e:  # noqa: BLE001
                                # DAPO is a baseline comparator; a crash must
                                # not kill training. Flag and continue.
                                metrics.setdefault("dapo/filter_failed", 1.0)

                        # ---- GRESO history recording (Phase-1 baseline) ----
                        # After rewards are finalized, log one (prompt_text,
                        # mean_reward) observation per unique UID so the
                        # filter can decide what to skip on future steps.
                        if self.greso_filter is not None:
                            try:
                                _tr_g = batch.batch["token_level_rewards"]
                                _rew_per = (
                                    _tr_g.sum(dim=-1).detach().cpu().numpy()
                                )
                                _uids_g = batch.non_tensor_batch.get(
                                    "uid", None
                                )
                                _ids_g = batch.batch.get("input_ids", None)
                                if _uids_g is not None and _ids_g is not None:
                                    _uids_arr = np.array(
                                        [str(u) for u in _uids_g]
                                    )
                                    _unique_u, _first_idx = np.unique(
                                        _uids_arr, return_index=True
                                    )
                                    for _u, _fi in zip(
                                        _unique_u, _first_idx
                                    ):
                                        _grp = _uids_arr == _u
                                        _avg = float(_rew_per[_grp].mean())
                                        _ptxt = self.tokenizer.decode(
                                            _ids_g[int(_fi)],
                                            skip_special_tokens=True,
                                        )
                                        self.greso_filter.record(
                                            _ptxt, _avg
                                        )
                                    metrics.update(
                                        self.greso_filter.stats()
                                    )
                                else:
                                    metrics.setdefault(
                                        "greso/missing_uid_or_ids", 1.0
                                    )
                            except Exception as _greso_rec_err:  # noqa: BLE001
                                metrics.setdefault(
                                    "greso/record_failed", 1.0
                                )
                                print(
                                    f"[ray_trainer] GRESO recording error: "
                                    f"{_greso_rec_err}"
                                )

                        # ---- ARRoL quality-head step (Phase-1 baseline) ----
                        # Compute log-prob summary features over the first
                        # L_detect response tokens per rollout; use as input
                        # to the ARRoL head. In warmup, train head on all
                        # rollouts. Post-warmup, predict → prune → train on
                        # kept → MASK predicted-fail token_level_rewards.
                        if self.arrol_head is not None:
                            try:
                                _olp = batch.batch.get("old_log_probs", None)
                                _rmask = batch.batch.get(
                                    "response_mask", None
                                )
                                _tr_a = batch.batch["token_level_rewards"]
                                if _olp is None or _rmask is None:
                                    metrics.setdefault(
                                        "arrol/missing_inputs", 1.0
                                    )
                                else:
                                    _L = int(self._arrol_l_detect)
                                    _T_resp = _olp.shape[-1]
                                    _L_eff = min(_L, _T_resp)
                                    _olp_prefix = (
                                        _olp[:, :_L_eff]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy()
                                    )
                                    _rmk_prefix = (
                                        _rmask[:, :_L_eff]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy()
                                    )
                                    # Weighted mean / std / min / max over
                                    # VALID positions (response_mask==1). If
                                    # a rollout has no valid prefix tokens,
                                    # fall back to zeros (safe defaults).
                                    _B_a = _olp_prefix.shape[0]
                                    _feat = np.zeros(
                                        (_B_a, self._arrol_feature_dim),
                                        dtype=np.float32,
                                    )
                                    for _bi in range(_B_a):
                                        _mask_bi = _rmk_prefix[_bi] > 0
                                        if _mask_bi.sum() < 1:
                                            continue
                                        _lp_valid = _olp_prefix[_bi][_mask_bi]
                                        _feat[_bi, 0] = float(
                                            _lp_valid.mean()
                                        )
                                        _feat[_bi, 1] = float(_lp_valid.std())
                                        _feat[_bi, 2] = float(_lp_valid.min())
                                        _feat[_bi, 3] = float(_lp_valid.max())
                                    import torch as _th_a
                                    _feat_t = _th_a.from_numpy(_feat)
                                    # Binary labels from rewards > 0.5 threshold
                                    _rew_sum = (
                                        _tr_a.sum(dim=-1)
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy()
                                    )
                                    _labels = (_rew_sum > 0.5).astype(
                                        np.float32
                                    )
                                    if self.arrol_head.in_warmup:
                                        # Warmup: full verification, update calibrator
                                        # + head. No pruning. Update head's calibrator
                                        # so it can calibrate post-warmup predictions.
                                        _raw_w = self.arrol_head.forward_raw(_feat_t)
                                        self.arrol_head.calibrator.update(_raw_w, _labels)
                                        _loss = self.arrol_head.train_step(
                                            _feat_t, _labels
                                        )
                                        metrics["arrol/train_loss"] = _loss
                                        metrics["arrol/prune_rate_step"] = 0.0
                                        metrics["arrol/n_pruned_step"] = 0.0
                                        metrics["arrol/pred_mean"] = float(_raw_w.mean())
                                    elif _arrol_fp_used:
                                        # Fast-path pre-reward path: reuse cached
                                        # raw scores + keep_mask. Update calibrator
                                        # with KEPT subset's labels (only those were
                                        # actually verified) + train head on kept.
                                        _probs_a = _arrol_fp_probs_a
                                        _keep_a = _arrol_fp_keep_a
                                        _prune_cnt = int((~_keep_a).sum())
                                        metrics["arrol/n_pruned_step"] = float(_prune_cnt)
                                        metrics["arrol/prune_rate_step"] = float(
                                            _prune_cnt / max(1, _keep_a.size)
                                        )
                                        metrics["arrol/pred_mean"] = float(
                                            _probs_a.mean()
                                        )
                                        metrics["arrol/fastpath_used"] = 1.0
                                        if _keep_a.any():
                                            # Re-compute raw on kept for calibrator
                                            # update (calibrator needs raw scores,
                                            # not calibrated q values).
                                            _raw_kept = self.arrol_head.forward_raw(
                                                _arrol_fp_feat_t[_keep_a]
                                            )
                                            self.arrol_head.calibrator.update(
                                                _raw_kept, _labels[_keep_a]
                                            )
                                            _loss = self.arrol_head.train_step(
                                                _arrol_fp_feat_t[_keep_a],
                                                _labels[_keep_a],
                                            )
                                            metrics["arrol/train_loss"] = _loss
                                    else:
                                        # Post-warmup but fast-path disabled (e.g.,
                                        # missing old_log_probs). Fall back to
                                        # post-reward pruning: predict → calibrate →
                                        # sample_survival → mask-out pruned rewards.
                                        _raw_n = self.arrol_head.forward_raw(_feat_t)
                                        if self.arrol_head.calibrator.n_observed() >= 16:
                                            _probs_a = self.arrol_head.calibrator.calibrate(_raw_n)
                                        else:
                                            _probs_a = _raw_n
                                        _keep_a = self.arrol_head.sample_survival(_probs_a)
                                        _prune_cnt = int((~_keep_a).sum())
                                        metrics[
                                            "arrol/n_pruned_step"
                                        ] = float(_prune_cnt)
                                        metrics[
                                            "arrol/prune_rate_step"
                                        ] = float(
                                            _prune_cnt / max(1, _keep_a.size)
                                        )
                                        metrics["arrol/pred_mean"] = float(
                                            _probs_a.mean()
                                        )
                                        if _keep_a.any():
                                            self.arrol_head.calibrator.update(
                                                _raw_n[_keep_a], _labels[_keep_a]
                                            )
                                            _loss = self.arrol_head.train_step(
                                                _feat_t[_keep_a],
                                                _labels[_keep_a],
                                            )
                                            metrics["arrol/train_loss"] = _loss
                                        if _prune_cnt > 0:
                                            _prune_t = _th_a.tensor(
                                                ~_keep_a,
                                                dtype=_th_a.bool,
                                                device=_tr_a.device,
                                            )
                                            batch.batch[
                                                "token_level_rewards"
                                            ][_prune_t] = 0.0
                                    metrics.update(self.arrol_head.stats())
                            except Exception as _arrol_err:  # noqa: BLE001
                                metrics.setdefault(
                                    "arrol/filter_failed", 1.0
                                )
                                print(
                                    f"[ray_trainer] ARRoL head step error: "
                                    f"{_arrol_err}"
                                )

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                        # ARRoL faithful: collect labels from verifier rewards,
                        # train head, log diagnostics. Decisions blob lives in
                        # batch.meta_info (set by the rollout worker). Labels:
                        # binary on R_ext > 0.
                        if (self.arrol_faithful_head is not None
                                and "arrol_faithful_decisions_blob" in batch.meta_info):
                            try:
                                import pickle as _pkl
                                _decisions = _pkl.loads(batch.meta_info["arrol_faithful_decisions_blob"])
                                _diag = _pkl.loads(batch.meta_info.get("arrol_faithful_diag_blob", b"\x80\x04}\x94."))
                                # Build a uid → final-reward map. Since decisions
                                # are per-rollout (one entry per L_detect crossing),
                                # we look up by request_id which corresponds to a
                                # uid prefix or full uid in vLLM's bookkeeping.
                                # For v1: associate each decision with the FIRST
                                # rollout in the batch for that prompt — close
                                # enough since pruned rollouts don't get rewards
                                # anyway. Improvements deferred to v2.
                                if "uid" in batch.non_tensor_batch:
                                    _uids = list(batch.non_tensor_batch["uid"])
                                    _rewards_per_row = batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
                                    _uid_to_reward = dict(zip(_uids, _rewards_per_row.tolist()))
                                else:
                                    _uid_to_reward = {}
                                _n_labelled = 0
                                for d in _decisions:
                                    if not d["keep"]:
                                        continue  # pruned rollouts have no reward
                                    label = float(_uid_to_reward.get(d["request_id"], 0.0) > 0)
                                    self.arrol_faithful_head.record_label(
                                        torch.from_numpy(d["hidden"]), label
                                    )
                                    _n_labelled += 1
                                _train_metrics = self.arrol_faithful_head.train_step(
                                    torch.device("cpu")
                                )
                                if _train_metrics:
                                    metrics.update(_train_metrics)
                                metrics.update(_diag)
                                metrics["arrol_faithful/n_labelled_this_step"] = float(_n_labelled)
                            except Exception as _arrol_err:
                                metrics.setdefault("arrol_faithful/post_step_failed", 1.0)
                                print(f"[ray_trainer] ARRoL post-step failed: {_arrol_err}")

                        # DUET SNIPS correction (paper Eq. 1): scale per-rollout
                        # advantages by 1/p_q^pre · 1/p_{q,i}^len. The weight is
                        # stashed in batch.batch by the DUET allocation hook;
                        # only present when DUET is active. We multiply both
                        # advantages and returns so any downstream consumer (PPO
                        # actor and critic) sees a unbiased gradient under the
                        # joint MNAR. GRPO's group-normalization already happened
                        # at compute_advantage above on the un-weighted rewards
                        # — this is the correct order (SNIPS weights the gradient,
                        # not the reward).
                        if "duet_snips_weight" in batch.batch:
                            _snips_w = batch.batch["duet_snips_weight"]
                            if _snips_w.shape[0] != batch.batch["advantages"].shape[0]:
                                # Defensive: shape mismatch means upstream replicate
                                # is out of sync; skip SNIPS and log.
                                metrics.setdefault("duet/snips_skipped_shape_mismatch", 1.0)
                            else:
                                # ε-keep SNIPS reweighting (paper §4.3, post
                                # 2026-04-28 fix): rollouts kept by ε-exploration
                                # despite no marker get an additional 1/abort_eps
                                # factor on top of 1/p_pre. We rely on the LP's
                                # explicit `eps_kept` flag (propagated via
                                # `duet_eps_kept` non_tensor_batch entry), NOT
                                # ~saw_marker — natural-EOS-before-K2 rollouts
                                # also have saw_marker=False but propensity 1.
                                _eps_kept_arr = batch.non_tensor_batch.get(
                                    "duet_eps_kept", None
                                )
                                _abort_eps = (
                                    float(self.config.duet.get("abort_eps", 0.05))
                                    if hasattr(self.config, "duet") else 0.05
                                )
                                if _eps_kept_arr is not None and _abort_eps > 0:
                                    import numpy as _np_local
                                    _eps_kept_mask = _np_local.asarray(
                                        _eps_kept_arr, dtype=bool
                                    )
                                    _n_eps_kept = int(_eps_kept_mask.sum())
                                    if _n_eps_kept > 0:
                                        _eps_idx = torch.from_numpy(
                                            _np_local.where(_eps_kept_mask)[0]
                                        ).to(_snips_w.device).long()
                                        _snips_w = _snips_w.clone()
                                        _snips_w[_eps_idx] = (
                                            _snips_w[_eps_idx] * (1.0 / _abort_eps)
                                        )
                                        # Numerical safety cap (post 2026-05-01 v6 fix):
                                        # Worst case 1/(eps_pre × abort_eps) can be 2000
                                        # at small abort_eps. Production observed max
                                        # was ~402 in long runs; the prior 200 cap was
                                        # too aggressive (clipped signal V5 vs prod).
                                        # Raised to 1000: still well above production's
                                        # natural max (so no clipping in normal traffic)
                                        # but below the theoretical NaN cliff at 2000.
                                        _snips_w = torch.clamp(_snips_w, max=1000.0)
                                        batch.batch["duet_snips_weight"] = _snips_w
                                    metrics["duet/n_epsilon_kept"] = float(_n_eps_kept)
                                    metrics["duet/epsilon_kept_rate"] = float(
                                        _eps_kept_mask.mean()
                                    )
                                    metrics["duet/epsilon_inflation_factor"] = float(
                                        1.0 / _abort_eps
                                    )
                                batch.batch["advantages"] = (
                                    batch.batch["advantages"] * _snips_w.unsqueeze(-1)
                                )
                                batch.batch["returns"] = (
                                    batch.batch["returns"] * _snips_w.unsqueeze(-1)
                                )
                                metrics["duet/snips_weight_mean"] = float(_snips_w.mean())
                                metrics["duet/snips_weight_max"] = float(_snips_w.max())
                                metrics["duet/snips_weight_min"] = float(_snips_w.min())

                        # DUET v3: zero advantages + response_mask for aborted
                        # rollouts (T2 K2-grace abort path). This:
                        #   1. zeros policy loss contribution (no gradient via
                        #      the PG term — already true via advantages=0)
                        #   2. zeros KL loss contribution (response_mask=0 →
                        #      masked_mean drops these tokens entirely)
                        #   3. shortens effective batch (forward still happens
                        #      for tensor-shape consistency, but contributes
                        #      no gradient signal)
                        # ε-kept rollouts (saw_marker=False but kept by ε
                        # exploration) are NOT zeroed; they carry full SNIPS
                        # weight to preserve unbiasedness of the gradient
                        # estimator on the no-marker subset.
                        _did_abort_arr = batch.non_tensor_batch.get(
                            "duet_did_abort", None
                        )
                        if _did_abort_arr is not None:
                            import numpy as _np_local
                            _abort_mask = _np_local.asarray(_did_abort_arr, dtype=bool)
                            n_aborted = int(_abort_mask.sum())
                            if n_aborted > 0:
                                _abort_idx = torch.from_numpy(
                                    _np_local.where(_abort_mask)[0]
                                ).to(batch.batch["advantages"].device).long()
                                batch.batch["advantages"].index_fill_(0, _abort_idx, 0.0)
                                batch.batch["returns"].index_fill_(0, _abort_idx, 0.0)
                                if "response_mask" in batch.batch:
                                    batch.batch["response_mask"].index_fill_(0, _abort_idx, 0)
                            metrics["duet/n_aborted"] = float(n_aborted)
                            metrics["duet/abort_rate"] = float(n_aborted) / float(len(_abort_mask))

                        _saw_marker_arr = batch.non_tensor_batch.get(
                            "duet_saw_marker", None
                        )
                        if _saw_marker_arr is not None:
                            import numpy as _np_local
                            _marker_mask = _np_local.asarray(_saw_marker_arr, dtype=bool)
                            metrics["duet/marker_rate"] = float(_marker_mask.mean())

                        # DUET v3: update online K1/K2 estimator from observed
                        # response lengths on KEPT-NOT-ABORTED rollouts (paper
                        # algorithm line 247, post 2026-04-28 fix). Includes
                        # both marker-completed AND ε-kept rollouts; only true
                        # aborts are excluded. Uses sliding window of last
                        # `max_history` lengths; refits K1=p30, K2=p80 every
                        # `update_every` steps.
                        _est = getattr(self, "_duet_k_estimator", None)
                        if _est is not None and "response_mask" in batch.batch:
                            import numpy as _np_local
                            _rl = batch.batch["response_mask"].sum(dim=-1).cpu().numpy()
                            _abort_arr = batch.non_tensor_batch.get("duet_did_abort")
                            for _idx in range(len(_rl)):
                                if _abort_arr is not None and bool(_abort_arr[_idx]):
                                    continue
                                _est["lengths"].append(int(_rl[_idx]))
                            # Trim history
                            if len(_est["lengths"]) > _est["max_history"]:
                                _est["lengths"] = _est["lengths"][-_est["max_history"]:]
                            # Update K1/K2 every N steps from observed quantiles
                            if (self.global_steps - _est["last_update_step"]
                                    >= _est["update_every"]
                                    and len(_est["lengths"]) >= 32):
                                _arr = _np_local.array(_est["lengths"])
                                _new_k1 = int(_np_local.quantile(_arr, 0.30))
                                _new_k2 = int(_np_local.quantile(_arr, 0.80))
                                # Sanity: K1 < K2 < max_response, K1 >= min_tokens floor
                                _new_k1 = max(_duet_min_tokens if False else 32, _new_k1)
                                _new_k2 = max(_new_k1 + 50, _new_k2)
                                _est["current_k1"] = _new_k1
                                _est["current_k2"] = _new_k2
                                _est["last_update_step"] = self.global_steps

                        # DUET per-prompt σ̂_q^obs / L̂_q updates (paper §4.1, §4.2).
                        # σ̂_q^obs is std_i(A_{q,i} · Σ_ℓ log π_θ(y_{q,i,ℓ}|q))
                        # over kept rollouts within a prompt group. Updates feed
                        # the next step's allocation via DuetPromptState helpers.
                        # Piggybacks on the same kept-rollout filter used by K1/K2.
                        if (self._duet_prompt_state is not None
                                and "response_mask" in batch.batch
                                and "advantages" in batch.batch
                                and "old_log_probs" in batch.batch):
                            import numpy as _np_local
                            from duet.duet_prompt_state import (
                                update_l as _duet_upd_l,
                                update_s as _duet_upd_s,
                            )
                            _adv = batch.batch["advantages"]
                            _olp = batch.batch["old_log_probs"]
                            _msk = batch.batch["response_mask"]
                            # Per-rollout scalar g_i = A_{q,i} · Σ_ℓ log π (paper §4.1).
                            # Use token-mean of advantages for a stable scalar A_i;
                            # response_mask was zeroed for aborts so they collapse to 0.
                            _msk_sum = _msk.sum(dim=-1).clamp_min(1)
                            _A_i = (_adv * _msk).sum(dim=-1) / _msk_sum
                            _slp_i = (_olp * _msk).sum(dim=-1)
                            _g_i = (_A_i * _slp_i).detach().cpu().numpy()
                            _rl_obs = _msk.sum(dim=-1).cpu().numpy()
                            _abort_arr2 = batch.non_tensor_batch.get("duet_did_abort")
                            _uids = batch.non_tensor_batch.get("uid")
                            _ei = batch.non_tensor_batch.get("extra_info")
                            if _uids is not None and _ei is not None:
                                _uids_str = _np_local.asarray(_uids).astype(str)
                                # Paper alg line 247: σ̂^obs uses kept-not-aborted
                                # rollouts (marker-completed AND ε-kept). Only
                                # exclude aborted (where response_mask was zeroed).
                                _keep = (
                                    ~_np_local.asarray(_abort_arr2, bool)
                                    if _abort_arr2 is not None
                                    else _np_local.ones(len(_uids_str), bool)
                                )
                                # Group by uid; std needs ≥2 kept rollouts.
                                for _u in _np_local.unique(_uids_str):
                                    _grp = (_uids_str == _u) & _keep
                                    _grp_n = int(_grp.sum())
                                    if _grp_n < 2:
                                        continue
                                    # extra_info constant within a uid group
                                    # (replicated from same prompt at allocation).
                                    _gidx = int(_np_local.where(_uids_str == _u)[0][0])
                                    _info = _ei[_gidx] if _gidx < len(_ei) else None
                                    if not isinstance(_info, dict):
                                        continue
                                    _q_idx = str(_info.get("index", ""))
                                    if not _q_idx:
                                        continue
                                    _sigma_t = float(_np_local.std(_g_i[_grp], ddof=0))
                                    _duet_upd_s(
                                        self._duet_prompt_state, _q_idx, _sigma_t
                                    )
                                    # L̂_q: per-rollout kept length within the group.
                                    for _li in _np_local.where(_grp)[0]:
                                        _duet_upd_l(
                                            self._duet_prompt_state, _q_idx,
                                            float(_rl_obs[_li]),
                                        )

                        _inpo_calib_dir = os.environ.get("INPO_CALIBRATION_DUMP_DIR")
                        if _inpo_calib_dir:
                            try:
                                self._dump_inpo_calibration(batch, _inpo_calib_dir)
                            except Exception as _calib_err:
                                print(
                                    f"[inpo-calib-dump] step "
                                    f"{self.global_steps} failed: {_calib_err}"
                                )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if self.config.get("ttrl", {}).get("enable", False):
                        from verl.trainer.ppo.ttrl_utils import apply_original_gt, compute_ttrl_metrics
                        batch = apply_original_gt(batch)
                        reward_tensor_original, reward_extra_infos_dict_original = compute_reward(batch, self.reward_fn)
                        batch.batch["token_level_scores_original"] = reward_tensor_original
                        # Compute ttrl metrics
                        ttrl_metrics = compute_ttrl_metrics(batch, self.config.ttrl.n_samples_per_prompt)
                        for key, value in ttrl_metrics.items():
                                metrics.update({f"train/{key}": value})

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        final_validation_data_source_regex = None
                        if is_last_step:
                            final_validation_data_source_regex = self.config.trainer.get(
                                "final_validation_data_source_regex", None
                            )
                            if final_validation_data_source_regex is not None:
                                final_validation_data_source_regex = str(final_validation_data_source_regex).strip() or None
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate(
                                data_source_regex=final_validation_data_source_regex
                            )
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        # Print val metrics to console on every val (not only
                        # is_last_step). The "Final validation metrics" pprint
                        # at the end of fit() doesn't always fire (off-by-one
                        # in trainer loop exit), so this ensures any val run
                        # is observable in the console log even when the
                        # final-print path is skipped.
                        pprint(f"Validation metrics step:{self.global_steps}: {val_metrics}")

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # Cumulative efficiency tracking (train-only time + rollouts).
                # train_time excludes testing + save_checkpoint so cells can be
                # compared on training-proper cost.
                _step_train_time = (
                    steps_duration
                    - timing_raw.get("testing", 0.0)
                    - timing_raw.get("save_checkpoint", 0.0)
                )
                self._custom_train_time_cum_s += max(0.0, float(_step_train_time))
                # Rollouts PRODUCED by vLLM this step = total rollouts in batch.
                # For GRPO: train_batch × rollout_n. For DAPO oversample:
                # gen_batch × rollout_n (= ~1.5× GRPO). For ARRoL: same as
                # GRPO (fast-path saves REWARD calls, not rollouts).
                try:
                    _step_produced = int(batch.batch["responses"].shape[0])
                except Exception:
                    _step_produced = 0
                self._custom_rollouts_produced_cum += _step_produced
                # Rollouts USED (contributed non-zero advantage) = produced
                # minus dropped/pruned. For GRPO=produced. For DAPO: subtract
                # dapo/kept_rollouts complement. For ARRoL: subtract pruned.
                _step_used = _step_produced
                if "dapo/kept_rollouts" in metrics:
                    _step_used = int(metrics["dapo/kept_rollouts"])
                elif "arrol/n_pruned_step" in metrics:
                    _step_used = max(0, _step_produced - int(metrics["arrol/n_pruned_step"]))
                self._custom_rollouts_used_cum += _step_used

                metrics["custom/train_time_cum_s"] = self._custom_train_time_cum_s
                metrics["custom/rollouts_produced_cum"] = float(self._custom_rollouts_produced_cum)
                metrics["custom/rollouts_used_cum"] = float(self._custom_rollouts_used_cum)
                metrics["custom/step_train_time_s"] = float(max(0.0, _step_train_time))
                metrics["custom/step_rollouts_produced"] = float(_step_produced)
                metrics["custom/step_rollouts_used"] = float(_step_used)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # BSV propensity CSV sidecar (DEFINE_SPEC_v3 §3.2). Append every
                # pair's {step, pair_idx, p_invoke, I_v, R_a, R_b, p_self, p_verif}
                # row accumulated on this step. IPS / DR evaluators consume this
                # file post-training.
                if self.bsv_gate is not None and self._bsv_propensity_rows:
                    self._flush_bsv_propensity_rows()

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
