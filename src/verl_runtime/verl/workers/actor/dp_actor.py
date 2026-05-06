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
Single Process Actor
"""

import itertools
import logging
import os
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_bsv_loss, compute_cpr_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        def _get_micro_batches(data: DataProto) -> Tuple[list, Optional[list]]:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(
                        batch=batch, max_token_len=max_token_len
                    )

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices
                else:
                    num_micro_batches = batch.batch_size[0] // micro_batch_size
                    micro_batches_dp = data.chunk(num_micro_batches)
                    return micro_batches_dp, None
            elif use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        sirl_distill_enabled = data.meta_info.get("sirl_distill_enabled", False)
        sirl_distill_loss_coeff = data.meta_info.get("sirl_distill_loss_coeff", 1.0)
        if sirl_distill_enabled:
            select_keys.extend([
                "sirl_imp_input_ids",
                "sirl_imp_attention_mask",
                "sirl_imp_position_ids",
                "sirl_imp_responses",
                "sirl_gate_mask",
                "sirl_gate_weight",
            ])

        # SIRL-Pref DPO mode: pull precomputed pair logprobs from the batch so
        # the actor can compute the DPO loss alongside (or in place of) pg_loss.
        sirl_pref_dpo_mode = (
            data.meta_info.get("sirl_pref_enabled", False)
            and data.meta_info.get("sirl_pref_loss", "") == "dpo"
            and data.meta_info.get("sirl_pref_dpo_logprobs_ready", False)
        )

        # CPR DPO mode: the soft-label DPO loss replaces PPO policy loss.
        cpr_dpo_mode = data.meta_info.get("cpr_enabled", False)
        cpr_beta = float(data.meta_info.get("cpr_beta", 0.15))
        cpr_eta = float(data.meta_info.get("cpr_eta", 0.0))
        cpr_anchor_mode = str(data.meta_info.get("cpr_anchor_mode", "symmetric"))
        # BSV overlay: when active, the pair loss dispatches on a per-pair
        # invocation indicator I_v rather than blending a continuous α inside
        # p_pref. p_pref was already overwritten to the BSV formula in the
        # trainer (ray_trainer.py); we still route through compute_bsv_loss
        # here so the BSV-specific metrics (iv_mean, p_self/p_verif means)
        # land in the actor metrics dict.
        bsv_mode = cpr_dpo_mode and bool(data.meta_info.get("bsv_enabled", False))
        if sirl_pref_dpo_mode:
            select_keys.extend([
                "sirl_pref_revised_logprobs",
                "sirl_pref_original_logprobs",
                "sirl_pref_revised_ref_logprobs",
                "sirl_pref_original_ref_logprobs",
                "sirl_pref_revised_loss_mask",
                "sirl_pref_original_loss_mask",
                "sirl_pref_pair_weight",
                "sirl_pref_label",
                "sirl_pref_skip_mask",
            ])
        if cpr_dpo_mode:
            select_keys.extend([
                "cpr_p_pref",
                "cpr_a_input_ids",
                "cpr_a_attention_mask",
                "cpr_a_position_ids",
                "cpr_a_responses",
                "cpr_a_loss_mask",
                "cpr_b_input_ids",
                "cpr_b_attention_mask",
                "cpr_b_position_ids",
                "cpr_b_responses",
                "cpr_b_loss_mask",
                "cpr_a_logprobs",
                "cpr_b_logprobs",
                "cpr_a_ref_logprobs",
                "cpr_b_ref_logprobs",
            ])
        if bsv_mode:
            select_keys.extend([
                "bsv_p_self",
                "bsv_p_verif",
                "bsv_I_v",
            ])

        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(
                            batch=batch_tensordict_for_rearrange, max_token_len=max_token_len
                        )

                        for current_original_indices, text_mb_td in zip(
                            textual_indices, rearranged_text_micro_batches_tds
                        ):
                            current_mm_inputs_list = [
                                all_multi_modal_inputs_list[idx] for idx in current_original_indices
                            ]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        )
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    micro_batch_metrics = {}

                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                data[k] = [
                                    {kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v
                                ]
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    response_mask = data["response_mask"]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    # SIRL distillation: zero ALL advantages — no PPO term when
                    # distill mode is enabled (distill + KL only).
                    if sirl_distill_enabled:
                        advantages = torch.zeros_like(advantages)

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    # CPR replaces (not adds to) the PPO policy loss. Running
                    # the PPO forward here would (a) waste compute, and more
                    # importantly (b) leave a dangling autograd graph + an
                    # FSDP+LoRA writeback record that subsequent ops (e.g.
                    # post-step validation `summon_full_params`) trip on with
                    # "Cannot writeback when the parameter shape changes".
                    # Stub PPO scalars so the downstream metrics dict still
                    # has the expected keys.
                    if cpr_dpo_mode:
                        zero = torch.zeros((), device=get_device_id(), dtype=torch.float32)
                        pg_loss = zero
                        pg_clipfrac = zero
                        ppo_kl = zero
                        pg_clipfrac_lower = zero
                        policy_loss = zero
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy
                        )

                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                        if self.config.policy_loss.loss_mode == "vanilla":
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                        else:
                            policy_loss_fn = get_policy_loss_fn(loss_mode)
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                                old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, self.config
                            )

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            # compute policy loss
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_prob = data["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # SIRL distillation loss: weighted SFT on gated improvement tokens
                    if sirl_distill_enabled:
                        gate_mask = data["sirl_gate_mask"]
                        gate_weight = data["sirl_gate_weight"]
                        n_gated = gate_mask.float().sum().clamp(min=1)

                        if gate_mask.any():
                            imp_micro_batch = {
                                "input_ids": data["sirl_imp_input_ids"],
                                "attention_mask": data["sirl_imp_attention_mask"],
                                "position_ids": data["sirl_imp_position_ids"],
                                "responses": data["sirl_imp_responses"],
                            }
                            _, imp_log_probs = self._forward_micro_batch(
                                imp_micro_batch, temperature=temperature,
                            )
                            # Build response mask from the improvement attention mask
                            imp_resp_len = data["sirl_imp_responses"].size(1)
                            imp_resp_mask = data["sirl_imp_attention_mask"][:, -imp_resp_len:]
                            # Per-sample mean negative log-prob (cross-entropy)
                            per_sample_ce = (-imp_log_probs * imp_resp_mask).sum(dim=-1) / (
                                imp_resp_mask.sum(dim=-1).clamp(min=1)
                            )
                            # Weighted by gate weight, masked to gated samples only
                            distill_loss = (gate_weight * per_sample_ce * gate_mask.float()).sum() / n_gated
                        else:
                            distill_loss = torch.tensor(0.0, device=policy_loss.device)

                        policy_loss = policy_loss + sirl_distill_loss_coeff * distill_loss
                        micro_batch_metrics["actor/distill_loss"] = distill_loss.detach().item()
                        micro_batch_metrics["actor/gate_fraction"] = gate_mask.float().mean().item()

                    if cpr_dpo_mode:
                        required_cpr_keys = (
                            "cpr_p_pref",
                            "cpr_a_input_ids",
                            "cpr_a_attention_mask",
                            "cpr_a_position_ids",
                            "cpr_a_responses",
                            "cpr_a_loss_mask",
                            "cpr_b_input_ids",
                            "cpr_b_attention_mask",
                            "cpr_b_position_ids",
                            "cpr_b_responses",
                            "cpr_b_loss_mask",
                            "cpr_a_ref_logprobs",
                            "cpr_b_ref_logprobs",
                        )
                        missing_cpr_keys = [key for key in required_cpr_keys if key not in data]
                        if missing_cpr_keys:
                            raise RuntimeError(
                                "CPR enabled but paired fresh-forward tensors are missing from the actor micro-batch: "
                                + ", ".join(missing_cpr_keys)
                            )

                        cpr_a_batch = {
                            "input_ids": data["cpr_a_input_ids"],
                            "attention_mask": data["cpr_a_attention_mask"],
                            "position_ids": data["cpr_a_position_ids"],
                            "responses": data["cpr_a_responses"],
                        }
                        cpr_b_batch = {
                            "input_ids": data["cpr_b_input_ids"],
                            "attention_mask": data["cpr_b_attention_mask"],
                            "position_ids": data["cpr_b_position_ids"],
                            "responses": data["cpr_b_responses"],
                        }

                        # --- FSDP+LoRA fix (debate 2026-04-19) -----------------
                        # Two sequential _forward_micro_batch calls each trigger
                        # FSDP unflatten/flatten under use_orig_params; with LoRA
                        # the second forward's writeback lands on a shape that
                        # does not match the base flat view, firing the assertion
                        # "Cannot writeback when parameter shape changes" at the
                        # next summon_full_params (post-step validation).
                        #
                        # Fix: concat the paired batches along dim=0 into ONE
                        # forward, split the output. One FSDP cycle per step.
                        # Standard DPO-pair trick (see TRL DPOTrainer).
                        b_a = cpr_a_batch["input_ids"].size(0)
                        b_b = cpr_b_batch["input_ids"].size(0)
                        t_resp_a = cpr_a_batch["responses"].size(-1)
                        t_resp_b = cpr_b_batch["responses"].size(-1)
                        # The concat+split slicing assumes the prompt region has
                        # the same width in a and b (so "last t_resp tokens" of
                        # each padded row is the response region). This holds
                        # today because build_cpr_paired_dataproto uses a shared
                        # questions list; it breaks under prefix_branch pair-
                        # source (revise-prompt vs original). Fast-fail here
                        # rather than silently slice the wrong window.
                        assert (
                            cpr_a_batch["input_ids"].size(-1) - t_resp_a
                            == cpr_b_batch["input_ids"].size(-1) - t_resp_b
                        ), (
                            "CPR paired concat requires equal prompt widths in a and b "
                            f"(seq_a={cpr_a_batch['input_ids'].size(-1)}, resp_a={t_resp_a}; "
                            f"seq_b={cpr_b_batch['input_ids'].size(-1)}, resp_b={t_resp_b})"
                        )

                        def _pad_last(t, new_len, pad_val=0):
                            if t.size(-1) >= new_len:
                                return t
                            pad_shape = list(t.shape)
                            pad_shape[-1] = new_len - t.size(-1)
                            return torch.cat(
                                [t, torch.full(pad_shape, pad_val, dtype=t.dtype, device=t.device)],
                                dim=-1,
                            )

                        t_seq = max(
                            cpr_a_batch["input_ids"].size(-1),
                            cpr_b_batch["input_ids"].size(-1),
                        )
                        t_resp = max(t_resp_a, t_resp_b)
                        cpr_cat = {
                            "input_ids": torch.cat([
                                _pad_last(cpr_a_batch["input_ids"], t_seq),
                                _pad_last(cpr_b_batch["input_ids"], t_seq),
                            ], dim=0),
                            "attention_mask": torch.cat([
                                _pad_last(cpr_a_batch["attention_mask"], t_seq, pad_val=0),
                                _pad_last(cpr_b_batch["attention_mask"], t_seq, pad_val=0),
                            ], dim=0),
                            "position_ids": torch.cat([
                                _pad_last(cpr_a_batch["position_ids"], t_seq, pad_val=0),
                                _pad_last(cpr_b_batch["position_ids"], t_seq, pad_val=0),
                            ], dim=0),
                            "responses": torch.cat([
                                _pad_last(cpr_a_batch["responses"], t_resp),
                                _pad_last(cpr_b_batch["responses"], t_resp),
                            ], dim=0),
                        }
                        _, cpr_logp_cat = self._forward_micro_batch(
                            micro_batch=cpr_cat, temperature=temperature, calculate_entropy=False
                        )
                        # cpr_logp_cat: [b_a + b_b, t_resp]. Split and truncate.
                        cpr_logp_a = cpr_logp_cat[:b_a, :t_resp_a]
                        cpr_logp_b = cpr_logp_cat[b_a:b_a + b_b, :t_resp_b]
                        # --- end FSDP+LoRA fix --------------------------------
                        if bsv_mode:
                            cpr_loss, cpr_metrics = compute_bsv_loss(
                                logp_a=cpr_logp_a,
                                logp_b=cpr_logp_b,
                                ref_logp_a=data["cpr_a_ref_logprobs"],
                                ref_logp_b=data["cpr_b_ref_logprobs"],
                                mask_a=data["cpr_a_loss_mask"],
                                mask_b=data["cpr_b_loss_mask"],
                                p_self=data["bsv_p_self"],
                                p_verif=data["bsv_p_verif"],
                                I_v=data["bsv_I_v"],
                                beta=cpr_beta,
                                eta=cpr_eta,
                                anchor_mode=cpr_anchor_mode,
                            )
                        else:
                            cpr_loss, cpr_metrics = compute_cpr_loss(
                                logp_a=cpr_logp_a,
                                logp_b=cpr_logp_b,
                                ref_logp_a=data["cpr_a_ref_logprobs"],
                                ref_logp_b=data["cpr_b_ref_logprobs"],
                                mask_a=data["cpr_a_loss_mask"],
                                mask_b=data["cpr_b_loss_mask"],
                                p_pref=data["cpr_p_pref"],
                                beta=cpr_beta,
                                eta=cpr_eta,
                                anchor_mode=cpr_anchor_mode,
                            )
                        policy_loss = cpr_loss
                        micro_batch_metrics["actor/cpr_loss"] = cpr_loss.detach().item()
                        for key, value in cpr_metrics.items():
                            micro_batch_metrics[key] = value.detach().item() if torch.is_tensor(value) else float(value)

                    # -------------------------------------------------------
                    # TODO-SIRL-PREF-DPO: fresh-forward DPO loss integration.
                    # -------------------------------------------------------
                    # The DPO loss requires per-token log-probs of (x+prefix, s+)
                    # and (x+prefix, s-) under the CURRENT module state (so
                    # autograd flows through to actor parameters). The
                    # precomputed `data["sirl_pref_*_logprobs"]` tensors are
                    # detached and MUST NOT be used as the loss surface — they
                    # are retained on the batch for metrics/diagnostics only.
                    #
                    # Correct wiring requires a fresh _forward_micro_batch on
                    # the paired (x+prefix, s±) input_ids and integrating
                    # those logps into compute_sirl_pref_dpo_loss. That's a
                    # non-trivial change to the micro-batch loop (pair-aware
                    # batching, attention/position handling for the paired
                    # sequences, loss aggregation under dynamic_bsz). Tracked
                    # separately; do NOT call compute_sirl_pref_dpo_loss here.
                    if sirl_pref_dpo_mode:
                        # Emit a metric so training dashboards show DPO mode
                        # is active even before the loss is wired.
                        micro_batch_metrics["sirl_pref/dpo_mode_active"] = 1.0

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
