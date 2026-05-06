# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any, Dict, List, Union

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)
        max_num_seqs = self.config.get("max_num_seqs", None)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # ARRoL faithful (paper §4.3, isolated patch): when the trainer sets
        # arrol_faithful_active=True in meta_info, install the mid-decode
        # pruning patch BEFORE generation and tear it down AFTER. The patch
        # is in a separate module (arrol_vllm_patch.py) and only imported
        # under this flag — non-faithful paths see zero overhead.
        _arrol_faithful_active = bool(prompts.meta_info.get("arrol_faithful_active", False))
        _arrol_faithful_diagnostics: dict | None = None
        _arrol_faithful_local_head = None
        _arrol_faithful_decisions: list = []
        if _arrol_faithful_active:
            import pickle as _pkl_arrol
            from .arrol_vllm_patch import apply_arrol_patch
            from duet.arrol_faithful_head import (
                ArrolFaithfulConfig as _AFConfig,
                ArrolFaithfulHead as _AFHead,
            )
            # Reconstruct head from the trainer-shipped blob.
            _cfg_dict = _pkl_arrol.loads(prompts.meta_info["arrol_faithful_cfg_blob"])
            _arrol_faithful_local_head = _AFHead(_AFConfig(**_cfg_dict))
            _arrol_faithful_local_head.load_state_dict(
                _pkl_arrol.loads(prompts.meta_info["arrol_faithful_head_blob"])
            )
            # Restore _step_count from trainer (otherwise the worker-side head
            # is reconstructed in eternal warmup → decide_keep always returns
            # True → n_pruned=0). Confirmed bug in arrol_tiny v1.
            _arrol_faithful_local_head._step_count = int(
                prompts.meta_info.get("arrol_faithful_step_count", 0)
            )
            # Move head to CUDA so post-warmup MLP forward matches the device
            # of hidden_state (which arrives from vLLM forward on cuda:0).
            # Without this, decide_keep raises addmm device-mismatch and the
            # patch swallows the exception. Confirmed bug in arrol_tiny v2.
            if torch.cuda.is_available():
                _arrol_faithful_local_head = _arrol_faithful_local_head.to("cuda")
            _arrol_faithful_local_head.eval()
            _arrol_l_detect = int(prompts.meta_info.get("arrol_faithful_l_detect", 512))
            # Capturing closure: stash decisions for trainer to consume.
            # Hidden states from the backbone forward come in bf16; we cast to
            # float32 once for both the head's MLP (which carries fp32 weights)
            # AND for the .numpy() persistence (numpy has no bfloat16 dtype —
            # unconverted bfloat16 raises "unsupported ScalarType BFloat16",
            # which the patch swallows in its try/except, leaving n_seen=0).
            def _arrol_callback(req_id, hidden):
                hidden_f32 = hidden.detach().to(torch.float32)
                keep, p = _arrol_faithful_local_head.decide_keep(hidden_f32)
                _arrol_faithful_decisions.append({
                    "request_id": req_id,
                    "hidden": hidden_f32.cpu().numpy(),
                    "p_keep": p,
                    "keep": keep,
                })
                return keep, p
            apply_arrol_patch(
                self.inference_engine,
                callback=_arrol_callback,
                l_detect=_arrol_l_detect,
            )

        # DUET joint controller (paper §5): the trainer pre-replicates the
        # batch by per-prompt n_q and stashes per-replica STOP THRESHOLDS in
        # non_tensor_batch (one per replica). When present we attach a
        # per-prompt DuetStopProcessor LogitsProcessor that forces EOS when
        # the chosen confidence signal crosses the threshold (paper §5.2
        # T2). vLLM uses the SHARED config.response_length max_tokens cap
        # for all prompts → uniform response shape across DP workers.
        _duet_stop_threshold = non_tensor_batch.pop("duet_stop_threshold", None)
        _duet_signal_mode = prompts.meta_info.get("duet_stop_signal_mode")
        _duet_eps_len = float(prompts.meta_info.get("duet_stop_eps_len", 0.05))
        _duet_min_tokens = int(prompts.meta_info.get("duet_stop_min_tokens", 32))
        _duet_top_k = int(prompts.meta_info.get("duet_stop_top_k", 3))
        _duet_hysteresis_k = int(prompts.meta_info.get("duet_stop_hysteresis_k", 1))
        # v3 marker-gated knobs (None disables → falls back to v2 LP behavior)
        _duet_marker_domain = prompts.meta_info.get("duet_stop_marker_domain", None)
        _duet_k1 = prompts.meta_info.get("duet_stop_k1", None)
        _duet_k2 = prompts.meta_info.get("duet_stop_k2", None)
        _duet_grace = int(prompts.meta_info.get("duet_stop_grace", 150))
        _duet_abort_eps = float(prompts.meta_info.get("duet_stop_abort_eps", 0.05))
        # Hash-based RNG seed for ε-keep lottery (post 2026-04-28 fix). Combines
        # a per-run base, the current global_step, and the row index — so the
        # ε_abort Bernoulli outcome is independent across batches even when
        # row indices repeat. Falls back to row index alone when meta_info
        # absent (legacy / unit-test path).
        _duet_rng_base = int(prompts.meta_info.get("duet_stop_rng_seed_base", 0))
        _duet_rng_step = int(prompts.meta_info.get("duet_stop_global_step", 0))
        if _duet_stop_threshold is not None and len(_duet_stop_threshold) != len(vllm_inputs):
            raise RuntimeError(
                f"DUET stop_threshold length {len(_duet_stop_threshold)} != "
                f"vllm_inputs length {len(vllm_inputs)}; trainer-side "
                f"replication is out of sync with rollout shape."
            )

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            if _duet_stop_threshold is not None and _duet_signal_mode is not None:
                # DUET active: build per-prompt SamplingParams each carrying its
                # own DuetStopProcessor. SHARED max_tokens=config.response_length
                # so response shape is uniform across DP workers (no concat issue).
                from copy import copy as _copy
                from duet.duet_logits_processor import DuetStopProcessor
                from duet.duet_marker_detector import make_detector as _make_md
                _eos = int(eos_token_id) if not isinstance(eos_token_id, list) else int(eos_token_id[0])
                # v3: build a single shared marker detector if domain is set
                # (stateless across rollouts → safe to share). When None,
                # processor falls back to v2 behavior.
                _shared_marker_detector = None
                if _duet_marker_domain is not None:
                    # vLLMRollout (this class) doesn't store self.tokenizer
                    # (only vLLMAsyncRollout does). Pull tokenizer from the
                    # vLLM engine directly — it owns one for its own decoding.
                    try:
                        _md_tokenizer = self.inference_engine.get_tokenizer()
                    except Exception:
                        # Fallback path for older vLLM versions
                        try:
                            _md_tokenizer = self.inference_engine.llm_engine.tokenizer.tokenizer
                        except Exception:
                            _md_tokenizer = None
                    if _md_tokenizer is not None:
                        _shared_marker_detector = _make_md(
                            tokenizer=_md_tokenizer,
                            domain=str(_duet_marker_domain),
                        )
                    else:
                        # Detector unavailable → fall back to legacy v2 LP
                        # behavior. Print once for diagnosis.
                        if os.environ.get("DUET_TRACE", "0") == "1":
                            print("[DUET-TRACE] marker detector tokenizer unavailable; "
                                  "falling back to v2 LP", flush=True)
                _per_prompt_sp = []
                _per_prompt_lps = []   # keep refs to read did_abort/saw_marker after
                for _i, _thr in enumerate(_duet_stop_threshold):
                    _sp = _copy(self.sampling_params)
                    _sp.n = 1  # DUET pre-replicates; vLLM must not multiply
                    # Per-rollout RNG seed: hash(base, step, row_idx). Mask to
                    # 31 bits so it fits in random.Random's int domain.
                    _row_rng = (
                        hash(("duet_eps_keep", _duet_rng_base,
                              _duet_rng_step, _i)) & 0x7FFFFFFF
                    )
                    _lp = DuetStopProcessor(
                        eos_token_id=_eos,
                        threshold=float(_thr),
                        signal_mode=str(_duet_signal_mode),
                        eps_len=_duet_eps_len,
                        min_tokens=_duet_min_tokens,
                        rng_seed=_row_rng,
                        top_k=_duet_top_k,
                        hysteresis_k=_duet_hysteresis_k,
                        marker_detector=_shared_marker_detector,
                        k1=int(_duet_k1) if _duet_k1 is not None else None,
                        k2=int(_duet_k2) if _duet_k2 is not None else None,
                        grace_window=_duet_grace,
                        abort_eps=_duet_abort_eps,
                    )
                    _sp.logits_processors = [_lp]
                    _per_prompt_sp.append(_sp)
                    _per_prompt_lps.append(_lp)
                if os.environ.get("DUET_TRACE", "0") == "1":
                    print(f"[DUET-TRACE] vllm generate with per-prompt LP: "
                          f"|inputs|={len(vllm_inputs)}, signal={_duet_signal_mode}, "
                          f"thr_range=[{min(_duet_stop_threshold):.3f}, "
                          f"{max(_duet_stop_threshold):.3f}]", flush=True)
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=_per_prompt_sp,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )
            else:
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            # Skip post-generate _repeat_interleave when DUET supplied a
            # per-prompt SamplingParams list (each with n=1) — the trainer
            # already pre-replicated by n_q. vLLM returned exactly len(idx)
            # outputs, so repeating idx by config.n would create a shape
            # mismatch (idx grows, response stays).
            if (self.sampling_params.n > 1 and do_sample
                    and _duet_stop_threshold is None):
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(
                        non_tensor_batch["tools_kwargs"], self.sampling_params.n
                    )
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(
                        non_tensor_batch["interaction_kwargs"], self.sampling_params.n
                    )
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(
                        non_tensor_batch["raw_prompt"], self.sampling_params.n
                    )

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # DUET v3: stash per-rollout marker/abort flags for trainer-side
        # gradient masking. did_abort=True → response_mask zeroed → zero
        # gradient contribution AND no PPO compute on these tokens.
        # saw_marker=True → reward signal expected to be valid → keep at full
        # SNIPS weight. Plumbed through non_tensor_batch (length matches
        # batch_size since each prompt has one LP).
        if (_duet_stop_threshold is not None
                and _duet_signal_mode is not None
                and _per_prompt_lps):
            non_tensor_batch["duet_did_abort"] = np.array(
                [bool(lp.did_abort) for lp in _per_prompt_lps],
                dtype=bool,
            )
            non_tensor_batch["duet_saw_marker"] = np.array(
                [bool(lp.saw_marker) for lp in _per_prompt_lps],
                dtype=bool,
            )
            # Post 2026-04-28 fix: ε-kept rollouts have saw_marker=False AND
            # did_abort=False AND eps_kept=True. The 1/abort_eps SNIPS factor
            # is applied to *only* these (not natural-EOS-before-K2 rollouts).
            non_tensor_batch["duet_eps_kept"] = np.array(
                [bool(getattr(lp, "eps_kept", False)) for lp in _per_prompt_lps],
                dtype=bool,
            )

        # ARRoL faithful: tear down the patch and ship decisions back to trainer
        # for label collection + online BCE training. Decisions blob lives in
        # the OUTPUT DataProto's meta_info (NOT non_tensor_batch — the latter
        # has a strict batch_size invariant that a length-1 blob would violate).

        _output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        if _arrol_faithful_active:
            import pickle as _pkl_arrol
            from .arrol_vllm_patch import deactivate_arrol_patch
            _arrol_faithful_diagnostics = deactivate_arrol_patch()
            _output.meta_info["arrol_faithful_decisions_blob"] = _pkl_arrol.dumps(
                _arrol_faithful_decisions
            )
            _output.meta_info["arrol_faithful_diag_blob"] = _pkl_arrol.dumps(
                _arrol_faithful_diagnostics or {}
            )
        return _output


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
