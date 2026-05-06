# shellcheck shell=bash
# Per-model overrides anchored for the three supported architectures.
#
# Sourced by scripts/run_grpo.sh and scripts/run_duet.sh. Sets:
#   MODEL_ALIAS, GPU_MEM_UTIL, MAX_NUM_SEQS, ROLLOUT_TP, GPUS_PER_NODE,
#   EXTRA_ARGS=(...)        # any additional --foo bar pairs
#
# Usage in caller:
#   source "$(dirname "${BASH_SOURCE[0]}")/_model_config.sh"
#   apply_model_config "${MODEL:-qwen3-1.7b-base}"
#
# Lock: any model not in the case below is rejected at call time. Add a new
# branch here to support an additional model.

apply_model_config() {
    local model="${1:?MODEL alias required (qwen3-1.7b-base | qwen3-4b-base | llama3.2-3b-instruct)}"
    EXTRA_ARGS=()
    case "$model" in
        qwen3-1.7b-base)
            MODEL_ALIAS="qwen3-1.7b-base"
            GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
            MAX_NUM_SEQS="${MAX_NUM_SEQS:-0}"
            ROLLOUT_TP="${ROLLOUT_TP:-1}"
            ;;
        qwen3-4b-base)
            MODEL_ALIAS="qwen3-4b-base"
            GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
            # 4B + V0 vLLM thrashes KV-cache without a max_num_seqs cap.
            MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
            ROLLOUT_TP="${ROLLOUT_TP:-2}"
            ;;
        llama3.2-3b-instruct)
            MODEL_ALIAS="llama3.2-3b-instruct"
            GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"
            MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
            ROLLOUT_TP="${ROLLOUT_TP:-2}"
            # Llama-3.2-3B-Instruct uses a chat template; pass through chat-style
            # prompt encoding via verl's default tokenizer config.
            ;;
        *)
            echo "[model-config] Unsupported model alias: $model" >&2
            echo "[model-config] Supported: qwen3-1.7b-base, qwen3-4b-base, llama3.2-3b-instruct" >&2
            return 2
            ;;
    esac

    # Default to all available GPUs in CUDA_VISIBLE_DEVICES.
    if [[ -z "${GPUS_PER_NODE:-}" ]]; then
        if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            GPUS_PER_NODE=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
        else
            GPUS_PER_NODE=1
        fi
    fi

    return 0
}
