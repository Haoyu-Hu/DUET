#!/usr/bin/env bash
# Run a GRPO experiment with a single model on the math train + multi-eval
# benchmark suite. This is the canonical baseline launcher.
#
# Usage:
#   bash scripts/run_grpo.sh                                   # 1.7B default
#   MODEL=qwen3-4b-base bash scripts/run_grpo.sh
#   MODEL=llama3.2-3b-instruct bash scripts/run_grpo.sh
#   bash scripts/run_grpo.sh --episodes 4 --train-batch-size 128
#
# Environment knobs:
#   MODEL          qwen3-1.7b-base (default) | qwen3-4b-base | llama3.2-3b-instruct
#   SEED           integer (default 0)
#   TAG            run tag (default grpo_<model>)
#   GPU_MEM_UTIL   override vLLM gpu_memory_utilization
#   MAX_NUM_SEQS   override vLLM max_num_seqs
#   ROLLOUT_TP     override vLLM tensor_model_parallel_size
#   GPUS_PER_NODE  override count (default = #CUDA_VISIBLE_DEVICES)
#
# vLLM engine: V1 (default). Set VLLM_USE_V1=0 manually if needed for debugging.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Match the original launch_sirl.sh env-var setup.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DUET_GLOBAL_SEED="${SEED:-0}"
export PYTHONHASHSEED="${SEED:-0}"

# shellcheck source=_model_config.sh
source "$REPO_ROOT/scripts/_model_config.sh"
apply_model_config "${MODEL:-qwen3-1.7b-base}"

SEED="${SEED:-0}"
TAG="${TAG:-grpo_${MODEL_ALIAS}}"

ARGS=(
    --model "$MODEL_ALIAS"
    --dataset math-train-clean
    --val-dataset math500-full
    --additional-val-datasets aime2024-full,gsm8k-test,humaneval-full,gpqa-diamond
    --max-samples 7497
    --max-prompt-length 512
    --max-response-length 3072
    --train-batch-size 128
    --mini-batch-size 512
    --samples-per-prompt 8
    --val-samples-per-prompt 4
    --temperature 0.9
    --beta1 0.9
    --beta2 0.999
    --episodes 4
    --test-freq 30
    --save-freq 0
    --adv-estimator grpo
    --reward-manager prime
    --verifier-mode ground_truth
    --actor-lr 3e-6
    --gpus-per-node "$GPUS_PER_NODE"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-seqs "$MAX_NUM_SEQS"
    --rollout-tp-size "$ROLLOUT_TP"
    --seed "$SEED"
    --run-tag "$TAG"
    --experiment-id "$TAG"
    --project duet
    "${EXTRA_ARGS[@]}"
)

echo "[run_grpo] model=$MODEL_ALIAS gpus=$GPUS_PER_NODE seed=$SEED tag=$TAG"
exec python src/run_duet_verl.py "${ARGS[@]}" --launch "$@"
