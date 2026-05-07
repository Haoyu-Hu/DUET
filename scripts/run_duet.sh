#!/usr/bin/env bash
# Run a DUET experiment with a single model on the math train + multi-eval
# benchmark suite.
#
# DUET requires per-request vLLM LogitsProcessors, which the V1 engine in
# vLLM 0.9.2 does not support. This wrapper FORCES V0 (VLLM_USE_V1=0) before
# vLLM is imported.
#
# Usage:
#   bash scripts/run_duet.sh                                   # 1.7B default
#   DUET_BUDGET=0.25 bash scripts/run_duet.sh
#   MODEL=qwen3-4b-base DUET_BUDGET=0.5 bash scripts/run_duet.sh
#
# Environment knobs:
#   MODEL              qwen3-1.7b-base (default) | qwen3-4b-base | llama3.2-3b-instruct
#   SEED               integer (default 0)
#   TAG                run tag (default duet_<model>_b<budget>)
#   DUET_BUDGET        fraction of GRPO full-budget (0.25 / 0.5 / 1.0; default 0.5)
#   DUET_SURROGATE     path to ridge weights JSON (default outputs/duet/ridge_weights.json)
#   DUET_MARKER_DOMAIN "none" (default; allocator+max-tokens only — paper-faithful baseline)
#                      "math" / "code" / "generic" (also enables marker-gated abort)
#   DUET_GRACE_WINDOW  marker-gated grace tokens after K2 (default 150;
#                      irrelevant when MARKER_DOMAIN=none)
#   DUET_K_WARMUP      σ̂ warmup observation count (default 1; do not regress to 2)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# DUET requires V0 vLLM. Must be set BEFORE python imports vLLM.
export VLLM_USE_V1=0

# shellcheck source=_model_config.sh
source "$REPO_ROOT/scripts/_model_config.sh"
apply_model_config "${MODEL:-qwen3-1.7b-base}"

SEED="${SEED:-0}"
DUET_BUDGET="${DUET_BUDGET:-0.5}"
DUET_SURROGATE="${DUET_SURROGATE:-outputs/duet/ridge_weights.json}"
DUET_MARKER_DOMAIN="${DUET_MARKER_DOMAIN:-none}"
DUET_GRACE_WINDOW="${DUET_GRACE_WINDOW:-150}"
DUET_K_WARMUP="${DUET_K_WARMUP:-1}"
TAG="${TAG:-duet_${MODEL_ALIAS}_b$(echo "$DUET_BUDGET" | tr '.' 'p')}"

# DUET pre-replicates per-prompt rollouts in ray_trainer; vLLM SamplingParams.n
# must stay 1 to avoid double-replication.
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
    --samples-per-prompt 1
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
    --duet-enable
    --duet-budget "$DUET_BUDGET"
    --duet-surrogate-path "$DUET_SURROGATE"
    --duet-eps-pre 0.05
    --duet-eps-len 0.05
    --duet-stop-signal max_prob
    --duet-stop-floor 0.5
    --duet-min-tokens 32
    --duet-threshold-mode linear
    --duet-top-k 3
    --duet-hysteresis-k 1
    --duet-marker-domain "$DUET_MARKER_DOMAIN"
    --duet-grace-window "$DUET_GRACE_WINDOW"
    --duet-abort-eps 0.05
    --duet-k-history 1024
    --duet-k-update-every 10
    --duet-n-min 1
    --duet-n-max 32
    --duet-ref-rollout-n 8
    --duet-rng-seed "$SEED"
    "${EXTRA_ARGS[@]}"
)

# σ̂ warmup observation count (cuts training time, see paper §6 ablation).
export DUET_S_HAT_K_WARMUP="$DUET_K_WARMUP"

echo "[run_duet] model=$MODEL_ALIAS gpus=$GPUS_PER_NODE seed=$SEED tag=$TAG"
echo "[run_duet] budget=$DUET_BUDGET surrogate=$DUET_SURROGATE k_warmup=$DUET_K_WARMUP marker_domain=$DUET_MARKER_DOMAIN"
echo "[run_duet] VLLM_USE_V1=$VLLM_USE_V1 (DUET requires V0)"
exec python src/run_duet_verl.py "${ARGS[@]}" --launch "$@"
