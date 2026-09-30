#!/usr/bin/env bash
# Minimal V1 verification: GRPO (full budget) vs DUET on the same vLLM V1
# engine, same model, data, seed and hardware, then a side-by-side summary.
#
# Usage:
#   bash scripts/verify_v1.sh                      # GRPO + DUET@50%, seed 0
#   DUET_BUDGETS="0.5 1.0" SEEDS="0 1 2" bash scripts/verify_v1.sh
#   SMOKE=1 bash scripts/verify_v1.sh              # ~8 steps each: plumbing check
#
# Env knobs: MODEL (default qwen3-1.7b-base), SEEDS (default "0"),
#   DUET_BUDGETS (default "0.5"), SMOKE (0/1), SKIP_GRPO (0/1).
# Writes experiments/<date>/*/ as usual and prints the summary from
# scripts/compare_runs.py at the end (also saved next to the runs).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-qwen3-1.7b-base}"
SEEDS="${SEEDS:-0}"
DUET_BUDGETS="${DUET_BUDGETS:-0.5}"
SMOKE="${SMOKE:-0}"
SKIP_GRPO="${SKIP_GRPO:-0}"
export MODEL

SMOKE_ARGS=()
SUFFIX="v1"
if [[ "$SMOKE" == "1" ]]; then
    # 1024 prompts x 1 epoch at batch 128 = 8 steps; validate at the end.
    SMOKE_ARGS=(--max-samples 1024 --episodes 1 --test-freq 8)
    SUFFIX="v1smoke"
fi

TB_ARGS=()
for SEED in $SEEDS; do
    export SEED
    if [[ "$SKIP_GRPO" != "1" ]]; then
        TAG="grpo_${MODEL}_${SUFFIX}_s${SEED}" bash scripts/run_grpo.sh "${SMOKE_ARGS[@]}"
        TB_ARGS+=(--run "GRPO s${SEED}=grpo_${MODEL}_${SUFFIX}_s${SEED}")
    fi
    for B in $DUET_BUDGETS; do
        BT="$(echo "$B" | tr '.' 'p')"
        TAG="duet_${MODEL}_b${BT}_${SUFFIX}_s${SEED}" DUET_BUDGET="$B" \
            bash scripts/run_duet.sh "${SMOKE_ARGS[@]}"
        TB_ARGS+=(--run "DUET ${B} s${SEED}=duet_${MODEL}_b${BT}_${SUFFIX}_s${SEED}")
    done
done

python scripts/compare_runs.py "${TB_ARGS[@]}" \
    | tee "experiments/verify_${SUFFIX}_$(date +%Y%m%d_%H%M%S).txt"
