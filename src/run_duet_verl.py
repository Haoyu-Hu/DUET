#!/usr/bin/env python
"""DUET verl pipeline entry point.

Generates a self-contained launcher under
``verl_workspace/duet/generated/run_duet_verl[_<exp_id>].sh`` and optionally
invokes it directly via ``--launch``.

Typical usage (called by scripts/run_grpo.sh or scripts/run_duet.sh):

    python src/run_duet_verl.py \\
        --dataset math-train-clean --val-dataset math500-full \\
        --model qwen3-1.7b-base --adv-estimator grpo \\
        --episodes 4 --samples-per-prompt 8 \\
        --train-batch-size 128 --launch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from duet import DEFAULT_MODEL_ALIAS  # noqa: E402
from duet.pipeline import (  # noqa: E402
    DEFAULT_VERL_RUNTIME_ROOT,
    DEFAULT_WORKSPACE_ROOT,
    PROJECT_ROOT,
    run_duet_predictable_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DUET verl pipeline entry.")

    # Dataset
    parser.add_argument("--dataset", default="math-train-clean",
                        help="Training dataset alias (or use --dataset-path for a custom file).")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--val-dataset", default=None,
                        help="Validation dataset alias. If unset, falls back to deterministic "
                             "80/20 split of --dataset.")
    parser.add_argument("--val-dataset-path", default=None)
    parser.add_argument("--additional-val-datasets", default=None,
                        help="Comma-separated extra val aliases (e.g. 'gsm8k-test,humaneval-full'). "
                             "Each emits a distinct data_source tag for per-set val metrics.")
    parser.add_argument("--max-samples", type=int, default=7497)
    parser.add_argument("--task-name", default=None, help="Override auto-derived data tag.")

    # Model
    parser.add_argument("--model", default=DEFAULT_MODEL_ALIAS,
                        help="Alias from duet.model_store: qwen3-1.7b-base | qwen3-4b-base | "
                             "llama3p2-3b-instruct.")
    parser.add_argument("--auto-download-model", action="store_true",
                        help="Fetch the model from HuggingFace if not present locally.")

    # Runtime paths
    parser.add_argument("--output-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--runtime-root", default=str(DEFAULT_VERL_RUNTIME_ROOT),
                        help="Path to vendored verl runtime (contains verl/ package).")
    parser.add_argument("--experiments-root", default=str(PROJECT_ROOT / "experiments"))
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--experiment-date", default=None)
    parser.add_argument("--launch", action="store_true")

    # Method selection
    parser.add_argument("--adv-estimator", default="grpo",
                        choices=["grpo"],
                        help="Advantage estimator. Locked to grpo at this stage; "
                             "DUET / DAPO / ARRoL / VIP all build on a GRPO-style advantage.")
    parser.add_argument("--experiment-id", default="",
                        help="Optional suffix for experiment name and generated launcher filename.")

    # Training hyperparameters
    parser.add_argument("--project", default="duet")
    parser.add_argument("--episodes", type=int, default=4, help="Total training epochs.")
    parser.add_argument("--samples-per-prompt", type=int, default=8, help="rollout.n")
    parser.add_argument("--val-samples-per-prompt", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--mini-batch-size", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=3072)
    parser.add_argument("--actor-lr", type=float, default=3e-6)
    parser.add_argument("--critic-lr", type=float, default=9e-6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", default="true")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--save-freq", type=int, default=0)
    parser.add_argument("--test-freq", type=int, default=30)
    parser.add_argument("--compute-val-aux", action="store_true", default=False)
    parser.add_argument("--trainer-logger", default="['console','tensorboard']")
    parser.add_argument("--seed", type=int, default=0)

    # Throughput tuning
    parser.add_argument("--use-dynamic-bsz", default="true")
    parser.add_argument("--ppo-max-token-len-per-gpu", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=0,
                        help="vLLM scheduler concurrent-sequence cap. Set to 128 for 4B + V0 "
                             "DUET runs to bypass KV-cache preemption.")
    parser.add_argument("--model-dtype", default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--rollout-tp-size", type=int, default=1)

    # Optimizer
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="Set to 0 to disable LoRA and run full-parameter training.")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-target-modules", default="all-linear")

    # Reward / verifier
    parser.add_argument("--reward-manager", default="prime",
                        choices=["naive", "prime"])
    parser.add_argument("--verifier-mode", default="ground_truth",
                        choices=["ground_truth", "llm_judge"])

    # ---- DUET joint controller (paper §5) ----
    parser.add_argument("--duet-enable", action="store_true", default=False,
                        help="Enable the DUET joint controller (paper §5). Pre-rollout per-prompt "
                             "rollout-count + per-prompt max-token cap derived from a ridge surrogate.")
    parser.add_argument("--duet-budget", type=float, default=0.5,
                        help="Token budget as a fraction of GRPO's full-budget step "
                             "(B / B_full). 0.25 / 0.5 / 1.0 are the paper sweep values.")
    parser.add_argument("--duet-surrogate-path", type=str,
                        default="outputs/duet/ridge_weights.json",
                        help="Path to the pre-fit ridge weights JSON.")
    parser.add_argument("--duet-eps-pre", type=float, default=0.05)
    parser.add_argument("--duet-eps-len", type=float, default=0.05)
    parser.add_argument("--duet-rng-seed", type=int, default=0)
    parser.add_argument("--duet-bisection-iters", type=int, default=10)
    parser.add_argument("--duet-stop-signal", type=str, default="max_prob",
                        choices=["max_prob", "entropy", "mean_logprob", "margin", "top_k_mass"])
    parser.add_argument("--duet-threshold-mode", type=str, default="linear",
                        choices=["linear", "log_scale", "batch_percentile"])
    parser.add_argument("--duet-top-k", type=int, default=3)
    parser.add_argument("--duet-hysteresis-k", type=int, default=1)
    parser.add_argument("--duet-marker-domain", type=str, default="math",
                        choices=["none", "math", "code", "generic"],
                        help="If set (math/code/generic), enables marker-gated termination: "
                             "the LP only fires AFTER an answer marker is detected. 'none' "
                             "= legacy un-marker-gated stop.")
    parser.add_argument("--duet-grace-window", type=int, default=150,
                        help="Tokens after K2 to wait for a marker before aborting.")
    parser.add_argument("--duet-abort-eps", type=float, default=0.05)
    parser.add_argument("--duet-k-history", type=int, default=1024)
    parser.add_argument("--duet-k-update-every", type=int, default=10)
    parser.add_argument("--duet-n-min", type=int, default=1)
    parser.add_argument("--duet-n-max", type=int, default=32)
    parser.add_argument("--duet-ref-rollout-n", type=int, default=8,
                        help="Reference per-prompt rollout count used to compute "
                             "B_full = M × ref_rollout_n × max_response_length.")
    parser.add_argument("--duet-stop-floor", type=float, default=0.5)
    parser.add_argument("--duet-min-tokens", type=int, default=32)

    return parser


def _apply_global_seed() -> int:
    """Seed torch / numpy / python-random from DUET_GLOBAL_SEED."""
    import os as _os
    import random as _random
    try:
        seed = int(_os.environ.get("DUET_GLOBAL_SEED", "0"))
    except ValueError:
        seed = 0
    _random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def main() -> int:
    seed = _apply_global_seed()
    print(f"[run_duet_verl] applied DUET_GLOBAL_SEED={seed}", flush=True)
    parser = build_parser()
    args = parser.parse_args()
    return run_duet_predictable_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
