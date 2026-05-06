"""DUET verl pipeline.

Generates a self-contained bash launcher under
``<workspace>/duet/generated/run_duet_verl[_<exp_id>].sh`` that calls
``verl.trainer.main_ppo`` with DUET-specific hydra overrides.

LoRA is enabled by default (rank=32, alpha=16, target=all-linear). Pass
``--lora-rank 0`` to ``run_duet_verl.py`` to run full-parameter training.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .data_utils import DatasetError, resolve_dataset
from .model_store import (
    ResolvedModelReference,
    ensure_model_available,
    hf_env_overrides,
    resolve_model_reference,
    shell_export_lines,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT / "verl_workspace"
DEFAULT_VERL_RUNTIME_ROOT = PROJECT_ROOT / "src" / "verl_runtime"


@dataclass
class PipelineArtifacts:
    method: str
    runtime_root: Path
    launch_cwd: Path
    script_path: Path
    metadata_path: Path
    staged_files: list[Path]
    required_model_paths: list[Path]
    script_text: str
    metadata: dict[str, Any]


def _resolve_tasks(dataset: str, dataset_path: str | None, max_samples: int) -> tuple[str, list[dict[str, Any]], Path]:
    """Resolve dataset to (kind, tasks, source_path). Accepts math or code."""
    dataset_type, tasks, source_path = resolve_dataset(dataset, dataset_path, max_samples)
    if dataset_type not in ("math", "code"):
        raise DatasetError(
            f"{dataset} resolved to {dataset_type}; SIRL pipeline supports math or code only."
        )
    return dataset_type, tasks, source_path


def _resolve_math_tasks(dataset: str, dataset_path: str | None, max_samples: int) -> tuple[list[dict[str, Any]], Path]:
    dtype, tasks, source_path = _resolve_tasks(dataset, dataset_path, max_samples)
    if dtype != "math":
        raise DatasetError(f"{dataset} resolved to {dtype}; this entry point requires math data.")
    return tasks, source_path


def _dataset_stem(dataset: str, dataset_path: str | None) -> str:
    if dataset_path:
        return Path(dataset_path).stem.replace("_", "-")
    return dataset


def _normalize_run_component(value: str) -> str:
    return value.strip().replace("/", "-").replace("-", "_").replace(".", "_")


def _model_run_tag(model_ref: ResolvedModelReference) -> str:
    label = model_ref.alias or model_ref.requested or model_ref.local_path.name
    return _normalize_run_component(label)


def _task_no_split(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic 80/20 train/val split (seed 42 for reproducibility)."""
    import random as _random
    n = len(tasks)
    if n < 10:
        return list(tasks), list(tasks)
    perm = list(range(n))
    _random.Random(42).shuffle(perm)
    n_train = int(0.8 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    train_tasks = [tasks[i] for i in train_idx]
    val_tasks = [tasks[i] for i in val_idx]
    return train_tasks, val_tasks


def _write_json_array(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=True, indent=2)


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


def _make_verl_math_record(task: dict[str, Any], idx: int, split: str, data_source: str) -> dict[str, Any]:
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": task["question"]}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": task["answer"]},
        "extra_info": {"split": split, "index": task.get("task_id", f"{data_source}-{idx}")},
    }


_CODE_PROMPT_TEMPLATE = (
    "Write a Python function to solve the following task.\n\n"
    "{prompt}\n\n"
    "The function must be named `{entry_point}`. "
    "Return ONLY the Python code inside a single ```python ... ``` fenced block."
)

_APPS_PROMPT_TEMPLATE = (
    "Write a complete Python program that reads from standard input and writes "
    "to standard output to solve the following task.\n\n"
    "{prompt}\n"
    "{starter_hint}"
    "Return ONLY the Python code inside a single ```python ... ``` fenced block. "
    "The program must use `input()` / `sys.stdin` for input and `print()` for "
    "output — do NOT wrap the solution in a function unless required by the "
    "problem statement."
)


def _is_apps_task(task: dict[str, Any]) -> bool:
    """APPS rows carry an `input_output` JSON string and a `difficulty` tier,
    while MBPP rows carry a `test` assertion block and `entry_point`. Detect
    APPS by presence of input_output — it's the authoritative signal."""
    return bool(task.get("input_output"))


def _make_verl_code_record(task: dict[str, Any], idx: int, split: str, data_source: str) -> dict[str, Any]:
    """Build a verl data record for a code task (MBPP- or APPS-shaped).

    MBPP shape: ``{prompt, test, entry_point, canonical_solution}`` → ground_truth
      packs ``{test, entry_point}``; verifier is the mbpp_code sandbox.
    APPS shape: ``{prompt, input_output, difficulty, starter_code}`` → ground_truth
      packs ``{input_output, difficulty}``; verifier is the apps_code sandbox
      (stdio-driven, line-by-line exact-match).
    """
    if _is_apps_task(task):
        starter = task.get("starter_code") or ""
        starter_hint = (
            f"\nStarter code (use exactly as-is):\n```python\n{starter}\n```\n"
            if starter.strip()
            else ""
        )
        question = _APPS_PROMPT_TEMPLATE.format(
            prompt=task["prompt"], starter_hint=starter_hint
        )
        gt_payload = json.dumps(
            {
                "input_output": task.get("input_output", ""),
                "difficulty": task.get("difficulty", ""),
            },
            ensure_ascii=True,
        )
    else:
        question = _CODE_PROMPT_TEMPLATE.format(
            prompt=task["prompt"], entry_point=task.get("entry_point", "solution")
        )
        gt_payload = json.dumps(
            {"test": task.get("test", ""), "entry_point": task.get("entry_point", "")},
            ensure_ascii=True,
        )
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": gt_payload},
        "extra_info": {"split": split, "index": task.get("task_id", f"{data_source}-{idx}")},
    }


def _make_verl_record(
    task: dict[str, Any], idx: int, split: str, data_source: str, kind: str
) -> dict[str, Any]:
    if kind == "code":
        return _make_verl_code_record(task, idx, split, data_source)
    return _make_verl_math_record(task, idx, split, data_source)


def _make_prompt_answer_record(task: dict[str, Any]) -> dict[str, Any]:
    return {"prompt": task["question"], "answer": task["answer"]}


def _make_prompt_answer_code_record(task: dict[str, Any]) -> dict[str, Any]:
    """Parallel of _make_prompt_answer_record for code tasks (kept for parity
    of train.json / test.json sidecar files; not consumed by verl).

    APPS rows carry `input_output` + `difficulty` instead of `test` +
    `entry_point` — emit both shapes so the sidecar captures whichever is
    present.
    """
    if _is_apps_task(task):
        return {
            "prompt": task.get("prompt", ""),
            "input_output": task.get("input_output", ""),
            "difficulty": task.get("difficulty", ""),
            "starter_code": task.get("starter_code", ""),
        }
    return {
        "prompt": task.get("prompt", ""),
        "entry_point": task.get("entry_point", ""),
        "test": task.get("test", ""),
    }


def _quote(path: Path | str) -> str:
    return shlex.quote(str(path))


def _render_script(lines: list[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"


def _model_prelude_lines(model_ref: ResolvedModelReference) -> list[str]:
    return [
        *shell_export_lines(),
        f"MODEL_CHECKPOINT={_quote(model_ref.local_path)}",
        'if [ ! -d "$MODEL_CHECKPOINT" ]; then',
        f"  echo {shlex.quote(f'Missing local model checkpoint at {model_ref.local_path}.')} >&2",
        f"  echo {shlex.quote(f'Download it with: {model_ref.download_command}')} >&2",
        "  exit 2",
        "fi",
    ]


def _runtime_prelude_lines(runtime_root: Path, python_bin: Path) -> list[str]:
    package_path = runtime_root / "verl"
    return [
        f"PROJECT_ROOT={_quote(PROJECT_ROOT)}",
        f"RUNTIME_ROOT={_quote(runtime_root)}",
        f"PYTHON_BIN={_quote(python_bin)}",
        f"if [ ! -d {_quote(package_path)} ]; then",
        f"  echo {shlex.quote(f'Missing vendored verl runtime package at {package_path}.')} >&2",
        "  exit 2",
        "fi",
        'if [ ! -x "$PYTHON_BIN" ]; then',
        "  echo 'Missing launcher Python executable.' >&2",
        "  exit 2",
        "fi",
        'export PYTHONPATH="$RUNTIME_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
        "unset PYTHONHOME",
        "export PYTHONNOUSERSITE=1",
    ]


def _runtime_env_lines() -> list[str]:
    return [
        "unset RAY_ADDRESS",
        "unset RAY_NAMESPACE",
    ]


def _module_preflight_lines(module_names: list[str]) -> list[str]:
    return [
        '"$PYTHON_BIN" - <<\'PY\'',
        "import importlib.util",
        "import sys",
        f"module_names = {module_names!r}",
        "missing = [name for name in module_names if importlib.util.find_spec(name) is None]",
        "if missing:",
        "    raise SystemExit('Missing Python modules under ' + sys.executable + ': ' + ', '.join(missing))",
        "PY",
    ]


def _duet_override_lines(args: Any) -> list[str]:
    """Emit hydra overrides for the DUET joint controller (paper §5).

    Call only when ``--duet-enable``. All knobs live under ``+duet.*``;
    they are consumed in :class:`RayPPOTrainer.__init__` to construct the
    DUET controller (see ``verl/trainer/ppo/duet_controller.py``).
    """
    return [
        "  +duet.enable=True \\",
        f"  +duet.budget={args.duet_budget} \\",
        f"  +duet.surrogate_path={shlex.quote(args.duet_surrogate_path)} \\",
        f"  +duet.eps_pre={args.duet_eps_pre} \\",
        f"  +duet.eps_len={args.duet_eps_len} \\",
        f"  +duet.rng_seed={args.duet_rng_seed} \\",
        f"  +duet.bisection_iters={args.duet_bisection_iters} \\",
        f"  +duet.ref_rollout_n={args.duet_ref_rollout_n} \\",
        f"  +duet.stop_signal={shlex.quote(args.duet_stop_signal)} \\",
        f"  +duet.stop_floor={args.duet_stop_floor} \\",
        f"  +duet.min_tokens={args.duet_min_tokens} \\",
        f"  +duet.threshold_mode={shlex.quote(args.duet_threshold_mode)} \\",
        f"  +duet.top_k={args.duet_top_k} \\",
        f"  +duet.hysteresis_k={args.duet_hysteresis_k} \\",
        f"  +duet.marker_domain={shlex.quote(args.duet_marker_domain)} \\",
        f"  +duet.grace_window={args.duet_grace_window} \\",
        f"  +duet.abort_eps={args.duet_abort_eps} \\",
        f"  +duet.k_history={args.duet_k_history} \\",
        f"  +duet.k_update_every={args.duet_k_update_every} \\",
        f"  +duet.n_min={args.duet_n_min} \\",
        f"  +duet.n_max={args.duet_n_max} \\",
    ]


def _resolve_rollout_backend_lines() -> list[str]:
    return [
        "ROLLOUT_BACKEND=vllm",
        'if ! "$PYTHON_BIN" -c "import vllm.model_executor.models.registry" 2>/dev/null; then',
        "  ROLLOUT_BACKEND=hf",
        "  echo 'Warning: vLLM unavailable, falling back to HF rollout.' >&2",
        "fi",
    ]


def _make_experiment_dir(
    *,
    experiments_root: Path,
    run_tag: str | None,
    date_str: str | None,
) -> Path:
    """Create ``experiments/YYYYMMDD/duet_run_N/`` with auto-incrementing N."""
    date = date_str or datetime.now().strftime("%Y%m%d")
    day_dir = experiments_root / date

    prefix = "duet_run_"
    if run_tag:
        experiment_dir = day_dir / f"{prefix}{run_tag}"
    else:
        day_dir.mkdir(parents=True, exist_ok=True)
        existing = [d.name for d in day_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
        max_n = 0
        for name in existing:
            try:
                max_n = max(max_n, int(name[len(prefix):]))
            except ValueError:
                pass
        experiment_dir = day_dir / f"{prefix}{max_n + 1}"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def _git_info(repo_path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"git_hash": None, "git_branch": None, "git_dirty": None}
    try:
        info["git_hash"] = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip() or None
        info["git_branch"] = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip() or None
        info["git_dirty"] = bool(subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    except Exception:
        pass
    return info


def _write_experiment_config(
    experiment_dir: Path,
    args: Any,
    model_ref: ResolvedModelReference,
    hyperparams: dict[str, Any],
    method_specific: dict[str, Any],
) -> Path:
    config = {
        "experiment_id": experiment_dir.name,
        "date": experiment_dir.parent.name,
        "method": "duet",
        "condition": "predictable",
        "model": model_ref.alias or model_ref.local_path.name,
        "model_path": str(model_ref.local_path),
        "dataset": getattr(args, "dataset", "unknown"),
        **_git_info(PROJECT_ROOT),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": hyperparams,
        "method_specific": method_specific,
        "checkpoint_dir": str(experiment_dir / "checkpoints"),
    }
    config_path = experiment_dir / "experiment_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return config_path


def _build_duet_predictable(args: Any) -> PipelineArtifacts:
    runtime_root = Path(args.runtime_root)
    if not (runtime_root / "verl").is_dir():
        raise FileNotFoundError(f"Vendored verl runtime not found at {runtime_root / 'verl'}")

    # .absolute() (not .resolve()) — venv/bin/python is a symlink to the system
    # interpreter; resolving it would leak /usr/bin/python3.12 into the generated
    # launcher, bypassing the venv's site-packages.
    python_bin = Path(args.python_bin or sys.executable).expanduser().absolute()

    # Auto-download model if missing (or just resolve reference)
    if args.auto_download_model:
        model_ref = ensure_model_available(args.model, auto_download=True)
    else:
        model_ref = resolve_model_reference(args.model, require_exists=False)

    # Dataset split resolution:
    # - If --val-dataset (or --val-dataset-path) is set, load train and val
    #   from pre-computed split files; skip the 80/20 fallback split entirely.
    # - Otherwise, load a single dataset and 80/20 it deterministically.
    val_dataset_name = getattr(args, "val_dataset", None)
    val_dataset_path_arg = getattr(args, "val_dataset_path", None)
    if val_dataset_name or val_dataset_path_arg:
        train_kind, train_tasks, source_path = _resolve_tasks(
            args.dataset, args.dataset_path, args.max_samples
        )
        val_kind, val_tasks, _ = _resolve_tasks(
            val_dataset_name or args.dataset, val_dataset_path_arg, args.max_samples
        )
        if train_kind != val_kind:
            raise DatasetError(
                f"train/val dataset kinds differ: train={train_kind!r} val={val_kind!r}"
            )
        dataset_kind_resolved = train_kind
    else:
        dataset_kind_resolved, tasks, source_path = _resolve_tasks(
            args.dataset, args.dataset_path, args.max_samples
        )
        train_tasks, val_tasks = _task_no_split(tasks)
    data_tag = args.task_name or f"{_dataset_stem(args.dataset, args.dataset_path).upper()}-SIRL"
    # When a separate val dataset is provided, tag test records with its name so
    # TB metrics surface as val-core/<VAL_NAME>-SIRL/... instead of borrowing
    # the train tag (which previously made mbpp-test val read as MBPP-TRAIN-SIRL).
    if val_dataset_name and val_dataset_name != args.dataset:
        val_data_tag = f"{_dataset_stem(val_dataset_name, val_dataset_path_arg).upper()}-SIRL"
    else:
        val_data_tag = data_tag

    method_root = Path(args.output_root) / "duet"
    data_root = method_root / data_tag
    train_json = data_root / "train.json"
    test_json = data_root / "test.json"
    train_parquet = data_root / "train.parquet"
    test_parquet = data_root / "test.parquet"

    if dataset_kind_resolved == "code":
        raw_train_records = [{"source": data_tag, **_make_prompt_answer_code_record(task)} for task in train_tasks]
        raw_test_records = [{"source": val_data_tag, **_make_prompt_answer_code_record(task)} for task in val_tasks]
    else:
        raw_train_records = [{"source": data_tag, **_make_prompt_answer_record(task)} for task in train_tasks]
        raw_test_records = [{"source": val_data_tag, **_make_prompt_answer_record(task)} for task in val_tasks]
    _write_json_array(train_json, raw_train_records)
    _write_json_array(test_json, raw_test_records)

    train_records = [_make_verl_record(task, idx, "train", data_tag, dataset_kind_resolved) for idx, task in enumerate(train_tasks)]
    test_records = [_make_verl_record(task, idx, "test", val_data_tag, dataset_kind_resolved) for idx, task in enumerate(val_tasks)]
    _write_parquet(train_parquet, train_records)
    _write_parquet(test_parquet, test_records)

    # Additional val datasets for cross-dataset evaluation (T17).
    # Each alias is resolved to its own parquet + distinct data_source tag,
    # so verl's validation loop emits per-source metrics
    # (val-core/<tag>/acc/mean@1 etc.). Primary test set remains the `source`
    # of the training checkpoint; additional sets only contribute to eval.
    additional_val_parquets: list[Path] = []
    additional_val_kinds: set[str] = set()
    additional_val_arg = getattr(args, "additional_val_datasets", None)
    if additional_val_arg:
        additional_aliases = [a.strip() for a in additional_val_arg.split(",") if a.strip()]
        for alias in additional_aliases:
            extra_kind, extra_tasks, _ = _resolve_tasks(alias, None, None)
            # Mixed-kind val (e.g. math train + HumanEval code val) is allowed;
            # the unified_router_reward dispatches per data_source. Only math
            # and code are supported — other kinds would need a new reward path.
            if extra_kind not in ("math", "code"):
                raise DatasetError(
                    f"additional val dataset {alias!r} has kind {extra_kind!r}; "
                    f"unified router supports math or code only."
                )
            additional_val_kinds.add(extra_kind)
            extra_tag = f"{_dataset_stem(alias, None).upper()}-SIRL"
            extra_parquet = data_root / f"test_{alias.replace('-', '_')}.parquet"
            extra_records = [
                _make_verl_record(task, idx, "test", extra_tag, extra_kind)
                for idx, task in enumerate(extra_tasks)
            ]
            _write_parquet(extra_parquet, extra_records)
            additional_val_parquets.append(extra_parquet)
    mixed_val_kinds = bool(additional_val_kinds - {dataset_kind_resolved})

    experiment_dir = _make_experiment_dir(
        experiments_root=Path(args.experiments_root),
        run_tag=args.run_tag,
        date_str=args.experiment_date,
    )
    output_dir = experiment_dir / "checkpoints"

    adv_estimator = args.adv_estimator
    experiment_id = (args.experiment_id or "")

    suffix = f"-{experiment_id}" if experiment_id else ""
    experiment_name = f"{data_tag}-{adv_estimator}{suffix}-{_model_run_tag(model_ref)}"

    # Throughput tuning: resolve auto values from total sequence length
    total_seq_len = args.max_prompt_length + args.max_response_length
    # vLLM V1 appends the sampled token BEFORE running stop-checks, so a request
    # that legally reaches (prompt + max_response) length briefly produces a
    # len=(total_seq_len+1) sequence and trips an assert in the engine
    # ("Sampled token IDs exceed the max model length"). Give the engine 1 token
    # of head-room. Keep the PPO/ref token budgets on the real sequence length.
    rollout_max_model_len = total_seq_len + 1
    ppo_max_token_len = (
        int(args.ppo_max_token_len_per_gpu) if int(getattr(args, "ppo_max_token_len_per_gpu", 0)) > 0
        else total_seq_len * 2
    )
    max_num_batched_tokens = (
        int(args.max_num_batched_tokens) if int(getattr(args, "max_num_batched_tokens", 0)) > 0
        else max(total_seq_len * 4, 8192)
    )
    # Chunked-prefill guard: vllm_rollout_spmd.py:135 rejects configs where
    # max_num_batched_tokens < max_model_len. Raise the batched-tokens budget
    # if the head-room bump pushed past it.
    if max_num_batched_tokens < rollout_max_model_len:
        max_num_batched_tokens = rollout_max_model_len
    use_dynamic_bsz = str(getattr(args, "use_dynamic_bsz", "true")).lower() in ("true", "1", "yes")
    use_dynamic_bsz_str = "True" if use_dynamic_bsz else "False"

    # FSDP master-weight dtype. verl's default is fp32 for the actor, which doubles
    # the FSDP.summon_full_params footprint used by the LoRA sharding manager
    # (fsdp_vllm.py:139). For LoRA training on long-context SIRL this peak is the
    # dominant OOM source when vLLM already pins >50% of VRAM.
    model_dtype = getattr(args, "model_dtype", "bf16") or "bf16"

    # vLLM tensor-parallel size for rollout. N×TP groups each share weights and
    # KV cache across N GPUs instead of every GPU holding its own replica. For
    # 4B+ on 8 GPUs, TP=2 roughly halves rollout wall-time (follow-ups included).
    rollout_tp_size = int(getattr(args, "rollout_tp_size", 1) or 1)
    if rollout_tp_size < 1:
        rollout_tp_size = 1
    if args.gpus_per_node % rollout_tp_size != 0:
        raise ValueError(
            f"--rollout-tp-size={rollout_tp_size} must evenly divide "
            f"--gpus-per-node={args.gpus_per_node}."
        )

    # Persistent per-run artifacts inside the experiment dir: train.log (tee of
    # stdout) and tensorboard events. experiment_config.json is already written
    # alongside. These survive the run regardless of Ray session cleanup.
    train_log_path = experiment_dir / "train.log"
    tensorboard_dir = experiment_dir / "tensorboard"
    ray_logs_backup = experiment_dir / "ray_worker_logs"

    _write_experiment_config(
        experiment_dir, args, model_ref,
        hyperparams={
            "train_examples": len(train_tasks), "val_examples": len(val_tasks),
            "actor_lr": args.actor_lr, "kl_loss_coef": 0.001, "kl_loss_type": "low_var_kl",
            "kl_ctrl_coef": 0.0, "rollout_n": args.samples_per_prompt,
            "train_batch_size": args.train_batch_size, "total_epochs": args.episodes,
            "max_prompt_length": args.max_prompt_length, "max_response_length": args.max_response_length,
            "temperature": args.temperature, "gpu_memory_utilization": args.gpu_memory_utilization,
            "save_freq": args.save_freq, "resume_mode": "disable",
            "advantage_estimator": adv_estimator,
            "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
            "optim_betas": [args.beta1, args.beta2],
            "model_dtype": model_dtype,
            "ppo_max_token_len_per_gpu": ppo_max_token_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "rollout_tensor_model_parallel_size": rollout_tp_size,
        },
        method_specific={
            "adv_estimator": adv_estimator,
        },
    )

    # Verifier selection. This clean repo supports math (ttrl_math) and
    # code (mbpp_code / humaneval-style asserts) verifiers; LLM-judge is
    # math-only. The mixed router (math+code in a single run) and the
    # standalone code router were retired alongside the SIRL/CPR/BSV cleanup.
    _verifier_mode = getattr(args, "verifier_mode", "ground_truth")
    if mixed_val_kinds:
        raise DatasetError(
            "Mixed math+code val datasets require the retired unified router; "
            "run math and code datasets in separate experiments."
        )
    if dataset_kind_resolved == "code":
        reward_path = runtime_root / "verl" / "utils" / "reward_score" / "mbpp_code" / "__init__.py"
    elif _verifier_mode == "llm_judge":
        reward_path = runtime_root / "verl" / "utils" / "reward_score" / "llm_judge_math" / "__init__.py"
    else:
        reward_path = runtime_root / "verl" / "utils" / "reward_score" / "ttrl_math" / "__init__.py"

    lora_lines: list[str] = []
    if args.lora_rank > 0:
        # FSDP+LoRA+vLLM-sync interaction: default load_format=dummy_dtensor
        # leaves base_sync_done=False, which forces __collect_lora_params down
        # the whole-module summon_full_params path in fsdp_vllm.py:139. Under
        # FSDP1 + use_orig_params=True + PEFT this fires a FlatParam shape
        # assertion at vLLM-sync time (see doc/cpr_smoke_findings.md, 04-19
        # update). Switching to safetensors primes base_sync_done=True, which
        # routes LoRA collection through layered_summon_lora_params (per-
        # submodule), sidestepping the assertion.
        lora_lines = [
            f"  actor_rollout_ref.model.lora_rank={args.lora_rank} \\",
            f"  actor_rollout_ref.model.lora_alpha={args.lora_alpha} \\",
            f"  actor_rollout_ref.model.target_modules={shlex.quote(args.lora_target_modules)} \\",
            "  +actor_rollout_ref.actor.fsdp_config.use_orig_params=True \\",
            "  actor_rollout_ref.rollout.load_format=safetensors \\",
            "  actor_rollout_ref.rollout.layered_summon=True \\",
        ]

    script_text = _render_script(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            *_runtime_prelude_lines(runtime_root, python_bin),
            *_runtime_env_lines(),
            *_module_preflight_lines(["hydra", "ray", "pyarrow"]),
            *_model_prelude_lines(model_ref),
            f"REWARD_FN_PATH={_quote(reward_path)}",
            # Code-domain reward mode (binary vs fractional). Always exported
            # when the resolved dataset kind is "code" so Ray workers inherit
            # it before importing the mbpp_code module. Harmless when the run
            # is on math (mbpp_code isn't imported).
            *(
                [f"export BSV_MBPP_REWARD_MODE={_quote(getattr(args, 'bsv_mbpp_reward_mode', 'binary'))}"]
                if dataset_kind_resolved == "code"
                else []
            ),
            *(
                [
                    f"export BSV_LLM_JUDGE_MODE={_quote(getattr(args, 'llm_judge_mode', 'stub_noisy'))}",
                    f"export BSV_LLM_JUDGE_URL={_quote(getattr(args, 'llm_judge_url', 'http://localhost:8000'))}",
                    f"export BSV_LLM_JUDGE_NOISE={getattr(args, 'llm_judge_noise', 0.10)}",
                    f"export BSV_LLM_JUDGE_MODEL={_quote(getattr(args, 'llm_judge_model', 'Qwen/Qwen3-4B-Instruct'))}",
                    f"export BSV_LLM_JUDGE_SEED={getattr(args, 'seed', 0) if hasattr(args, 'seed') else 0}",
                    # OpenAI-API mode requires OPENAI_API_KEY to be visible in the
                    # Ray worker env. We do NOT write the key into the generated
                    # script (it would persist on disk and in experiment
                    # manifests). Instead, fail fast with a clear message if
                    # it's not already exported in the invoking shell. Ray
                    # workers inherit env from this script's process, which
                    # inherits from the caller — typically a login shell that
                    # sourced ~/.bashrc.
                    *(
                        [
                            'if [[ -z "${OPENAI_API_KEY:-}" ]]; then',
                            '  echo "[launcher] ERROR: --llm-judge-mode openai requires OPENAI_API_KEY exported in the shell env." >&2',
                            '  echo "[launcher] Set it (e.g. in ~/.bashrc) and invoke from a shell that has it." >&2',
                            '  exit 3',
                            'fi',
                            # Do NOT echo the key; just confirm it is present.
                            'echo "[launcher] OPENAI_API_KEY present (len=${#OPENAI_API_KEY}); passed to Ray workers via env inheritance."',
                        ]
                        if getattr(args, "llm_judge_mode", "stub_noisy") == "openai"
                        else []
                    ),
                ]
                if _verifier_mode == "llm_judge"
                else []
            ),
            *_resolve_rollout_backend_lines(),
            'cd "$PROJECT_ROOT"',
            "unset VLLM_ATTENTION_BACKEND",
            'export VLLM_USE_V1="${VLLM_USE_V1:-1}"',  # respect pre-set (e.g., DUET cells set =0)
            f"export TENSORBOARD_DIR={_quote(tensorboard_dir)}",
            f"EXPERIMENT_LOG={_quote(train_log_path)}",
            f'mkdir -p {_quote(tensorboard_dir)} "$(dirname "$EXPERIMENT_LOG")"',
            f'echo "[launcher] tee-ing stdout/stderr to $EXPERIMENT_LOG"',
            "",
            'set +e',
            '("$PYTHON_BIN" -m verl.trainer.main_ppo \\',
            "  --config-name='ppo_trainer_ttrl.yaml' \\",
            f"  data.train_files=[\"{train_parquet}\"] \\",
            f"  data.val_files=[{','.join(chr(34) + str(p) + chr(34) for p in [test_parquet, *additional_val_parquets])}] \\",
            f"  data.max_prompt_length={args.max_prompt_length} \\",
            f"  data.max_response_length={args.max_response_length} \\",
            f"  data.train_batch_size={args.train_batch_size} \\",
            "  data.filter_overlong_prompts=True \\",
            "  data.truncation='error' \\",
            # Suffix prompt is overrideable via INPO_OVERRIDE_SUFFIX_PROMPT env var.
            # Default is the math suffix (preserves all math experiments unchanged);
            # code-task cells set this to "\nProvide your solution as a single complete Python code block."
            # Wrapping: outer single-quoted (shlex.quote), inner double-quoted for hydra string parsing.
            f"  +data.suffix_prompt={shlex.quote(chr(34) + os.environ.get('INPO_OVERRIDE_SUFFIX_PROMPT', '\\nPlease reason step by step, and put your final answer within \\boxed{{}}.') + chr(34))} \\",
            '  actor_rollout_ref.model.path="$MODEL_CHECKPOINT" \\',
            "  actor_rollout_ref.model.enable_gradient_checkpointing=True \\",
            "  actor_rollout_ref.model.use_remove_padding=True \\",
            *lora_lines,
            f"  actor_rollout_ref.actor.ppo_mini_batch_size={args.mini_batch_size} \\",
            f"  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.micro_batch_size} \\",
            f"  actor_rollout_ref.actor.use_dynamic_bsz={use_dynamic_bsz_str} \\",
            "  actor_rollout_ref.actor.use_kl_loss=True \\",
            f"  actor_rollout_ref.actor.optim.lr={args.actor_lr} \\",
            "  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \\",
            "  actor_rollout_ref.actor.optim.warmup_style='cosine' \\",
            f"  +actor_rollout_ref.actor.optim.betas=[{args.beta1},{args.beta2}] \\",
            # KL loss coef is overrideable via INPO_OVERRIDE_KL_LOSS_COEF (default 0.001 — math).
            # Code-task cells use 0.005 (5x stronger anchor; binary-reward variance protection).
            f"  actor_rollout_ref.actor.kl_loss_coef={os.environ.get('INPO_OVERRIDE_KL_LOSS_COEF', '0.001')} \\",
            "  actor_rollout_ref.actor.kl_loss_type=low_var_kl \\",
            "  actor_rollout_ref.actor.fsdp_config.param_offload=False \\",
            "  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \\",
            f"  +actor_rollout_ref.actor.fsdp_config.model_dtype={model_dtype} \\",
            f"  actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ppo_max_token_len} \\",
            f"  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.micro_batch_size} \\",
            f"  actor_rollout_ref.ref.log_prob_use_dynamic_bsz={use_dynamic_bsz_str} \\",
            f"  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={ppo_max_token_len} \\",
            "  actor_rollout_ref.ref.fsdp_config.param_offload=True \\",
            f"  +actor_rollout_ref.ref.fsdp_config.model_dtype={model_dtype} \\",
            '  actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND" \\',
            f"  actor_rollout_ref.rollout.temperature={args.temperature} \\",
            "  actor_rollout_ref.rollout.top_p=0.95 \\",
            "  actor_rollout_ref.rollout.enforce_eager=False \\",
            "  actor_rollout_ref.rollout.free_cache_engine=False \\",
            f"  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.micro_batch_size} \\",
            # Keep rollout-side log_prob on static batching: dynamic_bsz here
            # doesn't help SIRL follow-up generation (pure generate_sequences)
            # and can disturb vLLM's scheduler when prompts are long.
            f"  actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tp_size} \\",
            f"  actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization} \\",
            f"  actor_rollout_ref.rollout.n={args.samples_per_prompt} \\",
            "  actor_rollout_ref.rollout.val_kwargs.do_sample=True \\",
            f"  actor_rollout_ref.rollout.val_kwargs.n={args.val_samples_per_prompt} \\",
            "  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \\",
            f"  actor_rollout_ref.rollout.val_kwargs.temperature={args.temperature} \\",
            f"  actor_rollout_ref.rollout.max_model_len={rollout_max_model_len} \\",
            f"  actor_rollout_ref.rollout.max_num_batched_tokens={max_num_batched_tokens} \\",
            *([f"  actor_rollout_ref.rollout.max_num_seqs={int(args.max_num_seqs)} \\"]
              if int(getattr(args, "max_num_seqs", 0) or 0) > 0 else []),
            f"  critic.optim.lr={args.critic_lr} \\",
            "  critic.model.use_remove_padding=True \\",
            '  critic.model.path="$MODEL_CHECKPOINT" \\',
            "  critic.model.enable_gradient_checkpointing=True \\",
            f"  critic.ppo_micro_batch_size_per_gpu={args.micro_batch_size} \\",
            f"  critic.use_dynamic_bsz={use_dynamic_bsz_str} \\",
            f"  critic.ppo_max_token_len_per_gpu={ppo_max_token_len} \\",
            "  critic.model.fsdp_config.param_offload=False \\",
            "  critic.model.fsdp_config.optimizer_offload=False \\",
            f"  +critic.model.fsdp_config.model_dtype={model_dtype} \\",
            "  algorithm.kl_ctrl.kl_coef=0.00 \\",
            f"  algorithm.adv_estimator={adv_estimator} \\",
            '  custom_reward_function.path="$REWARD_FN_PATH" \\',
            "  custom_reward_function.name=reward_func \\",
            f"  reward_model.reward_manager={shlex.quote(getattr(args, 'reward_manager', 'naive'))} \\",
            "  ttrl.enable=False \\",
            # ---- DUET joint controller (paper §5) ----
            *(
                _duet_override_lines(args)
                if getattr(args, "duet_enable", False)
                else []
            ),
            f"  trainer.logger={shlex.quote(args.trainer_logger)} \\",
            f"  trainer.project_name={shlex.quote(args.project)} \\",
            f"  trainer.experiment_name={shlex.quote(experiment_name)} \\",
            f"  trainer.n_gpus_per_node={args.gpus_per_node} \\",
            "  trainer.nnodes=1 \\",
            f"  trainer.save_freq={args.save_freq} \\",
            "  trainer.max_actor_ckpt_to_keep=1 \\",
            f"  trainer.test_freq={args.test_freq} \\",
            "  trainer.resume_mode=disable \\",
            f"  trainer.default_local_dir={shlex.quote(str(output_dir))} \\",
            f"  +trainer.compute_val_aux={str(args.compute_val_aux).lower()} \\",
            f"  trainer.total_epochs={args.episodes} \"$@\") 2>&1 | tee \"$EXPERIMENT_LOG\"",
            'TRAIN_EXIT=${PIPESTATUS[0]}',
            # Ray worker stdout holds the step metrics; the driver tee captures
            # only the controller. Back them up into the experiment dir so they
            # survive /tmp/ray/ cleanup and are browsable alongside train.log.
            f'RAY_LOGS_BACKUP={_quote(ray_logs_backup)}',
            'LATEST_RAY_SESSION="$(ls -td /tmp/ray/session_* 2>/dev/null | head -1)"',
            'if [ -n "$LATEST_RAY_SESSION" ] && [ -d "$LATEST_RAY_SESSION/logs" ]; then',
            '    mkdir -p "$RAY_LOGS_BACKUP"',
            '    cp "$LATEST_RAY_SESSION"/logs/worker-*.out "$RAY_LOGS_BACKUP/" 2>/dev/null || true',
            '    cp "$LATEST_RAY_SESSION"/logs/worker-*.err "$RAY_LOGS_BACKUP/" 2>/dev/null || true',
            '    echo "[launcher] Copied Ray worker logs to $RAY_LOGS_BACKUP"',
            'fi',
            'set -e',
            'exit "$TRAIN_EXIT"',
        ]
    )

    method_root_gen = method_root / "generated"
    script_suffix = f"_{experiment_id}" if experiment_id else ""
    script_path = method_root_gen / f"run_duet_verl{script_suffix}.sh"
    metadata_path = method_root_gen / f"duet_metadata{script_suffix}.json"

    metadata = {
        "method": "duet",
        "split": "predictable",
        "source_dataset": str(source_path),
        "train_examples": len(train_tasks),
        "val_examples": len(val_tasks),
        "data_tag": data_tag,
        "experiment_dir": str(experiment_dir),
        "runtime_root": str(runtime_root),
        "launcher_python": str(python_bin),
        "model_alias": model_ref.alias,
        "model_repo_id": model_ref.repo_id,
        "model_local_path": str(model_ref.local_path),
        "model_checkpoint_present": model_ref.exists,
        "adv_estimator": adv_estimator,
        "lora_rank": args.lora_rank,
    }

    return PipelineArtifacts(
        method="duet",
        runtime_root=runtime_root,
        launch_cwd=PROJECT_ROOT,
        script_path=script_path,
        metadata_path=metadata_path,
        staged_files=[train_json, test_json, train_parquet, test_parquet],
        required_model_paths=[model_ref.local_path],
        script_text=script_text,
        metadata=metadata,
    )


def _write_artifacts(artifacts: PipelineArtifacts) -> None:
    artifacts.script_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.script_path.write_text(artifacts.script_text, encoding="utf-8")
    artifacts.script_path.chmod(0o755)
    artifacts.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.metadata_path.write_text(
        json.dumps(artifacts.metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def run_duet_predictable_pipeline(args: Any) -> int:
    try:
        artifacts = _build_duet_predictable(args)
    except DatasetError as exc:
        print(f"[pipeline] Dataset error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[pipeline] Missing file: {exc}", file=sys.stderr)
        return 2

    _write_artifacts(artifacts)
    print(f"[pipeline] Generated launcher: {artifacts.script_path}")
    print(f"[pipeline] Metadata: {artifacts.metadata_path}")
    print(f"[pipeline] Experiment: {artifacts.metadata['experiment_dir']}")
    print(f"[pipeline] Model checkpoint: {artifacts.metadata['model_local_path']}")

    if args.launch:
        missing = [p for p in artifacts.required_model_paths if not p.exists()]
        if missing:
            print(f"[pipeline] Cannot launch: missing model(s): {', '.join(str(p) for p in missing)}", file=sys.stderr)
            return 2
        env = os.environ.copy()
        env.update(hf_env_overrides())
        print(f"[pipeline] Launching {artifacts.script_path}")
        result = subprocess.run(["bash", str(artifacts.script_path)], cwd=artifacts.launch_cwd, env=env)
        return result.returncode
    return 0
