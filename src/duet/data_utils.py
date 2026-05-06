"""Dataset resolution for SIRL experiments.

Supported datasets (alias -> filename under sorted_upload/datasets/):

math:
    math500-full, math500-mini, aime2024-full, aime2024-mini

code:
    humaneval-full, humaneval-mini, mbpp-full, mbpp-mini

mcq:
    mmlu-pro-validation

open-ended:
    alpaca-eval
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DatasetError(RuntimeError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets"

DATASET_SPECS: dict[str, tuple[str, str]] = {
    "math500-full": ("math", "math500_full.jsonl"),
    "math500-mini": ("math", "math500_mini.jsonl"),
    # Pre-computed stratified splits (seed=42, subject × level); see setup/prepare_splits.py.
    "math500-strat-train": ("math", "math500_strat_train.jsonl"),
    "math500-strat-val": ("math", "math500_strat_val.jsonl"),
    # N9-scale math corpora (Hendrycks MATH all 7 configs + GSM8k); see setup/prepare_splits.py.
    "math-train": ("math", "math_train.jsonl"),
    "math-train-clean": ("math", "math_train_clean.jsonl"),
    "math-test": ("math", "math_test.jsonl"),
    "gpqa-diamond": ("math", "gpqa_diamond.jsonl"),
    "gsm8k-train": ("math", "gsm8k_train.jsonl"),
    "gsm8k-test": ("math", "gsm8k_test.jsonl"),
    "aime2024-full": ("math", "aime2024_full.jsonl"),
    "aime2024-mini": ("math", "aime2024_mini.jsonl"),
    "humaneval-full": ("code", "humaneval_full.jsonl"),
    "humaneval-mini": ("code", "humaneval_mini.jsonl"),
    "mbpp-full": ("code", "mbpp_full.jsonl"),
    "mbpp-mini": ("code", "mbpp_mini.jsonl"),
    # Natural HF MBPP splits (train+validation / test); see setup/prepare_splits.py.
    "mbpp-train": ("code", "mbpp_train.jsonl"),
    "mbpp-test": ("code", "mbpp_test.jsonl"),
    # N9-scale code corpus (APPS reupload mirror); see setup/prepare_splits.py.
    # Note: APPS is stdio-driven — the sandbox verifier reads `input_output` (JSON
    # string of {inputs:[...], outputs:[...]}) instead of `test` assertions.
    # apps-test is not persisted (per-problem IO can be MBs); eval-500 is the
    # stratified held-out set produced from it.
    "apps-train": ("code", "apps_train.jsonl"),
    "apps-eval-500": ("code", "apps_eval_500.jsonl"),
    # InPo Phase-1 eval split (see src/duet/data_code_phase1.py, doc/inpo_impl_dataset.md).
    # apps-eval-500-equal: 500 rows equal-stratified (166/167/167) across difficulties,
    # problem IDs at datasets/apps_eval_500_equal_ids.txt for reproducibility.
    # NOTE: for training, use `apps-train` (full 4,450 rows) with --max-prompt-length=1024;
    # trainer handles truncation. `apps-train-1024` below is a hard-filter ablation variant.
    "apps-train-1024": ("code", "apps_train_1024.jsonl"),
    "apps-eval-500-equal": ("code", "apps_eval_500_equal.jsonl"),
    "mmlu-pro-validation": ("mcq", "mmlu_pro_validation.jsonl"),
    "alpaca-eval": ("open_ended", "alpaca_eval.jsonl"),
}


def list_available_datasets() -> list[str]:
    return sorted(DATASET_SPECS.keys())


def default_dataset_path(dataset: str) -> Path:
    if dataset not in DATASET_SPECS:
        raise DatasetError(f"Unknown dataset '{dataset}'.")
    return DATASET_DIR / DATASET_SPECS[dataset][1]


def dataset_kind(dataset: str) -> str:
    if dataset not in DATASET_SPECS:
        raise DatasetError(f"Unknown dataset '{dataset}'.")
    return DATASET_SPECS[dataset][0]


def _load_jsonl(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise DatasetError(
            f"Dataset file not found: {path}. "
            f"Run: python setup/download_datasets.py --dataset <name>"
        )
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_samples and len(items) >= max_samples:
                break
    if not items:
        raise DatasetError(f"Dataset file is empty: {path}")
    return items


def _check_fields(items: list[dict[str, Any]], fields: set[str], dataset_type: str) -> None:
    missing = [idx for idx, item in enumerate(items) if not fields.issubset(item.keys())]
    if missing:
        show = ", ".join(str(x) for x in missing[:5])
        raise DatasetError(
            f"{dataset_type} dataset missing required fields {sorted(fields)} in rows: {show}"
        )


def resolve_dataset(
    dataset: str,
    dataset_path: str | None = None,
    max_samples: int | None = None,
) -> tuple[str, list[dict[str, Any]], Path]:
    name = dataset.strip().lower()
    if dataset_path:
        path = Path(dataset_path)
        items = _load_jsonl(path, max_samples)
        first = items[0]
        if {"prompt", "test", "entry_point"}.issubset(first):
            dtype = "code"
        elif {"question", "answer"}.issubset(first) and "options" in first:
            dtype = "mcq"
        elif {"question", "answer"}.issubset(first):
            dtype = "math"
        elif {"instruction", "output"}.issubset(first):
            dtype = "open_ended"
        else:
            raise DatasetError(
                f"Could not infer dataset type from fields: {sorted(first.keys())}"
            )
        return dtype, items, path

    if name not in DATASET_SPECS:
        raise DatasetError(
            f"Unsupported dataset '{dataset}'. Available: {', '.join(list_available_datasets())}"
        )
    dtype = dataset_kind(name)
    path = default_dataset_path(name)
    items = _load_jsonl(path, max_samples)

    if dtype == "code":
        # Two supported code schemas:
        #   MBPP-style: {prompt, test, entry_point, canonical_solution}
        #   APPS-style: {prompt, input_output, difficulty}  (stdio-driven)
        mbpp_fields = {"prompt", "test", "entry_point"}
        apps_fields = {"prompt", "input_output"}
        if mbpp_fields.issubset(items[0].keys()):
            _check_fields(items, mbpp_fields, "code(mbpp)")
        elif apps_fields.issubset(items[0].keys()):
            _check_fields(items, apps_fields, "code(apps)")
        else:
            raise DatasetError(
                f"Code dataset row 0 missing either MBPP-style {sorted(mbpp_fields)} or "
                f"APPS-style {sorted(apps_fields)}. Got fields: {sorted(items[0].keys())}"
            )
    elif dtype == "math":
        _check_fields(items, {"question", "answer"}, dtype)
    elif dtype == "mcq":
        _check_fields(items, {"question", "options", "answer"}, dtype)
    elif dtype == "open_ended":
        _check_fields(items, {"instruction", "output"}, dtype)

    return dtype, items, path
