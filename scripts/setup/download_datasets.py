#!/usr/bin/env python
"""Build DUET training + evaluation datasets.

Writes JSONL files under ``<repo>/datasets/`` from HuggingFace sources.
Aliases match what duet/data_utils.py expects:

    math-train-clean    -> RUC-AIBOX/STILL-3-Preview-RL-Data + cleaning  (train)
    math500-full        -> HuggingFaceH4/MATH-500          (val/eval)
    aime2024-full       -> HuggingFaceH4/aime_2024         (eval)
    gsm8k-test          -> openai/gsm8k                    (eval)
    gpqa-diamond        -> Idavidrein/gpqa                 (eval)
    humaneval-full      -> openai/openai_humaneval         (eval)

Usage:
    python scripts/setup/download_datasets.py --list
    python scripts/setup/download_datasets.py --dataset math-train-clean
    python scripts/setup/download_datasets.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets"


def _load(repo: str, split: str, **kwargs: Any):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies first: bash scripts/setup/install.sh"
        ) from exc
    return load_dataset(repo, split=split, **kwargs)


def _write(records: list[dict[str, Any]], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def prepare_math500() -> Path:
    ds = _load("HuggingFaceH4/MATH-500", "test")
    rows = [
        {
            "task_id": f"math500-{i}",
            "question": r["problem"],
            "answer": r["answer"],
            "level": r.get("level"),
            "subject": r.get("subject"),
        }
        for i, r in enumerate(ds)
    ]
    return _write(rows, DATASET_DIR / "math500_full.jsonl")


def prepare_math_train_clean() -> Path:
    ds = _load("RUC-AIBOX/STILL-3-Preview-RL-Data", "train")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, r in enumerate(ds):
        q = (r.get("question") or r.get("problem") or "").strip()
        a = (r.get("answer") or r.get("gt_answer") or "").strip()
        if not q or not a or q in seen:
            continue
        seen.add(q)
        rows.append({
            "task_id": f"math-train-{i}",
            "question": q,
            "answer": a,
            "level": r.get("level"),
            "subject": r.get("subject", r.get("type")),
        })
    return _write(rows, DATASET_DIR / "math_train_clean.jsonl")


def prepare_aime2024() -> Path:
    ds = _load("HuggingFaceH4/aime_2024", "train")
    rows = [
        {"task_id": f"aime2024-{i}", "question": r["problem"], "answer": str(r["answer"])}
        for i, r in enumerate(ds)
    ]
    return _write(rows, DATASET_DIR / "aime2024_full.jsonl")


_GSM_FINAL = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)


def prepare_gsm8k_test() -> Path:
    ds = _load("openai/gsm8k", "test", name="main")
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(ds):
        m = _GSM_FINAL.search(r["answer"])
        gt = m.group(1).strip() if m else r["answer"].strip()
        rows.append({"task_id": f"gsm8k-test-{i}", "question": r["question"], "answer": gt})
    return _write(rows, DATASET_DIR / "gsm8k_test.jsonl")


def prepare_gpqa_diamond() -> Path:
    ds = _load("Idavidrein/gpqa", "train", name="gpqa_diamond")
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(ds):
        question = r["Question"]
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        rows.append({
            "task_id": f"gpqa-diamond-{i}",
            "question": question,
            "answer": r["Correct Answer"],
            "choices": choices,
            "subject": r.get("Subdomain"),
        })
    return _write(rows, DATASET_DIR / "gpqa_diamond.jsonl")


def prepare_humaneval() -> Path:
    ds = _load("openai/openai_humaneval", "test")
    rows = [
        {
            "task_id": r["task_id"],
            "prompt": r["prompt"],
            "entry_point": r["entry_point"],
            "test": r["test"],
            "canonical_solution": r.get("canonical_solution", ""),
        }
        for r in ds
    ]
    return _write(rows, DATASET_DIR / "humaneval_full.jsonl")


HANDLERS: dict[str, Callable[[], Path]] = {
    "math-train-clean": prepare_math_train_clean,
    "math500-full":     prepare_math500,
    "aime2024-full":    prepare_aime2024,
    "gsm8k-test":       prepare_gsm8k_test,
    "gpqa-diamond":     prepare_gpqa_diamond,
    "humaneval-full":   prepare_humaneval,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", action="append", default=[],
                        help="Dataset alias (repeatable).")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"Datasets (output dir: {DATASET_DIR}):")
        for name in HANDLERS:
            out = DATASET_DIR / (name.replace("-", "_") + ".jsonl")
            mark = "[present]" if out.exists() else ""
            print(f"  {name:<20} -> {out.name}  {mark}")
        return 0

    targets = sorted(HANDLERS) if args.all else args.dataset
    if not targets:
        parser.print_usage(sys.stderr)
        print("\nPass --dataset <name>, --all, or --list.", file=sys.stderr)
        return 2

    for name in targets:
        if name not in HANDLERS:
            print(f"[error] unknown dataset: {name}", file=sys.stderr)
            return 2
        print(f"[prepare] {name}")
        out = HANDLERS[name]()
        print(f"[done]    {name} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
