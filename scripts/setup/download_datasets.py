#!/usr/bin/env python
"""Build DUET training + evaluation datasets.

Writes JSONL files under ``<repo>/datasets/`` from HuggingFace sources.
Aliases match what duet/data_utils.py expects:

    math-train-clean    -> Hendrycks MATH train (lighteval/MATH or hendrycks/competition_math),
                            deduplicated against MATH-500 test                  (train)
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


_BOXED_RE = re.compile(r"\\boxed\s*\{")


def _extract_boxed(solution: str) -> str | None:
    """Return the content of the LAST ``\\boxed{...}`` in the solution.

    Handles arbitrarily nested braces by counting depth. Returns None if no
    boxed answer is found.
    """
    if not solution:
        return None
    matches = list(_BOXED_RE.finditer(solution))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        ch = solution[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i].strip()
        i += 1
    return None


# Subject configs in the order the DUET training file was built (alphabetical
# by config name). Iteration order drives ``math-train-<subject>-<idx>``
# task_id assignment, so this MUST match the source-of-truth builder
# (``setup/prepare_splits.py::math_full``).
_HENDRYCKS_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def prepare_math_train_clean() -> Path:
    """Hendrycks MATH train, deduplicated vs MATH-500 test.

    Source: ``EleutherAI/hendrycks_math`` (7 subject configs; train split).
    Mirrors the authoritative builder in ``setup/prepare_splits.py::math_full``
    so the produced ``math_train_clean.jsonl`` is byte-aligned with the file
    used by all DUET / GRPO experiments.

    Pipeline:
      1. Iterate the 7 configs in fixed order; preserve in-config row order.
      2. Extract the answer from the LAST ``\\boxed{...}`` in ``solution``
         (drop rows with no boxed answer — ttrl_math can't grade them).
      3. Normalize ``"Level 5"`` -> ``5`` (int).
      4. Dedup against MATH-500 ``problem`` strings to remove the 1 leakage row.
      5. Emit ``task_id = "math-train-<subject>-<idx>"`` (per-subject index,
         pre-dedup) — matches the working file's task_ids exactly.
    """
    from datasets import load_dataset

    seen_in_test: set[str] = set()
    try:
        test_ds = _load("HuggingFaceH4/MATH-500", "test")
        seen_in_test = {r["problem"].strip() for r in test_ds}
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    dropped_no_boxed = dropped_leakage = 0
    for cfg in _HENDRYCKS_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        for idx, row in enumerate(ds):
            solution = row.get("solution") or ""
            answer = _extract_boxed(solution)
            if answer is None:
                dropped_no_boxed += 1
                continue
            # Preserve raw problem text (including trailing \n if present) —
            # matches the working math_train_clean.jsonl byte layout.
            problem = row.get("problem") or ""
            if problem.strip() in seen_in_test:
                dropped_leakage += 1
                continue
            raw_level = row.get("level")
            level: Any = raw_level
            if isinstance(raw_level, str) and raw_level.startswith("Level "):
                try:
                    level = int(raw_level.split()[-1])
                except ValueError:
                    pass
            rows.append({
                "task_id": f"math-train-{cfg}-{idx}",
                "question": problem,
                "answer": answer,
                "level": level,
                "subject": row.get("type") or cfg,
                "solution": solution,
            })
    if dropped_no_boxed:
        print(f"[math-train-clean] dropped {dropped_no_boxed} rows (no \\boxed{{}})")
    if dropped_leakage:
        print(f"[math-train-clean] dropped {dropped_leakage} rows leaking MATH-500")
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
    """GPQA-Diamond as multiple-choice with single-letter ground truth.

    Matches the format the math verifier (ttrl_math) can grade: the prompt
    enumerates four options, the canonical answer is "A"/"B"/"C"/"D" so the
    model's `\\boxed{<letter>}` extraction matches.
    """
    import random as _random
    ds = _load("Idavidrein/gpqa", "train", name="gpqa_diamond")
    rng = _random.Random(42)
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(ds):
        question = r["Question"].strip()
        opts = [
            r["Correct Answer"],
            r["Incorrect Answer 1"],
            r["Incorrect Answer 2"],
            r["Incorrect Answer 3"],
        ]
        order = list(range(4))
        rng.shuffle(order)
        letters = ["A", "B", "C", "D"]
        rendered_choices = "\n".join(
            f"({letters[j]}) {opts[order[j]]}" for j in range(4)
        )
        correct_letter = letters[order.index(0)]
        prompt = (
            f"{question}\n\n\n{rendered_choices}\n\n"
            "Please write your final answer in the form of "
            "\\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}"
        )
        rows.append({
            "task_id": f"gpqa-diamond-{i}",
            "question": prompt,
            "answer": correct_letter,
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
