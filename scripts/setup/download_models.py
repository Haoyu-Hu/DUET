#!/usr/bin/env python
"""Download DUET target models from Hugging Face.

Supports the three models anchored by this repo: qwen3-1.7b-base (default),
qwen3-4b-base, llama3.2-3b-instruct. Each is cached under
``<repo>/model_checkpoints/<alias>/``.

Usage:
    python scripts/setup/download_models.py --list
    python scripts/setup/download_models.py --model qwen3-1.7b-base
    python scripts/setup/download_models.py --all
    python scripts/setup/download_models.py --model meta-llama/Llama-3.2-3B-Instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from duet.model_store import (  # noqa: E402
    MODEL_SPECS,
    download_model_snapshot,
    list_model_specs,
    resolve_model_reference,
)

SUPPORTED = ["qwen3-1.7b-base", "qwen3-4b-base", "llama3.2-3b-instruct"]


def list_models() -> int:
    print(f"{'alias':<24} {'repo_id':<40} present")
    print("-" * 80)
    for spec in list_model_specs():
        if spec.alias not in SUPPORTED:
            continue
        resolved = resolve_model_reference(spec.alias, require_exists=False)
        mark = "yes" if resolved.exists else "no"
        print(f"{spec.alias:<24} {spec.repo_id:<40} {mark}")
    return 0


def download_one(name: str, force: bool) -> int:
    resolved = resolve_model_reference(name, require_exists=False)
    if resolved.alias not in SUPPORTED and resolved.alias is not None:
        print(f"[warn] {name} resolves to alias {resolved.alias!r} which is not "
              f"in the supported set {SUPPORTED}; downloading anyway.")
    if resolved.exists and not force:
        print(f"[skip] {name} already present at {resolved.local_path}")
        return 0
    print(f"[download] {name} ({resolved.repo_id}) -> {resolved.local_path}")
    download_model_snapshot(name)
    print(f"[done]     {name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", action="append", default=[],
                        help="Alias or HF repo id (repeatable).")
    parser.add_argument("--all", action="store_true",
                        help="Download all three supported models.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        return list_models()

    targets = list(SUPPORTED) if args.all else list(args.model)
    if not targets:
        parser.print_usage(sys.stderr)
        print("\nPick: --model <alias>, --all, or --list.", file=sys.stderr)
        return 2

    rc = 0
    for name in targets:
        rc |= download_one(name, args.force)
    return rc


if __name__ == "__main__":
    sys.exit(main())
