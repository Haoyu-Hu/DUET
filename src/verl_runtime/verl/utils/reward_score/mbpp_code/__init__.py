"""MBPP / HumanEval-style code-execution reward.

Signature matches ``ttrl_math.reward_func`` so the same verl reward_manager
dispatch path can swap between math and code by pointing ``REWARD_FN_PATH``
at this module. Ground truth is a JSON-encoded ``{test, entry_point}``
(packed in ``pipeline._make_verl_code_record``). The candidate is executed
as a Python script in a subprocess with a wall-clock timeout; score is 1.0
iff all asserts in ``test`` pass and the script exits 0.

Safety: subprocess isolation (fresh interpreter per call), hard timeout,
no shell. Do NOT run untrusted code from external sources through this path.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from typing import Any

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)

DEFAULT_TIMEOUT_S = float(os.environ.get("BSV_MBPP_TIMEOUT_S", "5"))

# Reward mode selector — set via env var at launch time.
#   binary     : score = 1.0 iff all asserts pass, else 0.0 (default).
#   fractional : score = (num passing asserts) / (total asserts), 4-level on
#                MBPP which uniformly has 3 asserts per task (→ {0, 1/3, 2/3, 1}).
# The toggle is read at every compute_score call so a single process can serve
# both modes if BSV_MBPP_REWARD_MODE is flipped between runs; in practice the
# verl reward_manager imports this module once, so the mode is fixed for the run.
REWARD_MODE_ENV = "BSV_MBPP_REWARD_MODE"
_VALID_REWARD_MODES = ("binary", "fractional")


def _reward_mode() -> str:
    mode = os.environ.get(REWARD_MODE_ENV, "binary").strip().lower()
    if mode not in _VALID_REWARD_MODES:
        raise ValueError(
            f"{REWARD_MODE_ENV}={mode!r} invalid; choose one of {_VALID_REWARD_MODES}"
        )
    return mode


def _extract_code(completion: str) -> str:
    """Extract Python source from a model completion.

    Preference order:
      1. First ```python ...``` fenced block.
      2. First ``` ...``` fenced block (no language hint).
      3. The raw completion (best-effort).
    """
    if not completion:
        return ""
    m = _CODE_FENCE_RE.search(completion)
    if m:
        return m.group(1).strip()
    return completion.strip()


def _exec_script(script: str, timeout: float) -> tuple[bool, str]:
    """Run a python script in a fresh subprocess; return (exit0, detail)."""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
            fh.write(script)
            path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                timeout=timeout,
                text=True,
                check=False,
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if proc.returncode == 0:
            return True, "ok"
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return False, "fail: " + " | ".join(tail)[:400]
    except subprocess.TimeoutExpired:
        return False, f"timeout>{timeout}s"
    except Exception as exc:  # pragma: no cover — defensive
        return False, f"exec_error: {exc}"


_CHECK_DEF_RE = re.compile(r"^\s*def\s+check\s*\(", re.MULTILINE)


def _maybe_invoke_check(test: str, entry_point: str) -> str:
    """Append ``check(<entry_point>)`` for HumanEval-style tests.

    HumanEval tests are written as ``def check(candidate): assert ...`` and
    rely on the harness invoking ``check(<entry_point>)`` to actually run
    the assertions. MBPP tests use module-level ``assert`` statements and
    don't need this. Without the invocation, HumanEval scripts trivially
    exit with code 0 (function defined, never called) and every completion
    scores 1.0.
    """
    if entry_point and _CHECK_DEF_RE.search(test):
        return f"check({entry_point})\n"
    return ""


def _run_candidate_binary(
    code: str, test: str, entry_point: str = "", timeout: float = DEFAULT_TIMEOUT_S,
) -> tuple[float, str]:
    """Binary mode: score=1.0 iff all asserts in ``test`` pass together, else 0.0."""
    if not code.strip():
        return 0.0, "no_code_extracted"
    script = code + "\n\n" + test + "\n" + _maybe_invoke_check(test, entry_point)
    ok, detail = _exec_script(script, timeout)
    return (1.0, detail) if ok else (0.0, detail)


def _split_asserts(test: str) -> list[str]:
    """Split a test string into individual assert statements (one per line).

    Preserves continuation lines by accumulating until a new assert keyword or EOF.
    MBPP tests are always single-line asserts so this is mostly a newline split,
    but we handle multi-line asserts defensively.
    """
    if not test:
        return []
    lines = test.split("\n")
    asserts: list[str] = []
    buf: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("assert ") or stripped.startswith("assert("):
            if buf:
                asserts.append("\n".join(buf).rstrip())
            buf = [line]
        elif buf:
            buf.append(line)
    if buf:
        asserts.append("\n".join(buf).rstrip())
    return [a for a in asserts if a.strip()]


def _run_candidate_fractional(
    code: str, test: str, entry_point: str = "", timeout: float = DEFAULT_TIMEOUT_S
) -> tuple[float, str]:
    """Fractional mode: run each assert independently, return passes/total.

    Each assert is executed in its own subprocess with the candidate code
    prepended, so candidate-defined symbols (e.g. helper classes like MBPP's
    ``Pair``) remain in scope. Scores are on a {0, 1/N, ..., N/N} grid where N
    is the number of asserts in the test string (3 for every MBPP row).

    HumanEval-style tests (``def check(candidate): assert ...``) are
    pre-flattened: the assert bodies are dedented out of the ``check``
    function body and re-substituted with ``<entry_point>`` for ``candidate``,
    so each assert can run independently in its own subprocess.
    """
    if not code.strip():
        return 0.0, "no_code_extracted"
    if entry_point and _CHECK_DEF_RE.search(test):
        # Flatten HumanEval-style ``def check(candidate): ...`` into
        # module-level asserts that reference the entry point directly.
        flat = re.sub(r"^\s*def\s+check\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\)\s*:\s*$",
                      "", test, count=1, flags=re.MULTILINE)
        # Dedent any line that was inside the ``check`` body and rebind
        # ``candidate`` -> entry_point.
        dedented = []
        for line in flat.splitlines():
            stripped = line.lstrip()
            if stripped:
                dedented.append(stripped.replace("candidate", entry_point))
            else:
                dedented.append("")
        test = "\n".join(dedented)
    asserts = _split_asserts(test)
    if not asserts:
        return 0.0, "no_asserts_parsed"
    # Per-assert timeout budget: split the total evenly so the whole compute_score
    # stays bounded by DEFAULT_TIMEOUT_S, avoiding 3× wall-time on pathological
    # candidates that infinite-loop on every assert.
    per_assert_timeout = max(1.0, timeout / len(asserts))
    passes = 0
    fail_snippets: list[str] = []
    for idx, a in enumerate(asserts):
        ok, detail = _exec_script(code + "\n\n" + a + "\n", per_assert_timeout)
        if ok:
            passes += 1
        elif len(fail_snippets) < 2:
            # Keep first couple of failure details for debugging, dropped otherwise.
            fail_snippets.append(f"assert{idx}:{detail[:80]}")
    n = len(asserts)
    score = passes / n
    detail = f"passes={passes}/{n}"
    if fail_snippets:
        detail += " | " + " ; ".join(fail_snippets)
    return score, detail


def _run_candidate(
    code: str, test: str, entry_point: str = "", timeout: float = DEFAULT_TIMEOUT_S,
) -> tuple[float, str]:
    """Dispatch to binary or fractional runner based on BSV_MBPP_REWARD_MODE."""
    if _reward_mode() == "fractional":
        return _run_candidate_fractional(code, test, entry_point, timeout)
    return _run_candidate_binary(code, test, entry_point, timeout)


def compute_score(model_response: str, ground_truth: Any) -> dict[str, Any]:
    """Score a single code rollout.

    Returns a dict with the same keys as ``ttrl_math.compute_score`` so that
    verl's reward_manager and the BSV metric layer consume it uniformly.
    """
    # Parse ground_truth: accept JSON string {test, entry_point} OR a bare
    # test string with entry_point living in extra_info (not used here).
    test_src = ""
    entry_point = ""
    if isinstance(ground_truth, dict):
        test_src = ground_truth.get("test", "")
        entry_point = ground_truth.get("entry_point", "")
    elif isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
            if isinstance(gt, dict):
                test_src = gt.get("test", "")
                entry_point = gt.get("entry_point", "")
            else:
                test_src = ground_truth
        except json.JSONDecodeError:
            test_src = ground_truth

    code = _extract_code(model_response or "")
    format_score = 1.0 if code else 0.0

    if not test_src:
        return {
            "score": 0.0,
            "format_score": format_score,
            "acc": False,
            "extracted_gt": entry_point,
            "pred": code[:200],
        }

    score, detail = _run_candidate(code, test_src, entry_point)
    # acc remains binary ("all asserts passed") for compatibility with verl's
    # reward_manager and the paper's pass@1 headline, independent of reward mode.
    acc = score >= 1.0 - 1e-9
    return {
        "score": float(score),
        "format_score": format_score,
        "acc": bool(acc),
        "extracted_gt": entry_point,
        "pred": code[:200],
        "detail": detail,
        "reward_mode": _reward_mode(),
    }


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    """verl reward_manager entry point. Mirrors ttrl_math.reward_func."""
    try:
        res = compute_score(solution_str, ground_truth)
        if isinstance(res, dict):
            return res
        if isinstance(res, (int, float, bool)):
            return float(res)
        return float(res[0])
    except Exception as exc:
        print(f"[ERROR] mbpp_code.reward_func failed: {exc}")
        traceback.print_exc()
        raise
