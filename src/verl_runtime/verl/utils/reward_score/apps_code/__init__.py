"""APPS stdio-driven code-execution reward.

APPS problems are stdin/stdout driven: each problem comes with a JSON-encoded
``{inputs: [str, ...], outputs: [str, ...]}`` fixture. The candidate is
executed in a subprocess; each input is piped into stdin, and the captured
stdout is compared against the expected output (whitespace-tolerant,
line-by-line). Score = pass-rate across test cases (or binary via env toggle).

Signature matches ``mbpp_code.reward_func`` and ``ttrl_math.reward_func`` so
the same verl reward_manager dispatch path can swap between them.

Safety: subprocess isolation (fresh interpreter per test case), hard per-case
timeout, no shell. Do NOT run untrusted code from external sources through
this path.
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

# Per-test-case timeout. Default 6s matches codeparrot/APPS-eval conventions;
# with up to 3 test cases per problem on the eval subset the worst-case wall
# is ~18s per rollout, well under the verl reward_manager queue budget.
DEFAULT_TIMEOUT_S = float(os.environ.get("BSV_APPS_TIMEOUT_S", "6"))

# Cap the number of test cases per rollout. APPS problems occasionally embed
# hundreds of IO pairs; running all of them serially blows up rollout latency.
# 10 is a pragmatic tradeoff: enough coverage to catch wrong answers, short
# enough that a rollout group of 8 doesn't stall the training loop.
MAX_TEST_CASES = int(os.environ.get("BSV_APPS_MAX_TESTS", "10"))

REWARD_MODE_ENV = "BSV_APPS_REWARD_MODE"
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


def _normalize_output(s: str) -> list[str]:
    """Line-by-line normalization matching the APPS convention used by
    codeparrot's eval: strip trailing whitespace from each line, drop trailing
    empty lines. Does NOT apply numeric tolerance — APPS is exact-match.
    """
    if s is None:
        return []
    lines = s.replace("\r\n", "\n").split("\n")
    # Strip trailing whitespace on each line.
    lines = [ln.rstrip() for ln in lines]
    # Drop trailing empties.
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _outputs_equal(got: str, expected: str) -> bool:
    """Compare two output strings using APPS exact-match-on-normalized-lines."""
    return _normalize_output(got) == _normalize_output(expected)


def _wrap_in_function(code: str) -> str:
    """Wrap code inside `def __sol(): ...` and call it.

    APPS canonical solutions (and many model completions) assume they are
    executed inside a function — some end with bare `return` statements to
    exit early, which is a SyntaxError at module level. codeparrot's eval
    wraps code in a function for exactly this reason. We apply the same fix
    only as a fallback when a naive run produced 'return' outside function.
    """
    indented = "\n".join("    " + ln if ln else "" for ln in code.split("\n"))
    return f"def __sol():\n{indented}\n\n__sol()\n"


def _exec_with_stdin(script: str, stdin_input: str, timeout: float) -> tuple[bool, str]:
    """Run a python script with stdin input; return (exit0, stdout_or_stderr_tail).

    On a `SyntaxError: 'return' outside function` we automatically retry with
    the script wrapped in a `def __sol(): ...; __sol()` harness — this is the
    codeparrot/APPS-eval convention and recovers solutions authored under the
    wrapper assumption.
    """
    def _run(src: str) -> subprocess.CompletedProcess:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
            fh.write(src)
            path = fh.name
        try:
            return subprocess.run(
                [sys.executable, path],
                input=stdin_input,
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

    try:
        proc = _run(script)
        if proc.returncode == 0:
            return True, proc.stdout
        err = proc.stderr or ""
        # Retry under a function wrapper if the only problem was a module-level
        # `return`. Do not retry for other SyntaxErrors or exceptions.
        if "'return' outside function" in err:
            proc = _run(_wrap_in_function(script))
            if proc.returncode == 0:
                return True, proc.stdout
            err = proc.stderr or ""
        tail = err.strip().splitlines()[-3:]
        return False, "exit!=0: " + " | ".join(tail)[:300]
    except subprocess.TimeoutExpired:
        return False, f"timeout>{timeout}s"
    except Exception as exc:  # pragma: no cover — defensive
        return False, f"exec_error: {exc}"


def _run_candidate_on_cases(
    code: str,
    inputs: list[str],
    outputs: list[str],
    timeout: float = DEFAULT_TIMEOUT_S,
    max_cases: int = MAX_TEST_CASES,
) -> tuple[int, int, str]:
    """Run candidate against each (input, output) pair; return (passes, total, detail)."""
    if not code.strip():
        return 0, 0, "no_code_extracted"
    if not inputs or not outputs:
        return 0, 0, "no_test_cases"
    n = min(len(inputs), len(outputs), max_cases)
    passes = 0
    fail_snippets: list[str] = []
    for idx in range(n):
        stdin_val = inputs[idx] or ""
        # APPS fixtures sometimes omit a trailing newline on the last line of
        # stdin; most solutions read via input() which tolerates EOF, but some
        # use sys.stdin.read().splitlines() which benefits from consistent
        # newline termination. Ensure trailing \n.
        if not stdin_val.endswith("\n"):
            stdin_val = stdin_val + "\n"
        ok_exit, stdout_or_err = _exec_with_stdin(code, stdin_val, timeout)
        if not ok_exit:
            if len(fail_snippets) < 2:
                fail_snippets.append(f"case{idx}:{stdout_or_err[:80]}")
            continue
        if _outputs_equal(stdout_or_err, outputs[idx] or ""):
            passes += 1
        elif len(fail_snippets) < 2:
            got = _normalize_output(stdout_or_err)
            exp = _normalize_output(outputs[idx] or "")
            got_preview = (got[0] if got else "")[:40]
            exp_preview = (exp[0] if exp else "")[:40]
            fail_snippets.append(f"case{idx}:diff got={got_preview!r} exp={exp_preview!r}")
    detail = f"passes={passes}/{n}"
    if fail_snippets:
        detail += " | " + " ; ".join(fail_snippets)
    return passes, n, detail


def _parse_input_output(raw: Any) -> tuple[list[str], list[str]]:
    """Parse the APPS input_output field (JSON string or dict) into lists.

    Returns ([], []) for malformed inputs — upstream should drop such rows at
    dataset preparation time, but defensive behavior here avoids killing a
    training cell on one bad row.
    """
    if raw is None:
        return [], []
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str):
        if not raw.strip():
            return [], []
        try:
            # Lift the 4300-digit int limit for APPS fixtures with huge literals.
            try:
                sys.set_int_max_str_digits(1_000_000)
            except AttributeError:
                pass
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [], []
    else:
        return [], []
    if not isinstance(parsed, dict):
        return [], []
    inputs = parsed.get("inputs") or []
    outputs = parsed.get("outputs") or []
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return [], []
    # Coerce every entry to str (some fixtures use lists of numbers for outputs).
    def _coerce(x: Any) -> str:
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            return "\n".join(str(y) for y in x)
        return str(x)

    return [_coerce(i) for i in inputs], [_coerce(o) for o in outputs]


def compute_score(model_response: str, ground_truth: Any) -> dict[str, Any]:
    """Score a single APPS code rollout.

    Returns a dict with the same key set as ``mbpp_code.compute_score`` and
    ``ttrl_math.compute_score`` so the verl reward_manager consumes it
    uniformly.
    """
    # Parse ground_truth: accept JSON string {input_output, difficulty} OR a
    # bare input_output JSON. The pipeline packs the former.
    input_output: Any = ""
    difficulty = ""
    if isinstance(ground_truth, dict):
        input_output = ground_truth.get("input_output", "")
        difficulty = ground_truth.get("difficulty", "")
    elif isinstance(ground_truth, str):
        try:
            gt = json.loads(ground_truth)
            if isinstance(gt, dict):
                input_output = gt.get("input_output", "")
                difficulty = gt.get("difficulty", "")
            else:
                input_output = ground_truth
        except json.JSONDecodeError:
            input_output = ground_truth

    code = _extract_code(model_response or "")
    format_score = 1.0 if code else 0.0

    inputs, outputs = _parse_input_output(input_output)
    if not inputs:
        return {
            "score": 0.0,
            "format_score": format_score,
            "acc": False,
            "extracted_gt": difficulty,
            "pred": code[:200],
            "detail": "no_io_fixture",
            "reward_mode": _reward_mode(),
        }

    passes, total, detail = _run_candidate_on_cases(code, inputs, outputs)
    if total == 0:
        score = 0.0
    elif _reward_mode() == "fractional":
        score = passes / total
    else:  # binary
        score = 1.0 if passes == total else 0.0

    acc = (passes == total) and (total > 0)
    return {
        "score": float(score),
        "format_score": format_score,
        "acc": bool(acc),
        "extracted_gt": difficulty,
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
    """verl reward_manager entry point. Mirrors mbpp_code.reward_func."""
    try:
        res = compute_score(solution_str, ground_truth)
        if isinstance(res, dict):
            return res
        if isinstance(res, (int, float, bool)):
            return float(res)
        return float(res[0])
    except Exception as exc:
        print(f"[ERROR] apps_code.reward_func failed: {exc}")
        traceback.print_exc()
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": "",
            "pred": "",
            "detail": f"reward_func_error: {exc}",
        }
