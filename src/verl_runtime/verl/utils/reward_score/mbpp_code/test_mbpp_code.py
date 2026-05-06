"""Smoke tests for the MBPP sandbox reward.

Run with: /workspace/venv/bin/python -m pytest -q \
    src/verl_runtime/verl/utils/reward_score/mbpp_code/test_mbpp_code.py
"""

from __future__ import annotations

import json
import os

import pytest

from verl.utils.reward_score.mbpp_code import (
    REWARD_MODE_ENV,
    _extract_code,
    _run_candidate,
    _run_candidate_binary,
    _run_candidate_fractional,
    _split_asserts,
    compute_score,
    reward_func,
)


@pytest.fixture(autouse=True)
def _reset_reward_mode_env():
    """Isolate BSV_MBPP_REWARD_MODE between tests so one test's setenv can't
    leak into the next (tests run alphabetically; fractional tests set the env
    var and could otherwise poison later binary-mode tests)."""
    prev = os.environ.get(REWARD_MODE_ENV)
    os.environ.pop(REWARD_MODE_ENV, None)
    yield
    if prev is None:
        os.environ.pop(REWARD_MODE_ENV, None)
    else:
        os.environ[REWARD_MODE_ENV] = prev


# ---------- _extract_code ----------

def test_extract_code_fenced_python():
    comp = "Here is my answer:\n```python\ndef f(): return 42\n```\nDone."
    assert _extract_code(comp) == "def f(): return 42"


def test_extract_code_fenced_no_lang():
    comp = "```\nx = 1\n```"
    assert _extract_code(comp) == "x = 1"


def test_extract_code_no_fence():
    comp = "def f(): return 1"
    assert _extract_code(comp) == "def f(): return 1"


def test_extract_code_empty():
    assert _extract_code("") == ""
    assert _extract_code("   \n  ") == ""


# ---------- _run_candidate ----------

def test_run_candidate_passes():
    # Default mode = binary; dispatcher must return 1.0 on all-pass.
    code = "def add(a, b): return a + b"
    test = "assert add(2, 3) == 5"
    score, detail = _run_candidate(code, test)
    assert score == 1.0, detail


def test_run_candidate_fails():
    code = "def add(a, b): return a - b"
    test = "assert add(2, 3) == 5"
    score, _ = _run_candidate(code, test)
    assert score == 0.0


def test_run_candidate_timeout():
    code = "def f():\n    while True: pass"
    test = "f()"
    score, detail = _run_candidate(code, test, timeout=1)
    assert score == 0.0
    assert "timeout" in detail


def test_run_candidate_syntax_error():
    code = "def add(a, b: return a + b"  # missing paren
    test = "assert add(1, 2) == 3"
    score, _ = _run_candidate(code, test)
    assert score == 0.0


def test_run_candidate_empty_code():
    score, detail = _run_candidate("", "assert True")
    assert score == 0.0
    assert detail == "no_code_extracted"


# ---------- _split_asserts ----------

def test_split_asserts_single():
    assert _split_asserts("assert f(1) == 2") == ["assert f(1) == 2"]


def test_split_asserts_multi_mbpp_style():
    t = "assert g(1) == 1\nassert g(2) == 4\nassert g(3) == 9"
    assert _split_asserts(t) == [
        "assert g(1) == 1",
        "assert g(2) == 4",
        "assert g(3) == 9",
    ]


def test_split_asserts_empty():
    assert _split_asserts("") == []


def test_split_asserts_with_blank_and_comment_lines():
    # Lines that aren't asserts are attached to the preceding assert (defensive
    # behavior for multi-line asserts / inline comments).
    t = "assert x(1) == 1\n# comment line\nassert x(2) == 4"
    out = _split_asserts(t)
    assert len(out) == 2
    assert out[0].startswith("assert x(1)")
    assert out[1] == "assert x(2) == 4"


def test_split_asserts_paren_form():
    # `assert(...)` (no space) should still be recognized.
    assert _split_asserts("assert(f() == 1)") == ["assert(f() == 1)"]


# ---------- _run_candidate_binary (explicit path) ----------

def test_run_binary_all_pass():
    score, _ = _run_candidate_binary(
        "def f(x): return x+1",
        "assert f(1) == 2\nassert f(2) == 3\nassert f(3) == 4",
    )
    assert score == 1.0


def test_run_binary_one_fails_returns_zero():
    # Binary mode: any failure → 0.0 (even if 2/3 pass).
    score, _ = _run_candidate_binary(
        "def f(x): return x+1 if x < 3 else 999",
        "assert f(1) == 2\nassert f(2) == 3\nassert f(3) == 4",
    )
    assert score == 0.0


# ---------- _run_candidate_fractional ----------

def test_run_fractional_all_pass():
    score, detail = _run_candidate_fractional(
        "def f(x): return x+1",
        "assert f(1) == 2\nassert f(2) == 3\nassert f(3) == 4",
    )
    assert score == pytest.approx(1.0)
    assert "passes=3/3" in detail


def test_run_fractional_partial_two_of_three():
    # Function passes first 2 asserts, fails third.
    score, detail = _run_candidate_fractional(
        "def f(x): return x+1 if x < 3 else 999",
        "assert f(1) == 2\nassert f(2) == 3\nassert f(3) == 4",
    )
    assert score == pytest.approx(2 / 3)
    assert "passes=2/3" in detail


def test_run_fractional_all_fail():
    score, detail = _run_candidate_fractional(
        "def f(x): return -1",
        "assert f(1) == 1\nassert f(2) == 2",
    )
    assert score == 0.0
    assert "passes=0/2" in detail


def test_run_fractional_candidate_class_stays_in_scope_per_assert():
    # MBPP convention: candidate defines a class (e.g. Pair) that the tests
    # reference. Each assert runs in a fresh subprocess with the candidate
    # code prepended, so the class must be visible each time.
    code = """
class Pair:
    def __init__(self, a, b): self.a, self.b = a, b
def first(p): return p.a
"""
    test = (
        "assert first(Pair(3, 4)) == 3\n"
        "assert first(Pair(1, 2)) == 1\n"
        "assert first(Pair(9, 9)) == 9"
    )
    score, _ = _run_candidate_fractional(code, test)
    assert score == pytest.approx(1.0)


def test_run_fractional_empty_asserts_returns_zero():
    score, detail = _run_candidate_fractional("def f(): pass", "")
    assert score == 0.0
    assert detail == "no_asserts_parsed"


# ---------- Dispatcher (_run_candidate via BSV_MBPP_REWARD_MODE) ----------

def test_dispatcher_defaults_to_binary():
    # No env var set → binary mode. 2/3 passing → 0.0, not 2/3.
    score, _ = _run_candidate(
        "def f(x): return x+1 if x < 3 else 999",
        "assert f(1) == 2\nassert f(2) == 3\nassert f(3) == 4",
    )
    assert score == 0.0


def test_dispatcher_fractional_mode_env_var():
    os.environ[REWARD_MODE_ENV] = "fractional"
    score, _ = _run_candidate(
        "def f(x): return x+1 if x < 3 else 999",
        "assert f(1) == 2\nassert f(2) == 3\nassert f(3) == 4",
    )
    assert score == pytest.approx(2 / 3)


def test_dispatcher_rejects_invalid_mode():
    os.environ[REWARD_MODE_ENV] = "weighted"
    with pytest.raises(ValueError):
        _run_candidate("def f(): pass", "assert True")


def test_dispatcher_mode_case_insensitive():
    os.environ[REWARD_MODE_ENV] = "FRACTIONAL"
    score, _ = _run_candidate(
        "def f(x): return x", "assert f(1)==1\nassert f(2)==2"
    )
    assert score == 1.0


# ---------- compute_score ----------

def test_compute_score_correct_mbpp_like():
    gt = json.dumps({
        "test": "assert add(2, 3) == 5\nassert add(0, 0) == 0",
        "entry_point": "add",
    })
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    res = compute_score(completion, gt)
    assert res["score"] == 1.0
    assert res["acc"] is True
    assert res["format_score"] == 1.0
    assert res["extracted_gt"] == "add"


def test_compute_score_wrong_answer():
    gt = json.dumps({"test": "assert add(2, 3) == 5", "entry_point": "add"})
    completion = "```python\ndef add(a, b):\n    return a * b\n```"
    res = compute_score(completion, gt)
    assert res["score"] == 0.0
    assert res["acc"] is False


def test_compute_score_no_code_block():
    gt = json.dumps({"test": "assert f() == 1", "entry_point": "f"})
    res = compute_score("I don't know", gt)
    # Bare text is treated as code; asserts will fail → score 0 but format=1 (non-empty).
    assert res["score"] == 0.0
    assert res["acc"] is False


def test_compute_score_empty_completion():
    gt = json.dumps({"test": "assert True", "entry_point": "x"})
    res = compute_score("", gt)
    assert res["score"] == 0.0
    assert res["format_score"] == 0.0


def test_compute_score_dict_gt():
    # ground_truth already a dict (alternate wire format)
    gt = {"test": "assert greet() == 'hi'", "entry_point": "greet"}
    completion = "```python\ndef greet(): return 'hi'\n```"
    res = compute_score(completion, gt)
    assert res["score"] == 1.0


def test_compute_score_malformed_json_gt_treated_as_bare_test():
    # If ground_truth is a plain test string (non-JSON), use it directly.
    gt = "assert bare_fn() == 7"
    completion = "```python\ndef bare_fn(): return 7\n```"
    res = compute_score(completion, gt)
    assert res["score"] == 1.0


def test_compute_score_missing_test():
    gt = json.dumps({"test": "", "entry_point": "f"})
    completion = "```python\ndef f(): return 1\n```"
    res = compute_score(completion, gt)
    assert res["score"] == 0.0
    assert res["acc"] is False


# ---------- reward_func (verl entry point) ----------

def test_reward_func_signature():
    gt = json.dumps({"test": "assert g(1) == 2", "entry_point": "g"})
    res = reward_func(
        data_source="mbpp-full",
        solution_str="```python\ndef g(x): return x + 1\n```",
        ground_truth=gt,
        extra_info={"split": "train", "index": 0},
    )
    assert isinstance(res, dict)
    assert res["score"] == 1.0


def test_reward_func_multi_assert_mbpp_style():
    # Real MBPP-shaped: multiple asserts joined by newline, assume code defines all refs.
    gt = json.dumps({
        "test": (
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n"
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)"
        ),
        "entry_point": "similar_elements",
    })
    completion = (
        "```python\n"
        "def similar_elements(a, b):\n"
        "    return tuple(sorted(set(a) & set(b)))\n"
        "```"
    )
    res = reward_func("mbpp-full", completion, gt)
    assert res["score"] == 1.0


# ---------- compute_score under fractional mode ----------

def test_compute_score_fractional_partial_credit():
    # Candidate passes 1/2 asserts → score 0.5, acc=False (acc requires all pass).
    os.environ[REWARD_MODE_ENV] = "fractional"
    gt = json.dumps({
        "test": "assert f(1) == 1\nassert f(2) == 2",
        "entry_point": "f",
    })
    completion = "```python\ndef f(x):\n    return 1\n```"
    res = compute_score(completion, gt)
    assert res["score"] == pytest.approx(0.5)
    assert res["acc"] is False
    assert res["reward_mode"] == "fractional"


def test_compute_score_fractional_all_pass_acc_true():
    os.environ[REWARD_MODE_ENV] = "fractional"
    gt = json.dumps({"test": "assert f(1)==1\nassert f(2)==2", "entry_point": "f"})
    completion = "```python\ndef f(x): return x\n```"
    res = compute_score(completion, gt)
    assert res["score"] == pytest.approx(1.0)
    assert res["acc"] is True


def test_compute_score_binary_default_reports_mode_in_output():
    # Default binary mode should surface reward_mode="binary" in the result.
    gt = json.dumps({"test": "assert f(1)==1", "entry_point": "f"})
    completion = "```python\ndef f(x): return x\n```"
    res = compute_score(completion, gt)
    assert res["score"] == 1.0
    assert res["reward_mode"] == "binary"


# ---------- HumanEval-style tests (regression: bug 2026-05-01) ----------

_HUMANEVAL_TEST = """
def check(candidate):
    assert candidate(2) == 4
    assert candidate(3) == 9
    assert candidate(0) == 0
"""


def test_humaneval_style_correct_completion_passes():
    """Correct completion should pass the check(<entry_point>) invocation."""
    code = "def square(x):\n    return x * x\n"
    score, detail = _run_candidate_binary(code, _HUMANEVAL_TEST, entry_point="square")
    assert score == 1.0, f"correct code should pass, got {score=} {detail=}"


def test_humaneval_style_wrong_completion_fails():
    """Trivially wrong completion MUST fail. Bug 2026-05-01 returned 1.0 for
    every nonempty completion because check() was never invoked."""
    code = "def square(x):\n    return 0\n"  # always wrong
    score, detail = _run_candidate_binary(code, _HUMANEVAL_TEST, entry_point="square")
    assert score == 0.0, f"wrong code must fail, got {score=} {detail=}"


def test_humaneval_style_no_entry_function_fails():
    """Code that does not even define the entry point function must fail."""
    code = "x = 42\n"
    score, _ = _run_candidate_binary(code, _HUMANEVAL_TEST, entry_point="square")
    assert score == 0.0


def test_humaneval_style_empty_completion_fails():
    """Empty completion should fail (and be caught by the early no-code guard)."""
    score, detail = _run_candidate_binary("", _HUMANEVAL_TEST, entry_point="square")
    assert score == 0.0
    assert detail == "no_code_extracted"


def test_humaneval_style_no_entry_point_arg_falls_back():
    """If entry_point is empty (e.g., MBPP), do not append check() call.
    Test still defines check() but exits cleanly — that is correct MBPP
    behaviour: MBPP tests use module-level asserts, not a check() function."""
    code = "def square(x):\n    return 0\n"  # would-be wrong
    score, _ = _run_candidate_binary(code, _HUMANEVAL_TEST, entry_point="")
    # No invocation appended → exits 0 → score 1.0 (this is the legacy
    # behaviour reserved for MBPP-style tests where the harness contract
    # is module-level asserts only).
    assert score == 1.0


def test_compute_score_routes_entry_point_to_runner():
    """End-to-end via compute_score: entry_point in ground_truth dict must
    propagate so check() is invoked for HumanEval rows."""
    gt = json.dumps({"test": _HUMANEVAL_TEST, "entry_point": "square"})
    # Wrong code
    res_wrong = compute_score("```python\ndef square(x): return -1\n```", gt)
    assert res_wrong["score"] == 0.0
    assert res_wrong["acc"] is False
    # Correct code
    res_ok = compute_score("```python\ndef square(x): return x*x\n```", gt)
    assert res_ok["score"] == 1.0
    assert res_ok["acc"] is True


def test_humaneval_fractional_mode_handles_check_block():
    """Fractional mode flattens the check() body into per-assert subprocesses
    so HumanEval works the same way as MBPP."""
    os.environ[REWARD_MODE_ENV] = "fractional"
    try:
        code = "def square(x):\n    return x*x if x != 3 else 0\n"  # wrong on x=3
        score, detail = _run_candidate(code, _HUMANEVAL_TEST, entry_point="square")
        # 2 of 3 asserts pass (x=2 ok, x=3 fails, x=0 ok)
        assert abs(score - 2.0/3) < 1e-6, f"expected 2/3, got {score=} {detail=}"
    finally:
        os.environ.pop(REWARD_MODE_ENV, None)

