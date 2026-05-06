"""Tests for the APPS stdio sandbox reward path.

Run with:
  PYTHONPATH=/workspace/src/verl_runtime \\
    /workspace/venv/bin/python -m pytest -q \\
    src/verl_runtime/verl/utils/reward_score/apps_code/test_apps_code.py
"""

from __future__ import annotations

import json
import os

import pytest

from verl.utils.reward_score.apps_code import (
    _extract_code,
    _normalize_output,
    _outputs_equal,
    _parse_input_output,
    compute_score,
)


# --- small helpers ---------------------------------------------------------


def _fenced(code: str, lang: str = "python") -> str:
    return f"```{lang}\n{code}\n```"


def _io_gt(inputs: list[str], outputs: list[str], difficulty: str = "introductory") -> str:
    return json.dumps(
        {
            "input_output": json.dumps({"inputs": inputs, "outputs": outputs}),
            "difficulty": difficulty,
        }
    )


# --- _extract_code ---------------------------------------------------------


def test_extract_code_python_fence():
    assert _extract_code("prefix\n```python\nx = 1\n```\nsuffix") == "x = 1"


def test_extract_code_bare_fence():
    assert _extract_code("```\nprint('hi')\n```") == "print('hi')"


def test_extract_code_no_fence():
    assert _extract_code("print('hi')") == "print('hi')"


def test_extract_code_empty():
    assert _extract_code("") == ""
    assert _extract_code(None) == ""  # type: ignore[arg-type]


# --- _normalize_output / _outputs_equal ------------------------------------


def test_normalize_output_strips_trailing_whitespace():
    assert _normalize_output("hello   \nworld\t\n") == ["hello", "world"]


def test_normalize_output_drops_trailing_blank_lines():
    assert _normalize_output("a\nb\n\n\n") == ["a", "b"]


def test_normalize_output_handles_crlf():
    assert _normalize_output("a\r\nb\r\n") == ["a", "b"]


def test_normalize_output_empty():
    assert _normalize_output("") == []
    assert _normalize_output("   \n\n") == []


def test_outputs_equal_whitespace_tolerant():
    assert _outputs_equal("42\n", "42")
    assert _outputs_equal("42 \n", "42")
    assert _outputs_equal("1\n2\n3\n", "1\n2\n3")


def test_outputs_equal_strict_numbers():
    # APPS does NOT apply numeric tolerance — exact match required.
    assert not _outputs_equal("3.14\n", "3.140000")
    assert not _outputs_equal("42\n", "41")


# --- _parse_input_output ---------------------------------------------------


def test_parse_input_output_dict():
    inp, out = _parse_input_output({"inputs": ["1\n"], "outputs": ["2\n"]})
    assert inp == ["1\n"]
    assert out == ["2\n"]


def test_parse_input_output_json_string():
    raw = json.dumps({"inputs": ["5"], "outputs": ["25"]})
    inp, out = _parse_input_output(raw)
    assert inp == ["5"]
    assert out == ["25"]


def test_parse_input_output_coerce_list_output():
    # Some APPS fixtures store outputs as lists of lines.
    raw = json.dumps({"inputs": ["x"], "outputs": [["line1", "line2"]]})
    inp, out = _parse_input_output(raw)
    assert inp == ["x"]
    assert out == ["line1\nline2"]


def test_parse_input_output_malformed_returns_empty():
    assert _parse_input_output("not-json") == ([], [])
    assert _parse_input_output(None) == ([], [])
    assert _parse_input_output({"wrong": "keys"}) == ([], [])
    assert _parse_input_output("") == ([], [])


def test_parse_input_output_huge_integer_literal():
    """Python 3.11+ enforces 4300-digit limit on int↔str conversion by default;
    the sandbox must lift it so json.loads doesn't fail on APPS fixtures with
    very large integer literals.
    """
    big_int = "1" + "0" * 5000  # 5001 digits
    raw = json.dumps({"inputs": [big_int], "outputs": [big_int]})
    inp, out = _parse_input_output(raw)
    assert inp == [big_int]
    assert out == [big_int]


# --- compute_score: integration --------------------------------------------


ECHO_SOLUTION = """
import sys
data = sys.stdin.read().strip()
print(int(data) * 2)
""".strip()


def test_compute_score_binary_all_pass():
    gt = _io_gt(inputs=["3", "5", "10"], outputs=["6", "10", "20"])
    os.environ["BSV_APPS_REWARD_MODE"] = "binary"
    out = compute_score(_fenced(ECHO_SOLUTION), gt)
    assert out["acc"] is True
    assert out["score"] == 1.0
    assert out["reward_mode"] == "binary"


def test_compute_score_binary_some_fail_is_zero():
    # 3 inputs but one expected output is wrong → binary = 0.
    gt = _io_gt(inputs=["3", "5", "10"], outputs=["6", "WRONG", "20"])
    os.environ["BSV_APPS_REWARD_MODE"] = "binary"
    out = compute_score(_fenced(ECHO_SOLUTION), gt)
    assert out["acc"] is False
    assert out["score"] == 0.0


def test_compute_score_fractional_partial():
    gt = _io_gt(inputs=["3", "5", "10"], outputs=["6", "WRONG", "20"])
    os.environ["BSV_APPS_REWARD_MODE"] = "fractional"
    out = compute_score(_fenced(ECHO_SOLUTION), gt)
    assert out["acc"] is False
    assert abs(out["score"] - 2.0 / 3.0) < 1e-6
    assert out["reward_mode"] == "fractional"


def test_compute_score_no_code_extracted():
    os.environ["BSV_APPS_REWARD_MODE"] = "binary"
    out = compute_score("", _io_gt(inputs=["1"], outputs=["2"]))
    assert out["acc"] is False
    assert out["score"] == 0.0
    assert out["format_score"] == 0.0


def test_compute_score_no_io_fixture():
    # Empty ground truth → score 0 but no crash.
    os.environ["BSV_APPS_REWARD_MODE"] = "binary"
    out = compute_score(
        _fenced(ECHO_SOLUTION),
        json.dumps({"input_output": "", "difficulty": "introductory"}),
    )
    assert out["score"] == 0.0
    assert out["detail"] == "no_io_fixture"


def test_compute_score_timeout_is_fail_not_crash():
    # An infinite loop solution — sandbox must timeout and report 0, not hang or crash.
    os.environ["BSV_APPS_REWARD_MODE"] = "binary"
    os.environ["BSV_APPS_TIMEOUT_S"] = "2"  # 2s for speed
    try:
        looper = "while True:\n    pass"
        out = compute_score(_fenced(looper), _io_gt(inputs=["1"], outputs=["1"]))
        assert out["score"] == 0.0
        assert "timeout" in out["detail"]
    finally:
        os.environ.pop("BSV_APPS_TIMEOUT_S", None)


def test_compute_score_exception_is_fail():
    # Raises on any input — sandbox reports exit!=0 and score 0.
    os.environ["BSV_APPS_REWARD_MODE"] = "binary"
    raiser = "raise ValueError('boom')"
    out = compute_score(_fenced(raiser), _io_gt(inputs=["1"], outputs=["1"]))
    assert out["score"] == 0.0
    assert "exit!=0" in out["detail"]


def test_compute_score_stdin_parsing_multiline():
    # Verify a solution reading multiple lines via input() works.
    sol = """
n = int(input())
for _ in range(n):
    x = int(input())
    print(x * x)
""".strip()
    gt = _io_gt(
        inputs=["3\n2\n3\n4"],
        outputs=["4\n9\n16"],
    )
    os.environ["BSV_APPS_REWARD_MODE"] = "binary"
    out = compute_score(_fenced(sol), gt)
    assert out["acc"] is True
    assert out["score"] == 1.0


def test_compute_score_max_cases_cap(monkeypatch):
    """If a fixture has 100 IO pairs we should only run up to MAX_TEST_CASES."""
    inputs = [str(i) for i in range(100)]
    outputs = [str(i * 2) for i in range(100)]
    gt = _io_gt(inputs=inputs, outputs=outputs)
    monkeypatch.setenv("BSV_APPS_MAX_TESTS", "5")
    os.environ["BSV_APPS_REWARD_MODE"] = "fractional"
    # Re-import path reset: the env var is read at call time via MAX_TEST_CASES,
    # but it's captured at module import. To exercise the dynamic behavior we
    # reload the module.
    import importlib
    import verl.utils.reward_score.apps_code as apps_mod
    importlib.reload(apps_mod)
    out = apps_mod.compute_score(_fenced(ECHO_SOLUTION), gt)
    # detail encodes "passes=X/5"
    assert "/5" in out["detail"]
    # Restore default (10) for other tests.
    monkeypatch.delenv("BSV_APPS_MAX_TESTS", raising=False)
    importlib.reload(apps_mod)


# --- reward_mode validation ------------------------------------------------


def test_reward_mode_invalid_raises():
    os.environ["BSV_APPS_REWARD_MODE"] = "cubic_root"
    try:
        with pytest.raises(ValueError):
            compute_score(_fenced(ECHO_SOLUTION), _io_gt(["1"], ["2"]))
    finally:
        os.environ["BSV_APPS_REWARD_MODE"] = "binary"


# --- format / structure contract -------------------------------------------


def test_return_shape_matches_mbpp_code_contract():
    """The reward dict must expose the same keys the verl reward_manager
    expects (score, format_score, acc, extracted_gt, pred) so the dispatch
    path can swap MBPP/APPS without manager code changes.
    """
    os.environ["BSV_APPS_REWARD_MODE"] = "binary"
    out = compute_score(_fenced(ECHO_SOLUTION), _io_gt(["3"], ["6"]))
    required = {"score", "format_score", "acc", "extracted_gt", "pred"}
    assert required.issubset(out.keys())
    assert isinstance(out["score"], float)
    assert isinstance(out["acc"], bool)
