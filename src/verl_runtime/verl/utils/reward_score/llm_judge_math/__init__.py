# Copyright 2026 BSV authors
# Licensed under the Apache License, Version 2.0
"""LLM-as-judge reward function for math tasks.

Simulates a "no ground truth available, expert verifier is expensive"
scenario: an external LLM is asked whether a rollout's solution is
correct, replacing the exact-match ground-truth verifier. This makes
math a meaningful BSV testbed — the verifier cost is now non-trivial
and structurally similar to the code/APPS sandbox case.

Interface matches :mod:`ttrl_math`:

    reward_func(data_source, solution_str, ground_truth, extra_info=None, ...)
        -> dict with {score, format_score, acc, extracted_gt, pred}

Judge backends (selected by the ``BSV_LLM_JUDGE_MODE`` environment variable):

    * ``stub_noisy`` (default) — grades via exact-match then flips the
      result with probability ``BSV_LLM_JUDGE_NOISE`` (default 0.10).
      Designed for CI / smoke: no external dependency, deterministic
      under a seed, simulates a noisy expensive verifier.
    * ``http`` — POSTs a chat completion request to a vLLM
      OpenAI-compatible server at ``BSV_LLM_JUDGE_URL``
      (default http://localhost:8000). The user is expected to start
      the judge server separately, e.g.

          python -m vllm.entrypoints.openai.api_server \\
              --model Qwen/Qwen3-4B-Instruct --port 8000

Baseline comparison: always keep :mod:`ttrl_math` as the "ground truth"
arm; the LLM-judge is the realistic-cost arm. Both share the same
``reward_func`` return-dict schema so the training loop stays the same.
"""

from __future__ import annotations

import os
import random
import traceback
from typing import Any

# The verl custom_reward_function loader imports this file directly by
# path (importlib.util.spec_from_file_location), so relative imports fail.
# Load the ground-truth grader via its installed module path instead.
try:
    from verl.utils.reward_score.ttrl_math import compute_score as _gt_compute_score
except ImportError:
    # Fallback for local-package-only runs (tests that add src to sys.path).
    from ..ttrl_math import compute_score as _gt_compute_score  # type: ignore[no-redef]

__all__ = ["reward_func", "compute_score", "DEFAULT_JUDGE_PROMPT"]


DEFAULT_JUDGE_PROMPT = """You are a math expert evaluating a student's solution.

Problem:
{problem}

Student's solution:
{solution}

Is the student's final answer mathematically correct? You do NOT have access
to an answer key — judge based on your own mathematical knowledge.

Respond with ONLY a single token: YES if correct, NO if incorrect."""


def _get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _stub_noisy_score(model_response: str, gt_answer: str) -> dict[str, Any]:
    """Exact-match grade flipped with probability noise.

    Simulates an expensive but imperfect judge. Deterministic under the
    env var ``BSV_LLM_JUDGE_SEED`` combined with a hash of the response.
    """
    gt_result = _gt_compute_score(model_response, str(gt_answer))
    acc = bool(gt_result.get("acc", False))

    noise = _get_env_float("BSV_LLM_JUDGE_NOISE", 0.10)
    noise = max(0.0, min(1.0, noise))
    seed = _get_env_int("BSV_LLM_JUDGE_SEED", 0)
    # random.Random only accepts scalar seeds; hash the composite key.
    seed_key = f"{seed}|{model_response[:64]}|{str(gt_answer)[:64]}"
    rng = random.Random(seed_key)
    if rng.random() < noise:
        acc = not acc

    score = 1.0 if acc else 0.0
    # Name convention (post-rename): `acc` = GT exact-match (the real
    # accuracy); `judge_agrees` = the noisy judge's verdict used as the
    # training reward. `score` IS `judge_agrees` as a float (that's the
    # signal the training loop uses). This means cross-verifier
    # comparisons can uniformly read `val-core/acc/mean@4` and get GT
    # pass rate, regardless of which reward_fn was configured.
    gt_acc_bool = bool(gt_result.get("acc", False))
    return {
        "score": score,
        "format_score": gt_result.get("format_score", 0.0),
        "acc": gt_acc_bool,
        "judge_agrees": acc,
        "extracted_gt": gt_result.get("extracted_gt", str(gt_answer)),
        "pred": gt_result.get("pred", ""),
        "judge_mode": "stub_noisy",
        "judge_noise": noise,
    }


def _http_score(model_response: str, gt_answer: str, problem: str | None) -> dict[str, Any]:
    """Call an OpenAI-compatible judge server; default localhost:8000.

    The judge is prompted with the problem + solution only (NOT the
    ground truth) to simulate the real-world scenario. Ground truth is
    passed through for post-hoc analysis only (in the return dict).
    """
    import json
    import urllib.request
    import urllib.error

    base_url = os.environ.get("BSV_LLM_JUDGE_URL", "http://localhost:8000").rstrip("/")
    model_name = os.environ.get("BSV_LLM_JUDGE_MODEL", "Qwen/Qwen3-4B-Instruct")
    timeout_s = _get_env_float("BSV_LLM_JUDGE_TIMEOUT", 30.0)

    if problem is None:
        problem = "[problem text not provided; judge on solution plausibility]"

    prompt_body = DEFAULT_JUDGE_PROMPT.format(problem=problem, solution=model_response)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_body}],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        reply = body["choices"][0]["message"]["content"].strip().upper()
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
        # On judge failure, return a neutral (0.5 hard-label) verdict and flag
        # it. Do NOT raise — a single judge outage should not kill training.
        # The propensity CSV still records the attempt.
        gt_fallback = _gt_compute_score(model_response, str(gt_answer))
        return {
            "score": 0.0,
            "format_score": gt_fallback.get("format_score", 0.0),
            "acc": bool(gt_fallback.get("acc", False)),
            "judge_agrees": False,
            "extracted_gt": str(gt_answer),
            "pred": "",
            "judge_mode": "http_failed",
            "judge_error": str(e),
        }

    # Parse: first token should be YES or NO.
    judge_agrees = reply.startswith("YES")
    score = 1.0 if judge_agrees else 0.0

    gt_for_eval = _gt_compute_score(model_response, str(gt_answer))
    # `acc` reports GT exact-match (the real accuracy; what you want to
    # plot on the y-axis of the Pareto). `judge_agrees` is the judge's
    # YES-rate — the training signal, not accuracy. `score` IS
    # `judge_agrees` as float. Val metrics that aggregate `acc` now
    # compare cross-verifier cleanly.
    return {
        "score": score,
        "format_score": gt_for_eval.get("format_score", 0.0),
        "acc": bool(gt_for_eval.get("acc", False)),
        "judge_agrees": judge_agrees,
        "extracted_gt": str(gt_answer),
        "pred": reply,
        "judge_mode": "http",
    }


def _openai_score(model_response: str, gt_answer: str, problem: str | None) -> dict[str, Any]:
    """Call OpenAI's hosted API (default gpt-4o-mini) as the external verifier.

    Distinct from ``_http_score``:
      * Fixed endpoint: ``https://api.openai.com/v1/chat/completions``
      * Reads ``OPENAI_API_KEY`` from the environment at call time. The key
        is **never** logged, never written to the propensity CSV, never
        returned in the result dict. Absence of the key → graceful
        degrade with ``judge_mode='openai_no_key'``.
      * Retries once on 429 / 5xx with exponential backoff.

    Like the other modes, the judge sees only (problem, solution) — NOT
    the ground truth. Ground-truth accuracy is carried through as
    ``gt_acc`` for post-hoc comparison.
    """
    import json
    import time
    import urllib.request
    import urllib.error

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Never fail training on a missing key — degrade and flag.
        # Key set must match the success/failure returns below so
        # reward_extra_info lists stay same-length across mixed samples.
        gt_fallback = _gt_compute_score(model_response, str(gt_answer))
        return {
            "score": 0.0,
            "format_score": gt_fallback.get("format_score", 0.0),
            "acc": bool(gt_fallback.get("acc", False)),
            "judge_agrees": False,
            "extracted_gt": str(gt_answer),
            "pred": "",
            "judge_mode": "openai_no_key",
            "judge_model": os.environ.get("BSV_LLM_JUDGE_MODEL", "gpt-4o-mini"),
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
            "judge_error": "OPENAI_API_KEY not set in worker environment",
            "judge_error_type": "",
            "judge_error_code": "",
            "verifier_failed": True,
        }

    model_name = os.environ.get("BSV_LLM_JUDGE_MODEL", "gpt-4o-mini")
    timeout_s = _get_env_float("BSV_LLM_JUDGE_TIMEOUT", 30.0)
    endpoint = "https://api.openai.com/v1/chat/completions"

    if problem is None:
        problem = "[problem text not provided; judge on solution plausibility]"

    prompt_body = DEFAULT_JUDGE_PROMPT.format(problem=problem, solution=model_response)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_body}],
        "max_tokens": 2,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")

    def _once() -> tuple[str | None, Exception | None, int, dict | None, str, str]:
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            reply = body["choices"][0]["message"]["content"].strip().upper()
            return reply, None, 200, body.get("usage"), "", ""
        except urllib.error.HTTPError as he:
            # Parse OpenAI's error payload but extract ONLY the coarse
            # {error.type, error.code} fields — never `error.message`.
            # On 401 "invalid_api_key" OpenAI echoes a masked-but-partial
            # key prefix inside `error.message`, which would otherwise
            # propagate into training logs via `reward_extra_info`.
            etype, ecode = "", ""
            try:
                raw = he.read().decode("utf-8", errors="replace")
                parsed = json.loads(raw).get("error", {}) if raw else {}
                if isinstance(parsed, dict):
                    etype = str(parsed.get("type", ""))[:64]
                    ecode = str(parsed.get("code", ""))[:64]
            except (json.JSONDecodeError, ValueError, AttributeError):
                pass
            return None, he, he.code, None, etype, ecode
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
            return None, e, 0, None, "", ""

    reply, err, code, usage, etype, ecode = _once()
    # Retry once on rate-limit / transient server errors.
    if reply is None and code in (429, 500, 502, 503, 504):
        time.sleep(1.0)
        reply, err, code, usage, etype, ecode = _once()

    # Success and failure dicts MUST carry identical key sets. The caller
    # (NaiveRewardManager) appends every key to a per-key list in
    # `reward_extra_info`, and `process_validation_metrics` asserts every
    # list's length equals sample count. Mixed success/failure samples with
    # divergent key sets break that invariant and crash `_validate`.
    gt_for_eval = _gt_compute_score(model_response, str(gt_answer))
    if reply is None:
        return {
            "score": 0.0,
            "format_score": gt_for_eval.get("format_score", 0.0),
            "acc": bool(gt_for_eval.get("acc", False)),
            "judge_agrees": False,
            "extracted_gt": str(gt_answer),
            "pred": "",
            "judge_mode": "openai_failed",
            "judge_model": model_name,
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
            "judge_error": f"{type(err).__name__}:{code}" if err else "unknown",
            # Only surface sanitized error metadata. OpenAI's raw
            # `error.message` can include a masked-partial key prefix on
            # 401 — we intentionally drop it.
            "judge_error_type": etype,
            "judge_error_code": ecode,
            "verifier_failed": True,
        }

    # After the 20260420 rename: `acc` reports the true GT exact-match;
    # `judge_agrees` is the openai judge's YES-rate (the training signal).
    # `score` is `judge_agrees` cast to float (unchanged training path).
    # This keeps `val-core/acc/mean@4` comparable across GT / llm_judge
    # runs without the caller having to know which reward_fn produced
    # the row.
    judge_agrees = reply.startswith("YES")
    return {
        "score": 1.0 if judge_agrees else 0.0,
        "format_score": gt_for_eval.get("format_score", 0.0),
        "acc": bool(gt_for_eval.get("acc", False)),
        "judge_agrees": judge_agrees,
        "extracted_gt": str(gt_answer),
        "pred": reply,
        "judge_mode": "openai",
        "judge_model": model_name,
        # Token usage is the only "cost" signal we expose. No key ever.
        "judge_prompt_tokens": int(usage.get("prompt_tokens", 0)) if usage else 0,
        "judge_completion_tokens": int(usage.get("completion_tokens", 0)) if usage else 0,
        "judge_error": "",
        "judge_error_type": "",
        "judge_error_code": "",
        "verifier_failed": False,
    }


def compute_score(model_response: str, gt_answer: str, problem: str | None = None) -> dict[str, Any]:
    """Route to the configured judge backend.

    ``BSV_LLM_JUDGE_MODE`` values:
      * ``stub_noisy`` — exact-match + probability-p flip (default, no network).
      * ``http``       — local OpenAI-compatible server (e.g. self-hosted vLLM).
      * ``openai``     — hosted api.openai.com (default model ``gpt-4o-mini``).
    """
    mode = os.environ.get("BSV_LLM_JUDGE_MODE", "stub_noisy").strip().lower()
    if mode == "openai":
        return _openai_score(model_response, gt_answer, problem)
    if mode == "http":
        return _http_score(model_response, gt_answer, problem)
    return _stub_noisy_score(model_response, gt_answer)


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    """verl reward_func entry point.

    Signature matches ``ttrl_math.reward_func``. ``extra_info`` is the
    verl-supplied dict; for math tasks it typically contains a
    ``"prompt"`` or ``"question"`` key that we pass to the judge.
    """
    try:
        problem = None
        if isinstance(extra_info, dict):
            problem = extra_info.get("question") or extra_info.get("prompt") or extra_info.get("problem")
        res = compute_score(solution_str, str(ground_truth), problem=problem)
        if isinstance(res, dict):
            return res
        return float(res)
    except Exception as e:
        print(f"[ERROR] Error in llm_judge_math.reward_func: {e}")
        traceback.print_exc()
        # Degrade to zero-reward rather than killing training on an
        # individual rollout's judge failure.
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": str(ground_truth),
            "pred": "",
            "judge_mode": "error",
            "judge_error": str(e),
        }
