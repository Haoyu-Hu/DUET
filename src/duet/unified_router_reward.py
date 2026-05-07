"""Reward-function router: dispatches by data_source to ttrl_math (math) or
mbpp_code (HumanEval/MBPP-style code).

Used when the run mixes math and code val datasets (e.g. MATH train + MATH-500
+ AIME + GSM8K + GPQA math evals plus HumanEval code eval). verl's loader
imports this file by path via importlib, so all verl imports are absolute.

Routing: data_source tag is ``<DATASET-STEM>-DUET`` (set in pipeline.py).
Stems beginning with HUMANEVAL / MBPP / APPS route to the code reward;
everything else routes to ttrl_math.
"""

from __future__ import annotations

from verl.utils.reward_score.mbpp_code import reward_func as _mbpp_reward
from verl.utils.reward_score.ttrl_math import reward_func as _math_reward

_CODE_STEM_PREFIXES = ("HUMANEVAL", "MBPP", "APPS")


def _is_code_source(data_source) -> bool:
    if not isinstance(data_source, str):
        return False
    upper = data_source.upper()
    return any(upper.startswith(p) for p in _CODE_STEM_PREFIXES)


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    impl = _mbpp_reward if _is_code_source(data_source) else _math_reward
    return impl(
        data_source, solution_str, ground_truth,
        extra_info=extra_info,
        sandbox_fusion_url=sandbox_fusion_url,
        concurrent_semaphore=concurrent_semaphore,
    )
