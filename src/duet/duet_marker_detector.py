"""Response marker detector for DUET v3 verifier-aware T2 (paper §5.2).

The marker detector watches a rolling output stream during generation and
returns ``True`` once the model has emitted a syntactically valid answer
marker for the verifier's domain (e.g., ``\\boxed{...}`` for math, a closed
code block for code). The DUET stop processor uses this signal to gate
LP firing — never truncating mid-reasoning, only after the model has
committed to a verifier-scoreable output.

This addresses the v10/v2 failure mode where ``DuetStopProcessor`` cut
rollouts at ``min_tokens + threshold`` regardless of whether the response
contained an answer, producing zero-reward gradients and catastrophic
forgetting on hard math (AIME → 0.000). See discussion in
``doc/paper_agent_work/DUET_Implementation_v1.md`` §6 (v3 redesign).

**Cheapness budget**: detection is called every ``check_every`` tokens
(default 8) — not every token — and only decodes the most recent suffix
(default last 256 tokens) rather than the whole response. Per-detection
cost is ~50-200μs depending on suffix length and tokenizer.

Detection happens in DuetStopProcessor's ``__call__``; the detector itself
holds no state across rollouts.
"""

from __future__ import annotations

import re
from typing import List, Optional


# ---- Per-domain marker regexes -----------------------------------------

# Math: a complete \boxed{...} expression. Matches Hendrycks MATH / GSM8k /
# AIME conventions. The boxed content may itself contain braces (e.g.,
# \boxed{\frac{1}{2}}), so we allow a single level of nested {...}. Two
# levels covers ~all practical math answers (parens, braces, set notation).
_MATH_MARKER_PATTERNS = [
    # \boxed{ matched with up to 2 levels of nesting:
    #   \boxed{ ... { ... { ... } ... } ... }
    r"\\boxed\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}",
    r"\\boxed\[[^\[\]]*\]",
    r"##\s*Final\s+Answer",
    r"Final\s+answer\s*[:=]\s*\S",
]

# Code: a complete code block close (```...```). We accept either a closed
# fence pair OR a bare close-fence on its own line/end-of-text.
_CODE_MARKER_PATTERNS = [
    r"```\w+[\s\S]*?```",          # opening (with language tag) + closing fence pair
    r"```[\s\S]*?\n```",           # generic open + close pair
    r"\n```\s*$",                  # bare close fence at newline-anchored end
    r"```\s*\Z",                   # bare close fence at end-of-text
]

# Generic: end-of-reasoning signals that work across domains. Leading
# whitespace/newline optional so we catch start-of-string and mid-text.
_GENERIC_MARKER_PATTERNS = [
    r"<answer>[\s\S]*?</answer>",
    r"<final_answer>[\s\S]*?</final_answer>",
    r"(?:^|\s)Therefore,\s+the\s+answer\s+is\s+\S",
]


# Compiled regex per domain (compile once at import).
_DOMAIN_PATTERNS = {
    "math": [re.compile(p, re.MULTILINE) for p in
             _MATH_MARKER_PATTERNS + _GENERIC_MARKER_PATTERNS],
    "code": [re.compile(p, re.MULTILINE) for p in
             _CODE_MARKER_PATTERNS + _GENERIC_MARKER_PATTERNS],
    "generic": [re.compile(p, re.MULTILINE) for p in
                _GENERIC_MARKER_PATTERNS + _MATH_MARKER_PATTERNS],
}


SUPPORTED_DOMAINS = tuple(_DOMAIN_PATTERNS.keys())


class ResponseMarkerDetector:
    """Domain-keyed regex detector. Stateless (no per-rollout state).

    Args:
        tokenizer: HF tokenizer used to decode output_token_ids on demand.
        domain: one of SUPPORTED_DOMAINS. Picks the regex set.
        check_every: poll interval (in tokens). Default 8 → ~12% overhead at
            ~250μs per check vs 2ms per generation step.
        suffix_window: max tokens at the tail of the response to decode +
            scan. Default 256 covers all reasonable marker spans (a boxed
            answer is ~50 tokens, code block close ~10 tokens). Avoids
            re-decoding the whole response.
    """

    __slots__ = ("tokenizer", "patterns", "domain", "check_every", "suffix_window")

    def __init__(
        self,
        tokenizer,
        domain: str = "math",
        check_every: int = 8,
        suffix_window: int = 256,
    ) -> None:
        if domain not in _DOMAIN_PATTERNS:
            raise ValueError(
                f"domain must be one of {SUPPORTED_DOMAINS}, got {domain!r}"
            )
        self.tokenizer = tokenizer
        self.patterns = _DOMAIN_PATTERNS[domain]
        self.domain = domain
        self.check_every = max(1, int(check_every))
        self.suffix_window = max(16, int(suffix_window))

    def should_check(self, n_tokens: int) -> bool:
        """Cheap pre-filter: only run detection every ``check_every`` tokens."""
        return n_tokens > 0 and (n_tokens % self.check_every == 0)

    def detect(self, output_token_ids: List[int]) -> bool:
        """Decode the suffix and check for any matching marker.

        Returns True if any marker pattern matches in the decoded text.
        Caller should typically gate this by ``should_check(len(output))``
        to avoid per-token decoding cost.
        """
        if not output_token_ids:
            return False
        # Decode only the suffix to keep cost bounded
        suffix_ids = output_token_ids[-self.suffix_window:]
        try:
            text = self.tokenizer.decode(suffix_ids, skip_special_tokens=True)
        except Exception:
            # Defensive: tokenizer can occasionally fail on partial UTF-8
            return False
        return any(pat.search(text) for pat in self.patterns)


def make_detector(
    tokenizer,
    domain: str = "math",
    check_every: int = 8,
    suffix_window: int = 256,
) -> Optional[ResponseMarkerDetector]:
    """Factory wrapper that returns None for invalid configs (for graceful
    degradation in the rollout worker — if marker detection is misconfigured,
    we want LP to behave as a no-op rather than crash the rollout).
    """
    try:
        return ResponseMarkerDetector(
            tokenizer=tokenizer,
            domain=domain,
            check_every=check_every,
            suffix_window=suffix_window,
        )
    except Exception:
        return None
