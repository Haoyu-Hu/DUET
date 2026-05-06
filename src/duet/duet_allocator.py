"""DUET joint controller (paper §5.2): Theorem-1 cost-weighted Neyman
allocation + Theorem-2 marginal-token stop, both coordinated by a single
dual multiplier λ★ via Theorem 3's bisection.

Public API:

    >>> alloc = DuetAllocator(budget=0.5, eps_pre=0.05, eps_len=0.05,
    ...                       bisection_iters=10)
    >>> result = alloc.allocate(
    ...     surrogate_s=[0.8, 1.2, 0.5, 1.5],   # ŝ_q
    ...     mean_length=[300, 400, 250, 600],   # L̂_q
    ...     full_budget_tokens=300_000,         # B_full from GRPO uniform
    ... )
    >>> result.n_q          # per-prompt rollout count
    >>> result.max_tokens_q # per-prompt max-token cap
    >>> result.p_pre_q      # per-prompt SNIPS propensity for the gate
    >>> result.lambda_star  # solved fixed point

Implementation status:
- v1.a (this file): allocator math is correct (T1 Neyman + T3 bisection),
  but the trainer-side hook in ``ray_trainer.py`` is currently a stub
  that does NOT yet override vLLM SamplingParams. v1.b will wire
  ``result.max_tokens_q`` into the rollout, and v1.c adds the per-token
  Δ̂ stop callback. See ``doc/paper_agent_work/DUET_Implementation_v1.md``
  for the milestone breakdown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DuetAllocation:
    """Output of one ``DuetAllocator.allocate`` call.

    All vectors are 1-D, length = batch size, indexed in the same order
    the caller passed ``surrogate_s``.
    """

    n_q: list[int]              # per-prompt rollout count (int ≥ n_min)
    max_tokens_q: list[int]     # per-prompt max-token cap (int ≥ tau_min)
    p_pre_q: list[float]        # SNIPS propensity for the pre-rollout gate
    lambda_star: float          # solved Lagrangian multiplier
    budget_used: float          # Σ n_q · L̂_q (≈ B at convergence)
    saturation_pre: float       # fraction of prompts hitting n_min floor
    saturation_len: float       # fraction of prompts hitting tau_min floor


class DuetAllocator:
    """Cost-weighted Neyman allocator with joint-controller bisection.

    The bisection solves Φ(λ) = Σ_q n_q(λ) · L_q(λ) = B for λ★, where
    n_q(λ) = round((1/√λ) · ŝ_q / √L̂_q) clipped to [n_min, n_max].
    We bisect on log λ in [-20, 20] (real range) since Φ is monotone
    decreasing in λ (Theorem 3 IVT argument).
    """

    def __init__(
        self,
        budget: float,
        *,
        eps_pre: float = 0.05,
        eps_len: float = 0.05,
        n_min: int = 1,
        n_max: int = 32,
        tau_min: int = 32,
        tau_max: int = 4096,
        bisection_iters: int = 10,
    ) -> None:
        if not 0.0 < budget <= 1.0:
            raise ValueError(f"budget must be in (0, 1], got {budget}")
        if not 0.0 < eps_pre <= 1.0:
            raise ValueError(f"eps_pre must be in (0, 1], got {eps_pre}")
        if not 0.0 < eps_len <= 1.0:
            raise ValueError(f"eps_len must be in (0, 1], got {eps_len}")
        self.budget = float(budget)
        self.eps_pre = float(eps_pre)
        self.eps_len = float(eps_len)
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.tau_min = int(tau_min)
        self.tau_max = int(tau_max)
        self.bisection_iters = int(bisection_iters)

    # ------------------------------------------------------------------ public

    def allocate(
        self,
        surrogate_s: Sequence[float],
        mean_length: Sequence[float],
        full_budget_tokens: float,
    ) -> DuetAllocation:
        """Return the joint (n_q, max_tokens_q) allocation under budget B.

        Args:
            surrogate_s: ŝ_q for each prompt in the batch (length M).
            mean_length: L̂_q running-mean per-rollout length.
            full_budget_tokens: B_full = train_batch × rollout_n × mean_resp,
                the matched-uniform-allocation budget. The token budget
                is B = self.budget × B_full.
        """
        if len(surrogate_s) != len(mean_length):
            raise ValueError(
                f"surrogate_s and mean_length must match length: "
                f"{len(surrogate_s)} vs {len(mean_length)}"
            )
        if not surrogate_s:
            raise ValueError("empty batch passed to DuetAllocator.allocate")

        target_B = self.budget * float(full_budget_tokens)

        # Bisect log λ on a wide bracket. Φ(λ) decreases in λ (Theorem 3).
        log_lo, log_hi = -20.0, 20.0
        n_q = max_tokens_q = None
        for _ in range(self.bisection_iters):
            log_mid = 0.5 * (log_lo + log_hi)
            lam = math.exp(log_mid)
            n_q, max_tokens_q = self._allocate_at_lambda(lam, surrogate_s, mean_length)
            phi = sum(n_q[i] * mean_length[i] for i in range(len(n_q)))
            if phi > target_B:
                log_lo = log_mid
            else:
                log_hi = log_mid
        # Final allocation at the converged λ★
        lambda_star = math.exp(0.5 * (log_lo + log_hi))
        n_q, max_tokens_q = self._allocate_at_lambda(
            lambda_star, surrogate_s, mean_length
        )

        # SNIPS propensity for the pre-rollout gate. Under cost-weighted
        # Neyman, P(I_pre = 1) = clamp(n_q / n_uniform, ε_pre, 1) — a
        # prompt with above-average ŝ has propensity 1, below-average has
        # propensity ε_pre. We approximate n_uniform = mean(n_q) for the
        # in-batch reference (paper Appendix I's bounded-floor variant).
        n_uniform = max(1.0, sum(n_q) / len(n_q))
        p_pre_q = [
            max(self.eps_pre, min(1.0, float(n_q[i]) / n_uniform))
            for i in range(len(n_q))
        ]

        budget_used = float(sum(n_q[i] * mean_length[i] for i in range(len(n_q))))
        saturation_pre = sum(1 for x in n_q if x <= self.n_min) / len(n_q)
        saturation_len = sum(1 for x in max_tokens_q if x <= self.tau_min) / len(max_tokens_q)

        return DuetAllocation(
            n_q=list(n_q),
            max_tokens_q=list(max_tokens_q),
            p_pre_q=p_pre_q,
            lambda_star=float(lambda_star),
            budget_used=budget_used,
            saturation_pre=float(saturation_pre),
            saturation_len=float(saturation_len),
        )

    # --------------------------------------------------------------- internals

    def _allocate_at_lambda(
        self,
        lam: float,
        surrogate_s: Sequence[float],
        mean_length: Sequence[float],
    ) -> tuple[list[int], list[int]]:
        """Compute n_q, max_tokens_q at a given λ.

        Theorem 1 stationarity: n_q = ŝ_q / √(λ · L̂_q).
        Operational form (Theorem 3 substitution): per-prompt stop
        threshold = ŝ_q² / L̂_q, but lacking an online Δ̂ estimator at
        v1.a we use the joint-controller length cap implied by the
        Lagrangian: max_tokens_q = round((B/Σŝ√L̂) · √(L̂_q/λ) · L̂_q^{1/2}),
        which simplifies to max_tokens_q = ŝ_q · √(L̂_q / λ).
        """
        sqrt_lam = math.sqrt(max(lam, 1e-30))
        n_q: list[int] = []
        max_tokens_q: list[int] = []
        for s_q, l_q in zip(surrogate_s, mean_length):
            n_real = float(s_q) / (sqrt_lam * math.sqrt(max(float(l_q), 1.0)))
            n_q.append(max(self.n_min, min(self.n_max, int(round(n_real)))))
            tau_real = float(s_q) * math.sqrt(max(float(l_q), 1.0) / max(lam, 1e-30))
            max_tokens_q.append(
                max(self.tau_min, min(self.tau_max, int(round(tau_real))))
            )
        return n_q, max_tokens_q


def make_n_q_divisible(
    n_q_list: Sequence[int],
    n_gpus: int,
    n_min: int,
) -> tuple[list[int], str]:
    """Adjust ``n_q_list`` so that ``sum(adjusted) % n_gpus == 0``.

    Trims entries above ``n_min`` first (preserves total budget); falls back
    to padding the smallest entries up by 1 when trim cannot proceed (all
    entries at floor). Returns (adjusted_list, mode) where mode is one of
    "noop" / "trim" / "pad" / "trim+pad".

    Replaces the bounded single-pass trim at the call site that crashed
    D3 σ_obs at step 222 with sum=178, n_gpus=8 (`verl/protocol.py:699`
    AssertionError "only support equal chunk").
    """
    out = list(n_q_list)
    M = len(out)
    if n_gpus <= 1 or M == 0 or sum(out) % n_gpus == 0:
        return out, "noop"

    trimmed = padded = False
    residue = sum(out) % n_gpus

    # Multi-pass trim: each sweep visits indices in (-n_q, i) order,
    # decrementing entries above n_min by 1. Repeat while progress is made.
    while residue > 0:
        order = sorted(range(M), key=lambda i: (-out[i], i))
        progressed = False
        for i in order:
            if residue == 0:
                break
            if out[i] > n_min:
                out[i] -= 1
                residue -= 1
                progressed = True
        if not progressed:
            break
        trimmed = True

    # Pad-up fallback when trim cannot reach divisibility (everything at floor).
    if residue > 0:
        pad = n_gpus - residue
        order = sorted(range(M), key=lambda i: (out[i], i))
        for k in range(pad):
            out[order[k % M]] += 1
        padded = True

    if trimmed and padded:
        mode = "trim+pad"
    elif padded:
        mode = "pad"
    else:
        mode = "trim"

    assert sum(out) % n_gpus == 0, (
        f"DUET divisibility broken: sum={sum(out)} n_gpus={n_gpus} mode={mode}"
    )
    return out, mode
