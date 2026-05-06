"""DUET per-prompt running state (paper §4.1, §4.2).

Tracks two per-prompt online statistics, indexed by stable ``extra_info.index``
(matching VIP's per-prompt key convention at ``ray_trainer.py:842``):

    * σ̂_q^obs — Welford running mean of within-step rollout-variance estimates
                std_i(A_{q,i} · Σ_ℓ log π_θ(y_{q,i,ℓ}|q)). After ``k_warmup``
                observations, ``get_s`` returns the running mean; before, returns
                None and the caller falls back to ridge or σ_min.
    * L̂_q     — online mean of kept-not-aborted rollout lengths. ``get_l``
                returns the running mean once any observations exist.

Stored as inline dicts (not a class) to mirror the existing ``_duet_k_estimator``
pattern and avoid a new module/class layer. The "state" is just a mutable
dict; helpers take it as the first argument.

Keys in the state dict (see ``new_state``):
    s_obs:       q_idx -> {"n": int, "mean": float, "M2": float}  (Welford)
    L_hat:       q_idx -> {"n": int, "mean": float}               (online mean)
    k_warmup:    int  — min observations before get_s returns the mean
    max_tracked: int  — LRU cap; oldest entries evicted on overflow
    seen_total:  int  — monotonic count of unique prompts ever inserted
    order:       OrderedDict[q_idx, None] — LRU recency tracker
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


def new_state(*, k_warmup: int = 1, max_tracked: int = 50_000) -> dict[str, Any]:
    """Create a fresh per-prompt state dict."""
    return {
        "s_obs": {},
        "L_hat": {},
        "k_warmup": int(k_warmup),
        "max_tracked": int(max_tracked),
        "seen_total": 0,
        "order": OrderedDict(),
    }


def _touch(state: dict[str, Any], q_idx: str) -> None:
    """LRU-touch an entry (insert or move to end). Evict oldest on overflow."""
    order: OrderedDict = state["order"]
    if q_idx in order:
        order.move_to_end(q_idx)
    else:
        order[q_idx] = None
        state["seen_total"] += 1
        if len(order) > state["max_tracked"]:
            old, _ = order.popitem(last=False)
            state["s_obs"].pop(old, None)
            state["L_hat"].pop(old, None)


def update_s(state: dict[str, Any], q_idx: str, sigma_t: float) -> None:
    """Welford running-mean update of σ̂_q^obs.

    Skips empty q_idx (missing extra_info.index) and non-finite sigma_t.
    """
    if not q_idx:
        return
    if not (sigma_t == sigma_t and sigma_t < 1e12 and sigma_t > -1e12):  # NaN/Inf guard
        return
    rec = state["s_obs"].get(q_idx)
    if rec is None:
        rec = {"n": 0, "mean": 0.0, "M2": 0.0}
        state["s_obs"][q_idx] = rec
    rec["n"] += 1
    delta = sigma_t - rec["mean"]
    rec["mean"] += delta / rec["n"]
    rec["M2"] += delta * (sigma_t - rec["mean"])
    _touch(state, q_idx)


def update_l(state: dict[str, Any], q_idx: str, length: float) -> None:
    """Online-mean update of L̂_q over kept rollout lengths."""
    if not q_idx:
        return
    if not (length == length and length >= 0 and length < 1e9):
        return
    rec = state["L_hat"].get(q_idx)
    if rec is None:
        rec = {"n": 0, "mean": 0.0}
        state["L_hat"][q_idx] = rec
    rec["n"] += 1
    rec["mean"] += (float(length) - rec["mean"]) / rec["n"]
    _touch(state, q_idx)


def get_s(state: dict[str, Any], q_idx: str) -> float | None:
    """Return Welford running-mean σ̂_q^obs, or None until n >= k_warmup."""
    if not q_idx:
        return None
    rec = state["s_obs"].get(q_idx)
    if rec is None or rec["n"] < state["k_warmup"]:
        return None
    return float(rec["mean"])


def get_l(state: dict[str, Any], q_idx: str) -> float | None:
    """Return online-mean L̂_q, or None for cold prompts."""
    if not q_idx:
        return None
    rec = state["L_hat"].get(q_idx)
    if rec is None or rec["n"] < 1:
        return None
    return float(rec["mean"])


def stats(state: dict[str, Any]) -> dict[str, float]:
    """Return a small summary suitable for TB export.

    Keys:
        prompt_state_size: current number of tracked prompts (post-eviction)
        prompt_seen_count: monotonic count of unique prompts ever inserted
        s_obs_warmup_progress: fraction of tracked prompts with n >= k_warmup
    """
    s_obs = state["s_obs"]
    n_tracked = len(state["order"])
    n_warm = sum(1 for r in s_obs.values() if r["n"] >= state["k_warmup"])
    return {
        "prompt_state_size": float(n_tracked),
        "prompt_seen_count": float(state["seen_total"]),
        "s_obs_warmup_progress": float(n_warm) / float(max(n_tracked, 1)),
    }
