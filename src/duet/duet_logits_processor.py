"""Per-token Δ̂ stop callback for DUET (paper §5.2 + Theorem 2).

A vLLM LogitsProcessor that monitors a per-token confidence signal as a
practical proxy for the marginal variance reduction Δ_q(t) = U_q(t) - U_q(t+1).
When the signal indicates the model has become confident (small remaining
information), the processor forces the next token to be EOS, terminating the
rollout early. ε-length exploration ensures every reachable length has
positive propensity (paper Assumption A5).

Three signal modes are supported, selectable via ``signal_mode``:

- ``max_prob`` (default, fast):
    signal = max_v softmax(logits)_v
    stops when signal > threshold (high peak probability ⇔ low entropy ⇔
    small remaining information). ~10μs per call. **The recommended choice
    for production runs** — paper §5.2's adaptive stop with negligible
    per-token compute overhead.

- ``entropy`` (paper-pure, slow):
    signal = -Σ p_v log p_v over softmax(logits)
    stops when signal < threshold (low entropy ⇔ small remaining info).
    ~250μs per call (full softmax + log + sum + .item() forced sync).
    Use only for ablation studies; for full-batch training it adds
    ~30-90s per training step at production hyperparameters.

- ``mean_logprob`` (smoothed, slow-lagging):
    signal = running mean over the past ``history_window`` of per-token
    max-logprob values; stops when signal > threshold (sustained
    high confidence). ~10μs per call but tends to stop *later* than
    instantaneous signals (lag from the running mean), losing some
    speedup. Provided as a backup baseline; not recommended as default.

Per-prompt threshold can be derived from the surrogate ŝ_q via
``threshold_from_surrogate``; see paper §5.2 for the calibration.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import List, Optional

import torch


# Signal modes. Each maps to a (compute_signal_fn, stop_predicate_fn) pair.
SIGNAL_MAX_PROB = "max_prob"
SIGNAL_ENTROPY = "entropy"
SIGNAL_MEAN_LOGPROB = "mean_logprob"
SIGNAL_MARGIN = "margin"            # max_prob - second_max_prob; ~10μs
SIGNAL_TOP_K_MASS = "top_k_mass"    # Σ top-K softmax probs; ~15μs
SIGNAL_MODES = (
    SIGNAL_MAX_PROB,
    SIGNAL_ENTROPY,
    SIGNAL_MEAN_LOGPROB,
    SIGNAL_MARGIN,
    SIGNAL_TOP_K_MASS,
)

# Threshold-mapping modes for `assign_thresholds_batched` (V2).
THRESHOLD_LINEAR = "linear"            # paper-default: linear in s_hat²/L_hat
THRESHOLD_LOG_SCALE = "log_scale"      # log1p compression — spreads dynamic range
THRESHOLD_BATCH_PERCENTILE = "batch_percentile"  # quantile within batch
THRESHOLD_MODES = (THRESHOLD_LINEAR, THRESHOLD_LOG_SCALE, THRESHOLD_BATCH_PERCENTILE)


class DuetStopProcessor:
    """Per-rollout LogitsProcessor that forces EOS based on a confidence signal.

    Each instance is closed over per-rollout, so vLLM's per-prompt
    ``SamplingParams.logits_processors=[DuetStopProcessor(...)]`` pattern
    works without cross-rollout state interference.
    """

    __slots__ = (
        "eos_token_id",
        "threshold",
        "signal_mode",
        "eps_len",
        "min_tokens",
        "top_k",
        "hysteresis_k",
        "_consecutive_above",
        "_rng",
        "_did_stop",
        "_stopped_at",
        "_signal_history",
        "_history_window",
        # v3 marker-gated state
        "marker_detector",      # ResponseMarkerDetector or None
        "k1",                   # arm-LP-after-marker checkpoint
        "k2",                   # late-marker-OR-abort checkpoint
        "grace_window",         # extra tokens after K2 to await marker
        "abort_eps",            # ε-exploration: keep no-marker rollouts with prob ε
        "_marker_seen",         # internal: True once *real* marker detected
        "_decided_abort",       # internal: True once K2-grace abort decision made
        "_eps_kept",             # internal: True once ε-keep decision (lottery win past K2)
        "_k2_grace_passed",      # internal: True once we've made the K2-grace decision (latch)
        "_lp_armed",            # internal: True once LP allowed to fire
    )

    def __init__(
        self,
        eos_token_id: int,
        threshold: float,
        signal_mode: str = SIGNAL_MAX_PROB,
        eps_len: float = 0.05,
        min_tokens: int = 32,
        rng_seed: Optional[int] = None,
        history_window: int = 8,
        top_k: int = 3,
        hysteresis_k: int = 1,
        # v3 marker-gated args (all optional → backwards compatible)
        marker_detector=None,
        k1: Optional[int] = None,
        k2: Optional[int] = None,
        grace_window: int = 150,
        abort_eps: float = 0.05,
    ) -> None:
        if signal_mode not in SIGNAL_MODES:
            raise ValueError(
                f"signal_mode must be one of {SIGNAL_MODES}, got {signal_mode!r}"
            )
        self.eos_token_id = int(eos_token_id)
        self.threshold = float(threshold)
        self.signal_mode = signal_mode
        self.eps_len = float(eps_len)
        self.min_tokens = int(min_tokens)
        self.top_k = int(top_k)
        self.hysteresis_k = max(1, int(hysteresis_k))
        self._consecutive_above = 0
        self._rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
        self._did_stop = False
        self._stopped_at: Optional[int] = None
        self._history_window = int(history_window)
        self._signal_history: deque = deque(maxlen=self._history_window)
        # v3 marker-gated state machine. When marker_detector is None (legacy
        # mode), the processor falls back to the v2 behavior: LP arms at
        # min_tokens regardless of response content. When provided, LP only
        # arms after the marker is detected, and rollouts past K2+grace
        # without a marker are flagged for trainer-side abort + zero gradient.
        self.marker_detector = marker_detector
        self.k1 = int(k1) if k1 is not None else None
        self.k2 = int(k2) if k2 is not None else None
        self.grace_window = max(0, int(grace_window))
        self.abort_eps = max(0.0, min(1.0, float(abort_eps)))
        self._marker_seen = False
        self._decided_abort = False
        self._eps_kept = False
        self._k2_grace_passed = False
        self._lp_armed = False

    def __call__(
        self,
        prompt_token_ids: List[int],
        output_token_ids: List[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Modify ``logits`` / return modified logits per the chosen signal mode.

        v3 marker-gated state machine (when marker_detector is set):
          phase A: pre-marker. LP off. Just generate.
            - If len < K1, skip detection (cheap path).
            - At K1+, poll detector every check_every tokens.
            - If marker found, transition to phase B.
            - If len ≥ K2 and no marker, transition to phase C.
          phase B: marker-armed. LP fires per signal+threshold+hysteresis.
            - Same as v2 LP behavior, but only fires AFTER marker exists.
          phase C: K2-grace. Wait grace_window more tokens for marker.
            - If marker appears, transition to phase B.
            - If grace expires without marker, decide abort (with prob 1-ε).
              Abort = force EOS now AND set _decided_abort flag.

        When marker_detector is None: fall back to v2 behavior (LP arms at
        min_tokens regardless of content). Backwards compatible.
        """
        # Idempotent post-stop
        if self._did_stop:
            return logits

        n_tokens = len(output_token_ids)

        # ---- Legacy v2 path (no marker detector) ------------------------
        if self.marker_detector is None:
            if n_tokens < self.min_tokens:
                return logits
            return self._maybe_fire_lp(output_token_ids, logits)

        # ---- v3 marker-gated path ---------------------------------------
        # Always honor min_tokens hard floor (safety against degenerate
        # K1=0 configs)
        if n_tokens < self.min_tokens:
            return logits

        # Phase A: pre-marker. Detect periodically once past K1.
        # Skip detection once K2+grace has passed (ε-kept latch via flag).
        _k2_grace_passed = getattr(self, "_k2_grace_passed", False)
        if (not self._marker_seen and not _k2_grace_passed
                and self.k1 is not None and n_tokens >= self.k1):
            if self.marker_detector.should_check(n_tokens):
                if self.marker_detector.detect(output_token_ids):
                    self._marker_seen = True
                    self._lp_armed = True

        # Phase C/abort: past K2 + grace with no marker → decide abort
        if (not self._marker_seen and not _k2_grace_passed
                and self.k2 is not None
                and n_tokens >= self.k2 + self.grace_window):
            # Final marker check (in case detector missed due to
            # check_every skipping the just-emitted marker token)
            if self.marker_detector.detect(output_token_ids):
                self._marker_seen = True
                self._lp_armed = True
            else:
                # Abort with probability (1 - abort_eps). ε-keep preserves
                # SNIPS unbiasedness on the no-marker subset.
                if self._rng.random() >= self.abort_eps:
                    forced = torch.full_like(logits, -math.inf)
                    forced[self.eos_token_id] = 0.0
                    self._did_stop = True
                    self._decided_abort = True
                    self._stopped_at = n_tokens
                    return forced
                # ε-kept: let it run to natural EOS without further LP firing.
                # NOTE (paper-faithful, post 2026-04-28 fix): saw_marker stays False —
                # ε-kept rollouts are decoupled from "real marker detected". A separate
                # `_eps_kept` flag is exposed so downstream SNIPS reweighting can apply
                # 1/ε_abort to *only* true ε-kept rollouts (vs natural-EOS-before-K2).
                self._lp_armed = False
                self._eps_kept = True
                # Latch _did_stop=False (do nothing here); future detect() calls are
                # skipped via this token-position guard:
                self._k2_grace_passed = True

        # Phase B: marker armed → LP can fire on confidence signal
        if self._lp_armed:
            return self._maybe_fire_lp(output_token_ids, logits)

        return logits

    def _maybe_fire_lp(
        self,
        output_token_ids: List[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Run signal computation + hysteresis + ε-len decision and
        force EOS if all conditions met. Shared by v2 (legacy) and v3
        (post-marker) paths.
        """
        signal, should_stop = self._compute_signal_and_decide(logits)
        if should_stop:
            self._consecutive_above += 1
        else:
            self._consecutive_above = 0
        fire = should_stop and (self._consecutive_above >= self.hysteresis_k)
        if fire:
            if self._rng.random() < self.eps_len:
                self._consecutive_above = max(0, self.hysteresis_k - 1)
                return logits
            forced = torch.full_like(logits, -math.inf)
            forced[self.eos_token_id] = 0.0
            self._did_stop = True
            self._stopped_at = len(output_token_ids)
            return forced
        return logits

    @property
    def did_abort(self) -> bool:
        """True iff this rollout was terminated via the K2-grace abort
        path (vs marker-armed LP, ε-len, or natural EOS). Read by the
        rollout-worker after generation to flag training-side gradient
        masking."""
        return self._decided_abort

    @property
    def saw_marker(self) -> bool:
        """True iff a *real* marker was detected at any point during generation.
        Used for diagnostics + K-estimator filtering. Decoupled from ε-keep
        (post 2026-04-28 fix): ε-kept rollouts have saw_marker=False but
        eps_kept=True, so the SNIPS reweighting can target them precisely."""
        return self._marker_seen

    @property
    def eps_kept(self) -> bool:
        """True iff this rollout reached K2+grace without a marker AND won the
        ε_abort exploration lottery (kept instead of aborted). The trainer
        applies an additional 1/ε_abort SNIPS factor to exactly these rollouts
        to preserve unbiasedness on the marker-less subset."""
        return self._eps_kept

    # ------------------------------------------------------------------ signals

    def _compute_signal_and_decide(
        self, logits: torch.Tensor
    ) -> tuple[float, bool]:
        """Compute the chosen confidence signal and decide stop.

        Stop conventions per signal:
          - max_prob       : signal > threshold → stop (peakedness ⇒ low info)
          - entropy        : signal < threshold → stop (low entropy ⇒ low info)
          - mean_logprob   : signal > threshold → stop (sustained confidence)
        """
        if self.signal_mode == SIGNAL_MAX_PROB:
            # softmax over float32 for numerical stability; max + .item() = ~10μs
            probs = torch.softmax(logits.to(torch.float32), dim=-1)
            signal = probs.max().item()
            return signal, signal > self.threshold

        if self.signal_mode == SIGNAL_ENTROPY:
            # Full entropy computation; ~250μs per call
            probs = torch.softmax(logits.to(torch.float32), dim=-1)
            signal = -(probs * torch.log(probs.clamp_min(1e-9))).sum().item()
            return signal, signal < self.threshold

        if self.signal_mode == SIGNAL_MEAN_LOGPROB:
            # Use max-logprob as a per-step proxy for the SAMPLED token's
            # logprob (we don't see the sample inside the LP). Maintain a
            # running window so the signal is smoothed.
            log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
            current_max_log = log_probs.max().item()
            self._signal_history.append(current_max_log)
            signal = sum(self._signal_history) / len(self._signal_history)
            return signal, signal > self.threshold

        if self.signal_mode == SIGNAL_MARGIN:
            # max_prob - second_max_prob: the confidence GAP. Discriminates
            # "peaked but contested" (small margin, high uncertainty about
            # alternatives) from "peaked and unique" (large margin, true
            # commitment). Two top-1 calls ≈ ~10–15μs.
            probs = torch.softmax(logits.to(torch.float32), dim=-1)
            top2 = torch.topk(probs, k=2).values
            signal = float((top2[0] - top2[1]).item())
            return signal, signal > self.threshold

        if self.signal_mode == SIGNAL_TOP_K_MASS:
            # Σ top-K softmax probabilities. Larger K gives a smoother proxy for
            # entropy concentration; cheap (one topk + sum) ~15μs.
            probs = torch.softmax(logits.to(torch.float32), dim=-1)
            k = max(1, int(self.top_k))
            topk = torch.topk(probs, k=k).values
            signal = float(topk.sum().item())
            return signal, signal > self.threshold

        raise RuntimeError(f"unhandled signal_mode {self.signal_mode!r}")  # unreachable


def threshold_from_surrogate(
    s_hat: float,
    L_hat: float,
    *,
    signal_mode: str = SIGNAL_MAX_PROB,
    alpha: float = 1.0,
    floor: float = 0.05,
    ceiling: float = 0.99,
    threshold_mode: str = THRESHOLD_LINEAR,
) -> float:
    """Map a per-prompt surrogate (ŝ_q, L̂_q) to a stop threshold.

    Per-prompt mapping (no batch context). For batch-aware mappings (e.g.,
    ``batch_percentile``), use :func:`assign_thresholds_batched` which has
    access to the full distribution over the prompt batch.

    Modes (controlled by ``threshold_mode``):
      - ``linear`` (paper-default): linear in s_hat²/L_hat, divided by 1e8.
      - ``log_scale``: log1p(s_hat²/L_hat) compresses dynamic range so high-
        variance prompts don't saturate the ceiling. Better when info_rate
        spans 10× across the batch.

    Stop conventions per signal (recap):
      - max_prob, margin, mean_logprob, top_k_mass: signal > threshold → stop
      - entropy: signal < threshold → stop
    """
    if L_hat <= 0:
        return floor

    raw_info_rate = (s_hat ** 2) / max(float(L_hat), 1.0)

    if threshold_mode == THRESHOLD_LOG_SCALE:
        # log1p compresses the typical 10^7 – 10^9 raw range into ~16-21 nats,
        # then scale by alpha and pass through a sigmoid. Per-prompt thresholds
        # spread more evenly across [floor, ceiling] than the linear mapping.
        import math as _m
        compressed = alpha * _m.log1p(raw_info_rate) / 20.0  # ~[0, 1.05]
        sigmoid = 1.0 / (1.0 + _m.exp(-(compressed - 0.5) * 6.0))
        scaled = sigmoid  # in (0, 1)
    else:
        # Linear (paper-default).
        scaled = alpha * raw_info_rate / 1.0e8

    if signal_mode in (SIGNAL_MAX_PROB, SIGNAL_TOP_K_MASS):
        # Probability-scale [0, 1]. Higher info rate → higher confidence
        # threshold (stop only when very confident).
        if threshold_mode == THRESHOLD_LOG_SCALE:
            return max(floor, min(ceiling, floor + (ceiling - floor) * scaled))
        return max(floor, min(ceiling, 0.5 + 0.4 * scaled))

    if signal_mode == SIGNAL_MARGIN:
        # Margin (max - 2nd_max) lives in [0, 1]; large gap = high confidence.
        # Empirically smaller dynamic range than max_prob — center band tighter.
        if threshold_mode == THRESHOLD_LOG_SCALE:
            return max(floor, min(ceiling, floor + (ceiling - floor) * scaled))
        return max(floor, min(ceiling, 0.3 + 0.5 * scaled))

    if signal_mode == SIGNAL_MEAN_LOGPROB:
        # log space; thresholds typically in [-3, -0.1] (log of probability)
        if threshold_mode == THRESHOLD_LOG_SCALE:
            return max(-5.0, min(-0.05, -3.0 + 2.95 * scaled))
        return max(-5.0, min(-0.05, -1.5 + 1.0 * scaled))

    if signal_mode == SIGNAL_ENTROPY:
        # Higher info rate → higher entropy threshold (stop earlier even at
        # moderate uncertainty). Note: stop predicate is signal < threshold.
        if threshold_mode == THRESHOLD_LOG_SCALE:
            return max(floor, min(5.0, floor + (5.0 - floor) * scaled))
        return max(floor, min(5.0, 0.5 + 1.0 * scaled))

    return floor


def assign_thresholds_batched(
    s_hat_list,
    L_hat_list,
    *,
    signal_mode: str = SIGNAL_MAX_PROB,
    threshold_mode: str = THRESHOLD_LINEAR,
    alpha: float = 1.0,
    floor: float = 0.5,
    ceiling: float = 0.99,
):
    """Assign per-prompt thresholds across a batch.

    For ``threshold_mode`` ∈ {linear, log_scale}, this is a list-comprehension
    over :func:`threshold_from_surrogate` (per-prompt independent).

    For ``threshold_mode = batch_percentile``, the threshold for prompt q is
    set to the empirical CDF position of its info_rate within the batch:
    rank-based mapping linearly into [floor, ceiling]. This guarantees per-
    batch threshold dispersion regardless of the absolute info_rate scale —
    rescues the per-prompt discrimination claim when raw info_rate is
    compressed (e.g., math-train-clean ridge fit).

    Args:
        s_hat_list: per-prompt surrogate σ predictions (length M)
        L_hat_list: per-prompt surrogate L̂ predictions (length M)
        signal_mode: SIGNAL_MAX_PROB | SIGNAL_MARGIN | ... (controls band)
        threshold_mode: linear | log_scale | batch_percentile
        alpha, floor, ceiling: scalars (same as threshold_from_surrogate)

    Returns:
        list[float] of thresholds, length M.
    """
    if len(s_hat_list) != len(L_hat_list):
        raise ValueError(
            f"length mismatch: s_hat_list={len(s_hat_list)} "
            f"L_hat_list={len(L_hat_list)}"
        )

    if threshold_mode != THRESHOLD_BATCH_PERCENTILE:
        return [
            threshold_from_surrogate(
                float(s),
                float(L),
                signal_mode=signal_mode,
                alpha=alpha,
                floor=floor,
                ceiling=ceiling,
                threshold_mode=threshold_mode,
            )
            for s, L in zip(s_hat_list, L_hat_list)
        ]

    # batch_percentile: rank-based per-batch dispersion.
    info_rates = [
        (s ** 2) / max(float(L), 1.0)
        for s, L in zip(s_hat_list, L_hat_list)
    ]
    n = len(info_rates)
    if n == 1:
        # Degenerate: single prompt → midpoint
        mid = (floor + ceiling) / 2.0
        if signal_mode == SIGNAL_MEAN_LOGPROB:
            return [-1.5]
        if signal_mode == SIGNAL_ENTROPY:
            return [(floor + 5.0) / 2.0]
        return [mid]

    sorted_idx = sorted(range(n), key=lambda i: info_rates[i])
    # quantile rank in [0, 1]
    rank_pos = [0.0] * n
    for rank, idx in enumerate(sorted_idx):
        rank_pos[idx] = rank / max(n - 1, 1)

    # Map rank → threshold band depending on signal_mode
    out = []
    for q in rank_pos:
        if signal_mode in (SIGNAL_MAX_PROB, SIGNAL_TOP_K_MASS, SIGNAL_MARGIN):
            out.append(floor + (ceiling - floor) * q)
        elif signal_mode == SIGNAL_MEAN_LOGPROB:
            out.append(-3.0 + 2.95 * q)
        elif signal_mode == SIGNAL_ENTROPY:
            out.append(floor + (5.0 - floor) * q)
        else:
            out.append(floor + (ceiling - floor) * q)
    return out
