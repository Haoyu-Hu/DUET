"""DUET marker-gated stop rule as a vLLM V1 batch-level logits processor.

V1 (vLLM >= 0.10.1) rejects per-request ``SamplingParams.logits_processors``;
instead one processor instance per engine sees the whole ``[num_reqs, vocab]``
logits tensor and learns about requests through ``BatchUpdate``s. This module
ports ``duet_logits_processor.DuetStopProcessor`` (the V0 rule the paper
numbers came from) to that API without changing its decisions:

* Per request we keep exactly the V0 state machine (min_tokens floor, marker
  poll every ``check_every`` tokens from K1, K2+grace abort with an eps coin,
  post-marker confidence signal + hysteresis + eps_len), drawing from a
  per-request ``random.Random(seed)`` in the same order as V0. Given the same
  logits and seed, a request is stopped at the same token with the same flags.
* The confidence signal is computed for all armed rows in one batched GPU op
  (one small device->host copy per step) instead of one ``.item()`` sync per
  row per token as in V0.
* Per-request arguments travel in ``SamplingParams.extra_args["duet"]``.
  Outcome flags (did_abort / saw_marker / eps_kept) are published to the
  module-level ``FLAGS`` registry keyed by the request's ``uid``; the rollout
  worker reads them after ``LLM.generate`` (verl's sync rollout runs vLLM with
  ``external_launcher``, i.e. in the same process as this processor).
* State is keyed by uid, not by batch slot, so it survives preemption and
  re-admission (the request is re-``added`` with its output so far).

Requests without ``extra_args["duet"]`` (e.g. validation, GRPO) are untouched.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import deque
from typing import Any, Optional

import torch

from duet.duet_logits_processor import (
    SIGNAL_ENTROPY,
    SIGNAL_MARGIN,
    SIGNAL_MAX_PROB,
    SIGNAL_MEAN_LOGPROB,
    SIGNAL_MODES,
    SIGNAL_TOP_K_MASS,
)
from duet.duet_marker_detector import make_detector

try:  # vLLM >= 0.10.1
    from vllm.v1.sample.logits_processor import (
        BatchUpdate,
        LogitsProcessor,
        MoveDirectionality,
    )
except ImportError:  # CPU unit tests without vLLM: minimal stand-ins
    from duet._v1_lp_shim import BatchUpdate, LogitsProcessor, MoveDirectionality  # noqa: F401

EXTRA_ARGS_KEY = "duet"

# uid -> {"did_abort", "saw_marker", "eps_kept", "stopped_at"}; filled by the
# processor, read (and cleared) by the rollout worker after each generate().
FLAGS: dict[str, dict[str, Any]] = {}


def stable_seed(*parts: Any) -> int:
    """Process-independent 31-bit seed (Python's hash() is salted per process)."""
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & 0x7FFFFFFF


def pop_flags(uids: list[str]) -> list[dict[str, Any]]:
    """Return and forget the flags for ``uids`` (missing uid -> all False)."""
    default = {"did_abort": False, "saw_marker": False, "eps_kept": False, "stopped_at": None}
    return [FLAGS.pop(u, dict(default)) for u in uids]


class _Req:
    """Per-request mirror of DuetStopProcessor's fields."""

    __slots__ = ("uid", "eos", "threshold", "signal_mode", "eps_len", "min_tokens", "top_k",
                 "hysteresis_k", "k1", "k2", "grace", "abort_eps", "rng", "output_ids",
                 "consecutive", "history", "did_stop", "stopped_at", "marker_seen",
                 "decided_abort", "eps_kept", "k2_grace_passed", "lp_armed", "domain",
                 "last_n", "pending_force")

    def __init__(self, a: dict[str, Any], output_ids: list[int]):
        self.uid = str(a["uid"])
        self.eos = int(a["eos_token_id"])
        self.threshold = float(a["threshold"])
        self.signal_mode = str(a.get("signal_mode", SIGNAL_MAX_PROB))
        if self.signal_mode not in SIGNAL_MODES:
            raise ValueError(f"unknown DUET signal_mode {self.signal_mode!r}")
        self.eps_len = float(a.get("eps_len", 0.05))
        self.min_tokens = int(a.get("min_tokens", 32))
        self.top_k = int(a.get("top_k", 3))
        self.hysteresis_k = max(1, int(a.get("hysteresis_k", 1)))
        self.k1 = None if a.get("k1") is None else int(a["k1"])
        self.k2 = None if a.get("k2") is None else int(a["k2"])
        self.grace = max(0, int(a.get("grace_window", 150)))
        self.abort_eps = max(0.0, min(1.0, float(a.get("abort_eps", 0.05))))
        self.rng = random.Random(int(a["rng_seed"]))
        dom = a.get("marker_domain")
        self.domain = None if dom in (None, "", "none") else str(dom)
        self.output_ids = output_ids
        self.consecutive = 0
        self.history: deque = deque(maxlen=int(a.get("history_window", 8)))
        self.did_stop = False
        self.stopped_at: Optional[int] = None
        self.marker_seen = self.decided_abort = self.eps_kept = False
        self.k2_grace_passed = self.lp_armed = False
        self.last_n = -1
        self.pending_force = False

    def publish(self) -> None:
        FLAGS[self.uid] = {"did_abort": self.decided_abort, "saw_marker": self.marker_seen,
                           "eps_kept": self.eps_kept, "stopped_at": self.stopped_at}


class DuetV1StopProcessor(LogitsProcessor):
    """Batch-level port of DuetStopProcessor (see module docstring)."""

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        self.device = device
        self._tokenizer_name = None
        self._trust_remote_code = False
        if vllm_config is not None and getattr(vllm_config, "model_config", None) is not None:
            mc = vllm_config.model_config
            self._tokenizer_name = mc.tokenizer
            self._trust_remote_code = bool(getattr(mc, "trust_remote_code", False))
        self._detectors: dict[str, Any] = {}
        self._slot_uid: dict[int, str] = {}   # persistent-batch slot -> uid
        self._reqs: dict[str, _Req] = {}      # uid -> state (survives preemption)

    # -- vLLM interface ------------------------------------------------------
    @classmethod
    def validate_params(cls, params) -> None:  # vLLM >= 0.30 calls this
        a = (getattr(params, "extra_args", None) or {}).get(EXTRA_ARGS_KEY)
        if a is not None:
            for key in ("uid", "eos_token_id", "threshold", "rng_seed"):
                if key not in a:
                    raise ValueError(f"extra_args['{EXTRA_ARGS_KEY}'] missing {key!r}")

    def is_argmax_invariant(self) -> bool:
        return False  # it forces EOS

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if batch_update is None:
            return
        for slot in batch_update.removed:  # order: removed -> added -> moved
            self._slot_uid.pop(slot, None)
        for added in batch_update.added:
            slot, params, output_ids = added[0], added[1], added[-1]
            a = (getattr(params, "extra_args", None) or {}).get(EXTRA_ARGS_KEY)
            if a is None:
                self._slot_uid.pop(slot, None)
                continue
            uid = str(a["uid"])
            req = self._reqs.get(uid)
            if req is None:
                req = self._reqs[uid] = _Req(a, output_ids)
                req.publish()
            else:  # re-admitted after preemption: keep state, rebind output list
                req.output_ids = output_ids
            self._slot_uid[slot] = uid
        for src, dst, direction in batch_update.moved:
            s_uid, d_uid = self._slot_uid.pop(src, None), self._slot_uid.pop(dst, None)
            if s_uid is not None:
                self._slot_uid[dst] = s_uid
            if direction == MoveDirectionality.SWAP and d_uid is not None:
                self._slot_uid[src] = d_uid
        # Drop state for requests that finished (no longer in any slot) and
        # already published their flags.
        live = set(self._slot_uid.values())
        for uid in [u for u, r in self._reqs.items() if u not in live and r.did_stop]:
            del self._reqs[uid]

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._slot_uid:
            return logits
        force: list[tuple[int, int]] = []          # (slot, eos)
        armed: list[tuple[int, _Req]] = []
        for slot, uid in self._slot_uid.items():
            req = self._reqs[uid]
            n = len(req.output_ids)
            if req.did_stop:
                # V0 returns logits untouched once stopped; the EOS we forced
                # ends the request. Re-force only if the same position is
                # evaluated again before that EOS lands.
                if req.pending_force and n == req.stopped_at:
                    force.append((slot, req.eos))
                continue
            if n == req.last_n:  # same position again (chunked recompute): no new draws
                continue
            req.last_n = n
            verdict = self._pre_signal(req, n)
            if verdict == "force":
                force.append((slot, req.eos))
            elif verdict == "signal":
                armed.append((slot, req))
        if armed:
            force.extend(self._signal_step(logits, armed))
        if force:
            rows = torch.tensor([s for s, _ in force], device=logits.device, dtype=torch.long)
            eos = torch.tensor([e for _, e in force], device=logits.device, dtype=torch.long)
            logits[rows] = -math.inf
            logits[rows, eos] = 0.0
        return logits

    # -- V0 state machine, split around the (batched) signal ----------------
    def _detector(self, domain: str):
        det = self._detectors.get(domain)
        if det is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self._tokenizer_name,
                                                trust_remote_code=self._trust_remote_code)
            det = self._detectors[domain] = make_detector(tok, domain=domain)
        return det

    def _pre_signal(self, req: _Req, n: int) -> Optional[str]:
        """DuetStopProcessor.__call__ up to the LP; returns 'force'|'signal'|None."""
        if n < req.min_tokens:
            return None
        det = self._detector(req.domain) if req.domain else None
        if det is None:
            return "signal"  # V0 legacy path (no marker detector): LP armed from min_tokens
        if (not req.marker_seen and not req.k2_grace_passed
                and req.k1 is not None and n >= req.k1):
            if det.should_check(n) and det.detect(req.output_ids):
                req.marker_seen = req.lp_armed = True
                req.publish()
        if (not req.marker_seen and not req.k2_grace_passed
                and req.k2 is not None and n >= req.k2 + req.grace):
            if det.detect(req.output_ids):
                req.marker_seen = req.lp_armed = True
                req.publish()
            elif req.rng.random() >= req.abort_eps:
                req.did_stop = req.decided_abort = req.pending_force = True
                req.stopped_at = n
                req.publish()
                return "force"
            else:
                req.lp_armed = False
                req.eps_kept = req.k2_grace_passed = True
                req.publish()
        return "signal" if req.lp_armed else None

    def _signal_step(self, logits: torch.Tensor, armed: list[tuple[int, _Req]]):
        """Batched _compute_signal_and_decide + hysteresis + eps_len draw."""
        slots = torch.tensor([s for s, _ in armed], device=logits.device, dtype=torch.long)
        sub = logits.index_select(0, slots).to(torch.float32)
        modes = {r.signal_mode for _, r in armed}
        sig: dict[str, list[float]] = {}
        if modes & {SIGNAL_MAX_PROB, SIGNAL_MARGIN, SIGNAL_TOP_K_MASS, SIGNAL_ENTROPY}:
            probs = torch.softmax(sub, dim=-1)
            if SIGNAL_MAX_PROB in modes:
                sig[SIGNAL_MAX_PROB] = probs.max(dim=-1).values.tolist()
            if SIGNAL_MARGIN in modes:
                top2 = torch.topk(probs, k=2, dim=-1).values
                sig[SIGNAL_MARGIN] = (top2[:, 0] - top2[:, 1]).tolist()
            if SIGNAL_TOP_K_MASS in modes:
                kmax = max(max(1, r.top_k) for _, r in armed)
                topk = torch.topk(probs, k=kmax, dim=-1).values
                csum = torch.cumsum(topk, dim=-1)
                idx = torch.tensor([max(1, r.top_k) - 1 for _, r in armed], device=sub.device)
                sig[SIGNAL_TOP_K_MASS] = csum.gather(1, idx[:, None]).squeeze(1).tolist()
            if SIGNAL_ENTROPY in modes:
                sig[SIGNAL_ENTROPY] = (-(probs * torch.log(probs.clamp_min(1e-9))).sum(-1)).tolist()
        if SIGNAL_MEAN_LOGPROB in modes:
            sig[SIGNAL_MEAN_LOGPROB] = torch.log_softmax(sub, dim=-1).max(dim=-1).values.tolist()

        fired = []
        for i, (slot, req) in enumerate(armed):
            value = sig[req.signal_mode][i]
            if req.signal_mode == SIGNAL_MEAN_LOGPROB:
                req.history.append(value)
                value = sum(req.history) / len(req.history)
            should_stop = value < req.threshold if req.signal_mode == SIGNAL_ENTROPY \
                else value > req.threshold
            req.consecutive = req.consecutive + 1 if should_stop else 0
            if should_stop and req.consecutive >= req.hysteresis_k:
                if req.rng.random() < req.eps_len:
                    req.consecutive = max(0, req.hysteresis_k - 1)
                    continue
                req.did_stop = req.pending_force = True
                req.stopped_at = len(req.output_ids)
                req.publish()
                fired.append((slot, req.eos))
        return fired


def build_extra_args(*, uid: str, eos_token_id: int, threshold: float, signal_mode: str,
                     eps_len: float, min_tokens: int, top_k: int, hysteresis_k: int,
                     marker_domain: Optional[str], k1: Optional[int], k2: Optional[int],
                     grace_window: int, abort_eps: float, rng_seed: int) -> dict[str, Any]:
    """extra_args payload for one DUET request (primitives only: vLLM deep-copies it)."""
    return {EXTRA_ARGS_KEY: {
        "uid": uid, "eos_token_id": int(eos_token_id), "threshold": float(threshold),
        "signal_mode": str(signal_mode), "eps_len": float(eps_len),
        "min_tokens": int(min_tokens), "top_k": int(top_k), "hysteresis_k": int(hysteresis_k),
        "marker_domain": marker_domain, "k1": k1, "k2": k2,
        "grace_window": int(grace_window), "abort_eps": float(abort_eps),
        "rng_seed": int(rng_seed),
    }}
