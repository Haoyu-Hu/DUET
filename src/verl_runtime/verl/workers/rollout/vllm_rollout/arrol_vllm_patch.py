"""ARRoL faithful — isolated vLLM mid-decode pruning patch (paper §4.3).

**Strict isolation contract:**

This module performs reversible monkey-patching of vLLM's request lifecycle to
implement ARRoL's mid-decode pruning at position L_detect (paper §4.3). The
patch is applied via ``apply_arrol_patch(callback, l_detect)`` and torn down
via ``deactivate_arrol_patch()``. When NOT activated, importing this module
has zero effect on vLLM behavior — no class methods are altered, no global
state is touched.

**Design constraints (per user 2026-04-25):**

1. *Distinguishable*: lives in a SEPARATE file from ``vllm_rollout_spmd.py``
   (the standard rollout module). The standard module imports + activates
   this patch ONLY when ``meta_info["arrol_faithful_active"] = True`` is
   set in the gen_batch (which only happens under ``--arrol-faithful``).
2. *Doesn't influence other experiments*: when the activation flag is not
   set, this module is never imported. ``--arrol-enable`` (Phase-1) still
   uses ``arrol_head.py`` + post-hoc reward masking — completely independent
   code path.
3. *Reversible*: ``deactivate_arrol_patch()`` restores the original methods.
   Idempotent (safe to call multiple times).

**Mechanism (paper §4.3 verbatim):**

> "the backend evaluates [the rollout's] quality and samples a pruning
> decision according to the survival probability. Pruned rollouts are
> immediately removed from the request pool, so the scheduler can
> reallocate the freed capacity."

We implement this via:

A. **Hidden-state capture**: monkey-patch the model's ``compute_logits``
   method (which receives both hidden states and the per-request position
   info). When a request crosses position ``l_detect``, we extract its
   hidden state vector at that position into a dispatch buffer.

B. **Pruning decision**: at the end of each ``LLMEngine.step()``, the
   buffered (request_id, hidden_state) pairs are passed to the user's
   callback (typically ``ArrolFaithfulHead.decide_keep``), which returns
   keep/prune.

C. **Request-pool removal**: pruned requests are aborted via
   ``engine.abort_request(request_id)``. The vLLM scheduler reclaims
   the KV-cache slots immediately on the next step.

**Reproducibility caveat (paper note):**

This implementation is pinned to vLLM 0.9.2 v1 engine. Version upgrades
(0.10+) may break the patch since vLLM's internal scheduler API is not
guaranteed stable. The patch is intentionally narrow: only ``compute_logits``
on the model and ``step`` on the engine are touched, both with explicit
backup-and-restore. See ``doc/paper_agent_work/ARRoL_Faithful_Implementation.md``
for the full design, deviations from the paper, and validation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class _ArrolPatchState:
    """Module-level state carrying the active patch + saved originals.

    Lifetime: created on apply_arrol_patch(), cleared on deactivate_arrol_patch().
    Holds: callback fn, L_detect threshold, hidden-state buffer, original methods.
    """

    callback: Callable[[str, torch.Tensor], tuple[bool, float]]
    l_detect: int
    # Buffer of (request_id, hidden_state) captured this step
    hidden_state_buffer: dict = None
    # Map request_id → propensity p_keep (for downstream SNIPS log)
    propensity_log: dict = None
    # Counters for diagnostics
    n_pruned: int = 0
    n_kept: int = 0
    n_seen: int = 0
    n_callback_failed: int = 0  # callback raised; row never reached n_seen
    n_crossings_without_buffer: int = 0  # crossing detected but no hidden in buffer
    # Saved originals (for restore)
    original_compute_logits: Optional[Callable] = None
    original_step: Optional[Callable] = None
    patched_engine_class: Optional[type] = None
    patched_model_obj: Optional[object] = None

    def __post_init__(self) -> None:
        if self.hidden_state_buffer is None:
            self.hidden_state_buffer = {}
        if self.propensity_log is None:
            self.propensity_log = {}


# Module-level singleton — None when patch is not active.
_PATCH_STATE: Optional[_ArrolPatchState] = None


def is_active() -> bool:
    return _PATCH_STATE is not None


def apply_arrol_patch(
    inference_engine,
    callback: Callable[[str, torch.Tensor], tuple[bool, float]],
    l_detect: int = 512,
) -> None:
    """Install the patch on a specific vLLM ``LLM`` instance.

    Args:
        inference_engine: the ``vllm.LLM`` object whose ``llm_engine`` we patch
        callback: function (request_id: str, hidden_state: torch.Tensor)
                  → (keep: bool, p_keep_propensity: float). Called at L_detect
                  crossings; if keep=False the request is aborted.
        l_detect: token position at which the head scores (paper default 512)

    Raises:
        RuntimeError: if a patch is already active (call deactivate first).
    """
    global _PATCH_STATE
    if _PATCH_STATE is not None:
        raise RuntimeError(
            "ARRoL patch already active; call deactivate_arrol_patch() first."
        )

    state = _ArrolPatchState(callback=callback, l_detect=int(l_detect))

    # ---- 1. Patch model's compute_logits to capture hidden states -----------
    # The model object is at inference_engine.llm_engine.model_executor.driver_worker.model_runner.model
    # (or similar — exact path varies between vLLM v0/v1 engines).
    try:
        model_runner = inference_engine.llm_engine.model_executor.driver_worker.model_runner
        model = model_runner.model
    except AttributeError as e:
        raise RuntimeError(
            f"Could not locate model object inside vLLM engine; "
            f"vLLM version may have changed the API path. ({e})"
        )

    print(f"[arrol-patch] INSTALL apply_arrol_patch: model={type(model).__name__} "
          f"l_detect={l_detect}", flush=True)

    state.patched_model_obj = model
    state.original_compute_logits = model.compute_logits

    def _patched_compute_logits(self, hidden_states, sampling_metadata):
        """Capture hidden states at L_detect crossings, then delegate to original."""
        # sampling_metadata carries per-request position info; depending on
        # vLLM version this is sampling_metadata.seq_groups or .selected_token_indices.
        # We use the generic approach: each row in hidden_states corresponds to
        # one in-flight sequence. The scheduler tracks the position of each
        # sequence; we infer the "did this row hit L_detect?" via the request
        # state below in _patched_engine_step. Here we just stash hidden_states.
        try:
            state.hidden_state_buffer["_pending_hidden_states"] = hidden_states.detach()
        except Exception:
            # Defensive: if buffering fails, do not break generation
            pass
        return state.original_compute_logits(hidden_states, sampling_metadata)

    # Bind the patched function as a method on the model instance
    import types
    model.compute_logits = types.MethodType(_patched_compute_logits, model)

    # ---- 2. Patch engine.step() to act on captured hidden states ------------
    engine = inference_engine.llm_engine
    state.patched_engine_class = type(engine)
    state.original_step = type(engine).step

    # Step counter for diagnostics
    state._step_calls = 0  # type: ignore[attr-defined]

    def _patched_engine_step(self_engine):
        """Run original step, then evaluate L_detect crossings + abort pruned."""
        state._step_calls = getattr(state, "_step_calls", 0) + 1  # type: ignore[attr-defined]
        if state._step_calls in (1, 5, 50, 200):  # quiet but informative
            print(f"[arrol-patch] STEP #{state._step_calls} "
                  f"scheduler_type={type(self_engine.scheduler).__name__} "
                  f"is_list={isinstance(self_engine.scheduler, list)}",
                  flush=True)
        # Snapshot which requests are at exactly L_detect BEFORE the step's
        # forward (so we can match by request_id after the step completes).
        # Prefer scheduler.running for v0; .running_requests for v1.
        before_step_positions: dict = {}
        try:
            scheduler = self_engine.scheduler[0] if isinstance(self_engine.scheduler, list) else self_engine.scheduler
            if hasattr(scheduler, "running_requests"):
                # v1 engine
                for req in scheduler.running_requests.values():
                    before_step_positions[req.request_id] = req.num_computed_tokens
            elif hasattr(scheduler, "running"):
                # v0 engine
                for seq_group in scheduler.running:
                    for seq in seq_group.get_seqs():
                        before_step_positions[seq_group.request_id] = seq.get_len()
            else:
                if state._step_calls == 1:
                    print(f"[arrol-patch] WARN no .running or .running_requests on "
                          f"scheduler type={type(scheduler).__name__}", flush=True)
        except Exception as _scheduler_err:
            if state._step_calls in (1, 2):
                print(f"[arrol-patch] WARN scheduler introspection raised: "
                      f"{_scheduler_err!r}", flush=True)
            return state.original_step(self_engine)

        # Run the original step
        outputs = state.original_step(self_engine)

        # Identify L_detect crossings: requests whose pre-step position was
        # < L_detect AND post-step position is >= L_detect.
        crossings: list[str] = []
        try:
            scheduler = self_engine.scheduler[0] if isinstance(self_engine.scheduler, list) else self_engine.scheduler
            if hasattr(scheduler, "running_requests"):
                for req in scheduler.running_requests.values():
                    pre = before_step_positions.get(req.request_id, 0)
                    post = req.num_computed_tokens
                    if pre < state.l_detect <= post:
                        crossings.append(req.request_id)
            elif hasattr(scheduler, "running"):
                for seq_group in scheduler.running:
                    pre = before_step_positions.get(seq_group.request_id, 0)
                    for seq in seq_group.get_seqs():
                        post = seq.get_len()
                        if pre < state.l_detect <= post:
                            crossings.append(seq_group.request_id)
                            break
        except Exception:
            pass

        # For each crossing, fire callback and abort if pruned.
        pending = state.hidden_state_buffer.get("_pending_hidden_states")
        if pending is not None and crossings:
            # Match crossings to rows in `pending`. vLLM's batch-row order
            # follows the scheduler's running list, so we match by enumeration.
            # NOTE: this is a heuristic matching; precise per-request mapping
            # would require deeper integration with sampling_metadata. For v1
            # of the patch we use the first-crossing-row heuristic and
            # validate per-request alignment in M3.f.5 smoke.
            for i, request_id in enumerate(crossings):
                if i >= pending.shape[0]:
                    break
                hidden = pending[i]
                try:
                    keep, p_keep = state.callback(request_id, hidden)
                    state.propensity_log[request_id] = p_keep
                    state.n_seen += 1
                    if keep:
                        state.n_kept += 1
                    else:
                        state.n_pruned += 1
                        # Request-pool removal: paper §4.3
                        try:
                            self_engine.abort_request(request_id)
                        except Exception as _e:
                            # If abort fails, the rollout continues — log loss
                            print(f"[arrol-patch] WARN abort failed for {request_id}: {_e}")
                except Exception as _cb_err:
                    state.n_callback_failed += 1
                    print(f"[arrol-patch] WARN callback failed: {_cb_err}")
            # Clear buffer for next step
            state.hidden_state_buffer.pop("_pending_hidden_states", None)

        return outputs

    # Replace step on the class (vLLM step is an instance method dispatched via class)
    setattr(state.patched_engine_class, "step", _patched_engine_step)

    _PATCH_STATE = state


def deactivate_arrol_patch() -> Optional[dict]:
    """Restore originals and return diagnostic counters.

    Returns:
        Dict of counters (n_pruned, n_kept, n_seen, prune_rate) or None if
        no patch was active.
    """
    global _PATCH_STATE
    if _PATCH_STATE is None:
        return None

    state = _PATCH_STATE
    _step_calls = getattr(state, "_step_calls", 0)
    diagnostics = {
        "arrol_faithful/n_pruned": state.n_pruned,
        "arrol_faithful/n_kept": state.n_kept,
        "arrol_faithful/n_seen": state.n_seen,
        "arrol_faithful/n_callback_failed": state.n_callback_failed,
        "arrol_faithful/n_step_calls": _step_calls,
        "arrol_faithful/prune_rate": (
            state.n_pruned / state.n_seen if state.n_seen > 0 else 0.0
        ),
        "arrol_faithful/propensities": dict(state.propensity_log),
    }
    print(f"[arrol-patch] DEACTIVATE n_step_calls={_step_calls} n_seen={state.n_seen} "
          f"n_pruned={state.n_pruned} n_kept={state.n_kept} "
          f"n_callback_failed={state.n_callback_failed}", flush=True)

    # Restore model.compute_logits
    try:
        if state.patched_model_obj is not None and state.original_compute_logits is not None:
            state.patched_model_obj.compute_logits = state.original_compute_logits
    except Exception as _e:
        print(f"[arrol-patch] WARN restore compute_logits failed: {_e}")

    # Restore engine class step()
    try:
        if state.patched_engine_class is not None and state.original_step is not None:
            setattr(state.patched_engine_class, "step", state.original_step)
    except Exception as _e:
        print(f"[arrol-patch] WARN restore step failed: {_e}")

    _PATCH_STATE = None
    return diagnostics


def get_propensity_log() -> dict:
    """Return the per-request keep-propensities recorded during the active patch.

    Used downstream by SNIPS correction. Empty dict if patch not active.
    """
    if _PATCH_STATE is None:
        return {}
    return dict(_PATCH_STATE.propensity_log)
