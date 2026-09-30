"""Parity: the V1 batch-level DUET stop processor must make exactly the same
decisions as the V0 per-request DuetStopProcessor the paper numbers came from.

Both see identical logits and token streams per request. The V1 side runs in a
simulated engine with requests joining/leaving, slot swaps and one-way moves,
preemption + re-admission, and repeated evaluation of the same position
(chunked recompute). For every request we compare the token at which it was
stopped (or its natural end) and the did_abort / saw_marker / eps_kept flags.

Run: python -m pytest tests/test_duet_v1_logits_processor.py -q
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duet import duet_v1_logits_processor as v1  # noqa: E402
from duet.duet_logits_processor import DuetStopProcessor, SIGNAL_MODES  # noqa: E402
from duet.duet_marker_detector import make_detector  # noqa: E402

BatchUpdate, MoveDirectionality = v1.BatchUpdate, v1.MoveDirectionality

# ---- toy vocabulary: decoding "\boxed{" "7" "}" yields a math marker --------
VOCAB = ["<eos>", "\\boxed{", "7", "}", "\n\n", " so", " x", " =", " 4", " wait",
         " let", " me", " check", " hmm", " the", " answer", " is", ".", " then",
         " we", " get", " step", " 1", " 2", " 3", " and", " but", " again", " ok",
         " thus", " done", " yes"]
EOS = 0
V = len(VOCAB)


class ToyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(VOCAB[i] for i in ids if not (skip_special_tokens and i == EOS))


DETECTOR = make_detector(ToyTokenizer(), domain="math")
MAX_LEN = 220


def token_stream(r: int) -> list[int]:
    """Scripted continuation for request r (independent of the logits)."""
    g = random.Random(1000 + r)
    n_nat = g.randint(40, MAX_LEN)            # natural EOS position
    toks = [g.randrange(4, V) for _ in range(n_nat)]
    if g.random() < 0.6:                      # 60% emit a boxed answer somewhere
        pos = g.randint(10, max(11, n_nat - 5))
        toks[pos:pos + 3] = [1, 2, 3]
        toks = toks[:n_nat]
    return toks + [EOS]


def logits_at(r: int, n: int) -> torch.Tensor:
    """Deterministic logits for request r at position n; sometimes very peaked."""
    g = torch.Generator().manual_seed(r * 100_003 + n)
    x = torch.randn(V, generator=g) * 1.5
    # confident stretches come in runs (so hysteresis_k > 1 can fire), with
    # occasional dips inside a run (so the counter's reset path is exercised)
    if ((n // 6) + r) % 2 == 0 and (n + r) % 5 != 0:
        x[int(torch.randint(1, V, (1,), generator=g))] += 9.0
    return x


def req_args(r: int) -> dict:
    g = random.Random(r)
    mode = SIGNAL_MODES[r % len(SIGNAL_MODES)]
    thr = {"entropy": g.uniform(0.3, 2.0), "mean_logprob": g.uniform(-2.0, -0.2)}.get(
        mode, g.uniform(0.3, 0.9))
    return dict(uid=f"u{r}", eos_token_id=EOS, threshold=thr, signal_mode=mode,
                eps_len=0.1, min_tokens=g.choice([8, 20]), top_k=g.choice([2, 3, 5]),
                hysteresis_k=g.choice([1, 3, 5]),
                marker_domain=None if r % 7 == 0 else "math",   # some legacy-path rows
                k1=30, k2=g.choice([60, 90]), grace_window=15, abort_eps=0.3,
                rng_seed=v1.stable_seed("test", r))


def is_forced_eos(row: torch.Tensor) -> bool:
    return row[EOS].item() == 0.0 and bool(torch.isinf(row[1:]).all())


def run_v0(r: int) -> dict:
    a = req_args(r)
    lp = DuetStopProcessor(
        eos_token_id=EOS, threshold=a["threshold"], signal_mode=a["signal_mode"],
        eps_len=a["eps_len"], min_tokens=a["min_tokens"], rng_seed=a["rng_seed"],
        top_k=a["top_k"], hysteresis_k=a["hysteresis_k"],
        marker_detector=DETECTOR if a["marker_domain"] else None,
        k1=a["k1"], k2=a["k2"], grace_window=a["grace_window"], abort_eps=a["abort_eps"])
    stream, out = token_stream(r), []
    while True:
        row = lp([], list(out), logits_at(r, len(out)).clone())
        if is_forced_eos(row):
            end = ("forced", len(out))
            break
        tok = stream[len(out)]
        out.append(tok)
        if tok == EOS:
            end = ("natural", len(out))
            break
    return {"end": end, "did_abort": lp.did_abort, "saw_marker": lp.saw_marker,
            "eps_kept": lp.eps_kept}


def run_v1(requests: list[int], seed: int, slots: int = 6) -> dict[int, dict]:
    proc = v1.DuetV1StopProcessor(None, torch.device("cpu"), False)
    proc._detectors["math"] = DETECTOR
    g = random.Random(seed)
    pending = list(requests)
    batch: list = []                     # slot -> request id or None
    outputs = {r: [] for r in requests}
    streams = {r: token_stream(r) for r in requests}
    results: dict[int, dict] = {}
    preempted: list[int] = []
    removed: list[int] = []

    def params(r):
        return SimpleNamespace(extra_args=v1.build_extra_args(**req_args(r)))

    while len(results) < len(requests):
        added, moved = [], []
        # admit (new or preempted) requests into free/new slots
        for queue in (preempted, pending):
            while queue and (None in batch or len(batch) < slots) and g.random() < 0.8:
                r = queue.pop(0)
                slot = batch.index(None) if None in batch else len(batch)
                if slot == len(batch):
                    batch.append(None)
                batch[slot] = r
                added.append((slot, params(r), [], outputs[r]))
        # random condense moves / swaps
        live = [i for i, r in enumerate(batch) if r is not None]
        if len(batch) > 1 and g.random() < 0.3:
            i, j = g.sample(range(len(batch)), 2)
            if batch[i] is not None and batch[j] is not None:
                batch[i], batch[j] = batch[j], batch[i]
                moved.append((i, j, MoveDirectionality.SWAP))
            elif batch[i] is not None and batch[j] is None:
                batch[j], batch[i] = batch[i], None
                moved.append((i, j, MoveDirectionality.UNIDIRECTIONAL))
        proc.update_state(BatchUpdate(batch_size=len(batch), removed=removed, added=added,
                                      moved=moved) if (removed or added or moved) else None)
        removed = []
        if not any(r is not None for r in batch):
            continue
        logits = torch.stack([logits_at(r, len(outputs[r])) if r is not None
                              else torch.zeros(V) for r in batch])
        logits = proc.apply(logits)
        if g.random() < 0.2:           # chunked recompute: same positions evaluated again
            again = torch.stack([logits_at(r, len(outputs[r])) if r is not None
                                 else torch.zeros(V) for r in batch])
            logits = proc.apply(again)
        for slot, r in enumerate(batch):
            if r is None:
                continue
            done = None
            if is_forced_eos(logits[slot]):
                done = ("forced", len(outputs[r]))
            else:
                tok = streams[r][len(outputs[r])]
                outputs[r].append(tok)
                if tok == EOS:
                    done = ("natural", len(outputs[r]))
            if done:
                f = v1.pop_flags([f"u{r}"])[0]
                results[r] = {"end": done, "did_abort": f["did_abort"],
                              "saw_marker": f["saw_marker"], "eps_kept": f["eps_kept"]}
                batch[slot] = None
                removed.append(slot)
            elif g.random() < 0.03:     # preempt: leave the batch, come back later
                batch[slot] = None
                removed.append(slot)
                preempted.append(r)
    return results


@pytest.mark.parametrize("seed", range(6))
def test_v1_matches_v0(seed):
    requests = list(range(seed * 50, seed * 50 + 50))
    got = run_v1(requests, seed)
    for r in requests:
        assert got[r] == run_v0(r), f"request {r} ({req_args(r)['signal_mode']})"


def test_paths_are_exercised():
    """Guard against a vacuous pass: every outcome type must occur."""
    outs = [run_v0(r) for r in range(300)]
    kinds = {
        "abort": any(o["did_abort"] for o in outs),
        "eps_kept": any(o["eps_kept"] for o in outs),
        "marker_then_signal_stop": any(o["saw_marker"] and o["end"][0] == "forced"
                                       and not o["did_abort"] for o in outs),
        "legacy_signal_stop": any(not o["saw_marker"] and o["end"][0] == "forced"
                                  and not o["did_abort"] for o in outs),
        "natural_end": any(o["end"][0] == "natural" for o in outs),
    }
    assert all(kinds.values()), kinds


def test_requests_without_duet_args_untouched():
    proc = v1.DuetV1StopProcessor(None, torch.device("cpu"), False)
    proc.update_state(BatchUpdate(batch_size=1, removed=[], moved=[],
                                  added=[(0, SimpleNamespace(extra_args=None), [], [5] * 500)]))
    x = torch.randn(1, V)
    assert torch.equal(proc.apply(x.clone()), x)


def test_stable_seed_is_process_independent():
    assert v1.stable_seed("duet_eps_keep", 0, 12, 3) == v1.stable_seed("duet_eps_keep", 0, 12, 3)
    assert v1.stable_seed("a", 1) != v1.stable_seed("a", 2)
    assert 0 <= v1.stable_seed("x") < 2 ** 31
    assert not math.isnan(v1.stable_seed("y"))


# ---- integration: slots managed by vLLM's real InputBatch --------------------
def _vllm_input_batch_available():
    try:
        from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch  # noqa: F401
        from vllm.v1.sample.logits_processor import LogitsProcessors  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _vllm_input_batch_available(), reason="vLLM not installed")
@pytest.mark.parametrize("seed", range(3))
def test_v1_matches_v0_with_real_input_batch(seed):
    """Same parity check, but batch slots, removals, condense moves and the
    BatchUpdates handed to update_state() all come from vLLM's own InputBatch."""
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    proc = v1.DuetV1StopProcessor(None, torch.device("cpu"), False)
    proc._detectors["math"] = DETECTOR
    ib = InputBatch(max_num_reqs=6, max_model_len=512, max_num_batched_tokens=512,
                    device=torch.device("cpu"), pin_memory=False, vocab_size=V,
                    block_sizes=[16], kernel_block_sizes=[16],
                    logitsprocs=LogitsProcessors([proc]),
                    logitsprocs_need_output_token_ids=True)
    g = random.Random(seed)
    requests = list(range(500 + seed * 40, 540 + seed * 40))
    pending, preempted = list(requests), []
    outputs = {r: [] for r in requests}
    streams = {r: token_stream(r) for r in requests}
    results: dict[int, dict] = {}

    def state(r):
        return CachedRequestState(
            req_id=f"r{r}", prompt_token_ids=[4, 5, 6], mm_features=[],
            sampling_params=SamplingParams(extra_args=v1.build_extra_args(**req_args(r))),
            generator=None, block_ids=([],), num_computed_tokens=0,
            output_token_ids=outputs[r])

    while len(results) < len(requests):
        for queue in (preempted, pending):
            while queue and ib.num_reqs < ib.max_num_reqs and g.random() < 0.8:
                ib.add_request(state(queue.pop(0)))
        ib.condense()
        ib.refresh_metadata()          # -> proc.update_state(BatchUpdate from vLLM)
        if ib.num_reqs == 0:
            continue
        rids = list(ib.req_ids[:ib.num_reqs])
        logits = torch.stack([logits_at(int(rid[1:]), len(outputs[int(rid[1:])])) for rid in rids])
        logits = proc.apply(logits)
        for slot, rid in enumerate(rids):
            r = int(rid[1:])
            done = None
            if is_forced_eos(logits[slot]):
                done = ("forced", len(outputs[r]))
            else:
                tok = streams[r][len(outputs[r])]
                outputs[r].append(tok)
                if tok == EOS:
                    done = ("natural", len(outputs[r]))
            if done:
                f = v1.pop_flags([f"u{r}"])[0]
                results[r] = {"end": done, "did_abort": f["did_abort"],
                              "saw_marker": f["saw_marker"], "eps_kept": f["eps_kept"]}
                ib.remove_request(rid)
            elif g.random() < 0.03:
                ib.remove_request(rid)
                preempted.append(r)
    for r in requests:
        assert results[r] == run_v0(r), f"request {r}"
