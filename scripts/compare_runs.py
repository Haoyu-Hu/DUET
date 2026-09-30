#!/usr/bin/env python
"""Side-by-side summary of finished runs from their TensorBoard logs.

    python scripts/compare_runs.py --run "GRPO=grpo_qwen3-1.7b-base_v1_s0" \
                                   --run "DUET 0.5=duet_qwen3-1.7b-base_b0p5_v1_s0"

A run is LABEL=TAG (looked up as experiments/*/duet_run_TAG/tensorboard, newest
date first) or LABEL=/path/to/tensorboard. The first run is the reference for
the speedup column. Reported, per run:
  * best val mean@N per benchmark over all validation checkpoints (the paper's
    headline metric), with the step it was reached at;
  * wall-clock = sum of timing_s/step over training steps (validation passes
    are timed separately by verl, so they are excluded), speedup vs run 1;
  * median generation time per step and mean response length;
  * DUET diagnostics when present (abort / marker / eps-kept rates, n_q range).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

VAL_RE = re.compile(r"^val-core/(?P<ds>[^/]+)/(?:acc|reward)/mean@(?P<n>\d+)$")
DUET_KEYS = ["duet/abort_rate", "duet/marker_rate", "duet/epsilon_kept_rate",
             "duet/n_q_mean", "duet/n_q_min", "duet/n_q_max"]


def resolve(spec: str) -> tuple[str, str]:
    label, _, target = spec.partition("=")
    if not target:
        label, target = spec, spec
    if os.path.isdir(target):
        return label, target
    hits = sorted(glob.glob(os.path.join("experiments", "*", f"duet_run_{target}", "tensorboard")))
    if not hits:
        sys.exit(f"no tensorboard dir for run {target!r} under experiments/*/duet_run_{target}/")
    return label, hits[-1]


def load(tb_dir: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for root, _, files in os.walk(tb_dir):
        if not any(f.startswith("events.out.tfevents") for f in files):
            continue
        acc = EventAccumulator(root, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            series[tag].extend((e.step, e.value) for e in acc.Scalars(tag))
    for tag in series:
        series[tag].sort()
    return series


def summarize(series) -> dict:
    out: dict = {"val": {}}
    for tag, pts in series.items():
        m = VAL_RE.match(tag)
        if m and pts:
            step, best = max(pts, key=lambda p: (p[1], -p[0]))
            out["val"][m["ds"]] = (100.0 * best, step, int(m["n"]))
    steps = [v for s, v in series.get("timing_s/step", []) if s > 0]
    out["wall_s"] = sum(steps) if steps else None
    out["n_steps"] = len(steps)
    gen = [v for _, v in series.get("timing_s/gen", [])]
    out["gen_med"] = statistics.median(gen) if gen else None
    rl = [v for _, v in series.get("response_length/mean", [])]
    out["resp_len"] = statistics.fmean(rl) if rl else None
    out["duet"] = {k: statistics.fmean(v for _, v in series[k]) for k in DUET_KEYS
                   if series.get(k)}
    if series.get("duet/n_q_max"):
        out["duet"]["duet/n_q_max"] = max(v for _, v in series["duet/n_q_max"])
    if series.get("duet/n_q_min"):
        out["duet"]["duet/n_q_min"] = min(v for _, v in series["duet/n_q_min"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="LABEL=TAG or LABEL=TB_DIR")
    args = ap.parse_args()
    runs = [(label, path, summarize(load(path))) for label, path in map(resolve, args.run)]

    datasets = sorted({ds for _, _, s in runs for ds in s["val"]})
    w = max(12, max(len(lbl) for lbl, _, _ in runs) + 2)
    print("best val mean@N (%) per benchmark  [step reached]")
    print("run".ljust(w) + "".join(ds[:18].rjust(20) for ds in datasets))
    for label, _, s in runs:
        cells = []
        for ds in datasets:
            if ds in s["val"]:
                acc, step, _ = s["val"][ds]
                cells.append(f"{acc:6.1f} [{step:>4}]".rjust(20))
            else:
                cells.append("—".rjust(20))
        print(label.ljust(w) + "".join(cells))

    print("\nwall-clock (sum of timing_s/step; validation excluded)")
    ref = runs[0][2]["wall_s"]
    print("run".ljust(w) + "steps".rjust(8) + "wall (min)".rjust(12) + "speedup".rjust(10)
          + "gen/step (s)".rjust(14) + "resp len".rjust(10))
    for label, _, s in runs:
        wall = s["wall_s"]
        sp = f"{ref / wall:.2f}x" if (ref and wall) else "—"
        print(label.ljust(w) + f"{s['n_steps']:>8}"
              + (f"{wall / 60:12.1f}" if wall else "—".rjust(12)) + sp.rjust(10)
              + (f"{s['gen_med']:14.1f}" if s["gen_med"] else "—".rjust(14))
              + (f"{s['resp_len']:10.0f}" if s["resp_len"] else "—".rjust(10)))

    duet_runs = [(lbl, s["duet"]) for lbl, _, s in runs if s["duet"]]
    if duet_runs:
        print("\nDUET diagnostics (run means; n_q range over run)")
        for label, d in duet_runs:
            print(f"  {label}: " + ", ".join(f"{k.split('/')[1]}={v:.3g}" for k, v in d.items()))
    print("\nsources:\n" + "\n".join(f"  {lbl}: {p}" for lbl, p, _ in runs))


if __name__ == "__main__":
    main()
