"""DUET surrogate ŝ_q loader (paper §5.1).

Loads a pre-fit linear ridge model that maps a single prompt feature
``f(q)`` (length-normalized base-model log-probability) to a positive
surrogate ``ŝ_q`` for the within-prompt gradient-trace standard deviation
σ_q. The fit itself is done offline by ``scripts/duet_fit_surrogate.py``
from existing GRPO calibration dumps + ``prompt_logprobs.parquet``.

Schema (``outputs/duet/ridge_weights.json``):

    {
        "feature":     "prompt_logprob_mean",
        "intercept":   <float>,        # α0 in paper
        "slope":       <float>,        # α1 in paper
        "sigma_min":   <float>,        # max(α0+α1·f, σ_min) clip floor
        "fit_metadata": {
            "n_train_prompts": <int>,
            "chi2_temporal":   <float>,
            "v_ratio_clipped": <float>,
            "fit_timestamp":   "<iso>",
            "source_dump":     "<path>"
        }
    }

If the file is missing or malformed the loader raises ``FileNotFoundError``
or ``DuetSurrogateError`` — DUET v1 explicitly requires the ridge to be
fit; the constant-σ̂ fallback was rejected during planning to avoid a
silent degradation to "uniform allocation under smaller B".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DuetSurrogateError(RuntimeError):
    """Raised when ridge weights are present but malformed."""


@dataclass(frozen=True)
class DuetSurrogate:
    intercept: float
    slope: float
    sigma_min: float
    feature: str = "prompt_logprob_mean"
    fit_metadata: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "DuetSurrogate":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"DUET ridge weights not found at {p}. "
                f"Run scripts/duet_fit_surrogate.py to produce it."
            )
        try:
            payload = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise DuetSurrogateError(f"weights file {p} is not valid JSON: {e}") from e
        for key in ("intercept", "slope", "sigma_min"):
            if key not in payload:
                raise DuetSurrogateError(
                    f"weights file {p} missing required key {key!r}"
                )
        return cls(
            intercept=float(payload["intercept"]),
            slope=float(payload["slope"]),
            sigma_min=float(payload["sigma_min"]),
            feature=str(payload.get("feature", "prompt_logprob_mean")),
            fit_metadata=payload.get("fit_metadata"),
        )

    def predict(self, feature_value: float) -> float:
        """Return ŝ_q = max(σ_min, α0 + α1·f(q)) for a single prompt."""
        raw = self.intercept + self.slope * float(feature_value)
        return max(self.sigma_min, raw)
