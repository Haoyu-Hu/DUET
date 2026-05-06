"""SIRL (Self-Improvement Reinforcement Learning) package."""

from .model_store import (
    DEFAULT_MODEL_ALIAS,
    MODEL_SPECS,
    ModelStoreError,
    OpenSourceModelSpec,
    ResolvedModelReference,
    download_model_snapshot,
    ensure_model_available,
    ensure_model_store_dirs,
    hf_env_overrides,
    list_model_specs,
    resolve_model_reference,
    shell_export_lines,
)

__all__ = [
    "DEFAULT_MODEL_ALIAS",
    "MODEL_SPECS",
    "ModelStoreError",
    "OpenSourceModelSpec",
    "ResolvedModelReference",
    "download_model_snapshot",
    "ensure_model_available",
    "ensure_model_store_dirs",
    "hf_env_overrides",
    "list_model_specs",
    "resolve_model_reference",
    "shell_export_lines",
]
