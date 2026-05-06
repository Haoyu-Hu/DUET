"""Model registry and HuggingFace resolution for SIRL experiments.

Six supported aliases:

Base:
  qwen3-1.7b-base, qwen3-4b-base, qwen3-8b-base

Instruct:
  qwen3-4b-instruct, qwen3-8b-instruct, llama3.1-8b-instruct

Callers can pass an alias, a `vendor/repo` HF id, or a local path.
Use `resolve_model_reference()` to get a `ResolvedModelReference` that
tells you whether the weights are already on disk, and
`download_model_snapshot()` to fetch them from HF when missing.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CHECKPOINT_ROOT = PROJECT_ROOT / "model_checkpoints"
RUNTIME_CACHE_ROOT = MODEL_CHECKPOINT_ROOT / "runtime_cache"
MANIFEST_FILENAME = "duet_model_manifest.json"

DEFAULT_MODEL_ALIAS = "qwen3-1.7b-base"


class ModelStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenSourceModelSpec:
    alias: str
    repo_id: str
    storage_subdir: str
    description: str
    revision: str | None = None

    def local_path(self, checkpoint_root: Path | None = None) -> Path:
        root = checkpoint_root or MODEL_CHECKPOINT_ROOT
        return root / self.storage_subdir


@dataclass(frozen=True)
class ResolvedModelReference:
    requested: str
    local_path: Path
    exists: bool
    managed: bool
    alias: str | None = None
    repo_id: str | None = None
    revision: str | None = None

    @property
    def manifest_path(self) -> Path:
        return self.local_path / MANIFEST_FILENAME

    @property
    def download_command(self) -> str:
        target = self.alias or self.repo_id or self.requested
        return f"python setup/download_models.py --model {shlex.quote(target)}"


MODEL_SPECS: dict[str, OpenSourceModelSpec] = {
    "qwen3-1.7b-base": OpenSourceModelSpec(
        alias="qwen3-1.7b-base",
        repo_id="Qwen/Qwen3-1.7B-Base",
        storage_subdir="qwen3-1.7b-base",
        description="Qwen3 1.7B base model (default SIRL actor).",
    ),
    "qwen3-4b-base": OpenSourceModelSpec(
        alias="qwen3-4b-base",
        repo_id="Qwen/Qwen3-4B-Base",
        storage_subdir="qwen3-4b-base",
        description="Qwen3 4B base model.",
    ),
    "qwen3-8b-base": OpenSourceModelSpec(
        alias="qwen3-8b-base",
        repo_id="Qwen/Qwen3-8B-Base",
        storage_subdir="qwen3-8b-base",
        description="Qwen3 8B base model (matches the tinker SIRL target).",
    ),
    "qwen3-14b-base": OpenSourceModelSpec(
        alias="qwen3-14b-base",
        repo_id="Qwen/Qwen3-14B-Base",
        storage_subdir="qwen3-14b-base",
        description="Qwen3 14B base model (LoRA-only for InPo appendix §A.2).",
    ),
    "llama3.2-3b-instruct": OpenSourceModelSpec(
        alias="llama3.2-3b-instruct",
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        storage_subdir="llama3.2-3b-instruct",
        description="Llama 3.2 3B instruct-tuned (gated). InPo cross-family appendix §B.",
    ),
    "llama3.2-3b": OpenSourceModelSpec(
        alias="llama3.2-3b",
        repo_id="meta-llama/Llama-3.2-3B",
        storage_subdir="llama3.2-3b",
        description="Llama 3.2 3B base (gated). InPo cross-family appendix §B — base preferred over Instruct for RLVR stability.",
    ),
    "qwen3-4b-instruct": OpenSourceModelSpec(
        alias="qwen3-4b-instruct",
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        storage_subdir="qwen3-4b-instruct",
        description="Qwen3 4B instruct-tuned model (2507 release).",
    ),
    "qwen3-8b-instruct": OpenSourceModelSpec(
        alias="qwen3-8b-instruct",
        repo_id="Qwen/Qwen3-8B-Instruct-2507",
        storage_subdir="qwen3-8b-instruct",
        description="Qwen3 8B instruct-tuned model (2507 release).",
    ),
    "llama3.1-8b-instruct": OpenSourceModelSpec(
        alias="llama3.1-8b-instruct",
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        storage_subdir="llama3.1-8b-instruct",
        description="Llama 3.1 8B instruct-tuned model.",
    ),
}


def ensure_model_store_dirs(checkpoint_root: Path | None = None) -> dict[str, Path]:
    root = checkpoint_root or MODEL_CHECKPOINT_ROOT
    runtime_root = root / "runtime_cache"
    paths = {
        "checkpoint_root": root,
        "runtime_root": runtime_root,
        "hf_home": runtime_root / "hf_home",
        "hf_hub": runtime_root / "hf_hub",
        "transformers": runtime_root / "transformers",
        "datasets": runtime_root / "datasets",
        "hf_modules": runtime_root / "hf_modules",
        "torch": runtime_root / "torch",
        "triton": runtime_root / "triton",
        "xdg": runtime_root / "xdg",
        "cuda": runtime_root / "cuda",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def hf_env_overrides(checkpoint_root: Path | None = None) -> dict[str, str]:
    paths = ensure_model_store_dirs(checkpoint_root)
    return {
        "HF_HOME": str(paths["hf_home"]),
        "HF_HUB_CACHE": str(paths["hf_hub"]),
        "HUGGINGFACE_HUB_CACHE": str(paths["hf_hub"]),
        "TRANSFORMERS_CACHE": str(paths["transformers"]),
        "HF_DATASETS_CACHE": str(paths["datasets"]),
        "HF_MODULES_CACHE": str(paths["hf_modules"]),
        "TORCH_HOME": str(paths["torch"]),
        "TRITON_CACHE_DIR": str(paths["triton"]),
        "XDG_CACHE_HOME": str(paths["xdg"]),
        "CUDA_CACHE_PATH": str(paths["cuda"]),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def shell_export_lines(checkpoint_root: Path | None = None) -> list[str]:
    exports = hf_env_overrides(checkpoint_root)
    ordered_keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_MODULES_CACHE",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
        "CUDA_CACHE_PATH",
        "TOKENIZERS_PARALLELISM",
        "HF_HUB_DISABLE_TELEMETRY",
    ]
    return [f"export {key}={shlex.quote(exports[key])}" for key in ordered_keys]


@contextmanager
def use_visible_hf_env(checkpoint_root: Path | None = None) -> Iterator[dict[str, str]]:
    overrides = hf_env_overrides(checkpoint_root)
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield overrides
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _sanitize_repo_id(repo_id: str) -> str:
    safe = repo_id.strip().lower().replace("/", "--")
    safe = re.sub(r"[^a-z0-9._-]+", "-", safe)
    safe = safe.strip("-")
    if not safe:
        raise ModelStoreError(f"Could not derive a storage name from repo id '{repo_id}'.")
    return safe


def _looks_like_local_path(value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    if candidate.startswith(("/", "./", "../", "~")):
        return True
    if candidate.startswith("model/") or candidate.startswith("models/"):
        return True
    if "\\" in candidate:
        return True
    return candidate.count("/") > 1


def _normalize_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def list_model_specs() -> list[OpenSourceModelSpec]:
    return sorted(MODEL_SPECS.values(), key=lambda item: item.alias)


def get_model_spec(model_name: str) -> OpenSourceModelSpec | None:
    if model_name in MODEL_SPECS:
        return MODEL_SPECS[model_name]
    for spec in MODEL_SPECS.values():
        if spec.repo_id == model_name:
            return spec
    return None


def resolve_model_reference(
    model_name: str,
    *,
    checkpoint_root: Path | None = None,
    require_exists: bool = False,
) -> ResolvedModelReference:
    root = checkpoint_root or MODEL_CHECKPOINT_ROOT
    requested = model_name.strip()
    if not requested:
        raise ModelStoreError("Model reference must not be empty.")

    if _looks_like_local_path(requested):
        local_path = _normalize_path(requested)
        resolved = ResolvedModelReference(
            requested=requested,
            local_path=local_path,
            exists=local_path.exists(),
            managed=str(local_path).startswith(str(root.resolve())),
        )
    else:
        spec = get_model_spec(requested)
        if spec is not None:
            local_path = spec.local_path(root)
            resolved = ResolvedModelReference(
                requested=requested,
                local_path=local_path,
                exists=local_path.exists(),
                managed=True,
                alias=spec.alias,
                repo_id=spec.repo_id,
                revision=spec.revision,
            )
        elif "/" in requested:
            local_path = root / _sanitize_repo_id(requested)
            resolved = ResolvedModelReference(
                requested=requested,
                local_path=local_path,
                exists=local_path.exists(),
                managed=True,
                repo_id=requested,
            )
        else:
            known = ", ".join(spec.alias for spec in list_model_specs())
            raise ModelStoreError(
                f"Unknown model alias '{requested}'. Known aliases: {known}. "
                "You may also pass a Hugging Face repo id or a local checkpoint path."
            )

    if require_exists and not resolved.exists:
        raise ModelStoreError(
            f"Model checkpoint not found at {resolved.local_path}. "
            f"Download it with `{resolved.download_command}`."
        )
    return resolved


def _clear_directory_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def cleanup_runtime_cache(checkpoint_root: Path | None = None) -> None:
    paths = ensure_model_store_dirs(checkpoint_root)
    for key in ("hf_hub", "hf_modules", "torch", "triton", "xdg", "cuda"):
        _clear_directory_contents(paths[key])


def _write_manifest(
    resolved: ResolvedModelReference,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "requested": resolved.requested,
        "alias": resolved.alias,
        "repo_id": resolved.repo_id,
        "revision": resolved.revision,
        "local_path": str(resolved.local_path),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "allow_patterns": allow_patterns or [],
        "ignore_patterns": ignore_patterns or [],
    }
    resolved.local_path.mkdir(parents=True, exist_ok=True)
    resolved.manifest_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def download_model_snapshot(
    model_name: str,
    *,
    checkpoint_root: Path | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> ResolvedModelReference:
    resolved = resolve_model_reference(model_name, checkpoint_root=checkpoint_root, require_exists=False)
    if not resolved.repo_id:
        raise ModelStoreError(
            f"Cannot download '{model_name}' because it resolved to a local path "
            f"({resolved.local_path}) rather than a managed repository id."
        )

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise ModelStoreError(
            "Downloading checkpoints requires 'huggingface_hub'. Install requirements first."
        ) from exc

    ensure_model_store_dirs(checkpoint_root)
    with use_visible_hf_env(checkpoint_root):
        snapshot_download(
            repo_id=resolved.repo_id,
            revision=resolved.revision,
            local_dir=str(resolved.local_path),
            local_dir_use_symlinks=False,
            cache_dir=str((checkpoint_root or MODEL_CHECKPOINT_ROOT) / "runtime_cache" / "hf_hub"),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    hidden_cache = resolved.local_path / ".cache"
    if hidden_cache.exists():
        shutil.rmtree(hidden_cache, ignore_errors=True)
    _write_manifest(resolved, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)
    cleanup_runtime_cache(checkpoint_root)
    return resolve_model_reference(model_name, checkpoint_root=checkpoint_root, require_exists=True)


def ensure_model_available(
    model_name: str,
    *,
    checkpoint_root: Path | None = None,
    auto_download: bool = True,
) -> ResolvedModelReference:
    """Resolve a model and optionally download if missing. Used by launchers."""
    resolved = resolve_model_reference(model_name, checkpoint_root=checkpoint_root, require_exists=False)
    if resolved.exists:
        return resolved
    if not auto_download:
        raise ModelStoreError(
            f"Model '{model_name}' not found at {resolved.local_path}. "
            f"Run: {resolved.download_command}"
        )
    print(f"[model_store] {model_name} not found locally — downloading from {resolved.repo_id}...")
    return download_model_snapshot(model_name, checkpoint_root=checkpoint_root)
