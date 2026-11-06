from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    docker_bin: str
    image: str
    specs_dir: Path
    default_timeout_seconds: int
    max_timeout_seconds: int
    max_source_bytes: int
    memory: str
    cpus: str
    pids_limit: int
    tmpfs_size: str
    max_output_bytes: int
    project_lib: Path | None = None


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    return max(minimum, min(value, maximum))


def load_settings() -> Settings:
    repository_root = Path(__file__).resolve().parents[2]
    default_specs = repository_root / "specs"
    specs_dir = (
        Path(os.environ.get("LEAN_VERIFIER_SPECS_DIR", default_specs))
        .expanduser()
        .resolve()
    )
    project_lib_raw = os.environ.get("LEAN_VERIFIER_PROJECT_LIB")
    project_lib = (
        Path(project_lib_raw).expanduser().resolve() if project_lib_raw else None
    )

    return Settings(
        docker_bin=os.environ.get("LEAN_VERIFIER_DOCKER_BIN", "docker"),
        image=os.environ.get("LEAN_VERIFIER_IMAGE", "local-lean-verifier:4.32.0"),
        specs_dir=specs_dir,
        default_timeout_seconds=_int_env("LEAN_VERIFIER_TIMEOUT", 120, 5, 600),
        max_timeout_seconds=_int_env("LEAN_VERIFIER_MAX_TIMEOUT", 300, 5, 900),
        max_source_bytes=_int_env(
            "LEAN_VERIFIER_MAX_SOURCE_BYTES", 250_000, 1_000, 5_000_000
        ),
        # Default 20g (was 2g, then 14g): safe_verify loads the full Mathlib
        # environment and kernel-replays every declaration; a small 1-theorem
        # file needs ~10GB RSS, but large corpus files need ~17GB (measured:
        # 14.4GB anon + 2.4GB file RSS when the cgroup OOM-killer fired at
        # 14g). 14g OOM-kills 67/74 corpus files. Override with
        # LEAN_VERIFIER_MEMORY. Keep sweep concurrency at 1 on a 27GB box.
        memory=os.environ.get("LEAN_VERIFIER_MEMORY", "20g"),
        cpus=os.environ.get("LEAN_VERIFIER_CPUS", "2"),
        pids_limit=_int_env("LEAN_VERIFIER_PIDS", 256, 32, 4096),
        tmpfs_size=os.environ.get("LEAN_VERIFIER_TMPFS_SIZE", "768m"),
        max_output_bytes=_int_env(
            "LEAN_VERIFIER_MAX_OUTPUT_BYTES", 2_000_000, 10_000, 20_000_000
        ),
        project_lib=project_lib,
    )
