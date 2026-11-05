from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from .config import Settings, load_settings


class VerifierInfrastructureError(RuntimeError):
    """The Docker verifier could not be started or returned invalid output."""


def clamp_timeout(requested: int | None, settings: Settings) -> int:
    if requested is None:
        return settings.default_timeout_seconds
    return max(5, min(int(requested), settings.max_timeout_seconds))


def validate_size(label: str, source: str, settings: Settings) -> None:
    size = len(source.encode("utf-8"))
    if size > settings.max_source_bytes:
        raise ValueError(
            f"{label} is {size} bytes; the configured maximum is "
            f"{settings.max_source_bytes} bytes"
        )
    if "\x00" in source:
        raise ValueError(f"{label} contains a NUL byte")


def resolve_spec(specs_dir: Path, spec_name: str) -> Path:
    """Resolve a spec underneath specs_dir without allowing path traversal."""
    if not spec_name or Path(spec_name).is_absolute():
        raise ValueError("spec_name must be a relative path")

    candidate = (specs_dir / spec_name).resolve()
    try:
        candidate.relative_to(specs_dir.resolve())
    except ValueError as exc:
        raise ValueError("spec_name escapes the configured specs directory") from exc

    if candidate.suffix != ".lean":
        raise ValueError("spec_name must end in .lean")
    if not candidate.is_file():
        raise FileNotFoundError(f"Unknown specification: {spec_name}")
    return candidate


def list_specs(settings: Settings | None = None) -> list[str]:
    settings = settings or load_settings()
    if not settings.specs_dir.is_dir():
        return []
    return sorted(
        str(path.relative_to(settings.specs_dir))
        for path in settings.specs_dir.rglob("*.lean")
        if path.is_file()
    )


class DockerVerifier:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()

    def _docker_command(self, container_name: str) -> list[str]:
        s = self.settings
        return [
            s.docker_bin,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={s.tmpfs_size}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            s.memory,
            "--memory-swap",
            s.memory,
            "--cpus",
            s.cpus,
            "--pids-limit",
            str(s.pids_limit),
            "--env",
            "HOME=/tmp/home",
            "--env",
            "TMPDIR=/tmp",
            "-i",
            s.image,
        ]

    async def _force_remove(self, container_name: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                self.settings.docker_bin,
                "rm",
                "-f",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=10)
        except (FileNotFoundError, TimeoutError):
            pass

    async def run(self, payload: dict[str, Any], timeout_seconds: int | None = None) -> dict[str, Any]:
        timeout = clamp_timeout(timeout_seconds, self.settings)
        payload = dict(payload)
        payload["timeout_seconds"] = timeout

        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        container_name = f"lean-verifier-{uuid.uuid4().hex[:20]}"
        command = self._docker_command(container_name)

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise VerifierInfrastructureError(
                f"Docker executable {self.settings.docker_bin!r} was not found"
            ) from exc

        try:
            # Small grace period covers Docker startup and shutdown. The worker applies
            # the requested timeout to Lean/SafeVerify themselves.
            stdout, stderr = await asyncio.wait_for(
                process.communicate(encoded), timeout=timeout + 20
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            await self._force_remove(container_name)
            return {
                "okay": False,
                "stage": "timeout",
                "error": f"Verification exceeded {timeout} seconds",
                "timed_out": True,
            }
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            await self._force_remove(container_name)
            raise

        if len(stdout) > self.settings.max_output_bytes:
            raise VerifierInfrastructureError(
                f"Verifier output exceeded {self.settings.max_output_bytes} bytes"
            )

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        try:
            result = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            detail = stderr_text[-4000:] or stdout_text[-4000:] or "no output"
            raise VerifierInfrastructureError(
                f"Verifier container returned invalid JSON (exit {process.returncode}): {detail}"
            ) from exc

        if not isinstance(result, dict):
            raise VerifierInfrastructureError("Verifier container returned a non-object JSON value")

        result.setdefault("container_exit_code", process.returncode)
        if process.returncode != 0:
            result["okay"] = False
            result.setdefault("stage", "infrastructure")
            result.setdefault(
                "error", f"Verifier container exited with status {process.returncode}"
            )
        if stderr_text:
            result.setdefault("container_stderr", stderr_text[-8000:])
        result.setdefault("image", self.settings.image)
        return result

    async def health(self, timeout_seconds: int = 30) -> dict[str, Any]:
        return await self.run({"mode": "health"}, timeout_seconds=timeout_seconds)

    async def verify(
        self,
        formal_statement: str,
        candidate: str,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        validate_size("formal_statement", formal_statement, self.settings)
        validate_size("candidate", candidate, self.settings)
        return await self.run(
            {
                "mode": "verify",
                "formal_statement": formal_statement,
                "candidate": candidate,
            },
            timeout_seconds=timeout_seconds,
        )
