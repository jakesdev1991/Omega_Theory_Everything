#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
omni_host_audit.py — Omni-Bridge FIRST-ACTION audit kit (directive §3,
§8, §9, §10, §66, §84).

RUN THIS ON THE GENTOO HOST (nvme0n1p7 system). It is strictly
READ-ONLY: no formatting, no mounting, no file modification. It collects
the storage/partition audit, the Hermes/Lucifer source discovery, and
the hardware/kernel inventory, then writes:

    reports/host-inventory.json           (host, kernel, hardware, fs)
    reports/existing-system-inventory.json (discovered components)
    reports/pre-migration-manifest.json    (paths + sizes + git state)
    reports/storage-audit.md               (p3 safety evidence report)

The p3 report does NOT decide to format anything. It assembles evidence
and a checklist; a human makes the §3 STOP/go decision.

Usage:
    python3 omni_host_audit.py                 # default roots
    python3 omni_host_audit.py --roots /home/jake /opt
    python3 omni_host_audit.py --out ./reports

Requires only Python 3.8+ (stdlib). Some commands are more informative
under sudo; the script degrades gracefully without it.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_ROOTS = [
    os.path.expanduser("~"),
    "/opt",
    "/srv",
    "/usr/local/src",
    "/var/lib",
    "/home",
]
SKIP_DIRS = {
    "proc",
    "sys",
    "dev",
    "run",
    "tmp",
    ".git/objects",
    "node_modules",
    "__pycache__",
    ".cache",
    "target",
    "build",
    "dist",
    ".venv",
    "site-packages",
}
SEARCH_NAMES = {
    "hermes",
    "lucifer",
    "omni",
    "omni-bridge",
    "bridge",
    "agents",
    "skills",
}
SEARCH_FILES = {
    "SOUL.md",
    "AGENTS.md",
    "pyproject.toml",
    "Cargo.toml",
    "CMakeLists.txt",
    "Makefile",
    "package.json",
}
MAX_HASH_BYTES = 64 * 1024 * 1024  # hash at most 64 MB per file
MAX_FILES_PER_ROOT = 50_000
TARGET_PARTITION = os.environ.get("OMNI_TARGET_PARTITION", "/dev/nvme0n1p3")


def run(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    """Run a command, capturing output; never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout[-20000:] if p.stdout else "",
            "stderr": p.stderr[-4000:] if p.stderr else "",
        }
    except FileNotFoundError:
        return {"cmd": cmd, "error": "not-found"}
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "error": "timeout"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"cmd": cmd, "error": str(exc)}


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Storage audit (§3)
# ---------------------------------------------------------------------------


def storage_audit() -> dict[str, Any]:
    out: dict[str, Any] = {"captured_at": now_iso()}
    out["lsblk"] = run(
        [
            "lsblk",
            "-e7",
            "-o",
            "NAME,PATH,SIZE,FSTYPE,FSVER,LABEL,UUID,MOUNTPOINTS,PARTUUID",
        ]
    )
    out["findmnt"] = run(["findmnt"])
    out["blkid"] = run(["blkid"])
    out["fstab"] = run(["cat", "/etc/fstab"])
    out["os_release"] = run(["cat", "/etc/os-release"])
    out["uname"] = run(["uname", "-a"])
    out["parted"] = run(["parted", "-l"])
    out["btrfs_version"] = run(["btrfs", "version"])

    # Evidence specifically about the target partition.
    target: dict[str, Any] = {
        "partition": TARGET_PARTITION,
        "blkid": run(["blkid", TARGET_PARTITION]),
    }
    # Is it referenced anywhere?
    refs: dict[str, Any] = {}
    fstab = out["fstab"].get("stdout", "")
    refs["in_fstab"] = TARGET_PARTITION in fstab or "nvme0n1p3" in fstab
    findmnt = out["findmnt"].get("stdout", "")
    refs["in_findmnt"] = "nvme0n1p3" in findmnt
    lsblk = out["lsblk"].get("stdout", "")
    for line in lsblk.splitlines():
        if "nvme0n1p3" in line:
            refs["lsblk_line"] = line.strip()
    # Does it carry a filesystem signature?
    blk = target["blkid"].get("stdout", "").strip()
    refs["has_filesystem_signature"] = bool(blk)
    refs["blkid_output"] = blk
    target["references"] = refs
    out["target_partition"] = target

    # §3 checklist (evidence only — the human decides).
    out["p3_safety_checklist"] = {
        "filesystem_exists": refs["has_filesystem_signature"],
        "appears_mounted": refs["in_findmnt"],
        "appears_in_fstab": refs["in_fstab"],
        "note": "Formatting requires ALL of: no meaningful data, no "
        "active mount/service usage, identifiable purpose, and "
        "an explicit human go decision (§3). If any check is "
        "uncertain: STOP.",
    }
    return out


# ---------------------------------------------------------------------------
# Kernel / hardware inventory (§8, §66)
# ---------------------------------------------------------------------------


def kernel_hardware() -> dict[str, Any]:
    probes = {
        "lscpu": ["lscpu"],
        "lspci": ["lspci", "-nn"],
        "lsusb": ["lsusb"],
        "free": ["free", "-h"],
        "meminfo": ["cat", "/proc/meminfo"],
        "numactl": ["numactl", "--hardware"],
        "cpufreq_driver": [
            "cat",
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
        ],
        "filesystems": ["cat", "/proc/filesystems"],
        "landlock": ["ls", "/sys/kernel/security/landlock"],
        "bpf_sysfs": ["ls", "/sys/fs/bpf"],
        "hugepages": ["cat", "/proc/meminfo"],
    }
    out: dict[str, Any] = {k: run(v) for k, v in probes.items()}
    fs = out["filesystems"].get("stdout", "")
    out["kernel_feature_flags"] = {
        "btrfs": "btrfs" in fs,
        "ext4": "ext4" in fs,
        "fuse": "fuse" in fs,
        "overlay": "overlay" in fs,
    }
    return out


# ---------------------------------------------------------------------------
# Hermes/Lucifer discovery (§9, §10)
# ---------------------------------------------------------------------------


def looks_relevant(path: Path) -> bool:
    name = path.name.lower()
    if name in SEARCH_NAMES:
        return True
    if name in SEARCH_FILES:
        return True
    if name.startswith("hermes") or name.startswith("lucifer"):
        return True
    if name.startswith("omni"):
        return True
    return False


def git_state(path: Path) -> dict[str, Any] | None:
    def g(*args: str) -> str | None:
        r = run(["git", "-C", str(path), *args], timeout=10)
        return r.get("stdout", "").strip() if r.get("returncode") == 0 else None

    branch = g("rev-parse", "--abbrev-ref", "HEAD")
    if branch is None:
        return None
    return {
        "branch": branch,
        "commit": g("rev-parse", "HEAD"),
        "dirty": bool(g("status", "--porcelain")),
        "remotes": g("remote", "-v"),
        "tags": g("tag", "--list"),
    }


def classify_component(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "absolute_path": str(path),
        "discovered_at": now_iso(),
    }
    markers: dict[str, bool] = {
        "language_python": (path / "pyproject.toml").exists()
        or (path / "setup.py").exists(),
        "language_rust": (path / "Cargo.toml").exists(),
        "language_cpp": (path / "CMakeLists.txt").exists(),
        "language_js": (path / "package.json").exists(),
        "has_makefile": (path / "Makefile").exists(),
        "has_soul_md": (path / "SOUL.md").exists(),
        "has_agents_md": (path / "AGENTS.md").exists(),
        "has_tests": any(path.glob("test*")) or any(path.glob("tests")),
        "has_config": (path / "config").exists(),
        "has_db": any(path.glob("*.db")) or any(path.glob("*.sqlite*")),
    }
    entry["markers"] = markers
    entry["languages"] = [
        k.split("_")[1] for k, v in markers.items() if k.startswith("language_") and v
    ]
    gs = git_state(path)
    if gs:
        entry["git"] = gs
    # Status classification is NOT inferred from names (§11): leave to
    # the engineer; record evidence only.
    entry["classification"] = "UNCLASSIFIED_PENDING_AUDIT"
    return entry


def discover_sources(roots: list[str]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set = set()
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        scanned = 0
        for dirpath, dirnames, filenames in os.walk(root_path):
            dp = Path(dirpath)
            scanned += 1
            if scanned > MAX_FILES_PER_ROOT:
                break
            # Prune noise.
            dirnames[:] = [
                d
                for d in dirnames
                if d not in SKIP_DIRS
                and not d.startswith(".")
                and d not in ("site-packages", "node_modules")
            ]
            if looks_relevant(dp):
                key = str(dp)
                if key not in seen:
                    seen.add(key)
                    found.append(classify_component(dp))
            for fn in filenames:
                if fn in SEARCH_FILES and dp not in seen:
                    key = str(dp)
                    if key not in seen:
                        seen.add(key)
                        found.append(classify_component(dp))
                    break
    return found


def pre_migration_manifest(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for comp in components:
        path = Path(comp["absolute_path"])
        total = 0
        count = 0
        sample_hashes: list[dict[str, Any]] = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and d != ".git"]
            for fn in filenames:
                fp = Path(dirpath) / fn
                try:
                    size = fp.stat().st_size
                except OSError:
                    continue
                total += size
                count += 1
                if len(sample_hashes) < 50 and size <= MAX_HASH_BYTES:
                    h = hashlib.sha256()
                    try:
                        with open(fp, "rb") as fh:
                            while True:
                                chunk = fh.read(1024 * 1024)
                                if not chunk:
                                    break
                                h.update(chunk)
                        sample_hashes.append(
                            {
                                "file": str(fp.relative_to(path)),
                                "sha256": h.hexdigest(),
                                "bytes": size,
                            }
                        )
                    except OSError:
                        continue
        manifest.append(
            {
                "path": str(path),
                "file_count": count,
                "total_bytes": total,
                "sample_file_hashes": sample_hashes,
                "git": comp.get("git"),
                "note": "full backup required before any migration (§10); "
                "this manifest is evidence, not the backup",
            }
        )
    return manifest


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def write_reports(
    outdir: Path,
    storage: dict[str, Any],
    hw: dict[str, Any],
    components: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "host-inventory.json").write_text(
        json.dumps({"storage": storage, "kernel_hardware": hw}, indent=2),
        encoding="utf-8",
    )
    (outdir / "existing-system-inventory.json").write_text(
        json.dumps({"captured_at": now_iso(), "components": components}, indent=2),
        encoding="utf-8",
    )
    (outdir / "pre-migration-manifest.json").write_text(
        json.dumps({"captured_at": now_iso(), "manifest": manifest}, indent=2),
        encoding="utf-8",
    )

    t = storage.get("target_partition", {})
    md = [
        "# Storage audit report (§3) — EVIDENCE ONLY",
        "",
        f"Generated: {now_iso()}  ",
        f"Target partition: `{TARGET_PARTITION}`",
        "",
        "## Raw evidence",
        "",
        "```",
        storage.get("lsblk", {}).get("stdout", ""),
        "```",
        "",
        "```",
        t.get("blkid", {}).get("stdout", "")
        or "(blkid: no output — partition may be blank OR unreadable without root)",
        "```",
        "",
        "## p3 checklist",
        "",
        "```json",
        json.dumps(storage.get("p3_safety_checklist", {}), indent=2),
        "```",
        "",
        "## Decision rule (directive §3)",
        "",
        "Format ONLY IF: no filesystem signature OR confirmed-empty "
        "data, no mount, no fstab/service usage, and the purpose of "
        "the partition is established with high confidence.  ",
        "**If any item is uncertain: STOP. Do not format.**",
        "",
        "## Discovered components",
        "",
    ]
    for c in components:
        md.append(
            f"- `{c['absolute_path']}` ({', '.join(c['languages']) or 'unknown'})"
        )
    if not components:
        md.append("- (none found — pass --roots or inspect manually)")
    (outdir / "storage-audit.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--out", default="./reports")
    args = ap.parse_args()

    print(f"[1/4] storage audit (target: {TARGET_PARTITION}) ...")
    storage = storage_audit()
    print("[2/4] kernel + hardware inventory ...")
    hw = kernel_hardware()
    print(f"[3/4] Hermes/Lucifer discovery under: {args.roots} ...")
    components = discover_sources(args.roots)
    print(f"      found {len(components)} candidate component(s)")
    print("[4/4] pre-migration manifest ...")
    manifest = pre_migration_manifest(components)

    outdir = Path(args.out)
    write_reports(outdir, storage, hw, components, manifest)
    print(f"\nReports written to {outdir}/:")
    for f in sorted(outdir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")
    print(
        "\nNEXT (per directive §76): review the reports, classify "
        "components (§11), back up BEFORE any change (§10), and make "
        "the p3 STOP/go decision explicitly (§3)."
    )
    print("This audit is read-only. Nothing was formatted, mounted, or modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
