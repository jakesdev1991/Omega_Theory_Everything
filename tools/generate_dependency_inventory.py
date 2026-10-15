#!/usr/bin/env python3
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
"""Generate the reproducible source-and-lockfile dependency inventory.

The output is CycloneDX-shaped JSON so downstream review tools can ingest it,
but it is intentionally labeled as a *source-and-lockfile inventory*. It is not
an assertion that every component license is acceptable, that every Python
specifier was resolved, or that operating-system/deployment dependencies have
been inventoried. See docs/SOFTWARE_SUPPLY_CHAIN.md.

Run:

    python tools/generate_dependency_inventory.py
    python tools/generate_dependency_inventory.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "sbom" / "omega-theory-everything.cdx.json"
DEFAULT_AS_OF = "2026-10-15"

# Every current npm workspace lives directly below the repository root. Discover
# its committed lockfile instead of maintaining a second hard-coded workspace
# list here; the runtime audit has a separate regression test for its explicit
# execution list.
NPM_LOCKFILES = tuple(sorted(ROOT.glob("*/package-lock.json")))
CARGO_MANIFESTS = tuple(sorted((ROOT / "rust").glob("**/Cargo.toml"))) + (
    ROOT / "desktop" / "src-tauri" / "Cargo.toml",
)
WORKFLOWS = tuple(sorted((ROOT / ".github" / "workflows").glob("*.yml")))
INPUT_PATHS = (
    ROOT / "requirements.txt",
    ROOT / "mcp" / "pyproject.toml",
    *NPM_LOCKFILES,
    *CARGO_MANIFESTS,
    ROOT / "lean_proofs" / "lake-manifest.json",
    *WORKFLOWS,
    ROOT / "app" / "vendor" / "ethers.umd.min.js",
)

REQUIREMENT_RE = re.compile(
    r"^\s*([A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?)\s*(.*?)\s*$"
)
QUOTED_VALUE_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
MCP_DEPENDENCIES_RE = re.compile(
    r"^dependencies\s*=\s*\[(.*?)^\]", re.MULTILINE | re.DOTALL
)
CARGO_PACKAGE_NAME_RE = re.compile(r'^name\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
CARGO_PACKAGE_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
ACTION_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", re.MULTILINE)
ETHERS_VERSION_RE = re.compile(r'const version="([^"]+)"')


def relative(path: Path) -> str:
    """Return a stable POSIX-style path within the repository."""
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {relative(path)}")
    return data


def input_digest(paths: Iterable[Path]) -> str:
    """Hash inventory inputs with their paths, not generated output."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"inventory input is missing: {relative(path)}")
        digest.update(relative(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def property_value(name: str, value: str) -> dict[str, str]:
    return {"name": name, "value": value}


def component(
    *,
    bom_ref: str,
    name: str,
    version: str,
    component_type: str = "library",
    source: str,
    properties: Iterable[tuple[str, str]] = (),
    purl: str | None = None,
    external_references: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Create one stable, minimally claimed CycloneDX component."""
    values = [property_value("omega:source", source)]
    values.extend(property_value(key, value) for key, value in properties)
    result: dict[str, Any] = {
        "bom-ref": bom_ref,
        "type": component_type,
        "name": name,
        "version": version,
        "properties": sorted(values, key=lambda item: (item["name"], item["value"])),
    }
    if purl:
        result["purl"] = purl
    if external_references:
        result["externalReferences"] = external_references
    return result


def npm_package_name(package_path: str, metadata: dict[str, Any]) -> str | None:
    """Resolve an npm package name from v3 lockfile metadata."""
    explicit = metadata.get("name")
    if isinstance(explicit, str) and explicit:
        return explicit
    marker = "node_modules/"
    if marker not in package_path:
        return None
    return package_path.rsplit(marker, maxsplit=1)[1] or None


def npm_purl(name: str, version: str) -> str:
    return f"pkg:npm/{quote(name, safe='/')}@{quote(version, safe='')}"


def npm_components(lockfile: Path) -> list[dict[str, Any]]:
    lock = read_json(lockfile)
    packages = lock.get("packages")
    if not isinstance(packages, dict):
        raise ValueError(
            f"{relative(lockfile)} does not contain a package-lock v2/v3 packages object"
        )
    source = relative(lockfile)
    result: list[dict[str, Any]] = []
    for package_path, raw_metadata in sorted(packages.items()):
        if not package_path or not isinstance(raw_metadata, dict):
            continue
        name = npm_package_name(package_path, raw_metadata)
        version = raw_metadata.get("version")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
        ):
            continue
        details: list[tuple[str, str]] = [("omega:package-path", package_path)]
        if raw_metadata.get("dev") is True:
            details.append(("omega:development-only", "true"))
        if raw_metadata.get("optional") is True:
            details.append(("omega:optional", "true"))
        integrity = raw_metadata.get("integrity")
        if isinstance(integrity, str) and integrity:
            details.append(("omega:integrity", integrity))
        resolved = raw_metadata.get("resolved")
        if isinstance(resolved, str) and resolved:
            details.append(("omega:resolved", resolved))
        result.append(
            component(
                bom_ref=f"npm:{source}:{package_path}@{version}",
                name=name,
                version=version,
                source=source,
                properties=details,
                purl=npm_purl(name, version),
            )
        )
    return result


def requirement_component(
    name: str, specifier: str, source: str, index: int
) -> dict[str, Any]:
    exact = re.fullmatch(r"==\s*([A-Za-z0-9.+!_-]+)", specifier)
    version = exact.group(1) if exact else "unresolved"
    base_name, separator, extras = name.partition("[")
    properties = [("omega:version-specifier", specifier or "unconstrained")]
    if separator:
        properties.append(
            ("omega:extras", extras[:-1] if extras.endswith("]") else extras)
        )
    purl = (
        f"pkg:pypi/{quote(base_name, safe='')}@{quote(version, safe='')}"
        if exact
        else None
    )
    return component(
        bom_ref=f"python:{source}:{index}:{name}@{version}",
        name=base_name,
        version=version,
        source=source,
        properties=properties,
        purl=purl,
    )


def requirement_components(path: Path) -> list[dict[str, Any]]:
    """Inventory direct requirements without pretending range specs are locks."""
    result: list[dict[str, Any]] = []
    source = relative(path)
    for index, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        match = REQUIREMENT_RE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"cannot parse requirement at {source}:{index}: {raw_line}"
            )
        name, specifier = match.groups()
        result.append(requirement_component(name, specifier.strip(), source, index))
    return result


def mcp_components(path: Path) -> list[dict[str, Any]]:
    """Read the simple PEP 621 dependency array without adding a TOML dependency."""
    source = relative(path)
    text = path.read_text(encoding="utf-8")
    match = MCP_DEPENDENCIES_RE.search(text)
    if match is None:
        raise ValueError(f"could not find dependency array in {source}")
    result: list[dict[str, Any]] = []
    for index, raw_value in enumerate(QUOTED_VALUE_RE.findall(match.group(1)), start=1):
        requirement = raw_value.replace('\\"', '"')
        requirement_match = REQUIREMENT_RE.fullmatch(requirement)
        if requirement_match is None:
            raise ValueError(f"cannot parse MCP requirement at {source}: {requirement}")
        name, specifier = requirement_match.groups()
        result.append(requirement_component(name, specifier.strip(), source, index))
    return result


def cargo_components(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Inventory local Rust package manifests; no Cargo.lock is committed."""
    result: list[dict[str, Any]] = []
    for path in sorted(paths):
        source = relative(path)
        text = path.read_text(encoding="utf-8")
        name_match = CARGO_PACKAGE_NAME_RE.search(text)
        if name_match is None:
            continue  # workspace root has no [package] section
        version_match = CARGO_PACKAGE_VERSION_RE.search(text)
        version = version_match.group(1) if version_match else "workspace"
        name = name_match.group(1)
        result.append(
            component(
                bom_ref=f"cargo:{source}:{name}@{version}",
                name=name,
                version=version,
                component_type="application",
                source=source,
                properties=[("omega:manifest-only", "true")],
            )
        )
    return result


def lean_components(path: Path) -> list[dict[str, Any]]:
    manifest = read_json(path)
    packages = manifest.get("packages")
    if not isinstance(packages, list):
        raise ValueError(f"{relative(path)} does not contain a packages list")
    source = relative(path)
    result: list[dict[str, Any]] = []
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            continue
        name = package.get("name")
        revision = package.get("rev")
        url = package.get("url")
        if (
            not isinstance(name, str)
            or not isinstance(revision, str)
            or not isinstance(url, str)
        ):
            continue
        properties = [("omega:input-revision", str(package.get("inputRev", "")))]
        if package.get("inherited") is True:
            properties.append(("omega:inherited", "true"))
        result.append(
            component(
                bom_ref=f"lean:{name}@{revision}",
                name=name,
                version=revision,
                source=source,
                properties=properties,
                external_references=[{"type": "vcs", "url": url}],
            )
        )
    return result


def action_components(workflows: Iterable[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(workflows):
        source = relative(path)
        for index, action in enumerate(
            ACTION_RE.findall(path.read_text(encoding="utf-8")), start=1
        ):
            if action.startswith("./") or "@" not in action:
                continue
            name, revision = action.rsplit("@", maxsplit=1)
            if not name or not revision:
                continue
            url = f"https://github.com/{name}/tree/{revision}"
            result.append(
                component(
                    bom_ref=f"github-action:{source}:{index}:{name}@{revision}",
                    name=name,
                    version=revision,
                    component_type="framework",
                    source=source,
                    properties=[("omega:workflow-line-occurrence", str(index))],
                    external_references=[{"type": "vcs", "url": url}],
                )
            )
    return result


def vendored_components(path: Path) -> list[dict[str, Any]]:
    source = relative(path)
    match = ETHERS_VERSION_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"could not identify the vendored ethers version in {source}")
    version = match.group(1)
    return [
        component(
            bom_ref=f"vendored:{source}:ethers@{version}",
            name="ethers",
            version=version,
            source=source,
            properties=[("omega:vendored", "true")],
            purl=npm_purl("ethers", version),
        )
    ]


def build_bom(as_of: str = DEFAULT_AS_OF) -> dict[str, Any]:
    """Build deterministic JSON from every supported manifest/lockfile."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of):
        raise ValueError("--as-of must use YYYY-MM-DD")
    digest = input_digest(INPUT_PATHS)
    components: list[dict[str, Any]] = []
    components.extend(requirement_components(ROOT / "requirements.txt"))
    components.extend(mcp_components(ROOT / "mcp" / "pyproject.toml"))
    for lockfile in NPM_LOCKFILES:
        components.extend(npm_components(lockfile))
    components.extend(cargo_components(CARGO_MANIFESTS))
    components.extend(lean_components(ROOT / "lean_proofs" / "lake-manifest.json"))
    components.extend(action_components(WORKFLOWS))
    components.extend(
        vendored_components(ROOT / "app" / "vendor" / "ethers.umd.min.js")
    )
    components.sort(key=lambda item: item["bom-ref"])

    serial = uuid.uuid5(
        uuid.NAMESPACE_URL, f"omega-dependency-inventory:{as_of}:{digest}"
    )
    return {
        "$schema": "https://cyclonedx.org/schema/bom-1.5.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "timestamp": f"{as_of}T00:00:00Z",
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "tools/generate_dependency_inventory.py",
                        "version": "1",
                    }
                ]
            },
            "component": {
                "type": "application",
                "name": "Omega Theory Everything",
                "version": f"source-lockfile-inventory-{as_of}",
                "properties": [
                    property_value("omega:input-sha256", digest),
                    property_value(
                        "omega:inventory-scope",
                        "declared manifests, committed lockfiles, workflows, and vendored ethers bundle",
                    ),
                    property_value(
                        "omega:license-review", "not performed by generator"
                    ),
                    property_value("omega:component-count", str(len(components))),
                ],
            },
        },
        "components": components,
    }


def render(bom: dict[str, Any]) -> str:
    return json.dumps(bom, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="output JSON path"
    )
    parser.add_argument(
        "--as-of", default=DEFAULT_AS_OF, help="inventory date in YYYY-MM-DD form"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if output differs from regenerated JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        expected = render(build_bom(args.as_of))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"dependency inventory error: {error}", file=sys.stderr)
        return 2

    if args.check:
        try:
            actual = args.output.read_text(encoding="utf-8")
        except OSError as error:
            print(f"dependency inventory missing: {error}", file=sys.stderr)
            return 1
        if actual != expected:
            print(
                "dependency inventory is out of date; regenerate with "
                "python tools/generate_dependency_inventory.py",
                file=sys.stderr,
            )
            return 1
        print(f"dependency inventory is current: {relative(args.output)}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(f"wrote dependency inventory: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
