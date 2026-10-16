# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Entry point for the omega-hub MCP server."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from omega_mcp import mcp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omega-hub", description="Omega tri-token economy MCP simulation prototype"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport to use (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind host when using streamable-http; no built-in auth (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8029,
        help="HTTP bind port when using streamable-http (default: 8029)",
    )
    return parser


def configure_server(args: argparse.Namespace) -> None:
    """Apply HTTP bind settings before FastMCP starts its transport."""
    if args.transport == "streamable-http":
        mcp.settings.host = args.host
        mcp.settings.port = args.port


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_server(args)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
