"""Entry point for the omega-hub MCP server."""

from __future__ import annotations

import argparse

from omega_mcp import mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="omega-hub", description="Omega tri-token economy MCP hub"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport to use (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8029,
        help="HTTP bind port when using streamable-http (default: 8029)",
    )
    args = parser.parse_args()

    if args.transport == "streamable-http":
        # host/port are baked into FastMCP at construction time, not passed to run().
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
