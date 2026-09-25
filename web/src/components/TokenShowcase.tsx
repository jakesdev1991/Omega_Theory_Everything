// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"use client";

import { useState } from "react";
import { TOKENS, TokenInfo, WIRED_TOKENS } from "@/lib/types";

const TOKEN_ORDER: TokenInfo["role"][] = ["macro", "world", "amity"];

export function TokenShowcase({ variant = "wired" }: { variant?: "wired" | "economy" }) {
  const [hovered, setHovered] = useState<string | null>(null);
  const source = variant === "economy" ? TOKENS : WIRED_TOKENS;
  const ordered = TOKEN_ORDER.map((role) => source.find((token) => token.role === role)).filter(Boolean) as TokenInfo[];

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: "16px",
      }}
    >
      {ordered.map((token) => {
        const isHovered = hovered === token.id;

        return (
          <button
            key={token.id}
            style={{
              textAlign: "left",
              cursor: "pointer",
              border: `1px solid ${isHovered ? "var(--color-border-strong)" : "var(--color-border)"}`,
              borderRadius: "14px",
              padding: "22px 20px",
              background: isHovered
                ? "linear-gradient(180deg, rgba(255,255,255,0.03), transparent)"
                : "transparent",
              transition: "border-color 0.2s ease, background 0.2s ease, transform 0.2s ease",
              transform: isHovered ? "translateY(-3px)" : "none",
              position: "relative",
              overflow: "hidden",
            }}
            onMouseEnter={() => setHovered(token.id)}
            onMouseLeave={() => setHovered(null)}
          >
            <div
              style={{
                width: "38px",
                height: "38px",
                borderRadius: "10px",
                background: token.color,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                marginBottom: "14px",
                boxShadow: `0 0 24px ${token.color}33`,
              }}
            >
              <span
                style={{
                  color: "#07080c",
                  fontWeight: 700,
                  fontSize: token.symbol.length > 6 ? "11px" : "14px",
                  fontFamily: "ui-monospace, monospace",
                  letterSpacing: "0.02em",
                }}
              >
                {token.symbol}
              </span>
            </div>

            <h3
              style={{
                fontFamily: "ui-serif, Georgia, serif",
                fontSize: "17px",
                fontWeight: 500,
                margin: "0 0 6px",
                color: "var(--color-foreground)",
              }}
            >
              {token.name}
            </h3>

            <p
              style={{
                fontSize: "13px",
                color: "var(--color-muted)",
                lineHeight: 1.55,
                margin: "0 0 12px",
                display: "-webkit-box",
                WebkitLineClamp: 3,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
              }}
            >
              {token.purpose}
            </p>

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
              <span
                style={{
                  fontSize: "11px",
                  padding: "3px 8px",
                  borderRadius: "6px",
                  background: token.availability === "wired" ? "rgba(52,211,153,0.1)" : "rgba(167,139,250,0.12)",
                  color: token.availability === "wired" ? "var(--color-unlock)" : "var(--color-amity)",
                  fontFamily: "ui-monospace, monospace",
                  letterSpacing: "0.03em",
                }}
              >
                {token.availability === "wired" ? "Live rail" : "Separate scaffold"}
              </span>
              <span
                style={{
                  fontSize: "11px",
                  padding: "3px 8px",
                  borderRadius: "6px",
                  background: `${token.color}18`,
                  color: token.color,
                  fontFamily: "ui-monospace, monospace",
                  letterSpacing: "0.03em",
                }}
              >
                {token.network}
              </span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
