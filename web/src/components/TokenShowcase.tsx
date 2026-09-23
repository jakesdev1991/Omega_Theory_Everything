"use client";

import { useState } from "react";
import { TOKENS, TokenInfo } from "@/lib/types";

const TOKEN_ORDER: TokenInfo["role"][] = [
  "sovereign",
  "proof-of-work",
  "governance",
  "exchange",
  "macro",
];

export function TokenShowcase() {
  const [hovered, setHovered] = useState<string | null>(null);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: "16px",
      }}
    >
      {TOKEN_ORDER.map((role) => {
        const t = TOKENS.find((tok) => tok.role === role)!;
        const isHovered = hovered === t.id;

        return (
          <button
            key={t.id}
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
            onMouseEnter={() => setHovered(t.id)}
            onMouseLeave={() => setHovered(null)}
          >
            <div
              style={{
                width: "38px",
                height: "38px",
                borderRadius: "10px",
                background: t.color,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                marginBottom: "14px",
                boxShadow: `0 0 24px ${t.color}33`,
              }}
            >
              <span
                style={{
                  color: "#07080c",
                  fontWeight: 700,
                  fontSize: "14px",
                  fontFamily: "ui-monospace, monospace",
                  letterSpacing: "0.02em",
                }}
              >
                {t.symbol}
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
              {t.name}
            </h3>

            <p
              style={{
                fontSize: "13px",
                color: "var(--color-muted)",
                lineHeight: 1.55,
                margin: "0 0 12px",
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
              }}
            >
              {t.purpose}
            </p>

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
              <span
                style={{
                  fontSize: "11px",
                  padding: "3px 8px",
                  borderRadius: "6px",
                  background: "rgba(255,255,255,0.04)",
                  color: "var(--color-muted-strong)",
                  fontFamily: "ui-monospace, monospace",
                  letterSpacing: "0.03em",
                }}
              >
                {t.transferable ? "Transferable" : "Non-transferable"}
              </span>
              <span
                style={{
                  fontSize: "11px",
                  padding: "3px 8px",
                  borderRadius: "6px",
                  background: `${t.color}18`,
                  color: t.color,
                  fontFamily: "ui-monospace, monospace",
                  letterSpacing: "0.03em",
                }}
              >
                {t.role.replace("-", " ")}
              </span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
