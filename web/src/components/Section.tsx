// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
"use client";

import { useCallback, useEffect, useRef } from "react";

interface Props {
  title: string;
  subtitle?: string;
  eyebrow?: string;
  children?: React.ReactNode;
  id?: string;
}

export function Section({ title, subtitle, eyebrow, children, id: idProp }: Props) {
  const ref = useRef<HTMLAnchorElement>(null);
  const id = idProp ?? title.toLowerCase().replace(/[^a-z0-9]+/g, "-");

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLAnchorElement>) => {
      e.preventDefault();
      const el = document.getElementById(id);
      if (el) {
        const y = el.getBoundingClientRect().top + window.scrollY - 80;
        window.scrollTo({ top: y, behavior: "smooth" });
      }
    },
    [id]
  );

  // Smooth-scroll when the section is rendered via anchor navigation
  useEffect(() => {
    const el = document.getElementById(id);
    if (!el) return;
    const y = el.getBoundingClientRect().top + window.scrollY - 80;
    window.scrollTo({ top: y, behavior: "smooth" });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <a
      ref={ref}
      id={id}
      href={`#${id}`}
      style={{ display: "block", scrollMarginTop: "96px" }}
      onClick={handleClick}
    >
      <section style={{ padding: "96px 0 72px" }}>
        <div
          style={{
            maxWidth: "1100px",
            margin: "0 auto",
            padding: "0 24px",
          }}
        >
          {eyebrow && (
            <p
              style={{
                color: "var(--color-accent)",
                fontSize: "12px",
                letterSpacing: "0.16em",
                textTransform: "uppercase",
                fontWeight: 600,
                marginBottom: "18px",
                fontFamily: "ui-monospace, monospace",
              }}
            >
              {eyebrow}
            </p>
          )}

          <h2
            style={{
              fontFamily: "ui-serif, Georgia, Cambria, serif",
              fontSize: "clamp(28px, 4vw, 48px)",
              lineHeight: 1.1,
              fontWeight: 500,
              letterSpacing: "-0.015em",
              color: "var(--color-foreground)",
              margin: "0 0 20px",
              maxWidth: "760px",
            }}
          >
            {title}
          </h2>

          {subtitle && (
            <p
              style={{
                color: "var(--color-muted-strong)",
                fontSize: "clamp(15px, 1.6vw, 18px)",
                lineHeight: 1.65,
                maxWidth: "620px",
                margin: 0,
                fontWeight: 400,
              }}
            >
              {subtitle}
            </p>
          )}

          {children && (
            <div style={{ marginTop: "40px" }}>{children}</div>
          )}
        </div>
      </section>
    </a >
  );
}
