"use client";

import { useState, useCallback } from "react";
import Link from "next/link";

const NAV_LINKS = [
  { href: "/", label: "Home" },
  { href: "/economy", label: "Economy" },
  { href: "/novel", label: "The Novel" },
  { href: "/invest", label: "Invest" },
  { href: "/mcp", label: "MCP Hub" },
];

export function Navigation() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  const handleScroll = useCallback(() => {
    setScrolled(window.scrollY > 12);
  }, []);

  if (typeof window !== "undefined") {
    window.addEventListener("scroll", handleScroll, { passive: true });
  }

  return (
    <header
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 50,
        borderBottom: scrolled ? "1px solid var(--color-border)" : "1px solid transparent",
        backdropFilter: scrolled ? "blur(14px)" : "none",
        backgroundColor: scrolled ? "rgba(7,8,12,0.82)" : "transparent",
        transition: "backdrop-filter 0.25s ease, background-color 0.25s ease, border-color 0.25s ease",
      }}
    >
      <nav
        style={{
          maxWidth: "1200px",
          margin: "0 auto",
          padding: "0 24px",
          height: "64px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
        aria-label="Primary navigation"
      >
        <Link
          href="/"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            color: "var(--color-foreground)",
            textDecoration: "none",
            fontWeight: 600,
            letterSpacing: "0.01em",
            fontSize: "15px",
          }}
        >
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              width: "28px",
              height: "28px",
              borderRadius: "6px",
              background: "linear-gradient(135deg, var(--color-accent), var(--color-omega))",
              color: "#07080c",
              fontWeight: 700,
              fontSize: "13px",
              fontFamily: "ui-monospace, monospace",
            }}
          >
            O
          </span>
          <span style={{ fontFamily: "ui-serif, Georgia, serif", fontWeight: 500 }}>
            Omega MCP Hub
          </span>
        </Link>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "28px",
          }}
          className="desktop-nav"
        >
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              style={{
                color: "var(--color-muted-strong)",
                textDecoration: "none",
                fontSize: "14px",
                fontWeight: 450,
                transition: "color 0.15s ease",
                padding: "6px 0",
                borderBottom: "2px solid transparent",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.color = "var(--color-foreground)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.color = "var(--color-muted-strong)")
              }
            >
              {link.label}
            </Link>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <Link
            href="/invest"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              padding: "9px 18px",
              borderRadius: "8px",
              background: "linear-gradient(135deg, var(--color-accent), #b8763d)",
              color: "#07080c",
              fontWeight: 600,
              fontSize: "13px",
              textDecoration: "none",
              letterSpacing: "0.02em",
              transition: "opacity 0.15s ease, transform 0.15s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.opacity = "0.9";
              e.currentTarget.style.transform = "translateY(-1px)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.opacity = "1";
              e.currentTarget.style.transform = "none";
            }}
          >
            Participate
          </Link>

          <button
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: "36px",
              height: "36px",
              borderRadius: "8px",
              border: "1px solid var(--color-border-strong)",
              background: "transparent",
              color: "var(--color-muted-strong)",
              cursor: "pointer",
              transition: "border-color 0.15s ease, color 0.15s ease, background 0.15s ease",
            }}
            onClick={() => setMenuOpen((o) => !o)}
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen}
          >
            <span style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <span
                style={{
                  display: "block",
                  height: "2px",
                  width: "16px",
                  background: menuOpen ? "var(--color-muted-strong)" : "var(--color-foreground)",
                  transition: "background 0.2s ease",
                  transform: menuOpen ? "rotate(45deg) translateY(5px)" : "none",
                }}
              />
              <span
                style={{
                  display: "block",
                  height: "2px",
                  width: "16px",
                  background: menuOpen ? "transparent" : "var(--color-foreground)",
                  transition: "background 0.2s ease",
                }}
              />
              <span
                style={{
                  display: "block",
                  height: "2px",
                  width: "16px",
                  background: menuOpen ? "var(--color-muted-strong)" : "var(--color-foreground)",
                  transition: "background 0.2s ease",
                  transform: menuOpen ? "rotate(-45deg) translateY(-5px)" : "none",
                }}
              />
            </span>
          </button>
        </div>
      </nav>

      {menuOpen && (
        <div
          style={{
            position: "fixed",
            top: "64px",
            left: 0,
            right: 0,
            background: "rgba(7,8,12,0.96)",
            backdropFilter: "blur(16px)",
            borderBottom: "1px solid var(--color-border)",
            padding: "16px 24px 24px",
            zIndex: 40,
          }}
        >
          <nav style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                style={{
                  padding: "12px 0",
                  color: "var(--color-muted-strong)",
                  fontSize: "16px",
                  fontWeight: 500,
                  borderBottom: "1px solid var(--color-border)",
                  transition: "color 0.15s ease",
                }}
                onClick={() => setMenuOpen(false)}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.color = "var(--color-foreground)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.color = "var(--color-muted-strong)")
                }
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
      )}

      <style>{`
        @media (min-width: 768px) {
          .desktop-nav { display: flex; }
        }
        @media (max-width: 767px) {
          .desktop-nav { display: none; }
        }
      `}</style>
    </header>
  );
}
