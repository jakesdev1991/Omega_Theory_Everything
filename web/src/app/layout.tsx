// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import type { Metadata } from "next";
import "./globals.css";
import { Navigation } from "@/components/Navigation";

export const metadata: Metadata = {
  title: {
    default: "The C.A.R.E. Economy — Call About Resuscitating Everyone",
    template: "%s | The C.A.R.E. Economy",
  },
  description:
    "The C.A.R.E. Economy: Call About Resuscitating Everyone. A sovereign, three-currency economy built on care. Read the manifesto and whitepaper, run the wallet, and test the economy. The public draft of Crucible: The Satoshi Protocol is licensed under CC BY 4.0; $OMEGA and TWC proofs power an optional reader flow, not a content gate.",
  icons: {
    icon: [{ url: "/favicon.svg", type: "image/svg+xml" }],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="h-full flex flex-col">
        <Navigation />
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}

function Footer() {
  return (
    <footer className="border-t border-border/40 py-8 text-sm text-muted">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-4 px-6">
        <p>
          The C.A.R.E. Economy — Call About Resuscitating Everyone. Operated
          by the open-source Omega MCP Hub.
        </p>
        <p className="text-xs">
          Crucible: The Satoshi Protocol —{" "}
          <span className="font-mono text-foreground/60">
            Public draft · CC BY 4.0
          </span>
        </p>
      </div>
    </footer>
  );
}
