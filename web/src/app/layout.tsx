import type { Metadata } from "next";
import "./globals.css";
import { Navigation } from "@/components/Navigation";

export const metadata: Metadata = {
  title: {
    default: "Omega MCP Hub — $OMEGA + TWC Unlock Rails",
    template: "%s | Omega MCP Hub",
  },
  description:
    "Omega MCP Hub currently wires two release-day currencies into the wallet-to-web unlock flow: $OMEGA on the EVM rail and TWC on Solana. Sign a verified proof and unlock Genesis Block: The Satoshi Protocol.",
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
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-6">
        <p>
          Omega MCP Hub — Open-source AI operating hub for a three-currency economy.
        </p>
        <p className="text-xs">
          Genesis Block: The Satoshi Protocol —{" "}
          <span className="font-mono text-foreground/60">
            Akash Varma&apos;s novel
          </span>
        </p>
      </div>
    </footer>
  );
}
