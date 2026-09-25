import type { Metadata } from "next";
import "./globals.css";
import { Navigation } from "@/components/Navigation";

export const metadata: Metadata = {
  title: {
    default: "The C.A.R.E. Economy — Call About Resuscitating Everyone",
    template: "%s | The C.A.R.E. Economy",
  },
  description:
    "The C.A.R.E. Economy: Call About Resuscitating Everyone. A sovereign, three-currency economy built on care. Read the manifesto and the extensive whitepaper, run the wallet, and test the whole economy. Currently wires $OMEGA (EVM) and TWC (Solana) release proofs to the token-gated novel Crucible: The Satoshi Protocol.",
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
            Akash Varma&apos;s novel
          </span>
        </p>
      </div>
    </footer>
  );
}
