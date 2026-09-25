import type { Metadata } from "next";

import { WalletDownloadPanel } from "@/components/WalletDownloadPanel";

export const metadata: Metadata = {
  title: "Wallet Downloads",
  description:
    "Download the Omega Wallet GUI: offline bundle with checksums and local launchers, installable web app, or native desktop builds — with SHA-256 verification for every artifact.",
};

export default function WalletDownloadPage() {
  return <WalletDownloadPanel />;
}
