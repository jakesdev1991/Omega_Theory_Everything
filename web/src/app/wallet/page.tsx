import type { Metadata } from "next";

import { WalletHub } from "@/components/WalletHub";

export const metadata: Metadata = {
  title: "Wallet",
  description:
    "The Omega release wallet GUI, served by this site: real BIP-39 keys, an encrypted local keystore, MetaMask and Phantom bridges, and signed release-day unlock proofs for $OMEGA and TWC.",
};

export default function WalletPage() {
  return <WalletHub />;
}
