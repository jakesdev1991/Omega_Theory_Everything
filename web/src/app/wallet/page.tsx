// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import type { Metadata } from "next";

import { WalletHub } from "@/components/WalletHub";

export const metadata: Metadata = {
  title: "Wallet",
  description:
    "The Omega wallet GUI, served by this site: real BIP-39 keys, an encrypted local keystore, MetaMask and Phantom bridges, and optional $OMEGA/TWC reader-proof demonstrations. The public novel requires no token or wallet.",
};

export default function WalletPage() {
  return <WalletHub />;
}
