// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
export interface TokenInfo {
  id: string;
  name: string;
  symbol: string;
  color: string;
  purpose: string;
  description: string;
  role: "macro" | "world" | "amity";
  transferable: boolean;
  network: string;
  availability: "wired" | "scaffold";
  notionalPrice?: string;
  unlockRequirement?: string;
}

export const TOKENS: TokenInfo[] = [
  {
    id: "omega",
    name: "Omega",
    symbol: "$OMEGA",
    color: "var(--color-omega)",
    purpose:
      "EVM-side macro-governance currency and one of the two currently wired optional reader-proof rails.",
    role: "macro",
    transferable: true,
    network: "EVM / Sepolia pilot",
    availability: "wired",
    notionalPrice: "Pilot rail — wallet signature + EVM proof",
    unlockRequirement:
      "Optional: sign an $OMEGA proof to try the verified reader flow. The public CC BY 4.0 manuscript needs no token or wallet.",
    description:
      "$OMEGA is the EVM-side anchor. Its current wallet-to-web proof flow is an optional reader demonstration; it does not restrict the public text of Crucible.",
  },
  {
    id: "twc",
    name: "Token of the World Citizen",
    symbol: "TWC",
    color: "var(--color-twc)",
    purpose:
      "Solana-side world-citizen currency and the second currently wired optional reader-proof rail.",
    role: "world",
    transferable: true,
    network: "Solana / Devnet pilot",
    availability: "wired",
    notionalPrice: "Pilot rail — wallet signature + Solana proof",
    unlockRequirement:
      "Optional: sign a TWC proof to try the verified reader flow. The public CC BY 4.0 manuscript needs no token or wallet.",
    description:
      "TWC is the Solana-side pilot currency. Its current proof flow demonstrates server-side verification for the optional reader; it does not restrict the public text.",
  },
  {
    id: "amity",
    name: "AMITY",
    symbol: "AMITY",
    color: "var(--color-amity)",
    purpose:
      "Bitcoin / Lightning / Taproot economy rail focused on dignified participation, repair, and exchange without judgment-first identity rules.",
    role: "amity",
    transferable: true,
    network: "Bitcoin / Lightning / Taproot testnet scaffold",
    availability: "scaffold",
    notionalPrice: "Separate scaffold — not wired into the optional reader-proof flow",
    unlockRequirement:
      "Not live as a reader-proof rail. AMITY remains a separate Bitcoin / Lightning / Taproot workstream; the public manuscript is not gated.",
    description:
      "AMITY is the third currency in the economy. In the current repository it exists as a separate Bitcoin / Lightning / Taproot scaffold, not a live reader-proof rail.",
  },
];

export const WIRED_TOKENS = TOKENS.filter((token) => token.availability === "wired");

export function tokenById(id: string): TokenInfo | undefined {
  return TOKENS.find((t) => t.id === id);
}
