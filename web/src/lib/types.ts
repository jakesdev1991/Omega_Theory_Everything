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
      "EVM-side macro-governance currency and one of the two currently wired release-day unlock rails.",
    role: "macro",
    transferable: true,
    network: "EVM / Sepolia pilot",
    availability: "wired",
    notionalPrice: "Pilot rail — wallet signature + EVM proof",
    unlockRequirement:
      "Sign an $OMEGA release-day proof from the wired EVM wallet flow to unlock the full novel.",
    description:
      "$OMEGA is the EVM-side anchor. In the current build it is one of the two wallet-to-web currencies that can sign a release-day proof and unlock Crucible in full.",
  },
  {
    id: "twc",
    name: "Token of the World Citizen",
    symbol: "TWC",
    color: "var(--color-twc)",
    purpose:
      "Solana-side world-citizen currency and the second currently wired release-day unlock rail.",
    role: "world",
    transferable: true,
    network: "Solana / Devnet pilot",
    availability: "wired",
    notionalPrice: "Pilot rail — wallet signature + Solana proof",
    unlockRequirement:
      "Sign a TWC release-day proof from a Solana wallet to unlock the full novel.",
    description:
      "TWC is the Solana-side release currency. In the current build it is wired from wallet proof creation through server-side signature verification for the novel unlock flow.",
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
    notionalPrice: "Separate scaffold — not wired into the live unlock flow yet",
    unlockRequirement:
      "Not live yet. AMITY remains a separate Bitcoin / Lightning / Taproot workstream until holder verification exists.",
    description:
      "AMITY is the third currency in the economy. In the current repository it exists as a separate Bitcoin / Lightning / Taproot scaffold, not a live wallet-to-web unlock rail.",
  },
];

export const WIRED_TOKENS = TOKENS.filter((token) => token.availability === "wired");

export function tokenById(id: string): TokenInfo | undefined {
  return TOKENS.find((t) => t.id === id);
}
