export interface TokenInfo {
  id: string;
  name: string;
  symbol: string;
  color: string;
  purpose: string;
  description: string;
  role: "sovereign" | "proof-of-work" | "governance" | "exchange" | "macro";
  transferable: boolean;
  notionalPrice?: string;
  unlockRequirement?: string;
}

export const TOKENS: TokenInfo[] = [
  {
    id: "sov",
    name: "Sovereign",
    symbol: "SOv",
    color: "var(--color-sov)",
    purpose:
      "Accounting and settlement layer — the ledger that holds the economy's state.",
    role: "sovereign",
    transferable: true,
    notionalPrice: "1 SOv = 1 unit of sovereign account",
    unlockRequirement:
      "Hold any token in the Omega economy to access the ledger registry.",
    description:
      "SOv is the money of account for the Omega MCP Hub. It records who did what, when, and in what volume. It is the spine of the tri-token economy — every other token references a SOv position.",
  },
  {
    id: "use",
    name: "Proof of Useful Work",
    symbol: "USE",
    color: "var(--color-use)",
    purpose:
      "Non-transferable receipt for useful computational or creative work performed for the economy.",
    role: "proof-of-work",
    transferable: false,
    notionalPrice: "1 USE = 1 verified unit of useful work",
    unlockRequirement:
      "Contribute to the economy (compute, content, research) to earn USE and unlock the receipts ledger.",
    description:
      "USE is not a currency. It is a signed receipt that says: this person did something useful. It cannot be bought, sold, or transferred — only earned. The novel unlocks progressively as your USE balance grows, because USE is the proof that you're part of the work.",
  },
  {
    id: "care",
    name: "Care",
    symbol: "CARE",
    color: "var(--color-care)",
    purpose:
      "Governance and stewardship token — the voice of the economy's long-term steward.",
    role: "governance",
    transferable: true,
    notionalPrice: "1 CARE = 1 stewardship vote weight",
    unlockRequirement:
      "Hold CARE to vote on protocol upgrades, treasury allocation, and the future of the novel's ecosystem.",
    description:
      "CARE is the governance layer. Token holders steward the protocol: they vote on what gets built, where treasury funds go, and whether new chapters of the novel are released to the public or kept gated. Holding CARE is not just a right — it is a responsibility.",
  },
  {
    id: "amity",
    name: "Amity",
    symbol: "AMITY",
    color: "var(--color-amity)",
    purpose:
      "Exchange-eligible representation of care — the liquidity layer that lets care participate in markets.",
    role: "exchange",
    transferable: true,
    notionalPrice: "1 AMITY = 1 unit of exchange-eligible care representation",
    unlockRequirement:
      "Hold AMITY or convert CARE to AMITY to access the exchange and novella marketplace.",
    description:
      "AMITY is CARE's exchange-facing twin. You cannot spend CARE directly in external markets, but you can represent it as AMITY — a transferable, liquid token that still carries the stewardship weight of CARE. AMITY holders unlock the mart: the marketplace of novella chapters, citations, and proofs.",
  },
  {
    id: "omega",
    name: "Omega",
    symbol: "$OMEGA",
    color: "var(--color-omega)",
    purpose:
      "Macro-governance scarce asset — the fixed-supply anchor of the entire economy.",
    role: "macro",
    transferable: true,
    notionalPrice: "1 $OMEGA = 1 macro-governance vote + scarcity premium",
    unlockRequirement:
      "Hold $OMEGA for full access: all chapters of Genesis Block unlocked, plus macro-governance rights.",
    description:
      "$OMEGA is the scarcity anchor. Fixed supply. Every holder of $OMEGA unlocks the complete novel on release day — no gating, no chapters withheld. $OMEGA holders are the founding stewards of the Omega MCP Hub and the Genesis Block story.",
  },
];

export function tokenById(id: string): TokenInfo | undefined {
  return TOKENS.find((t) => t.id === id);
}

export const TOKEN_BY_ROLE: Record<TokenInfo["role"], TokenInfo> = {
  sovereign: TOKENS[0],
  "proof-of-work": TOKENS[1],
  governance: TOKENS[2],
  exchange: TOKENS[3],
  macro: TOKENS[4],
};
