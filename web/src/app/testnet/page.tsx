import type { Metadata } from "next";

import { EconomyTestConsole } from "@/components/EconomyTestConsole";

export const metadata: Metadata = {
  title: "Economy Test Console",
  description:
    "Complete offline testing for the four-plane crypto economy: readiness across the $OMEGA, TWC, and AMITY rails, the slice A–G scenario suite, a valueless faucet, an audited action runner, and the Nostr integration surface.",
};

export default function TestnetPage() {
  return <EconomyTestConsole />;
}
