// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import type { Metadata } from "next";

import { AppStore } from "@/components/AppStore";

export const metadata: Metadata = {
  title: "App Store",
  description:
    "Static app-store frontend over the Nostr backplane: directory from kinds 31990/30017 published by the sovereign root key, NIP-90 job requests to the mobile execution node, and settlement of verified results into the TWC economy ledger.",
};

export default function StorePage() {
  return <AppStore />;
}
