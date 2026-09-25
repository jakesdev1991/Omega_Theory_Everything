/**
 * Nostr integration surface for the Omega economy.
 *
 * The economy's social layer (care circles, work receipts, governance
 * discussion) is designed to ride on top of a Nostr client. The client itself
 * is developed separately; this module is the contract it plugs into, so that
 * landing the client is a configuration change rather than a rewrite.
 *
 * Nothing here talks to a relay yet. Until NOSTR_RELAYS and
 * NOSTR_PUBLISHER_NPUB are configured the whole surface reports
 * `status: "not_configured"` and every consumer fails closed.
 */

export const ECONOMY_NOSTR_KINDS = {
  /** Participant notes and circle chatter (standard NIP-01 short note). */
  participantNote: 1,
  /** Long-form claims and research artifacts (NIP-23 parameterized replaceable). */
  longFormClaim: 30023,
  /** Provisional: settled TWC work receipt, published as evidence. */
  workReceipt: 31331,
  /** Provisional: CARE attestation summary (consent-scoped, privacy filtered). */
  careAttestation: 31332,
  /** Provisional: $OMEGA governance proposal mirror for public discussion. */
  governanceProposal: 31333,
  /** Provisional: engine audit event export (operator channel only). */
  auditEvent: 31334,
  /** App store directory: NIP-89 handler announcements published by the root key. */
  storeHandler: 31990,
  /** App store directory: NIP-99 generic classified listings (fallback index). */
  storeListing: 30017,
  /** NIP-90 Data Vending Machine job request range (store → mobile node). */
  dvmRequestMin: 5000,
  dvmRequestMax: 5999,
  /** NIP-90 job result range: result kind = request kind + 1000. */
  dvmResultMin: 6000,
  dvmResultMax: 6999,
} as const;

export type EconomyNostrKind = (typeof ECONOMY_NOSTR_KINDS)[keyof typeof ECONOMY_NOSTR_KINDS];

export interface NostrIntegrationStatus {
  configured: boolean;
  status: "ready" | "not_configured";
  relays: string[];
  publisherNpub: string | null;
  nip05Domain: string | null;
  storeRootNpub: string | null;
  kinds: typeof ECONOMY_NOSTR_KINDS;
  requiredFromClient: string[];
  notes: string[];
}

export const NOSTR_CLIENT_REQUIREMENTS = [
  "Connect to the relays listed in NOSTR_RELAYS (wss:// URLs, comma separated).",
  "Sign economy events with the key behind NOSTR_PUBLISHER_NPUB, or with the user's own key once identity handoff exists.",
  `Publish settled work receipts as kind ${ECONOMY_NOSTR_KINDS.workReceipt} with the receipt's artifact hash in tag ["e", artifactHash].`,
  `Publish governance mirrors as kind ${ECONOMY_NOSTR_KINDS.governanceProposal} tagged ["d", proposalId].`,
  "Never publish CARE content for a participant whose privacy level or consent scope forbids it; the website only emits consent-scoped summaries.",
  "Treat every relay event as untrusted input: the web API re-verifies signatures and never accepts a relay event as settlement authority.",
] as const;

export function getNostrIntegrationStatus(): NostrIntegrationStatus {
  const relays = (process.env.NOSTR_RELAYS ?? "")
    .split(",")
    .map((relay) => relay.trim())
    .filter((relay) => relay.startsWith("wss://"));
  const publisherNpub = process.env.NOSTR_PUBLISHER_NPUB?.trim() || null;
  const nip05Domain = process.env.NOSTR_NIP05_DOMAIN?.trim() || null;
  const storeRootNpub = process.env.NOSTR_STORE_ROOT_NPUB?.trim() || null;

  const configured = relays.length > 0;

  return {
    configured,
    status: configured ? "ready" : "not_configured",
    relays,
    publisherNpub,
    nip05Domain,
    storeRootNpub,
    kinds: ECONOMY_NOSTR_KINDS,
    requiredFromClient: [...NOSTR_CLIENT_REQUIREMENTS],
    notes: [
      configured
        ? "Relays configured. The Nostr client may publish economy events; the website will surface them once the client ships."
        : "No relays configured (set NOSTR_RELAYS). The economy social layer stays local until the Nostr client lands.",
      publisherNpub
        ? `Publisher identity configured: ${publisherNpub}.`
        : "No publisher npub configured (set NOSTR_PUBLISHER_NPUB) — events will be signed by the user's own client key.",
      storeRootNpub
        ? `App store root key configured: ${storeRootNpub}. The /store page lists handlers announced by this key.`
        : "No store root npub configured (set NOSTR_STORE_ROOT_NPUB) — /store lets you paste one per session.",
      "Provisional kinds 31331-31334 are placeholders pending a NIP allocation; they are namespaced to avoid collisions.",
      "The mobile execution node lives in mobile-node/ (Termux daemon): it answers NIP-90 job requests from /store and only runs allowlisted algorithms for operator keys.",
    ],
  };
}
