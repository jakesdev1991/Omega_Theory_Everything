<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# Token of the World Citizen — Solana Devnet pilot

This directory is the **Solana Devnet-only leg** of the C.A.R.E. Protocol pilot. It prepares one deliberately valueless standard SPL token named **Token of the World Citizen** with the visibly test-only symbol **`tTWC`**. The canonical proposed identity is **Token of the World Citizen (`TWC`)**, but the pilot must never be presented as a live/mainnet asset.

> **Not production software. Not a token sale. Not an investment or a promise of value.** No Devnet deployment has been performed from this repository. Trademark clearance, legal review, independent security review, explicit governance approval, and a separate mainnet specification are required before any commercial or mainnet use.

## What the issuer creates

The deployer uses the standard SPL Token Program (`Tokenkeg…`) and Metaplex Token Metadata; it deploys **no bespoke Solana program** and uses no Token-2022 extensions.

The single atomic Devnet transaction:

1. creates a fresh standard SPL mint with **9 decimals**;
2. initializes it with its **freeze authority set to `None`**;
3. creates Metaplex `Fungible` metadata with name `Token of the World Citizen`, symbol `tTWC`, zero seller fees, and `isMutable: false`;
4. creates the treasury's associated token account if needed and mints the fixed initial supply to it once; and
5. permanently sets the mint authority to `None`.

If any instruction fails, the transaction fails as a whole. The default test supply is **1,000,000,000 tTWC** (`1,000,000,000,000,000,000` base units). The issuer and verifier independently check that supply, mint authority, freeze authority, treasury ATA, metadata PDA, metadata fields, metadata immutability, and the off-chain metadata hash.

`isMutable: false` makes the Metaplex metadata data immutable. The metadata account still records its original update-authority public address for provenance; it is not a path to change immutable metadata.

## Prerequisites

- Node.js 22+ and npm. The project uses modern `@solana/kit` APIs, so it does **not** require the Solana CLI, Anchor, Rust, or `@solana/web3.js` v1.
- A newly generated, externally held **Devnet-only** payer keypair JSON file. Keep it outside this repository or in the ignored `solana/keys/` directory; it must be mode `0600` on Unix.
- Faucet Devnet SOL for that payer.
- An authorized public Devnet treasury address.
- A final publicly reachable HTTPS metadata JSON document, preferably content-addressed/immutable (for example, an Arweave or IPFS gateway URI).

Do **not** use a mainnet wallet, a production treasury, or a key pasted into chat, `.env`, source code, or a Git commit.

## Local validation

```bash
cd solana
npm ci
npm run check
npm audit --omit=dev --audit-level=high
```

The offline tests validate exact u64 supply arithmetic, configuration guards, metadata hash pinning and pilot wording, manifest safety, and the complete signed transaction plan. The plan test confirms that the serialized atomic issuance transaction stays inside Solana's packet limit.

## Prepare metadata before a deployment

1. Copy [`metadata/twc-devnet.template.json`](metadata/twc-devnet.template.json) outside the repository or to a new local working file.
2. Replace the placeholder image URI and review the document. It must retain name `Token of the World Citizen`, symbol `tTWC`, and an explicit Devnet/valueless warning.
3. Upload the exact final JSON to an immutable public HTTPS URI.
4. Hash the **exact bytes** uploaded:

   ```bash
   npm run metadata:hash -- path/to/final-twc-devnet.json
   ```

5. Record the resulting lowercase SHA-256 along with the final URI. The deployer fetches the URI without redirects, compares the bytes to that SHA-256, and refuses mismatches.

## Configuration and guarded deployment

```bash
cd solana
cp .env.example .env
chmod 600 /path/to/devnet-deployer.json
```

Fill only local values in `.env`:

```dotenv
SOLANA_DEPLOYER_KEYPAIR_PATH=/absolute/path/to/devnet-deployer.json
SOLANA_DEVNET_RPC_URL=https://api.devnet.solana.com
# SOLANA_DEVNET_WS_URL=  # optional; derived from the RPC URL when blank
TWC_TREASURY_ADDRESS=<authorized Devnet public address>
TWC_METADATA_URI=https://<immutable-public-uri>/twc-devnet.json
TWC_METADATA_SHA256=<64 hexadecimal characters>
TWC_INITIAL_SUPPLY=1000000000
```

`npm run deploy:devnet` runs only preflight and **will not broadcast** without two intentional acknowledgements. After an operator has independently reviewed the output, the treasury, the final metadata, the exact source commit, and the generated transaction parameters, the Devnet-only command is:

```bash
TWC_DEPLOY_CONFIRM=DEVNET_TWC_PILOT npm run deploy:devnet -- --confirm-devnet
```

The script verifies the RPC's Devnet genesis hash before it builds or broadcasts a deployment transaction, requires at least 0.05 Devnet SOL, and refuses to overwrite an existing local manifest. It creates no mainnet transaction and has no mainnet configuration.

On success, the script writes `deployments/twc-devnet.json`. Deployment manifests are intentionally Git-ignored: a local transaction must not become a repository claim or a canonical release merely because it occurred. Archive the public manifest, mint/metadata/ATA explorer links, transaction signature, metadata bytes/hash, source commit, and independent verification evidence in the operator's pilot record.

## Verify a deployed pilot

A verifier needs no private key. It uses the ignored manifest and independently queries Devnet:

```bash
npm run verify:devnet
# or: npm run verify:devnet -- /path/to/twc-devnet.json
```

Verification fails closed if the RPC is not Devnet, a mint/freeze authority remains, supply or treasury balance differs, the standard program/PDA differs, metadata is mutable or malformed, or the public JSON no longer matches the committed SHA-256.

## Operating limits and handoff

- `tTWC` is a valueless test artifact. Do not advertise it, list it, sell it, bridge it, or treat it as a right to a product, payment, governance power, or real-world service.
- This work does **not** make the existing web unlock route suitable for Solana. That route must later verify an independently sourced balance or claim event; it must not trust a client-supplied tier or address.
- A real-value or mainnet token requires a new specification, separate contracts/tooling review, approved treasury policy, legal/securities/compliance analysis, trademark clearance, economics review, external audit, incident process, and explicit approval.
- Review [`THREAT_MODEL.md`](THREAT_MODEL.md) before any Devnet broadcast. It describes residual risks and the failure/incident runbook.

## References

- [Solana mint authorities](https://solana.com/docs/tokens/basics/set-authority)
- [Solana Metaplex metadata guide](https://solana.com/docs/tokens/metaplex)
- [Metaplex token metadata creation](https://www.metaplex.com/docs/smart-contracts/token-metadata/mint)
- [Project naming notice](../docs/TRADEMARKS.md)
- [Broader staged launch plan](../launch/novel_day_one_plan.md)
