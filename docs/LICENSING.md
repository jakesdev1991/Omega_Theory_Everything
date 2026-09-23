# Licensing policy and scope

> This is a project policy summary, not legal advice or a signed commercial agreement. Have an intellectual-property lawyer review the transition and prepare any commercial license or contributor agreement before relying on it.

## Chosen model

The project is split between **open scientific materials** and **proprietary product materials**. There is no single license for the whole repository. The root [`LICENSE`](../LICENSE) is a scope index, not a grant covering everything.

| Scope | License / status | What it means |
|---|---|---|
| Scientific simulations, theory notes, formal proofs, typeset and plain-text proof companions, and listed project tooling | Apache License 2.0 | Commercial use is permitted under the license terms, including attribution, change-notice, and its express contributor patent-license provisions. No royalty is charged for use of these covered files under Apache-2.0. |
| `app/`, `rust/`, `evm/`, `solana/`, the Tri-Token blueprint, `whitepapers/`, `cpp/`, `rcod/`, and `omni-bridge/` | All rights reserved; `LicenseRef-Omega-Product-Proprietary` | No new general permission to copy, modify, distribute, or commercially exploit these materials is granted by this repository version. Commercial licenses may be negotiated under a separate signed agreement with a percentage-based royalty. |
| Project administration and legal notices | No separate product or source-code license | These documents explain the policy; they do not expand the grants above or license the product architecture. |

## Exact path scope

### Apache-2.0 scientific materials

The Apache-2.0 grant applies to these repository paths, to the extent the rights holder is authorized to license them:

- `Sim1_Emergent_Geometry.py`, `Sim2_Cosmology.py`, `Sim3_Dynamic_Scale.py`, `Sim4_Evolution.py`, `Sim5_Emergent_Gravity.py`, and `sim6_v14_depletion.py`;
- `Omega_Theory_Laymans_Guide.md` and `Omega_Theory_v4.0_Technical.md`;
- all files in `lean_proofs/`, `latex_docs/`, and `txt_proofs/`;
- `README.md`, `requirements.txt`, `update_discovery.sh`, `.gitignore`, and `.github/`.

The full license is [`LICENSES/Apache-2.0.txt`](../LICENSES/Apache-2.0.txt). The license's patent grant is limited to patent claims licensable by a contributor and necessarily infringed by the covered contribution or its combination with the work, as described in Apache-2.0 Section 3. The license does not grant trademark rights. Third-party content and dependencies are not relicensed by this list.

### All-rights-reserved product materials

The product scope is `app/**`, `rust/**`, `evm/**`, `solana/**`, `whitepapers/**`, `tri_token_sovereign_economy_blueprint.md`, `cpp/**`, `rcod/**`, and `omni-bridge/**`. These files carry or inherit the proprietary identifier `LicenseRef-Omega-Product-Proprietary`; see [`LICENSES/Omega-Product-Proprietary.txt`](../LICENSES/Omega-Product-Proprietary.txt).

The intended commercial route is an **advance, signed written license with percentage-based compensation payable to Jacob See**. This repository does not set a rate, royalty base (for example, gross or net receipts), minimum, term, reporting/audit terms, territory, sublicensing rights, or other deal terms. Those must be negotiated and written into the separate agreement. Until a suitable agreement is signed, this notice grants no commercial-use permission. Do not infer permission from the fact that the files are publicly viewable or downloadable.

For a commercial licensing inquiry, contact the rights holder through [github.com/jakesdev1991](https://github.com/jakesdev1991). Do not post confidential deal terms in a public issue. No email address is stated here because none has been verified for publication.

## Important limits on the royalty goal

Copyright generally protects original expression, not abstract ideas, facts, methods, systems, processes, or concepts. This repository can reserve rights in the actual text, illustrations, and code to the extent they are copyrightable and owned, but this notice alone cannot require royalties merely because someone independently implements an idea described in a public paper. Patent, trademark, contract, or other rights may have different requirements and limitations. Have counsel evaluate what can actually be licensed and enforced before promising exclusivity or royalties.

## Prior MIT license — material limit on this change

The public baseline commit [`c717dc5`](https://github.com/jakesdev1991/Omega_Theory_Everything/commit/c717dc53bd20db4a6290734f09a3db5bb93fa287) included a root MIT license alongside the science files, app, Rust prototype, whitepapers, and blueprint. The exact prior license text is retained at [`LICENSES/MIT-legacy.txt`](../LICENSES/MIT-legacy.txt).

This licensing change is prospective. It cannot retroactively withdraw the permissions already granted to recipients of copies distributed under the prior MIT-licensed version. The MIT license allowed commercial use, modification, and redistribution subject to retaining its notice; those earlier rights do not acquire a royalty because this repository now states a different policy. This means the new royalty model cannot guarantee control over use of the already-published MIT version. Removing files now would not revoke rights already granted. Get legal advice before representing the product materials as exclusively controlled or royalty-bearing.

## Public disclosure and patents

The repository and product materials are already publicly accessible on GitHub. Public disclosure can affect patent rights, with consequences that vary by jurisdiction and facts; this repository cannot undo disclosure. No patent-pending status is represented here. See [`PATENT-POSTURE.md`](PATENT-POSTURE.md) and [`PROVENANCE.md`](PROVENANCE.md), and consult patent counsel before any further disclosure or patent-related claim.

## Third-party material, contributions, and ownership

The grants above apply only to rights the rights holder owns or is authorized to license. They do not change third-party licenses. The provenance record is incomplete, and the local checkout contains only shallow Git history. No signed contributor license agreement is recorded in this repository. See [`PROVENANCE.md`](PROVENANCE.md) and [`../CONTRIBUTING.md`](../CONTRIBUTING.md) before accepting contributions or making ownership claims.
