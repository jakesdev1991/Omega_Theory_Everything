# Omega App Store — Terms of Use & End-User License

**Terms ID:** `omega-store-eula-1.0` · **Status:** DRAFT pending legal review · **Last updated:** 2026-09-25

> This draft was prepared as a project template. It is not legal advice and has not yet been reviewed by a licensed attorney. Items in [square brackets] must be completed, and the whole text reviewed by counsel, before it is relied on for paid licenses.

These terms govern your use of the Omega App Store storefront (the `/store` web page), the apps listed in it, the mobile execution node that runs those apps, and any license you receive for them (together, the **Store**). The Store is operated by **Jacob See** ("**we**", "**us**", the **Store Operator**). By checking the acceptance box in the storefront, obtaining or using a license, or sending a job request to a listed app, you ("**you**", the **Licensee**) agree to these terms. If you do not agree, do not use the Store.

## 1. Who may use the Store

You must be at least 18 years old, or the age of majority where you live, and able to enter a binding contract. If you use the Store for an organization, you confirm that you are authorized to bind it, and "you" includes that organization. You may not use the Store where doing so is prohibited by law, including applicable sanctions and export-control laws.

## 2. What you get: a limited license

**2.1 License grant.** For each app you hold a valid license for, we (or the app's publisher, through us) grant you a limited, personal, non-exclusive, non-transferable, non-sublicensable, revocable license to submit job requests to that app through the Store and to use the results it returns. The license lasts for the tier and term shown in your license record. Some apps are listed as operator-only; those can only be run by keys the node operator has authorized.

**2.2 License records.** Each license is a signed Nostr event (kind 31335) issued by the store key to your public key (npub). It records the app, tier, terms version, expiry (if any) and status. That signed record, together with any newer version of it, is the authoritative statement of your license. It is bound to your public key. Anyone who controls the matching private key (nsec) can use it.

**2.3 Tiers and trials.** Tiers (for example trial, standard or pro) may differ in which apps, features, limits or terms they include, as described in the listing. Trial licenses may be limited in time or scope and may end without notice.

**2.4 What is not licensed.** The license covers running the app through the Store. It does **not** give you any right to the app's source code, algorithms, models, the storefront or node software, documentation, or our names and marks. It does not transfer ownership of anything. Unless a listing expressly states an open-source license for particular materials, all apps and Store software are proprietary (`LicenseRef-Omega-Product-Proprietary`), and all rights not expressly granted are reserved.

## 3. Restrictions

You will not, and will not help anyone else to:

1. share, sell, rent, lend, transfer or publish your license or private key so that others can use a license issued to you;
2. copy, modify, or create derivative works of the apps or the Store software, or reverse engineer, decompile or disassemble them, except to the extent that applicable law expressly allows this despite this restriction;
3. circumvent, disable or tamper with licensing, signatures, rate limits, sandboxing, or other technical protections, or forge, replay or alter license or job events;
4. submit job requests that are unlawful, infringing, malicious, or designed to harm, overload, probe or exploit the node, relays, or other users;
5. use the Store or its outputs to build a competing store or service, or to benchmark it for publication, without our written consent;
6. use results in any way that violates law or the rights of others; or
7. remove or obscure any proprietary, copyright or license notice.

## 4. Your keys and the Nostr network

**4.1 Self-custody.** You control your private key. We never receive it and cannot recover it. If you lose it, you lose access to the licenses issued to it. We may, but do not have to, reissue a license to a new key when you ask. If your key is compromised, tell us promptly so we can revoke the affected licenses.

**4.2 Public data.** Nostr relays are public, third-party systems. **Your license records and your job requests, including the parameters you submit, are published as public events.** They may be copied, stored and indexed by anyone, indefinitely. Do not submit personal data, confidential information, or anything you are not entitled to share. We do not control relays and cannot delete events from them.

**4.3 Local storage.** The storefront keeps your operator key and your terms acceptance in your browser's local storage on your device. Clearing browser data deletes them.

## 5. Fees and payment

Licenses may be free, trial, or paid. When a listing states a price, the price, payment method, currency and refund policy shown at the time you obtain the license apply. [Refund policy to be completed before paid licenses are offered.] You are responsible for any taxes that apply to your purchase, other than taxes on our income. No license, job result or work receipt is an investment, security, deposit, or claim on any token or asset. Any test-network tokens shown in the Store (for example tTWC) have no monetary value.

## 6. Suspension, revocation and termination

We may suspend or revoke a license, by publishing a newer license record with status `revoked`, or refuse job requests if:

- you breach these terms;
- we must do so to comply with law or a legal order;
- your key appears to be compromised; or
- the app is withdrawn by its publisher or by us.

For paid licenses revoked for reasons other than your breach, we will offer a pro-rated refund or an equivalent license [subject to counsel review]. You may stop using the Store at any time. Sections 2.4, 3, 4.2, 5 (for amounts owed), 7, 8, 9 and 11 survive termination.

## 7. Availability and results — no warranty

The Store depends on volunteer relays, network connectivity, and a mobile execution node that may be offline, slow, or restarted at any time. **To the maximum extent permitted by law, the Store, the apps, and all results are provided "AS IS" and "AS AVAILABLE", without warranties of any kind, express or implied, including merchantability, fitness for a particular purpose, accuracy, and non-infringement.** Scientific simulations and audits produce computational outputs that may be incomplete or wrong. They are not professional, financial, medical, engineering or legal advice. You are responsible for checking any result before you rely on it.

## 8. Limitation of liability

To the maximum extent permitted by law:

- neither we nor any app publisher will be liable for any indirect, incidental, special, consequential, exemplary or punitive damages, or for lost profits, revenue, data or goodwill, arising out of or relating to the Store; and
- our total liability for all claims relating to the Store will not exceed the greater of the amount you paid us for the relevant license in the 12 months before the claim, or USD $50.

Some jurisdictions do not allow certain exclusions or limits, so some of the above may not apply to you. Nothing in these terms limits liability that cannot be limited by law.

## 9. Indemnity

You will defend and indemnify us and the app publishers against third-party claims arising from your breach of these terms, or your unlawful use of the Store or its results, to the extent permitted by law.

## 10. Changes to these terms

We may update these terms. Each material version has a new terms ID (for example `omega-store-eula-1.1`). Each license records the terms ID in force when it was issued. An app may require you to accept the current terms before a license is issued or renewed, and the storefront will ask you to do so. The terms that apply to an existing license are those recorded in it, unless you accept a newer version.

## 11. General

- **Governing law and venue:** [State/country to be completed by counsel], without regard to conflict-of-laws rules. [Dispute-resolution clause to be completed by counsel.]
- **Publisher terms:** a listing may add app-specific terms. If they conflict with these terms, these terms control, unless the app-specific terms give you more rights.
- **Entire agreement:** these terms, your license records, and any app-specific terms are the entire agreement between you and us about the Store.
- **Severability:** if a provision is unenforceable, the rest remains in effect.
- **No waiver:** failing to enforce a provision is not a waiver of it.
- **Assignment:** you may not assign these terms. We may assign them in connection with a reorganization or a transfer of the Store.
- **Trademarks:** project names such as Omega Theory™, C.A.R.E.™ and Lucifer–Hermes Omni-Bridge Prime™ are claimed marks and are not licensed to you (see `docs/TRADEMARKS.md`).

**Contact:** licensing questions and revocation requests go through [github.com/jakesdev1991](https://github.com/jakesdev1991). Do not post confidential information in public issues. [A dedicated contact address will be added once verified.]
