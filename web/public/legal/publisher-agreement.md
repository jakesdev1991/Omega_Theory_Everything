# Omega App Store — Publisher Agreement

**Agreement ID:** `omega-store-publisher-1.0` · **Status:** DRAFT template pending legal review · **Last updated:** 2026-09-25

> This is a template for listing third-party apps in the Omega App Store. It is not legal advice, and it is **not in force for anyone until it is completed and signed in writing by both parties**. Publishing it here does not grant any rights. Items in [square brackets] must be negotiated or completed, and the whole text reviewed by counsel, before signature.

This Publisher Agreement (the **Agreement**) is between **Jacob See** (the **Store Operator**) and the party identified in the signature block (the **Publisher**). It governs how the Publisher's software, algorithms or services (each an **App**) are listed, licensed and run through the Omega App Store (the **Store**).

## 1. Listing and distribution

**1.1 Appointment.** The Publisher appoints the Store Operator, on a non-exclusive basis unless Schedule A says otherwise, to list the Apps in the Store directory. The Store Operator will issue licenses to end users under the Store Terms of Use & End-User License (`omega-store-eula-1.0` or its successor, the **Store Terms**) and will route licensed job requests to the execution node that runs the App.

**1.2 License to the Store Operator.** The Publisher grants the Store Operator a worldwide, non-exclusive, royalty-free license, for the term of this Agreement, to:

- reproduce, install and run the Apps on Store execution nodes;
- display the App names, descriptions, icons and screenshots in the directory and in promotional material for the Store; and
- sublicense end users' right to use the Apps under the Store Terms.

**1.3 Directory mechanics.** Listings are kind 31990/30017 Nostr events signed by the store root key. Licenses are kind 31335 events signed by the store key, issued on the Publisher's behalf. The Store Operator controls these keys. The Publisher may ask for a listing to be added, changed or withdrawn, and the Store Operator will act on the request within [5] business days.

## 2. Ownership

The Publisher keeps all ownership of the Apps. The Store Operator keeps all ownership of the Store: the storefront, mobile-node software, licensing protocol, directory and brand. Neither party acquires rights in the other's materials except as expressly granted here. Feedback may be used freely by the recipient without obligation.

## 3. End-user licensing

**3.1** End users receive licenses under the Store Terms. The Publisher may add app-specific terms in the listing. Those terms may not conflict with the Store Terms except to give users more rights.

**3.2** The Publisher sets the tiers, prices, trial terms and license duration for each App within the limits the Store supports (Schedule A). The Store Operator may refuse or suspend a tier that is unlawful or that the Store cannot technically enforce.

**3.3** The Store Operator may revoke end-user licenses as the Store Terms allow. If the Publisher withdraws an App, licenses already issued stay valid until they expire, or [30] days after withdrawal if they have no expiry, unless the law or a security issue requires otherwise. [Refund allocation for withdrawn Apps to be agreed.]

## 4. Fees and revenue share

**4.1 Compensation to the Store Operator.** In return for distribution, licensing and execution, the Publisher will pay the Store Operator **[__]% of Gross Receipts**. Alternatively, where the Store Operator collects payment, the Store Operator will keep that percentage and remit the balance. "Gross Receipts" means all amounts paid by end users for licenses to, or use of, the Apps through the Store, less only refunds, chargebacks, and taxes collected from end users and paid to authorities. [Rate, base, minimum and currency to be negotiated. Consistent with the project's licensing policy, compensation is percentage-based and payable to Jacob See.]

**4.2 Payment rails.** Payment methods (for example Lightning/zaps, the C.A.R.E. token rails, or off-chain invoices) are listed in Schedule A. The license record's `payment` tag may reference the payment. No payment rail is operational until it is specified in Schedule A.

**4.3 Reporting and audit.** The party that collects payment will provide [monthly] statements within [15] days after the end of each period and pay amounts due within [30] days. Each party may audit the other's relevant records once per year on [30] days' notice, at its own cost. If the audit shows an underpayment of more than 5%, the underpaying party bears the audit cost.

**4.4 Taxes.** Each party is responsible for its own income taxes. [Withholding and VAT/GST treatment to be completed by counsel.]

## 5. Publisher obligations and warranties

The Publisher represents, warrants and agrees that:

1. it owns or has all rights needed to grant the licenses in this Agreement, and the Apps and listing materials do not infringe or misappropriate any third-party right;
2. it has disclosed every open-source or third-party component in the Apps, with its license, and none of those licenses requires the Store software to be disclosed or licensed on different terms;
3. the Apps contain no malware, backdoors, cryptominers, data exfiltration, or undisclosed network access, and they run within the node's sandbox rules: argv-only execution, parameters on stdin, time and output limits, and no secrets in the environment;
4. listing descriptions are accurate and not misleading, and the Publisher will keep them current;
5. the Apps comply with applicable law, including export control, sanctions, consumer-protection and privacy law. The Publisher will not design an App that asks users for personal data, because job parameters are published on public relays;
6. the Publisher will promptly report, and fix within a reasonable time, any security vulnerability that affects an App; and
7. the Publisher will not represent any Store output, receipt or token as an investment or as having monetary value unless that is lawful and approved in writing by the Store Operator.

## 6. Review, suspension and removal

The Store Operator may review Apps before and after listing. It may suspend or delist an App immediately if it reasonably believes the App:

- breaches this Agreement or the law;
- creates a security risk to the node, relays or users; or
- is the subject of a credible infringement claim.

It will notify the Publisher with reasons where it lawfully can.

## 7. Confidentiality

Each party will protect the other's non-public information disclosed under this Agreement with reasonable care, and use it only to perform this Agreement, for [3] years after disclosure (trade secrets: for as long as they remain trade secrets). Standard exclusions apply: information that is public, already known, independently developed, or rightfully received from a third party. Information may be disclosed where the law requires it. **Nothing published to Nostr relays is confidential.**

## 8. Indemnities

**8.1** The Publisher will defend and indemnify the Store Operator against third-party claims that an App or its listing materials infringe or misappropriate any right, or that result from the Publisher's breach of Section 5.

**8.2** The Store Operator will defend and indemnify the Publisher against third-party claims that the Store software (not the App) infringes a third party's copyright.

**8.3** The indemnified party must give prompt notice, allow the indemnifying party to control the defence, and cooperate reasonably.

## 9. Limitation of liability

Except for indemnity obligations, breach of confidentiality, or amounts owed under Section 4:

- neither party is liable for indirect, incidental, special, consequential or punitive damages, or for lost profits; and
- each party's total liability is limited to the amounts paid or payable to the Store Operator under this Agreement in the 12 months before the claim. [Cap to be reviewed by counsel.]

## 10. Term and termination

This Agreement starts when signed and continues for [one year], renewing automatically for successive one-year terms unless either party gives [60] days' notice. Either party may terminate on [30] days' written notice for material breach not cured within that period, or immediately if the other party becomes insolvent. Sections 2, 3.3, 4 (for amounts accrued), 7, 8, 9 and 11 survive.

## 11. General

- **Governing law and venue:** [to be completed by counsel].
- **Independent contractors:** the parties are independent contractors. There is no partnership, joint venture, agency or employment relationship.
- **Assignment:** neither party may assign this Agreement without consent, except that the Store Operator may assign it in connection with a transfer of the Store.
- **Entire agreement:** this Agreement, its Schedules and the Store Terms are the entire agreement between the parties on this subject. **Amendments must be in writing and signed by both parties.** A pull request, issue, Nostr event or chat message is not a signed writing.
- **Notices:** in writing to the addresses in the signature block.
- **Trademarks:** each party may use the other's names only to identify the App and the Store accurately. No other trademark license is granted (see `docs/TRADEMARKS.md`).

## Schedule A — Commercial terms (to be completed per Publisher)

| Item | Value |
|---|---|
| Apps (app ids / `d` tags) | [ ] |
| Exclusivity | Non-exclusive [or specify] |
| Tiers, prices, trial terms | [ ] |
| Store Operator revenue share | [__]% of Gross Receipts |
| Payment rails and collecting party | [ ] |
| Reporting period / payment terms | [monthly / 30 days] |
| App-specific end-user terms | [none / attach] |
| Execution node(s) and resource limits | [ ] |

## Signatures

| | Store Operator | Publisher |
|---|---|---|
| Name | Jacob See | [ ] |
| Signature | | |
| Date | | |
| Notice address | [ ] | [ ] |
