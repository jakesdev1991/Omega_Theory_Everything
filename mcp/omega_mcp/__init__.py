# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Omega MCP Hub — in-memory simulation prototype for the tri-token economy concepts.

Planes:
  - SOV  : simulated balances and accounting events
  - USE  : simulated non-transferable receipt records
  - CARE : proposed claims, labeled attestations, and appeals
  - AMITY: simulated entitlements and balances
  - OMEGA: simulated balances, locks, proposals, and votes

Mutations emit versioned audit metadata and snapshots expose current state. Event
payloads are not a complete replay log. No real keys, chain, services, or value
are involved; tool actions are not independently verified.

Run (from the repository root):
  uv run --directory mcp python -m omega_mcp
or, with the package installed in the active environment:
  python -m omega_mcp
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from mcp.server import FastMCP

# ---------------------------------------------------------------------------
# Type aliases / helpers
# ---------------------------------------------------------------------------

Snapshot = dict[str, Any]
Event = dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _eid() -> str:
    return str(uuid.uuid4())


def _require(condition: bool, message: str) -> None:
    """Enforce simulation invariants even when Python optimization is enabled."""
    if not condition:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# Minimal in-memory state
# ---------------------------------------------------------------------------


class OmegaState:
    """Append-only in-memory ledger simulation for the prototype."""

    def __init__(self) -> None:
        self.snapshots: dict[str, Snapshot] = {}
        self.events: list[Event] = []
        self.next_version: int = 1

        # ---- SOV plane ----
        self.sov_balances: dict[str, int] = defaultdict(int)
        self.sov_supply: int = 0

        # ---- USE plane (receipts keyed by receipt_id) ----
        self.use_receipts: dict[str, dict[str, Any]] = {}

        # ---- CARE plane ----
        self.care_claims: dict[str, dict[str, Any]] = {}
        self.care_attestations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.care_appeals: dict[str, list[dict[str, Any]]] = defaultdict(list)

        # ---- AMITY plane ----
        self.amity_balances: dict[str, int] = defaultdict(int)
        self.amity_supply: int = 0
        self.amity_entitlements: dict[str, dict[str, Any]] = {}

        # ---- OMEGA plane ----
        self.omega_balances: dict[str, int] = defaultdict(int)
        self.omega_supply: int = 0
        self.omega_locks: dict[str, dict[str, Any]] = {}
        self.omega_proposals: dict[str, dict[str, Any]] = {}
        self.omega_votes: dict[str, dict[str, Any]] = defaultdict(dict)

    # ---- versioning / event log ----

    def emit(self, plane: str, action: str, payload: dict[str, Any]) -> Event:
        e: Event = {
            "event_id": _eid(),
            "plane": plane,
            "action": action,
            "version": self.next_version,
            "timestamp": _now_iso(),
            "payload": payload,
        }
        self.events.append(e)
        self.next_version += 1
        return e

    def snapshot(self, name: str, data: Snapshot) -> None:
        self.snapshots[name] = {"name": name, "version": self.next_version - 1, **data}

    # ---- SOV ----

    def sov_mint(self, to: str, amount: int, reason: str) -> Snapshot:
        _require(amount > 0, "amount must be positive")
        self.sov_balances[to] += amount
        self.sov_supply += amount
        self.emit("SOV", "mint", {"to": to, "amount": amount, "reason": reason})
        return self.sov_snapshot()

    def sov_burn(self, from_address: str, amount: int, reason: str) -> Snapshot:
        _require(amount > 0, "amount must be positive")
        _require(self.sov_balances.get(from_address, 0) >= amount, "insufficient SOV")
        self.sov_balances[from_address] -= amount
        self.sov_supply -= amount
        self.emit(
            "SOV", "burn", {"from": from_address, "amount": amount, "reason": reason}
        )
        return self.sov_snapshot()

    def sov_transfer(self, from_address: str, to: str, amount: int) -> Snapshot:
        _require(amount > 0, "amount must be positive")
        _require(self.sov_balances.get(from_address, 0) >= amount, "insufficient SOV")
        self.sov_balances[from_address] -= amount
        self.sov_balances[to] += amount
        self.emit("SOV", "transfer", {"from": from_address, "to": to, "amount": amount})
        return self.sov_snapshot()

    def sov_snapshot(self) -> Snapshot:
        return {
            "balances": dict(self.sov_balances),
            "supply": self.sov_supply,
            "version": self.next_version - 1,
        }

    # ---- USE ----

    def use_issue(
        self,
        contributor: str,
        work_category: str,
        quantity: int,
        evidence_commitment: str,
        attestations: int = 1,
    ) -> Snapshot:
        _require(quantity > 0, "quantity must be positive")
        _require(attestations >= 1, "at least one attestation required")
        receipt_id = _eid()
        receipt: dict[str, Any] = {
            "receipt_id": receipt_id,
            "contributor": contributor,
            "work_category": work_category,
            "quantity": quantity,
            "evidence_commitment": evidence_commitment,
            "attestation_count": attestations,
            "issued_at": _now_iso(),
            "revoked": False,
        }
        self.use_receipts[receipt_id] = receipt
        self.emit(
            "USE",
            "issue",
            {
                "receipt_id": receipt_id,
                "contributor": contributor,
                "work_category": work_category,
                "quantity": quantity,
                "attestation_count": attestations,
            },
        )
        return self.use_snapshot()

    def use_revoke(self, receipt_id: str, reason: str) -> Snapshot:
        r = self.use_receipts.get(receipt_id)
        if r is None:
            raise ValueError("receipt not found")
        _require(not r["revoked"], "already revoked")
        r["revoked"] = True
        r["revoked_at"] = _now_iso()
        r["revoked_reason"] = reason
        self.emit("USE", "revoke", {"receipt_id": receipt_id, "reason": reason})
        return self.use_snapshot()

    def use_snapshot(self) -> Snapshot:
        return {
            "receipts": list(self.use_receipts.values()),
            "version": self.next_version - 1,
        }

    # ---- CARE ----

    def care_submit(
        self,
        participant: str,
        service_category: str,
        consent_scope: str,
        evidence_commitment: str,
    ) -> Snapshot:
        claim_id = _eid()
        claim: dict[str, Any] = {
            "claim_id": claim_id,
            "participant": participant,
            "service_category": service_category,
            "consent_scope": consent_scope,
            "evidence_commitment": evidence_commitment,
            "status": "proposed",
            "attestations": [],
            "created_at": _now_iso(),
        }
        self.care_claims[claim_id] = claim
        self.emit(
            "CARE",
            "submit",
            {
                "claim_id": claim_id,
                "participant": participant,
                "service_category": service_category,
                "consent_scope": consent_scope,
            },
        )
        return self.care_snapshot()

    def care_attest(self, claim_id: str, verifier: str, reason: str) -> Snapshot:
        c = self.care_claims.get(claim_id)
        if c is None:
            raise ValueError("claim not found")
        _require(c["status"] == "proposed", "claim is not open for attestation")
        verifier_label = verifier.strip()
        _require(bool(verifier_label), "verifier is required")
        normalized_verifier = verifier_label.casefold()
        _require(
            all(
                item["verifier"].casefold() != normalized_verifier
                for item in c["attestations"]
            ),
            "verifier has already attested this claim",
        )
        a = {
            "verifier": verifier_label,
            "reason": reason,
            "attested_at": _now_iso(),
        }
        self.care_attestations[claim_id].append(a)
        c["attestations"].append(a)
        if len(c["attestations"]) >= 2:
            c["status"] = "final"
        self.emit(
            "CARE",
            "attest",
            {
                "claim_id": claim_id,
                "verifier": verifier_label,
                "attestation_count": len(c["attestations"]),
                "status": c["status"],
            },
        )
        return self.care_snapshot()

    def care_appeal(self, claim_id: str, appellant: str, reason: str) -> Snapshot:
        c = self.care_claims.get(claim_id)
        if c is None:
            raise ValueError("claim not found")
        _require(bool(c["attestations"]), "claim must have at least one attestation")
        _require(c["status"] != "appealed", "claim is already under appeal")
        appeal_id = _eid()
        app: dict[str, Any] = {
            "appeal_id": appeal_id,
            "claim_id": claim_id,
            "appellant": appellant,
            "reason": reason,
            "filed_at": _now_iso(),
            "status": "open",
        }
        self.care_appeals[claim_id].append(app)
        c["status"] = "appealed"
        self.emit("CARE", "appeal", {"appeal_id": appeal_id, "claim_id": claim_id})
        return self.care_snapshot()

    def care_snapshot(self) -> Snapshot:
        claims = list(self.care_claims.values())
        return {
            "claims": claims,
            "attestation_counts": {
                k: len(v) for k, v in self.care_attestations.items()
            },
            "appeals": {k: len(v) for k, v in self.care_appeals.items()},
            "version": self.next_version - 1,
        }

    # ---- AMITY ----

    def amity_entitle(
        self, from_claim_id: str, to: str, amount: int, policy: str
    ) -> Snapshot:
        _require(amount > 0, "amount must be positive")
        claim = self.care_claims.get(from_claim_id)
        if claim is None:
            raise ValueError("CARE claim not found")
        _require(claim["status"] == "final", "CARE claim must be finalized")
        _require(
            not any(
                item["source_claim_id"] == from_claim_id
                for item in self.amity_entitlements.values()
            ),
            "CARE claim has already been converted",
        )
        eid = _eid()
        entitlement: dict[str, Any] = {
            "entitlement_id": eid,
            "source_claim_id": from_claim_id,
            "holder": to,
            "amount": amount,
            "policy": policy,
            "created_at": _now_iso(),
        }
        self.amity_entitlements[eid] = entitlement
        self.amity_balances[to] += amount
        self.amity_supply += amount
        self.emit(
            "AMITY",
            "entitle",
            {
                "entitlement_id": eid,
                "source_claim_id": from_claim_id,
                "to": to,
                "amount": amount,
                "policy": policy,
            },
        )
        return self.amity_snapshot()

    def amity_transfer(self, from_address: str, to: str, amount: int) -> Snapshot:
        _require(amount > 0, "amount must be positive")
        _require(
            self.amity_balances.get(from_address, 0) >= amount, "insufficient AMITY"
        )
        self.amity_balances[from_address] -= amount
        self.amity_balances[to] += amount
        self.emit(
            "AMITY", "transfer", {"from": from_address, "to": to, "amount": amount}
        )
        return self.amity_snapshot()

    def amity_snapshot(self) -> Snapshot:
        return {
            "balances": dict(self.amity_balances),
            "supply": self.amity_supply,
            "entitlements": list(self.amity_entitlements.values()),
            "version": self.next_version - 1,
        }

    # ---- OMEGA ----

    def omega_mint(
        self, to: str, amount: int, reason: str, protocol_version: str = "0.1.0"
    ) -> Snapshot:
        _require(amount > 0, "amount must be positive")
        self.omega_balances[to] += amount
        self.omega_supply += amount
        self.emit(
            "OMEGA",
            "mint",
            {
                "to": to,
                "amount": amount,
                "reason": reason,
                "protocol_version": protocol_version,
            },
        )
        return self.omega_snapshot()

    def omega_lock(
        self, owner: str, amount: int, duration_blocks: int, reason: str
    ) -> Snapshot:
        _require(amount > 0, "amount must be positive")
        _require(duration_blocks > 0, "duration_blocks must be positive")
        _require(self.omega_balances.get(owner, 0) >= amount, "insufficient OMEGA")
        self.omega_balances[owner] -= amount
        lock_id = _eid()
        lock: dict[str, Any] = {
            "lock_id": lock_id,
            "owner": owner,
            "amount": amount,
            "duration_blocks": duration_blocks,
            "locked_at": _now_iso(),
            "reason": reason,
            "weight": amount * max(1, duration_blocks),
            "active": True,
        }
        self.omega_locks[lock_id] = lock
        self.emit(
            "OMEGA",
            "lock",
            {
                "lock_id": lock_id,
                "owner": owner,
                "amount": amount,
                "duration_blocks": duration_blocks,
            },
        )
        return self.omega_snapshot()

    def omega_proposal(
        self,
        proposer: str,
        change_set: dict[str, Any],
        simulation_ref: str,
        rollback_plan: str,
    ) -> Snapshot:
        prop_id = _eid()
        prop: dict[str, Any] = {
            "proposal_id": prop_id,
            "proposer": proposer,
            "change_set": change_set,
            "simulation_ref": simulation_ref,
            "rollback_plan": rollback_plan,
            "status": "draft",
            "created_at": _now_iso(),
        }
        self.omega_proposals[prop_id] = prop
        self.emit(
            "OMEGA", "proposal_draft", {"proposal_id": prop_id, "proposer": proposer}
        )
        return self.omega_snapshot()

    def omega_vote(
        self, proposal_id: str, voter: str, support: bool, weight: int
    ) -> Snapshot:
        _require(weight > 0, "weight must be positive")
        p = self.omega_proposals.get(proposal_id)
        if p is None:
            raise ValueError("proposal not found")
        _require(
            p["status"] not in ("activated", "rejected"), "proposal already resolved"
        )
        p["status"] = "vote"
        self.omega_votes[proposal_id][voter] = {
            "support": support,
            "weight": weight,
            "voted_at": _now_iso(),
        }
        self.emit(
            "OMEGA",
            "vote",
            {
                "proposal_id": proposal_id,
                "voter": voter,
                "support": support,
                "weight": weight,
            },
        )
        return self.omega_snapshot()

    def omega_snapshot(self) -> Snapshot:
        return {
            "balances": dict(self.omega_balances),
            "supply": self.omega_supply,
            "locks": list(self.omega_locks.values()),
            "proposals": list(self.omega_proposals.values()),
            "version": self.next_version - 1,
        }

    # ---- house ----

    def ledger(self) -> Snapshot:
        return {
            "sov": self.sov_snapshot(),
            "use": self.use_snapshot(),
            "care": self.care_snapshot(),
            "amity": self.amity_snapshot(),
            "omega": self.omega_snapshot(),
            "event_count": len(self.events),
            "next_version": self.next_version,
        }

    def events_since(self, version: int) -> list[Event]:
        return [e for e in self.events if e["version"] > version]

    def full_event_log(self) -> list[Event]:
        return list(self.events)


# ---------------------------------------------------------------------------
# Tool descriptions used by both FastMCP and the manifest
# ---------------------------------------------------------------------------

TOOL_DESCRIPTIONS = {
    # SOV — simulated accounting only
    "sov_mint": "Add simulated SOV units to an address in this process's in-memory ledger. Requires a positive amount and reason; no real chain or value is involved.",
    "sov_burn": "Remove simulated SOV units from an address in the in-memory ledger. Requires a positive amount and sufficient balance.",
    "sov_transfer": "Move simulated SOV units between addresses in the in-memory ledger. Requires a positive amount and sufficient sender balance.",
    "sov_snapshot": "Inspect current in-memory SOV balances, simulated supply, and version.",
    # USE — simulated receipts only
    "use_issue": "Create a simulated, non-transferable proof-of-useful-work receipt. Requires a category, positive quantity, evidence commitment, and at least one claimed attestation.",
    "use_revoke": "Mark a simulated USE receipt revoked by ID with a reason; the audit event is append-only.",
    "use_snapshot": "Inspect simulated USE receipts and their claimed attestation counts.",
    # CARE — simulated claims only
    "care_submit": "Create a proposed Proof-of-Care claim in memory. This does not verify evidence or grant a real service or benefit.",
    "care_attest": "Add a simulated CARE attestation. Two distinct verifier labels move the claim to final; this is not cryptographic or independent verification.",
    "care_appeal": "File an in-memory appeal on a CARE claim with at least one attestation.",
    "care_snapshot": "Inspect in-memory CARE claims, attestation counts, and open appeals.",
    # AMITY — simulated entitlements only
    "amity_entitle": "Create simulated AMITY units from a finalized CARE claim under a policy label. The claim must exist and the amount must be positive; no real value is issued.",
    "amity_transfer": "Move simulated AMITY units between addresses in memory. Requires a positive amount and sufficient sender balance.",
    "amity_snapshot": "Inspect in-memory AMITY balances, simulated supply, and entitlements.",
    # OMEGA — simulated governance only
    "omega_mint": "Add simulated OMEGA units to an address with a reason and protocol version; no real token is issued.",
    "omega_lock": "Lock simulated OMEGA units for a positive duration in blocks; returns a prototype governance weight.",
    "omega_proposal": "Draft an in-memory governance proposal with a change set, simulation reference, and rollback plan.",
    "omega_vote": "Record a simulated vote on a proposal with voter, support flag, and positive weight.",
    "omega_snapshot": "Inspect in-memory OMEGA balances, simulated supply, locks, and proposals.",
    # house — snapshots and audit metadata
    "ledger": "Current in-memory simulation snapshot across all planes, plus event count and next version.",
    "events_since": "Versioned audit events after a given version; payloads are not a complete state-replay log.",
    "full_event_log": "Inspect the append-only audit events emitted by this process; they are not a complete state-replay log.",
}


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="omega-hub",
    instructions=(
        "Prototype simulation only. All state is in memory; no real chain, keys, "
        "services, or value are involved. Tool actions are not independently verified. "
        "The HTTP transport has no built-in authentication and defaults to loopback; "
        "do not expose it to untrusted networks."
    ),
)

_state = OmegaState()


def _documented_tool(function: Any) -> Any:
    """Register a tool with its shared, explicit MCP description."""
    return mcp.tool(description=TOOL_DESCRIPTIONS[function.__name__])(function)


# ---- SOV tools ----


@_documented_tool
def sov_mint(to: str, amount: int, reason: str) -> dict[str, Any]:
    try:
        snap = _state.sov_mint(to=to, amount=amount, reason=reason)
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def sov_burn(from_address: str, amount: int, reason: str) -> dict[str, Any]:
    try:
        snap = _state.sov_burn(from_address=from_address, amount=amount, reason=reason)
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def sov_transfer(from_address: str, to: str, amount: int) -> dict[str, Any]:
    try:
        snap = _state.sov_transfer(from_address=from_address, to=to, amount=amount)
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def sov_snapshot() -> dict[str, Any]:
    return {"ok": True, "snapshot": _state.sov_snapshot()}


# ---- USE tools ----


@_documented_tool
def use_issue(
    contributor: str,
    work_category: str,
    quantity: int,
    evidence_commitment: str,
    attestations: int = 1,
) -> dict[str, Any]:
    try:
        snap = _state.use_issue(
            contributor=contributor,
            work_category=work_category,
            quantity=quantity,
            evidence_commitment=evidence_commitment,
            attestations=attestations,
        )
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def use_revoke(receipt_id: str, reason: str) -> dict[str, Any]:
    try:
        snap = _state.use_revoke(receipt_id=receipt_id, reason=reason)
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def use_snapshot() -> dict[str, Any]:
    return {"ok": True, "snapshot": _state.use_snapshot()}


# ---- CARE tools ----


@_documented_tool
def care_submit(
    participant: str,
    service_category: str,
    consent_scope: str,
    evidence_commitment: str,
) -> dict[str, Any]:
    snap = _state.care_submit(
        participant=participant,
        service_category=service_category,
        consent_scope=consent_scope,
        evidence_commitment=evidence_commitment,
    )
    return {"ok": True, "snapshot": snap}


@_documented_tool
def care_attest(claim_id: str, verifier: str, reason: str) -> dict[str, Any]:
    try:
        snap = _state.care_attest(claim_id=claim_id, verifier=verifier, reason=reason)
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def care_appeal(claim_id: str, appellant: str, reason: str) -> dict[str, Any]:
    try:
        snap = _state.care_appeal(claim_id=claim_id, appellant=appellant, reason=reason)
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def care_snapshot() -> dict[str, Any]:
    return {"ok": True, "snapshot": _state.care_snapshot()}


# ---- AMITY tools ----


@_documented_tool
def amity_entitle(
    from_claim_id: str, to: str, amount: int, policy: str
) -> dict[str, Any]:
    try:
        snap = _state.amity_entitle(
            from_claim_id=from_claim_id, to=to, amount=amount, policy=policy
        )
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def amity_transfer(from_address: str, to: str, amount: int) -> dict[str, Any]:
    try:
        snap = _state.amity_transfer(from_address=from_address, to=to, amount=amount)
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def amity_snapshot() -> dict[str, Any]:
    return {"ok": True, "snapshot": _state.amity_snapshot()}


# ---- OMEGA tools ----


@_documented_tool
def omega_mint(
    to: str, amount: int, reason: str, protocol_version: str = "0.1.0"
) -> dict[str, Any]:
    try:
        snap = _state.omega_mint(
            to=to, amount=amount, reason=reason, protocol_version=protocol_version
        )
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def omega_lock(
    owner: str, amount: int, duration_blocks: int, reason: str
) -> dict[str, Any]:
    try:
        snap = _state.omega_lock(
            owner=owner, amount=amount, duration_blocks=duration_blocks, reason=reason
        )
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def omega_proposal(
    proposer: str,
    change_set: dict[str, Any],
    simulation_ref: str,
    rollback_plan: str,
) -> dict[str, Any]:
    snap = _state.omega_proposal(
        proposer=proposer,
        change_set=change_set,
        simulation_ref=simulation_ref,
        rollback_plan=rollback_plan,
    )
    return {"ok": True, "snapshot": snap}


@_documented_tool
def omega_vote(
    proposal_id: str, voter: str, support: bool, weight: int
) -> dict[str, Any]:
    try:
        snap = _state.omega_vote(
            proposal_id=proposal_id, voter=voter, support=support, weight=weight
        )
        return {"ok": True, "snapshot": snap}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


@_documented_tool
def omega_snapshot() -> dict[str, Any]:
    return {"ok": True, "snapshot": _state.omega_snapshot()}


# ---- house tools ----


@_documented_tool
def ledger() -> dict[str, Any]:
    return {"ok": True, "snapshot": _state.ledger()}


@_documented_tool
def events_since(version: int) -> dict[str, Any]:
    return {"ok": True, "events": _state.events_since(version)}


@_documented_tool
def full_event_log() -> dict[str, Any]:
    return {"ok": True, "events": _state.full_event_log()}


if __name__ == "__main__":
    mcp.run(transport="stdio")
