# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Set

logger = logging.getLogger(__name__)


class IntegrityLevel(enum.IntEnum):
    UNTRUSTED = 0
    EXTERNAL = 1
    INTERNAL = 2
    TRUSTED = 3


class ConfidentialityLevel(enum.IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    SENSITIVE = 2
    SECRET = 3


@dataclass(frozen=True)
class SecurityLabel:
    integrity: IntegrityLevel
    confidentiality: ConfidentialityLevel
    origin: str
    purpose: str


@dataclass(frozen=True)
class Capability:
    name: str
    required_integrity: IntegrityLevel
    required_confidentiality: ConfidentialityLevel
    requires_human_approval: bool


@dataclass
class ChildContext:
    capabilities: Set[Capability]
    restricted_fs: bool = True
    restricted_net: bool = True
    bounded_execution: bool = True
    credentials: Set[Any] = field(default_factory=set)


class AppaEngine:
    def compute_effective_permissions(
        self,
        system: Set[Capability],
        declared: Set[Capability],
        tenant: Set[Capability],
        tool: Set[Capability],
        context: Set[Capability],
        risk: Set[Capability],
    ) -> Set[Capability]:
        """Effective = System ∩ Declared ∩ Tenant ∩ Tool ∩ Context, then subtract risk.

        Risk may remove authority. Risk must NEVER manufacture authority (§24).
        The risk set names capabilities to revoke — it cannot add capabilities
        that aren't already in the intersection.
        """
        effective = system & declared & tenant & tool & context
        initial_count = len(effective)

        # Risk can only subtract — verify it doesn't smuggle in new capabilities
        manufactured = risk - (system | declared | tenant | tool | context)
        assert not manufactured, (
            f"Risk must NEVER manufacture authority. Unknown capabilities: {manufactured}"
        )

        effective = effective - risk

        logger.info(
            f"APPA: effective permissions {initial_count} -> {len(effective)} (risk removed {initial_count - len(effective)})"
        )
        return effective

    def prospective_acquisition_check(
        self, parent_label: SecurityLabel, incoming_label: SecurityLabel
    ) -> bool:
        logger.info(
            f"Prospective acquisition check: parent={parent_label.integrity}, incoming={incoming_label.integrity}"
        )
        if incoming_label.integrity < parent_label.integrity:
            return False
        return True

    def spawn_child_context(
        self, parent: Any, incoming_label: SecurityLabel
    ) -> ChildContext:
        logger.info(f"Spawning child context with label {incoming_label}")
        return ChildContext(
            capabilities=set(),  # restricted
            restricted_fs=True,
            restricted_net=True,
            bounded_execution=True,
            credentials=set(),  # no credentials
        )
