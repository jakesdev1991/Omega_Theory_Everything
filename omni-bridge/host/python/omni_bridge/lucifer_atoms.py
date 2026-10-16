# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ATOMSProfile:
    beliefs: str
    intentions: str
    desires: str
    emotions: str
    knowledge: str
    percepts: str
    non_literal: str

    def requires_empathic_routing(self) -> bool:
        """
        Determines if the prompt exhibits high frustration or false beliefs,
        necessitating empathic routing to 'Death' or 'Famine' over 'Conquest'.
        """
        critical_flags = [
            "frustration",
            "urgency",
            "error",
            "false assumption",
            "misunderstanding",
        ]
        for flag in critical_flags:
            if flag in self.emotions.lower() or flag in self.beliefs.lower():
                return True
        return False


class ATOMSEngine:
    """
    Abilities in Theory of Mind Space (ATOMS) Engine.
    Vectorizes unstated psychological states across 7 dimensions to de-noise obscure prompts.
    """

    def __init__(self):
        # In a full implementation, this would interface with a lightweight local LLM or classifier
        # to generate the 7-dimensional semantic vector.
        pass

    def generate_profile(
        self, user_prompt: str, context: Dict[str, Any] | None = None
    ) -> ATOMSProfile:
        """
        Analyzes the prompt and generates a 7-dimensional psychological profile.
        *Mock implementation for the architectural harness*
        """
        prompt_lower = user_prompt.lower()

        # Heuristics for the mock implementation
        emotions = "neutral"
        if "error" in prompt_lower or "failed" in prompt_lower or "!" in user_prompt:
            emotions = "high frustration or urgency"

        intentions = "Action execution and architecture implementation"
        if "?" in user_prompt and "why" in prompt_lower:
            intentions = "Explanation and theory crafting"

        return ATOMSProfile(
            beliefs="User assumes the Omni-Bridge is capable of unified execution.",
            intentions=intentions,
            desires="Sub-millisecond execution route or robust structural unification.",
            emotions=emotions,
            knowledge="Systems architecture, advanced AI alignment.",
            percepts="User interprets previous implementations as disjointed.",
            non_literal="Literal",
        )
