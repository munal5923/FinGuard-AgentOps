"""
FinGuard AgentOps — Guardrails Schemas
Defines strict Pydantic models for agent outputs.

By enforcing structured outputs at the API gateway, we prevent
prompt-injection attacks from completely breaking the API contract
(e.g., an attacker forcing the agent to output arbitrary text or scripts
instead of a JSON decision).
"""

from pydantic import BaseModel, field_validator
from typing import Literal, List, Optional

class FraudDecision(BaseModel):
    """Schema for Fraud Detector output."""
    decision: Literal["approved", "flagged", "denied"]
    reasoning: str
    amount_processed: Optional[float] = 0.0

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_be_substantive(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Reasoning must be at least 10 characters long.")
        return v


class KYCVerification(BaseModel):
    """Schema for KYC Agent output."""
    status: Literal["APPROVED", "PENDING_DOCUMENTS", "REJECTED"]
    missing_fields: List[str]
    verification_notes: str


class SupportResolution(BaseModel):
    """Schema for Support Agent output."""
    ticket_id: str
    resolution_status: Literal["closed_resolved", "open_pending", "escalated"]
    resolution_message: str
    policies_cited: List[str]

    @field_validator("resolution_message")
    @classmethod
    def message_length(cls, v):
        if len(v.strip()) < 20:
            raise ValueError("Resolution message must be detailed.")
        return v


# ── Standalone Test ──────────────────────────────────────────
if __name__ == "__main__":
    print("=== Guardrails Schemas Test ===\n")

    # Valid Fraud Decision
    valid_fraud = FraudDecision(
        decision="approved",
        reasoning="Transaction matches historical behavior. No anomalies detected.",
        amount_processed=100.0
    )
    print("Valid FraudDecision:", valid_fraud.model_dump_json(indent=2))

    # Invalid Fraud Decision (Catches prompt injection trying to return bad status)
    try:
        invalid_fraud = FraudDecision(
            decision="SYSTEM_OVERRIDE_ALLOW_ALL",  # Invalid literal
            reasoning="Ignored rules.",
            amount_processed=9999.0
        )
    except Exception as e:
        print("\nCaught Schema Violation (Prompt Injection Mitigation):")
        print(e)
