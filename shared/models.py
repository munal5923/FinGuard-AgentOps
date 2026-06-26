"""
FinGuard AgentOps — Shared Pydantic Models
Defines structured schemas used across agents and the API layer.
"""

from pydantic import BaseModel, field_validator
from typing import Literal, Optional
from datetime import datetime


class LoanDecision(BaseModel):
    """Structured output schema for the Loan Analyst agent."""
    decision: Literal["approved", "rejected"]
    reasoning: str
    confidence: float
    applicant_name: Optional[str] = None
    monthly_income: Optional[float] = None
    recommended_amount: Optional[float] = None

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return round(v, 2)

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Reasoning must be substantive (at least 10 characters)")
        return v


class LoanRequest(BaseModel):
    """Incoming loan assessment request metadata."""
    applicant_id: Optional[str] = None
    request_timestamp: datetime = None

    def __init__(self, **data):
        if data.get("request_timestamp") is None:
            data["request_timestamp"] = datetime.utcnow()
        super().__init__(**data)


class AgentHealthResponse(BaseModel):
    """Health check response for any agent."""
    agent: str
    status: Literal["running", "degraded", "paused", "isolated"]
    model: str
    version: str
    uptime_seconds: Optional[float] = None
