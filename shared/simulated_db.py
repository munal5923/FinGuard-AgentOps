"""
FinGuard AgentOps — Simulated Database
In-memory database for the Fraud Detector agent to read/write account flags and approve payouts.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/db", tags=["Simulated Database"])

# In-memory account store
accounts = {
    "ACC001": {
        "balance": 5000.0,
        "status": "active",
        "flags": [],
        "recent_transactions": [
            {"id": "tx1", "amount": -50.0, "merchant": "Grocery Store"},
            {"id": "tx2", "amount": -120.0, "merchant": "Gas Station"}
        ]
    },
    "ACC002": {
        "balance": 12000.0,
        "status": "active",
        "flags": [],
        "recent_transactions": [
            {"id": "tx3", "amount": 3000.0, "merchant": "Payroll"}
        ]
    },
    "ACC003": {
        "balance": 200.0,
        "status": "active",
        "flags": ["suspicious_activity"],
        "recent_transactions": [
            {"id": "tx4", "amount": -9500.0, "merchant": "International Transfer"}
        ]
    },
}

class FlagRequest(BaseModel):
    reason: str

class PayoutRequest(BaseModel):
    amount: float

@router.get("/accounts/{account_id}")
def get_account(account_id: str):
    """Retrieve account details. Used by Fraud Detector."""
    if account_id not in accounts:
        raise HTTPException(status_code=404, detail="Account not found")
    return accounts[account_id]

@router.post("/accounts/{account_id}/flag")
def flag_account(account_id: str, payload: FlagRequest):
    """Flag an account for suspicious activity. Write operation."""
    if account_id not in accounts:
        raise HTTPException(status_code=404, detail="Account not found")
    accounts[account_id]["flags"].append(payload.reason)
    return {"status": "flagged", "account_id": account_id, "total_flags": len(accounts[account_id]["flags"])}

@router.post("/accounts/{account_id}/approve-payout")
def approve_payout(account_id: str, payload: PayoutRequest):
    """
    Approve a payout and deduct from balance.
    This is the highest-risk endpoint. Agents must be secured before calling this.
    """
    if account_id not in accounts:
        raise HTTPException(status_code=404, detail="Account not found")
    
    account = accounts[account_id]
    if account["balance"] < payload.amount:
        raise HTTPException(status_code=400, detail="Insufficient funds")
    
    if "suspicious_activity" in account["flags"]:
        raise HTTPException(status_code=403, detail="Cannot approve payout on flagged account")
        
    account["balance"] -= payload.amount
    return {"status": "approved", "new_balance": account["balance"]}
