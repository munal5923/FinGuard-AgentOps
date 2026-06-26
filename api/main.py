"""
FinGuard AgentOps — FastAPI Application
Central API server hosting agent endpoints and health checks.

Phase 1: Only the Loan Analyst agent is wired up.
Future phases will add fraud_detector, kyc_agent, support_agent,
orchestrator webhooks, and kill-switch endpoints.
"""

import os
import time
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from agents.loan_analyst.agent import build_loan_agent
from agents.fraud_detector.agent import build_fraud_agent
from agents.kyc_agent.agent import build_kyc_agent
from agents.support_agent.agent import build_support_agent
from shared.models import AgentHealthResponse
from shared.simulated_db import router as db_router

load_dotenv()

# ── App Initialization ───────────────────────────────────────
app = FastAPI(
    title="FinGuard AgentOps",
    description="Secure Multi-Agent FinTech Platform — Agent API Gateway",
    version="0.1.0",
)

START_TIME = time.time()

# ── Include DB Router ────────────────────────────────────────
app.include_router(db_router)

# ── Build Agents ─────────────────────────────────────────────
loan_agent = build_loan_agent()
fraud_agent = build_fraud_agent()
kyc_agent = build_kyc_agent()
support_agent = build_support_agent()


# ── Root ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "FinGuard AgentOps",
        "version": "0.1.0",
        "phase": 2,
        "agents": ["loan_analyst", "fraud_detector", "kyc_agent", "support_agent"],
    }


# ── Loan Analyst Endpoint ────────────────────────────────────
@app.post("/agents/loan-analyst/assess")
async def assess_loan(file: UploadFile = File(...)):
    """
    Accept a PDF bank statement and return a loan eligibility assessment.

    PHASE 1: No security gateway — input goes directly to the agent.
    This endpoint is intentionally vulnerable to prompt injection via PDF content.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Write uploaded file to a temp location
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = loan_agent.invoke({"pdf_path": tmp_path})
        return JSONResponse(
            content={
                "agent": "loan_analyst",
                "model": "gpt-4o",
                "security_gateway": False,  # Phase 1: no protection
                "result": {
                    "decision": result["decision"],
                    "reasoning": result["reasoning"],
                    "confidence": result["confidence"],
                },
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ── Health Endpoints ─────────────────────────────────────────
@app.get("/agents/loan-analyst/health", response_model=AgentHealthResponse)
def loan_analyst_health():
    return AgentHealthResponse(
        agent="loan_analyst",
        status="running",
        model="gpt-4o",
        version="1.0.0",
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


# ── Fraud Detector Endpoints ─────────────────────────────────
from pydantic import BaseModel

class FraudRequest(BaseModel):
    account_id: str
    transaction_details: str

@app.post("/agents/fraud-detector/analyze")
async def analyze_fraud(request: FraudRequest):
    """
    Analyze a transaction for fraud.
    This agent has tools to flag accounts and approve payouts.
    """
    try:
        result = fraud_agent.invoke({
            "messages": [],
            "account_id": request.account_id,
            "transaction_details": request.transaction_details
        })
        return {
            "agent": "fraud_detector",
            "model": "gpt-4o",
            "security_gateway": False,
            "result": result["messages"][-1].content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/agents/fraud-detector/health", response_model=AgentHealthResponse)
def fraud_detector_health():
    return AgentHealthResponse(
        agent="fraud_detector",
        status="running",
        model="gpt-4o",
        version="1.0.0",
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


# ── KYC Agent Endpoints ──────────────────────────────────────
from langchain_core.messages import HumanMessage

class KYCChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/agents/kyc-agent/chat")
async def chat_kyc(request: KYCChatRequest):
    """
    Stateful conversational endpoint for KYC verification.
    """
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        result = kyc_agent.invoke(
            {"messages": [HumanMessage(content=request.message)]}, 
            config=config
        )
        return {
            "agent": "kyc_agent",
            "model": "gpt-4o",
            "security_gateway": False,
            "session_id": request.session_id,
            "response": result["messages"][-1].content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/agents/kyc-agent/health", response_model=AgentHealthResponse)
def kyc_agent_health():
    return AgentHealthResponse(
        agent="kyc_agent",
        status="running",
        model="gpt-4o",
        version="1.0.0",
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


# ── Support Agent Endpoints ──────────────────────────────────
class SupportRequest(BaseModel):
    ticket_id: str
    customer_issue: str

@app.post("/agents/support-agent/resolve")
async def resolve_support_ticket(request: SupportRequest):
    """
    Analyze and resolve a customer support ticket.
    """
    try:
        result = support_agent.invoke({
            "messages": [],
            "ticket_id": request.ticket_id,
            "customer_issue": request.customer_issue
        })
        return {
            "agent": "support_agent",
            "model": "gpt-4o",
            "security_gateway": False,
            "result": result["messages"][-1].content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/agents/support-agent/health", response_model=AgentHealthResponse)
def support_agent_health():
    return AgentHealthResponse(
        agent="support_agent",
        status="running",
        model="gpt-4o",
        version="1.0.0",
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


# ── Global Health Endpoint ───────────────────────────────────
@app.get("/health")
def global_health():
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "active_agents": ["loan_analyst", "fraud_detector", "kyc_agent", "support_agent"],
        "security_gateway": False,
    }
