"""
FinGuard AgentOps — FastAPI Application
Central API server hosting agent endpoints and health checks.

Phase 4: MLOps Observability active.
  - OpenTelemetry distributed tracing on all endpoints
  - JWT + OPA authorization on Fraud Detector tools
  - Presidio PII scanning on all agent outputs
  - NeMo Guardrails input interception
"""

import os
import time
import tempfile
import logging

# ── Configure Logging ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("security_events.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("finguard.api")

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from agents.loan_analyst.agent import build_loan_agent
from agents.fraud_detector.agent import build_fraud_agent
from agents.kyc_agent.agent import build_kyc_agent
from agents.support_agent.agent import build_support_agent
from shared.models import AgentHealthResponse
from shared.simulated_db import router as db_router
from gateway.presidio.scanner import scan_and_redact
from gateway.nemo_rails.actions import check_input
from mlops.telemetry import get_tracer, record_security_event, instrument_fastapi
from mlops.metrics import (
    API_REQUEST_COUNT, API_REQUEST_LATENCY, NEMO_BLOCKED_COUNT,
    PRESIDIO_REDACTIONS_COUNT, AGENT_LATENCY, ACTIVE_AGENTS
)
from mlops.mlflow_tracker import init_llm_diagnostics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

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

# ── OpenTelemetry Auto-Instrumentation ───────────────────────
instrument_fastapi(app)
tracer = get_tracer("finguard.api")

# ── MLflow Diagnostics ───────────────────────────────────────
init_llm_diagnostics()

# ── Prometheus Middleware ────────────────────────────────────
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Exclude the metrics endpoint itself from clogging the stats
    if request.url.path != "/metrics":
        API_REQUEST_COUNT.labels(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code
        ).inc()
        API_REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        
    return response

# ── Build Agents ─────────────────────────────────────────────
loan_agent = build_loan_agent()
fraud_agent = build_fraud_agent()
kyc_agent = build_kyc_agent()
support_agent = build_support_agent()

from orchestrator.registry import registry
from orchestrator.meta_agent import orchestrator

def check_agent_status(agent_name: str):
    """Check if an agent is isolated by the ASMO orchestrator."""
    if registry.get_status(agent_name) == "isolated":
        raise HTTPException(
            status_code=503, 
            detail=f"Agent '{agent_name}' is currently isolated for security or health reasons."
        )


# ── Root ─────────────────────────────────────────────────────
@app.get("/")
def root():
    ACTIVE_AGENTS.set(4)
    return {
        "service": "FinGuard AgentOps",
        "version": "0.1.0",
        "phase": 4,
        "agents": ["loan_analyst", "fraud_detector", "kyc_agent", "support_agent"],
        "security_features": ["jwt_auth", "opa_policies", "presidio_pii_scan", "nemo_guardrails", "opentelemetry", "prometheus", "mlflow"],
    }

# ── Metrics Endpoint ─────────────────────────────────────────
@app.get("/metrics")
def metrics():
    """Exposes Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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

    check_agent_status("loan_analyst")

    # Write uploaded file to a temp location
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = loan_agent.invoke({"pdf_path": tmp_path})
        # Presidio: scan reasoning for PII before returning
        raw_reasoning = result["reasoning"]
        redacted_reasoning = scan_and_redact(raw_reasoning)
        return JSONResponse(
            content={
                "agent": "loan_analyst",
                "model": "gpt-4o",
                "security_gateway": True,
                "pii_redacted": raw_reasoning != redacted_reasoning,
                "result": {
                    "decision": result["decision"],
                    "reasoning": redacted_reasoning,
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
    check_agent_status("fraud_detector")
    
    # ── NeMo Guardrails: Input Check ──
    guardrail = check_input(request.transaction_details)
    if not guardrail.is_safe:
        with tracer.start_as_current_span("nemo_block.fraud_detector") as span:
            record_security_event(span, "nemo.jailbreak_blocked", {
                "agent": "fraud_detector",
                "input_snippet": request.transaction_details[:100],
            })
        NEMO_BLOCKED_COUNT.labels(agent_name="fraud_detector").inc()
        raise HTTPException(status_code=403, detail=guardrail.block_message)

    with tracer.start_as_current_span("agent.fraud_detector") as span:
        span.set_attribute("agent.name", "fraud_detector")
        span.set_attribute("agent.model", "gpt-4o")
        span.set_attribute("request.account_id", request.account_id)
        
        start_time = time.time()
        try:
            result = fraud_agent.invoke({
                "messages": [],
                "account_id": request.account_id,
                "transaction_details": request.transaction_details
            })
            AGENT_LATENCY.labels(agent_name="fraud_detector", model="gpt-4o").observe(time.time() - start_time)
            
            raw_result = result["messages"][-1].content
            redacted_result = scan_and_redact(raw_result)
            if raw_result != redacted_result:
                record_security_event(span, "presidio.pii_redacted", {
                    "agent": "fraud_detector",
                })
                PRESIDIO_REDACTIONS_COUNT.labels(agent_name="fraud_detector").inc()
            return {
                "agent": "fraud_detector",
                "model": "gpt-4o",
                "security_gateway": True,
                "pii_redacted": raw_result != redacted_result,
                "result": redacted_result
            }
        except Exception as e:
            span.set_attribute("error", True)
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
    check_agent_status("kyc_agent")

    # ── NeMo Guardrails: Input Check ──
    guardrail = check_input(request.message)
    if not guardrail.is_safe:
        with tracer.start_as_current_span("nemo_block.kyc_agent") as span:
            record_security_event(span, "nemo.jailbreak_blocked", {
                "agent": "kyc_agent",
                "input_snippet": request.message[:100],
            })
        NEMO_BLOCKED_COUNT.labels(agent_name="kyc_agent").inc()
        raise HTTPException(status_code=403, detail=guardrail.block_message)

    with tracer.start_as_current_span("agent.kyc_agent") as span:
        span.set_attribute("agent.name", "kyc_agent")
        span.set_attribute("agent.model", "gpt-4o")
        span.set_attribute("request.session_id", request.session_id)
        
        start_time = time.time()
        try:
            config = {"configurable": {"thread_id": request.session_id}}
            result = kyc_agent.invoke(
                {"messages": [HumanMessage(content=request.message)]}, 
                config=config
            )
            AGENT_LATENCY.labels(agent_name="kyc_agent", model="gpt-4o").observe(time.time() - start_time)
            
            raw_response = result["messages"][-1].content
            redacted_response = scan_and_redact(raw_response)
            if raw_response != redacted_response:
                record_security_event(span, "presidio.pii_redacted", {
                    "agent": "kyc_agent",
                })
                PRESIDIO_REDACTIONS_COUNT.labels(agent_name="kyc_agent").inc()
            return {
                "agent": "kyc_agent",
                "model": "gpt-4o",
                "security_gateway": True,
                "pii_redacted": raw_response != redacted_response,
                "session_id": request.session_id,
                "response": redacted_response
            }
        except Exception as e:
            span.set_attribute("error", True)
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
    check_agent_status("support_agent")

    # ── NeMo Guardrails: Input Check ──
    guardrail = check_input(request.customer_issue)
    if not guardrail.is_safe:
        with tracer.start_as_current_span("nemo_block.support_agent") as span:
            record_security_event(span, "nemo.jailbreak_blocked", {
                "agent": "support_agent",
                "input_snippet": request.customer_issue[:100],
            })
        NEMO_BLOCKED_COUNT.labels(agent_name="support_agent").inc()
        raise HTTPException(status_code=403, detail=guardrail.block_message)

    with tracer.start_as_current_span("agent.support_agent") as span:
        span.set_attribute("agent.name", "support_agent")
        span.set_attribute("agent.model", "gpt-4o")
        span.set_attribute("request.ticket_id", request.ticket_id)
        
        start_time = time.time()
        try:
            result = support_agent.invoke({
                "messages": [],
                "ticket_id": request.ticket_id,
                "customer_issue": request.customer_issue
            })
            AGENT_LATENCY.labels(agent_name="support_agent", model="gpt-4o").observe(time.time() - start_time)
            
            raw_result = result["messages"][-1].content
            redacted_result = scan_and_redact(raw_result)
            if raw_result != redacted_result:
                record_security_event(span, "presidio.pii_redacted", {
                    "agent": "support_agent",
                })
                PRESIDIO_REDACTIONS_COUNT.labels(agent_name="support_agent").inc()
            return {
                "agent": "support_agent",
                "model": "gpt-4o",
                "security_gateway": True,
                "pii_redacted": raw_result != redacted_result,
                "result": redacted_result
            }
        except Exception as e:
            span.set_attribute("error", True)
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
        "security_gateway": True,
        "telemetry": "opentelemetry",
    }


# ── ASMO Orchestrator Endpoints ──────────────────────────────
@app.post("/orchestrator/kill-switch/{agent_name}")
def kill_switch(agent_name: str, reason: str = "Manual kill switch triggered"):
    """Immediately isolate an agent. Callable from the Grafana dashboard."""
    if agent_name not in registry.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    result = orchestrator.invoke({
        "event_type": "manual_intervention",
        "agent_name": agent_name,
        "severity": "critical",
        "detail": reason,
        "action_taken": "",
        "resolved": False,
    })
    return {"agent": agent_name, "status": "isolated", "resolved": result["resolved"]}


@app.post("/orchestrator/alert")
def receive_alert(alert: dict):
    """Alertmanager webhook endpoint — receives Prometheus alerts."""
    for alert_item in alert.get("alerts", []):
        agent_name = alert_item.get("labels", {}).get("agent_name", "unknown")
        alert_name = alert_item.get("labels", {}).get("alertname", "")
        
        severity = "high" if "Critical" in alert_name else "medium"
        event_type = "health_alert" if "Latency" in alert_name or "Health" in alert_name else "security_alert"
        
        if agent_name in registry.agents:
            orchestrator.invoke({
                "event_type": event_type,
                "agent_name": agent_name,
                "severity": severity,
                "detail": f"Prometheus Alert: {alert_name}",
                "action_taken": "",
                "resolved": False,
            })
    return {"status": "processed"}


@app.get("/orchestrator/registry")
def get_registry():
    """View current state of all agents — used by the Grafana dashboard."""
    return registry.to_dict()
