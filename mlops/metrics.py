"""
FinGuard AgentOps — Prometheus Metrics Engine
Defines custom metrics to track security events, API latency, and LLM usage.
These metrics are scraped by Prometheus and visualized in Grafana.
"""

from prometheus_client import Counter, Histogram, Gauge

# ── API Metrics ──────────────────────────────────────────────
API_REQUEST_COUNT = Counter(
    "finguard_api_requests_total",
    "Total number of API requests received",
    ["endpoint", "method", "status_code"]
)

API_REQUEST_LATENCY = Histogram(
    "finguard_api_request_duration_seconds",
    "Latency of API requests in seconds",
    ["endpoint"]
)

# ── Security Gateway Metrics ─────────────────────────────────
NEMO_BLOCKED_COUNT = Counter(
    "finguard_security_nemo_blocks_total",
    "Total number of prompt injections blocked by NeMo Guardrails",
    ["agent_name"]
)

OPA_DENIALS_COUNT = Counter(
    "finguard_security_opa_denials_total",
    "Total number of tool executions blocked by OPA",
    ["agent_name", "tool_name", "reason"]
)

PRESIDIO_REDACTIONS_COUNT = Counter(
    "finguard_security_presidio_redactions_total",
    "Total number of responses where PII was redacted by Presidio",
    ["agent_name"]
)

# ── Agent Performance Metrics ────────────────────────────────
AGENT_LATENCY = Histogram(
    "finguard_agent_invocation_duration_seconds",
    "Latency of LLM agent invocations in seconds",
    ["agent_name", "model"]
)

ACTIVE_AGENTS = Gauge(
    "finguard_active_agents",
    "Current number of healthy agents available to serve requests",
)

LLM_FALLBACK_COUNT = Counter(
    "finguard_llm_fallbacks_total",
    "Total number of times the self-healing orchestrator hot-swapped the model due to failure",
    ["primary_model", "fallback_model"]
)
