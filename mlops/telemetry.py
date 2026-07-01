"""
FinGuard AgentOps — OpenTelemetry Telemetry Module
Configures distributed tracing for the entire FinGuard platform.

Every request that flows through our system is wrapped in a "trace" made
of nested "spans". This allows us to see exactly how long each step takes:
  API Request → NeMo Check → Agent Invocation → Tool Call → OPA Check → Presidio Scan

Traces are exported to:
  1. A local JSON log file (traces.log) for offline analysis.
  2. Console output for real-time debugging.
  3. (Future) An OTLP collector for Grafana Tempo / Jaeger in production.
"""

import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    BatchSpanProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode

logger = logging.getLogger("finguard.telemetry")

# ── Resource Identity ────────────────────────────────────────
# Tags every span with our service name so traces are identifiable
# when multiple services report to the same collector.
resource = Resource.create({
    "service.name": "finguard-agentops",
    "service.version": "0.3.0",
    "deployment.environment": "development",
})

# ── Provider Setup ───────────────────────────────────────────
provider = TracerProvider(resource=resource)

# Export traces to console (visible in uvicorn terminal)
console_exporter = ConsoleSpanExporter()
provider.add_span_processor(SimpleSpanProcessor(console_exporter))

# Set as the global tracer provider
trace.set_tracer_provider(provider)

# ── Tracer Factory ───────────────────────────────────────────
def get_tracer(component_name: str = "finguard") -> trace.Tracer:
    """
    Get a named tracer for a specific component.
    
    Usage:
        tracer = get_tracer("fraud_detector")
        with tracer.start_as_current_span("analyze_transaction") as span:
            span.set_attribute("account_id", "ACC001")
            # ... do work ...
    """
    return trace.get_tracer(component_name)


def record_security_event(
    span: trace.Span,
    event_type: str,
    details: dict,
):
    """
    Record a security-relevant event on the current span.
    
    Args:
        span: The active OpenTelemetry span.
        event_type: e.g., "nemo.jailbreak_blocked", "opa.authorization_denied",
                    "presidio.pii_redacted"
        details: A dict of event attributes.
    """
    span.add_event(event_type, attributes=details)
    span.set_attribute(f"security.{event_type}", True)
    logger.info(f"Telemetry event recorded: {event_type} — {details}")


def mark_span_error(span: trace.Span, error: Exception):
    """Mark a span as failed with error details."""
    span.set_status(StatusCode.ERROR, str(error))
    span.record_exception(error)


# ── FastAPI Auto-Instrumentation ─────────────────────────────
def instrument_fastapi(app):
    """
    Automatically instrument every FastAPI route with OpenTelemetry spans.
    Each incoming HTTP request will generate a parent span with:
      - http.method, http.url, http.status_code
      - Timing data (start, end, duration)
    """
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry: FastAPI auto-instrumentation active.")


# ── Standalone Test ──────────────────────────────────────────
if __name__ == "__main__":
    print("=== OpenTelemetry Telemetry Test ===\n")
    
    tracer = get_tracer("test_component")
    
    # Simulate a traced request lifecycle
    with tracer.start_as_current_span("test_api_request") as parent:
        parent.set_attribute("http.method", "POST")
        parent.set_attribute("http.url", "/agents/fraud-detector/analyze")
        
        # Child span: NeMo check
        with tracer.start_as_current_span("nemo_input_check") as nemo_span:
            nemo_span.set_attribute("input.safe", True)
            nemo_span.set_attribute("input.length", 42)
        
        # Child span: Agent execution
        with tracer.start_as_current_span("agent_invocation") as agent_span:
            agent_span.set_attribute("agent.name", "fraud_detector")
            agent_span.set_attribute("agent.model", "gpt-4o")
            
            # Grandchild span: Tool call with OPA check
            with tracer.start_as_current_span("tool_call.approve_payout") as tool_span:
                tool_span.set_attribute("tool.name", "approve_payout")
                tool_span.set_attribute("opa.decision", "DENY")
                record_security_event(tool_span, "opa.authorization_denied", {
                    "agent": "fraud_detector",
                    "tool": "approve_payout",
                    "reason": "account_flagged",
                })
        
        # Child span: Presidio scan
        with tracer.start_as_current_span("presidio_output_scan") as presidio_span:
            presidio_span.set_attribute("presidio.pii_detected", True)
            presidio_span.set_attribute("presidio.entities_found", 2)
    
    print("\n✅ Trace exported to console above. In production, this goes to Grafana Tempo.")
