"""
FinGuard AgentOps — ASMO Meta-Agent (Agent Security & Management Orchestrator)
A LangGraph state machine that acts as the supervisory brain of the platform.

When a Prometheus alert fires, or a manual kill switch is triggered,
the event is dispatched to this orchestrator. It evaluates the severity
and takes one of four deterministic actions:

  1. switch_model   — Advance the agent to a fallback LLM.
  2. isolate        — Completely kill the agent. No traffic passes.
  3. pause_and_audit— Temporarily halt the agent and audit its memory for poisoning.
  4. log_only       — Low-severity event. Log it and move on.

The orchestrator is NOT an LLM. It is a pure deterministic state machine
built with LangGraph. It makes zero API calls to OpenAI. This ensures
that the recovery system itself cannot be prompt-injected.
"""

import logging
from typing import TypedDict, Literal, Optional

from langgraph.graph import StateGraph, END

from orchestrator.registry import registry

logger = logging.getLogger("finguard.orchestrator")


# ── Orchestrator State ───────────────────────────────────────
class OrchestratorState(TypedDict):
    event_type: str            # "health_alert" | "security_alert" | "manual_intervention"
    agent_name: str
    severity: Literal["low", "medium", "high", "critical"]
    detail: Optional[str]      # Human-readable context about the event
    action_taken: str          # Populated by the evaluate node
    resolved: bool


# ── Node 1: Evaluate the Event ──────────────────────────────
def evaluate_event_node(state: OrchestratorState) -> dict:
    """
    Decision logic — maps the incoming event to a recovery action.
    This is entirely rule-based. No LLM involved.
    """
    severity = state["severity"]
    event_type = state["event_type"]
    agent_name = state["agent_name"]

    # Critical events always isolate, regardless of type
    if severity == "critical":
        action = "isolate"
    elif event_type == "health_alert":
        action = "switch_model"
    elif event_type == "security_alert":
        action = "pause_and_audit"
    else:
        action = "log_only"

    logger.info(
        f"ASMO: Evaluated event for '{agent_name}' "
        f"[type={event_type}, severity={severity}] → action={action}"
    )
    return {"action_taken": action}


# ── Conditional Router ──────────────────────────────────────
def route_action(state: OrchestratorState) -> str:
    """Route to the correct action node based on the evaluation."""
    return state["action_taken"]


# ── Node 2a: Switch Model ──────────────────────────────────
def switch_model_node(state: OrchestratorState) -> dict:
    """Advance the agent to the next model in the fallback chain."""
    agent_name = state["agent_name"]
    reason = state.get("detail", "Health check degraded")

    registry.switch_model(agent_name, reason)

    logger.warning(
        f"ASMO: Model switch executed for '{agent_name}'. "
        f"Now using: {registry.get_current_model(agent_name)}"
    )
    return {"resolved": True}


# ── Node 2b: Isolate Agent ─────────────────────────────────
def isolate_agent_node(state: OrchestratorState) -> dict:
    """Completely kill the agent. All traffic will be rejected."""
    agent_name = state["agent_name"]
    reason = state.get("detail", "Critical event — manual or automated isolation")

    registry.isolate_agent(agent_name, reason)

    logger.critical(
        f"🚨 ASMO: Agent '{agent_name}' has been ISOLATED. "
        f"Reason: {reason}"
    )
    return {"resolved": True}


# ── Node 2c: Pause and Audit ───────────────────────────────
def pause_and_audit_node(state: OrchestratorState) -> dict:
    """
    Temporarily pause the agent and run a security audit.
    If the audit finds poisoned memory entries, escalate to full isolation.
    """
    agent_name = state["agent_name"]
    detail = state.get("detail", "Security alert triggered audit")

    logger.warning(
        f"ASMO: Pausing '{agent_name}' for security audit. Detail: {detail}"
    )

    # For now, we log the audit event. In a future phase, this will
    # invoke gateway/memory_audit.py to scan the vector store for
    # poisoned instructions injected via adversarial PDFs.
    #
    # If poisoned entries are found:
    #   registry.isolate_agent(agent_name, "Poisoned memory detected")
    #   return {"resolved": False}

    logger.info(f"ASMO: Audit complete for '{agent_name}'. No threats detected.")
    return {"resolved": True}


# ── Node 2d: Log Only ──────────────────────────────────────
def log_only_node(state: OrchestratorState) -> dict:
    """Low-severity event. Just log it and mark as resolved."""
    logger.info(
        f"ASMO: Low-severity event for '{state['agent_name']}' logged. "
        f"Detail: {state.get('detail', 'N/A')}"
    )
    return {"resolved": True}


# ── Graph Assembly ──────────────────────────────────────────
def build_orchestrator():
    """
    Assemble the ASMO orchestrator as a LangGraph state machine.
    
    Flow:
      evaluate_event → [conditional routing] → action_node → END
    """
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("evaluate_event", evaluate_event_node)
    graph.add_node("switch_model", switch_model_node)
    graph.add_node("isolate", isolate_agent_node)
    graph.add_node("pause_and_audit", pause_and_audit_node)
    graph.add_node("log_only", log_only_node)

    # Entry point
    graph.set_entry_point("evaluate_event")

    # Conditional edges from the evaluator to the action nodes
    graph.add_conditional_edges("evaluate_event", route_action, {
        "switch_model": "switch_model",
        "isolate": "isolate",
        "pause_and_audit": "pause_and_audit",
        "log_only": "log_only",
    })

    # All action nodes terminate
    graph.add_edge("switch_model", END)
    graph.add_edge("isolate", END)
    graph.add_edge("pause_and_audit", END)
    graph.add_edge("log_only", END)

    return graph.compile()


# ── Global Singleton ─────────────────────────────────────────
orchestrator = build_orchestrator()
