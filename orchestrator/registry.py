"""
FinGuard AgentOps — Agent Registry
Centralized state tracker for all agents in the platform.

The registry maintains the health status, current model assignment,
and isolation state of every agent. It is the single source of truth
that the ASMO orchestrator reads from and writes to when making
recovery or isolation decisions.
"""

import logging
from typing import Literal

from mlops.metrics import ACTIVE_AGENTS

logger = logging.getLogger("finguard.registry")

# Model fallback chain — ordered from most capable to fastest/cheapest
MODEL_CHAIN = [
    "gpt-4o",           # Primary — highest reasoning quality
    "gpt-4o-mini",      # Fallback — faster and cheaper
]


class AgentRegistry:
    """
    Global state registry for every agent in the FinGuard platform.
    
    Each agent entry tracks:
      - status: "running" | "degraded" | "isolated"
      - model_index: which position in MODEL_CHAIN the agent is currently using
      - health: float 0.0–1.0 representing the latest health check score
    """
    
    def __init__(self):
        self.agents = {
            "loan_analyst":   {"status": "running", "model_index": 0, "health": 1.0},
            "fraud_detector": {"status": "running", "model_index": 0, "health": 1.0},
            "kyc_agent":      {"status": "running", "model_index": 0, "health": 1.0},
            "support_agent":  {"status": "running", "model_index": 0, "health": 1.0},
        }
        # Set the initial active agent count
        ACTIVE_AGENTS.set(len(self.agents))

    def get_current_model(self, agent_name: str) -> str:
        """Return the model string the agent is currently assigned to."""
        idx = self.agents[agent_name]["model_index"]
        return MODEL_CHAIN[idx]

    def get_status(self, agent_name: str) -> str:
        return self.agents[agent_name]["status"]

    def switch_model(self, agent_name: str, reason: str):
        """
        Advance the agent to the next model in the fallback chain.
        If no more fallbacks exist, isolate the agent entirely.
        """
        current_idx = self.agents[agent_name]["model_index"]
        
        if current_idx + 1 >= len(MODEL_CHAIN):
            logger.critical(
                f"REGISTRY: {agent_name} has exhausted all fallback models. Isolating."
            )
            self.isolate_agent(agent_name, reason="All fallback models exhausted")
            return
        
        before_model = MODEL_CHAIN[current_idx]
        self.agents[agent_name]["model_index"] = current_idx + 1
        after_model = MODEL_CHAIN[current_idx + 1]
        self.agents[agent_name]["status"] = "degraded"
        
        logger.warning(
            f"REGISTRY: {agent_name} switched from {before_model} → {after_model}. "
            f"Reason: {reason}"
        )

    def isolate_agent(self, agent_name: str, reason: str = "Manual isolation"):
        """
        Completely isolate an agent — all traffic to it will be rejected.
        This is the kill switch.
        """
        self.agents[agent_name]["status"] = "isolated"
        self.agents[agent_name]["health"] = 0.0
        
        # Update the Prometheus gauge
        active = sum(1 for a in self.agents.values() if a["status"] != "isolated")
        ACTIVE_AGENTS.set(active)
        
        logger.critical(
            f"🚨 REGISTRY: {agent_name} ISOLATED. All traffic blocked. Reason: {reason}"
        )

    def restore_agent(self, agent_name: str):
        """
        Restore an isolated agent back to service with the primary model.
        """
        self.agents[agent_name]["status"] = "running"
        self.agents[agent_name]["model_index"] = 0
        self.agents[agent_name]["health"] = 1.0
        
        active = sum(1 for a in self.agents.values() if a["status"] != "isolated")
        ACTIVE_AGENTS.set(active)
        
        logger.info(f"REGISTRY: {agent_name} restored to service with {MODEL_CHAIN[0]}.")

    def update_health(self, agent_name: str, health: float):
        self.agents[agent_name]["health"] = health

    def to_dict(self) -> dict:
        """Serialize the registry for API responses."""
        return {
            name: {
                **info,
                "current_model": MODEL_CHAIN[info["model_index"]],
            }
            for name, info in self.agents.items()
        }


# ── Global Singleton ─────────────────────────────────────────
registry = AgentRegistry()
