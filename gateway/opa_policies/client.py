"""
FinGuard AgentOps — OPA Policy Client
Evaluates agent tool-call authorization using the Rego policy logic.

This client implements the OPA policy rules natively in Python,
mirroring the Rego definitions in agent_permissions.rego.
This avoids requiring an external OPA server for local development
while maintaining identical authorization semantics.

In production, this would call a running OPA server via HTTP.
"""

import logging
from typing import Optional

from shared.token_issuer import verify_token
from mlops.metrics import OPA_DENIALS_COUNT

logger = logging.getLogger("finguard.opa")

# ── Policy Rules (mirrors agent_permissions.rego) ────────────
# Maps agent_role -> set of allowed tools
ROLE_PERMISSIONS = {
    "loan_analyst": {"query_policies"},
    "fraud_detector": {"read_account", "flag_account", "approve_payout"},
    "kyc_agent": set(),  # No tool access
    "support_agent": {"search_policies", "resolve_ticket"},
}

# Tools that require additional context checks
CONTEXT_SENSITIVE_TOOLS = {
    "approve_payout": {
        "check": "account_not_flagged",
        "deny_reason": "BLOCKED: Cannot approve payout on a flagged account",
    }
}


class OPADecision:
    """Result of an OPA policy evaluation."""
    
    def __init__(self, allowed: bool, reason: str = ""):
        self.allowed = allowed
        self.reason = reason
    
    def __bool__(self):
        return self.allowed
    
    def __repr__(self):
        status = "ALLOW" if self.allowed else "DENY"
        return f"OPADecision({status}: {self.reason})"


def evaluate(
    token: str,
    tool_name: str,
    context: Optional[dict] = None,
) -> OPADecision:
    """
    Evaluate whether an agent (identified by JWT) is authorized
    to execute the specified tool with the given context.
    
    Args:
        token: A signed JWT from token_issuer.mint_agent_token().
        tool_name: The name of the tool the agent wants to call.
        context: Optional dict with runtime context, e.g.:
                 {"account_flagged": True, "account_id": "ACC003"}
    
    Returns:
        An OPADecision indicating whether the call is allowed or denied.
    """
    context = context or {}
    
    # Step 1: Verify the JWT token
    try:
        payload = verify_token(token)
    except ValueError as e:
        logger.warning(f"OPA: Token verification failed: {e}")
        OPA_DENIALS_COUNT.labels(agent_name="unknown", tool_name=tool_name, reason="auth_failure").inc()
        return OPADecision(False, f"BLOCKED: Invalid or expired token — {str(e)}")
    
    agent_role = payload.get("sub", "unknown")
    token_permissions = set(payload.get("permissions", []))
    
    # Step 2: Check if the tool is in the agent's JWT permissions
    if tool_name not in token_permissions:
        logger.warning(f"OPA: {agent_role} attempted unauthorized tool '{tool_name}'")
        return OPADecision(
            False,
            f"BLOCKED: Agent '{agent_role}' does not have permission to use tool '{tool_name}'"
        )
    
    # Step 3: Check role-based permissions (defense in depth)
    role_perms = ROLE_PERMISSIONS.get(agent_role, set())
    if tool_name not in role_perms:
        logger.warning(f"OPA: Role '{agent_role}' not authorized for tool '{tool_name}'")
        return OPADecision(
            False,
            f"BLOCKED: Role '{agent_role}' is not authorized for tool '{tool_name}'"
        )
    
    # Step 4: Context-sensitive checks
    if tool_name in CONTEXT_SENSITIVE_TOOLS:
        check_config = CONTEXT_SENSITIVE_TOOLS[tool_name]
        
        if check_config["check"] == "account_not_flagged":
            if context.get("account_flagged", False):
                logger.warning(
                    f"OPA: {agent_role} blocked from {tool_name} — account is flagged"
                )
                return OPADecision(False, check_config["deny_reason"])
    
    # All checks passed
    logger.info(f"OPA: ALLOW {agent_role} -> {tool_name}")
    return OPADecision(True, f"ALLOW: {agent_role} authorized for {tool_name}")


# ── Standalone Test ──────────────────────────────────────────
if __name__ == "__main__":
    from shared.token_issuer import mint_agent_token
    
    print("=== OPA Policy Client Test ===\n")
    
    # Test 1: Fraud detector reading an account (should ALLOW)
    token = mint_agent_token("fraud_detector")
    result = evaluate(token, "read_account")
    print(f"Test 1 - fraud_detector read_account: {result}")
    
    # Test 2: Fraud detector approving payout on clean account (should ALLOW)
    result = evaluate(token, "approve_payout", {"account_flagged": False})
    print(f"Test 2 - fraud_detector approve_payout (clean): {result}")
    
    # Test 3: Fraud detector approving payout on FLAGGED account (should DENY)
    result = evaluate(token, "approve_payout", {"account_flagged": True})
    print(f"Test 3 - fraud_detector approve_payout (flagged): {result}")
    
    # Test 4: Loan analyst trying to approve payout (should DENY)
    loan_token = mint_agent_token("loan_analyst")
    result = evaluate(loan_token, "approve_payout")
    print(f"Test 4 - loan_analyst approve_payout: {result}")
    
    # Test 5: Support agent resolving ticket (should ALLOW)
    support_token = mint_agent_token("support_agent")
    result = evaluate(support_token, "resolve_ticket")
    print(f"Test 5 - support_agent resolve_ticket: {result}")
    
    # Test 6: KYC agent trying any tool (should DENY)
    kyc_token = mint_agent_token("kyc_agent")
    result = evaluate(kyc_token, "read_account")
    print(f"Test 6 - kyc_agent read_account: {result}")
