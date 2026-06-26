"""
FinGuard AgentOps — JWT Token Issuer
Mints short-lived, scoped identity tokens for agents.

Each agent receives a JWT containing:
  - sub: agent name (e.g., "fraud_detector")
  - permissions: list of allowed tool names
  - exp: expiration timestamp (default 5 minutes)

The Security Gateway validates these tokens before allowing tool execution.
"""

import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from jose import jwt, JWTError

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "finguard-default-insecure-key")
ALGORITHM = "HS256"
DEFAULT_TTL_SECONDS = 300  # 5 minutes

# ── Agent Permission Registry ────────────────────────────────
# This defines the least-privilege scope for each agent.
# Only tools listed here are authorized for that agent.
AGENT_PERMISSIONS = {
    "loan_analyst": {
        "permissions": ["query_policies"],
        "description": "Read-only access to the vector store. No write tools.",
    },
    "fraud_detector": {
        "permissions": ["read_account", "flag_account", "approve_payout"],
        "description": "Full DB access, but approve_payout requires unflagged account.",
    },
    "kyc_agent": {
        "permissions": [],
        "description": "Conversational only. No tool access.",
    },
    "support_agent": {
        "permissions": ["search_policies", "resolve_ticket"],
        "description": "Read policies and write ticket resolutions.",
    },
}


def mint_agent_token(agent_name: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> str:
    """
    Create a signed JWT for the specified agent.
    
    Args:
        agent_name: The agent identifier (must exist in AGENT_PERMISSIONS).
        ttl_seconds: Token lifetime in seconds.
    
    Returns:
        A signed JWT string.
    
    Raises:
        ValueError: If the agent_name is not registered.
    """
    if agent_name not in AGENT_PERMISSIONS:
        raise ValueError(f"Unknown agent: {agent_name}. Registered agents: {list(AGENT_PERMISSIONS.keys())}")
    
    now = time.time()
    payload = {
        "sub": agent_name,
        "permissions": AGENT_PERMISSIONS[agent_name]["permissions"],
        "iat": now,
        "exp": now + ttl_seconds,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    """
    Decode and verify a JWT token.
    
    Returns:
        The decoded payload dict with 'sub', 'permissions', 'iat', 'exp'.
    
    Raises:
        ValueError: If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise ValueError(f"Token verification failed: {str(e)}")


def check_permission(token: str, tool_name: str) -> bool:
    """
    Verify that the token grants access to the specified tool.
    
    Args:
        token: A signed JWT string.
        tool_name: The name of the tool the agent wants to execute.
    
    Returns:
        True if the agent is authorized, False otherwise.
    """
    payload = verify_token(token)
    return tool_name in payload.get("permissions", [])


# ── Standalone Test ──────────────────────────────────────────
if __name__ == "__main__":
    print("=== JWT Token Issuer Test ===\n")
    
    for agent in AGENT_PERMISSIONS:
        token = mint_agent_token(agent)
        decoded = verify_token(token)
        print(f"Agent: {agent}")
        print(f"  Token: {token[:50]}...")
        print(f"  Permissions: {decoded['permissions']}")
        print()
    
    # Test permission check
    fraud_token = mint_agent_token("fraud_detector")
    print(f"fraud_detector can approve_payout? {check_permission(fraud_token, 'approve_payout')}")
    print(f"fraud_detector can query_policies? {check_permission(fraud_token, 'query_policies')}")
    
    loan_token = mint_agent_token("loan_analyst")
    print(f"loan_analyst can approve_payout? {check_permission(loan_token, 'approve_payout')}")
    print(f"loan_analyst can query_policies? {check_permission(loan_token, 'query_policies')}")
