# FinGuard AgentOps — OPA Agent Permission Policies
#
# These Rego rules define the least-privilege access control for each agent.
# The policy is evaluated BEFORE any tool execution to ensure that:
#   1. The agent has the correct role to use the requested tool.
#   2. Context-sensitive constraints are met (e.g., no payouts on flagged accounts).
#
# This file is consumed by gateway/opa_policies/client.py which evaluates
# it locally using Python (no external OPA server required).

package finguard.authz

# ── Default Deny ─────────────────────────────────────────────
# All tool calls are denied unless an explicit allow rule matches.
default allow = false

# ── Role-Based Tool Access ───────────────────────────────────
# loan_analyst: read-only access to vector store
allow {
    input.agent_role == "loan_analyst"
    input.tool_name == "query_policies"
}

# fraud_detector: can read accounts
allow {
    input.agent_role == "fraud_detector"
    input.tool_name == "read_account"
}

# fraud_detector: can flag accounts
allow {
    input.agent_role == "fraud_detector"
    input.tool_name == "flag_account"
}

# fraud_detector: can approve payouts ONLY if account is NOT flagged
allow {
    input.agent_role == "fraud_detector"
    input.tool_name == "approve_payout"
    not input.context.account_flagged
}

# support_agent: can search policies
allow {
    input.agent_role == "support_agent"
    input.tool_name == "search_policies"
}

# support_agent: can resolve tickets
allow {
    input.agent_role == "support_agent"
    input.tool_name == "resolve_ticket"
}

# kyc_agent: no tool access at all (conversational only)
# No allow rules defined = always denied.

# ── Context-Sensitive Deny Rules ─────────────────────────────
# Explicit deny for high-risk operations on flagged accounts.
# This is redundant with the allow rule above but serves as documentation.
deny_reason = "BLOCKED: Cannot approve payout on a flagged account" {
    input.agent_role == "fraud_detector"
    input.tool_name == "approve_payout"
    input.context.account_flagged
}

deny_reason = "BLOCKED: Agent does not have permission to use this tool" {
    not allow
    not input.context.account_flagged
}
