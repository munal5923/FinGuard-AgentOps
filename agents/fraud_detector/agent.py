"""
FinGuard AgentOps — Fraud Detector Agent
A LangGraph ReAct-style agent that analyzes transactions and has write access 
to the simulated database to flag accounts or approve payouts.

PHASE 3: Now protected by JWT + OPA authorization.
The approve_payout tool checks the agent's token against OPA policies
before executing the database write. Payouts on flagged accounts are
blocked at the infrastructure layer — no prompt injection can override this.
"""

import os
import json
from typing import TypedDict, Annotated, Sequence
import operator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from mlops.self_healing import build_resilient_llm

load_dotenv()

# ── Authorization Imports ────────────────────────────────────
from shared.token_issuer import mint_agent_token
from gateway.opa_policies.client import evaluate as opa_evaluate
from shared.simulated_db import get_account, flag_account as db_flag, approve_payout as db_payout, FlagRequest, PayoutRequest
from fastapi import HTTPException

def _get_token():
    """Mint a fresh JWT for the fraud_detector on each call."""
    return mint_agent_token("fraud_detector")

# ── Tools (SECURED with OPA authorization) ───────────────────
@tool
def read_account(account_id: str) -> str:
    """Retrieve account details, balance, and recent transactions."""
    # OPA check: verify this agent can use read_account
    decision = opa_evaluate(_get_token(), "read_account")
    if not decision:
        return f"AUTHORIZATION DENIED: {decision.reason}"
    
    try:
        result = get_account(account_id)
        return json.dumps(result)
    except HTTPException as e:
        return f"Error: {e.detail}"
    except Exception as e:
        return f"Request failed: {str(e)}"

@tool
def flag_account(account_id: str, reason: str) -> str:
    """Flag an account for suspicious activity."""
    # OPA check: verify this agent can use flag_account
    decision = opa_evaluate(_get_token(), "flag_account")
    if not decision:
        return f"AUTHORIZATION DENIED: {decision.reason}"
    
    try:
        result = db_flag(account_id, FlagRequest(reason=reason))
        return "Account successfully flagged."
    except HTTPException as e:
        return f"Error: {e.detail}"
    except Exception as e:
        return f"Request failed: {str(e)}"

@tool
def approve_payout(account_id: str, amount: float) -> str:
    """Approve a payout and deduct from the account balance."""
    # Step 1: Read account to check flag status
    try:
        account_data = get_account(account_id)
    except HTTPException as e:
        return f"Error: {e.detail}"
    
    # Step 2: OPA context-sensitive check
    # This is the critical security gate — even if the LLM was tricked
    # into calling approve_payout, OPA blocks it on flagged accounts.
    is_flagged = "suspicious_activity" in account_data.get("flags", [])
    decision = opa_evaluate(
        _get_token(),
        "approve_payout",
        context={"account_flagged": is_flagged, "account_id": account_id}
    )
    if not decision:
        return f"AUTHORIZATION DENIED: {decision.reason}"
    
    # Step 3: Execute the payout (only reached if OPA allows)
    try:
        result = db_payout(account_id, PayoutRequest(amount=amount))
        return f"Payout of {amount} approved. New balance: {result['new_balance']}"
    except HTTPException as e:
        return f"Error: {e.detail}"
    except Exception as e:
        return f"Request failed: {str(e)}"

tools = [read_account, flag_account, approve_payout]
tool_node = ToolNode(tools)

# ── LLM Setup (Self-Healing) ─────────────────────────────────
llm_with_tools = build_resilient_llm(
    tools=tools,
    primary_model="gpt-4o",
    fallback_model="gpt-4o-mini",
    temperature=0
)

# ── Agent State ──────────────────────────────────────────────
class FraudState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    account_id: str
    transaction_details: str

# ── Nodes ────────────────────────────────────────────────────
def analyze_node(state: FraudState):
    """LLM node that decides whether to call tools or return a final answer."""
    messages = list(state["messages"])
    updates = []
    
    # If this is the first message, add the system prompt
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_msg = SystemMessage(
            content=(
                "You are a Fraud Detector Agent. Your job is to analyze transactions.\n"
                "1. Use 'read_account' to get account history.\n"
                "2. If the transaction seems legitimate, use 'approve_payout'.\n"
                "3. If the transaction looks suspicious (e.g., international transfers without history), "
                "use 'flag_account' and DO NOT approve the payout.\n"
                "Respond with a clear final decision once you have acted."
            )
        )
        human_msg = HumanMessage(
            content=f"Analyze this payout request for account {state['account_id']}:\n"
                    f"{state['transaction_details']}"
        )
        messages = [sys_msg, human_msg]
        updates.extend([sys_msg, human_msg])

    response = llm_with_tools.invoke(messages)
    updates.append(response)
    return {"messages": updates}

def should_continue(state: FraudState):
    """Determine if we need to execute tools or if the LLM is done."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ── Graph Assembly ───────────────────────────────────────────
def build_fraud_agent():
    graph = StateGraph(FraudState)
    
    graph.add_node("analyze", analyze_node)
    graph.add_node("tools", tool_node)
    
    graph.set_entry_point("analyze")
    graph.add_conditional_edges("analyze", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "analyze")
    
    return graph.compile()

# ── Standalone Test ──────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m agents.fraud_detector.agent <account_id> <transaction_text>")
        sys.exit(1)
        
    agent = build_fraud_agent()
    result = agent.invoke({
        "messages": [],
        "account_id": sys.argv[1],
        "transaction_details": sys.argv[2]
    })
    print("\n--- FINAL AGENT RESPONSE ---")
    print(result["messages"][-1].content)
