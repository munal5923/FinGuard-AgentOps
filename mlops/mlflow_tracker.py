"""
FinGuard AgentOps — LLM Diagnostics with MLflow
Enables deep tracing of LangChain and LangGraph executions.

While OpenTelemetry gives us the high-level API request lifecycle, 
MLflow autologging intercepts the actual LLM payloads. This allows us to see:
  - Exact prompts sent to OpenAI
  - Exact JSON tool definitions provided to the LLM
  - Token counts and token latency
  - Raw responses from the LLM before any parsing

This is critical for diagnosing hallucination or prompt injection success rates.
"""

import os
import logging
import mlflow

logger = logging.getLogger("finguard.mlflow")

def init_llm_diagnostics():
    """
    Initialize MLflow tracking for LangChain.
    Call this once when the application starts.
    """
    # Create a local SQLite tracking database
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
    
    try:
        # Set the active experiment for FinGuard
        mlflow.set_experiment("finguard-agentops-traces")
        
        # Autolog intercepts all LangChain classes (ChatOpenAI, StateGraph, etc.)
        # and records their inputs/outputs as MLflow traces.
        mlflow.langchain.autolog(
            log_traces=True,
        )
        logger.info("MLflow Diagnostics: LangChain autologging enabled.")
        
    except Exception as e:
        logger.error(f"Failed to initialize MLflow diagnostics: {e}")

# ── Standalone Test ──────────────────────────────────────────
if __name__ == "__main__":
    print("=== MLflow LLM Diagnostics Test ===\n")
    
    init_llm_diagnostics()
    
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    print("Sending test request to LLM. Check the mlflow.db database for the trace...")
    response = llm.invoke([HumanMessage(content="Hello! This is a test trace for FinGuard MLflow.")])
    print(f"Response: {response.content}")
    print("\n✅ Trace recorded successfully.")
