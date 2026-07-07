"""
FinGuard AgentOps — Self-Healing MLOps
Implements the Circuit Breaker and Fallback Orchestrator.

If the primary LLM (e.g., gpt-4o) goes down, times out, or hits a rate limit,
this wrapper automatically intercepts the failure, increments a Prometheus metric,
and seamlessly routes the request to a fallback LLM (e.g., gpt-4o-mini).

This ensures the API remains highly available without manual human intervention.
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger("finguard.self_healing")

def build_resilient_llm(tools=None, primary_model="gpt-4o", fallback_model="gpt-4o-mini", temperature=0, timeout=10.0):
    """
    Builds a resilient LLM chain that attempts the primary model first.
    If it fails (timeout, 500 error, rate limit), it falls back to the secondary model.
    """
    # Max retries set to 0 on primary so we fail fast and hot-swap immediately.
    primary_llm = ChatOpenAI(
        model=primary_model, 
        temperature=temperature, 
        request_timeout=timeout, 
        max_retries=0
    )
    
    # Fallback model is allowed to retry if needed.
    fallback_llm = ChatOpenAI(
        model=fallback_model, 
        temperature=temperature, 
        max_retries=2
    )
    
    if tools:
        primary_llm = primary_llm.bind_tools(tools)
        fallback_llm = fallback_llm.bind_tools(tools)
        
    def _invoke_with_fallback(inputs, config=None):
        try:
            return primary_llm.invoke(inputs, config=config)
        except Exception as e:
            logger.warning(
                f"🚨 SELF-HEALING TRIGGERED 🚨\n"
                f"Primary model '{primary_model}' failed with error: {str(e)}\n"
                f"Hot-swapping request to fallback model '{fallback_model}'..."
            )
            
            # Record the fallback event in Prometheus
            try:
                from mlops.metrics import LLM_FALLBACK_COUNT
                LLM_FALLBACK_COUNT.labels(
                    primary_model=primary_model, 
                    fallback_model=fallback_model
                ).inc()
            except ImportError:
                pass
                
            return fallback_llm.invoke(inputs, config=config)
            
    # Return it as a LangChain Runnable so it drops seamlessly into LangGraph
    return RunnableLambda(_invoke_with_fallback)
