import pytest
import requests
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

# The endpoint we are testing for quality
API_URL = "http://localhost:8000/agents/fraud-detector/analyze"

def test_fraud_detector_quality():
    """
    Ensure the security layers (NeMo, Presidio) don't destroy the AI's 
    ability to give faithful and relevant answers to BENIGN prompts.
    """
    input_prompt = "Review the recent transfer of $500 to account ACC002."
    
    # Send request to our actual API
    response = requests.post(
        API_URL,
        json={"account_id": "ACC001", "transaction_details": input_prompt}
    )
    
    # We expect a 200 OK for a benign prompt
    assert response.status_code == 200, "Benign prompt was blocked!"
    
    actual_output = response.json().get("result", "")
    
    # Define the expected context (what the agent should base its answer on)
    retrieval_context = [
        "Account ACC001 has a clean history.",
        "Transfer of $500 is within normal limits for ACC001."
    ]

    # Create a DeepEval Test Case
    test_case = LLMTestCase(
        input=input_prompt,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    # Metric 1: Faithfulness (Is the output hallucinated?)
    faithfulness = FaithfulnessMetric(threshold=0.8)
    
    # Metric 2: Answer Relevancy (Did the agent actually answer the prompt?)
    relevancy = AnswerRelevancyMetric(threshold=0.8)

    # Run the assertions
    assert_test(test_case, [faithfulness, relevancy])
