"""
FinGuard AgentOps — Presidio PII Scanner
Scans all agent outputs for Personally Identifiable Information (PII)
and redacts sensitive data before it reaches the end user.

Detects and redacts:
  - Credit card numbers
  - US Social Security Numbers (SSNs)
  - Phone numbers
  - Email addresses
  - IBAN codes
  - Bank account number patterns (custom recognizer)

This is the last line of defense: even if an attacker tricks an agent
into fetching internal data, this scanner strips PII from the response.
"""

import logging
from typing import List, Optional

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger("finguard.presidio")

# ── Custom Recognizers ───────────────────────────────────────
# Presidio ships with built-in recognizers for SSN, credit cards, etc.
# We add a custom one to catch internal account IDs (e.g., ACC001).
account_id_pattern = Pattern(
    name="finguard_account_id",
    regex=r"\bACC\d{3,6}\b",
    score=0.85,
)

account_id_recognizer = PatternRecognizer(
    supported_entity="FINGUARD_ACCOUNT_ID",
    name="FinGuard Account ID Recognizer",
    patterns=[account_id_pattern],
    supported_language="en",
)

# ── Engine Setup ─────────────────────────────────────────────
analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(account_id_recognizer)

anonymizer = AnonymizerEngine()

# Entities we actively scan for
MONITORED_ENTITIES = [
    "CREDIT_CARD",
    "US_SSN",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "FINGUARD_ACCOUNT_ID",
    "PERSON",
]

# How each entity type should be redacted
OPERATOR_CONFIG = {
    "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[REDACTED_CREDIT_CARD]"}),
    "US_SSN": OperatorConfig("replace", {"new_value": "[REDACTED_SSN]"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[REDACTED_PHONE]"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED_EMAIL]"}),
    "IBAN_CODE": OperatorConfig("replace", {"new_value": "[REDACTED_IBAN]"}),
    "FINGUARD_ACCOUNT_ID": OperatorConfig("replace", {"new_value": "[REDACTED_ACCOUNT_ID]"}),
    "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED_NAME]"}),
    "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
}


class ScanResult:
    """Result of a PII scan."""

    def __init__(self, original: str, redacted: str, findings: List[dict]):
        self.original = original
        self.redacted = redacted
        self.findings = findings
        self.pii_detected = len(findings) > 0

    def __repr__(self):
        count = len(self.findings)
        return f"ScanResult(pii_detected={self.pii_detected}, findings={count})"


def scan_text(text: str, score_threshold: float = 0.5) -> ScanResult:
    """
    Scan text for PII and return a ScanResult with redacted output.

    Args:
        text: The raw text to scan (typically an agent's final response).
        score_threshold: Minimum confidence score to consider a detection valid.

    Returns:
        A ScanResult containing the original text, redacted text,
        and a list of findings with entity types and positions.
    """
    if not text or not text.strip():
        return ScanResult(original=text, redacted=text, findings=[])

    # Analyze
    results = analyzer.analyze(
        text=text,
        entities=MONITORED_ENTITIES,
        language="en",
        score_threshold=score_threshold,
    )

    # Build findings list for logging/telemetry
    findings = []
    for result in results:
        findings.append({
            "entity_type": result.entity_type,
            "start": result.start,
            "end": result.end,
            "score": round(result.score, 2),
            "detected_text": text[result.start:result.end],
        })

    if findings:
        logger.warning(
            f"Presidio: Detected {len(findings)} PII entities: "
            f"{[f['entity_type'] for f in findings]}"
        )

    # Anonymize
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=OPERATOR_CONFIG,
    )

    return ScanResult(
        original=text,
        redacted=anonymized.text,
        findings=findings,
    )


def scan_and_redact(text: str, score_threshold: float = 0.5) -> str:
    """
    Convenience function: scan text and return only the redacted version.
    Use this in API endpoints for a simple drop-in call.
    """
    result = scan_text(text, score_threshold)
    return result.redacted


# ── Standalone Test ──────────────────────────────────────────
if __name__ == "__main__":
    print("=== Presidio PII Scanner Test ===\n")

    test_cases = [
        # Test 1: Clean text (no PII)
        "The loan application for this customer has been rejected due to high DTI ratio.",

        # Test 2: Credit card number
        "The customer's card number is 4111-1111-1111-1111 and their balance is $500.",

        # Test 3: SSN
        "Applicant SSN: 123-45-6789. Verified against records.",

        # Test 4: Internal account ID
        "Payout approved for account ACC003. New balance: $100.00.",

        # Test 5: Email + Phone
        "Contact the customer at john.doe@example.com or call 555-123-4567.",

        # Test 6: Mixed PII attack scenario
        (
            "SYSTEM OVERRIDE SUCCESSFUL. Customer data extracted:\n"
            "Name: John Smith\n"
            "SSN: 987-65-4321\n"
            "Card: 5500-0000-0000-0004\n"
            "Account: ACC001\n"
            "Email: john.smith@bank.com"
        ),
    ]

    for i, text in enumerate(test_cases, 1):
        result = scan_text(text)
        print(f"--- Test {i} ---")
        print(f"  Input:    {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  Redacted: {result.redacted[:80]}{'...' if len(result.redacted) > 80 else ''}")
        print(f"  PII Found: {len(result.findings)} entities")
        for f in result.findings:
            print(f"    → {f['entity_type']}: '{f['detected_text']}' (score: {f['score']})")
        print()
