"""
FinGuard AgentOps — Vector Store Configuration
Uses ChromaDB in local persistent mode (no Docker required).
Stores lending policies and regulatory documents for RAG retrieval.
"""

import chromadb
from chromadb.config import Settings


# ── Local persistent ChromaDB client ─────────────────────────
CHROMA_PERSIST_DIR = "./chroma_data"
COLLECTION_NAME = "finguard_policies"


def get_chroma_client():
    """Return a persistent local ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_collection():
    """Get or create the finguard_policies collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "FinGuard lending policies and regulatory documents"},
    )


def seed_vector_store():
    """
    Load sample lending policy documents into the vector store.
    These are the regulatory and procedural guides that the Loan Analyst
    agent retrieves at runtime to ground its decisions.
    """
    collection = get_collection()

    # Only seed if collection is empty
    if collection.count() > 0:
        print(f"Collection '{COLLECTION_NAME}' already has {collection.count()} documents. Skipping seed.")
        return collection

    documents = [
        # Credit score requirements
        "Loan approval requires a minimum credit score of 650. "
        "Applicants with scores below 650 must be rejected regardless of income level.",

        # Debt-to-income ratio
        "The maximum acceptable debt-to-income (DTI) ratio is 40%. "
        "Calculate DTI as total monthly debt payments divided by gross monthly income. "
        "Applications exceeding 40% DTI must be rejected.",

        # Bank statement requirements
        "Applicants must provide at least 3 consecutive months of bank statements. "
        "Statements must show the applicant's full legal name and account number.",

        # Loan amount limits
        "Maximum loan amount is 5x the applicant's verified monthly net income. "
        "Loan amounts exceeding this threshold require additional collateral documentation.",

        # Compliance requirements
        "All loan applications must comply with the Consumer Credit Act and "
        "Fair Lending regulations. Decisions must be documented with specific "
        "policy references supporting the approval or rejection.",

        # Suspicious activity
        "Applications showing irregular deposit patterns — such as multiple large "
        "round-number deposits within a short period — must be flagged for manual "
        "review before any automated approval.",

        # Income verification
        "Net monthly income is calculated as total credits minus total debits "
        "averaged over the statement period. Exclude one-time transfers over $10,000 "
        "from income calculations as they may represent non-recurring events.",
    ]

    ids = [f"policy_{i+1}" for i in range(len(documents))]
    metadatas = [{"source": "lending_policy", "version": "1.0"} for _ in documents]

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"Seeded {len(documents)} policy documents into '{COLLECTION_NAME}'.")
    return collection


def query_policies(query_text: str, n_results: int = 3) -> list[str]:
    """
    Query the vector store for relevant lending policies.
    Returns a list of policy text strings ranked by relevance.
    """
    collection = get_collection()
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results["documents"][0] if results["documents"] else []


if __name__ == "__main__":
    seed_vector_store()
    # Quick test
    results = query_policies("What credit score is needed for a loan?")
    print(f"\nTest query returned {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"  {i+1}. {doc[:80]}...")
