<p align="center">
  <h1 align="center">FinGuard AgentOps</h1>
  <p align="center">
    A zero-trust, multi-agent AI platform for FinTech — secured by design, observable by default.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-LangChain-1C3C3C?style=flat-square" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Security-Defense--in--Depth-DC143C?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Active%20Research-orange?style=flat-square" />
</p>

---

## Overview

Most AI prototypes connect an LLM directly to a database and hope for the best. FinGuard takes a different approach.

FinGuard AgentOps is a production-grade platform that treats AI agents as **zero-trust actors**. Every agent is sandboxed behind cryptographic authorization, semantic input filtering, and automated output sanitization. The system is fully instrumented with distributed tracing, real-time security dashboards, and autonomous self-healing capabilities.

The project demonstrates how to safely deploy autonomous LLM agents in high-risk environments — where prompt injection, data exfiltration, and hallucination are not theoretical risks, but engineering constraints that must be solved at the infrastructure layer.

> **Note:** This is an active research project. The core platform (Phases 1–5), including the ASMO orchestrator and Kubernetes deployment templates, is fully implemented and functional. Additional phases covering adversarial red-teaming and cascading failure simulation are currently in development. See the [Roadmap](#roadmap) for details.

---

## Architecture

FinGuard is structured as four distinct layers, each building on the last.

### Agents

Four specialized LangGraph agents, each with a narrow scope of responsibility:

| Agent | Role | Tools |
|-------|------|-------|
| **Fraud Detector** | Analyzes transactions, flags suspicious accounts, approves or denies payouts | `read_account`, `flag_account`, `approve_payout` |
| **KYC Agent** | Stateful conversational agent for identity verification | None (conversational only) |
| **Support Agent** | Resolves customer tickets using internal policy documents | `search_policies`, `resolve_ticket` |
| **Loan Analyst** | Ingests PDF bank statements and returns loan eligibility assessments | PDF extraction pipeline |

### Security Gateway

A 4-layer defense-in-depth perimeter that wraps every agent interaction:

```
User Request
     │
     ▼
┌─────────────────────────────────────────────┐
│  Layer 1: NeMo Guardrails (Input Filter)    │  ← Semantic + LLM-based jailbreak detection
├─────────────────────────────────────────────┤
│  Layer 2: Agent Execution                   │
│    └─ Layer 2a: OPA + JWT (Authorization)   │  ← Cryptographic tool-call gating
├─────────────────────────────────────────────┤
│  Layer 3: Pydantic Schemas (Output Format)  │  ← Strict API contract enforcement
├─────────────────────────────────────────────┤
│  Layer 4: Presidio (Output Sanitization)    │  ← PII redaction before response delivery
└─────────────────────────────────────────────┘
     │
     ▼
  API Response
```

**How each layer works:**

- **NeMo Guardrails** — Intercepts prompts *before* they reach the LLM. Uses vector embedding similarity and a fast classifier model (`gpt-4o-mini`) to detect jailbreak attempts, even ones never seen before. Not regex-based.
- **OPA + JWT** — Agents receive short-lived, cryptographically signed tokens at startup. When an agent calls a tool (e.g., `approve_payout`), OPA verifies the token and checks runtime context (e.g., whether the account is flagged). A compromised LLM cannot forge or alter these tokens.
- **Pydantic Schemas** — Enforces strict output structure. If a prompt injection causes the LLM to return malformed data instead of the expected JSON contract, validation fails before the response is sent.
- **Microsoft Presidio** — Scans every outbound response for PII (SSNs, credit card numbers, emails, and custom FinGuard account IDs). Detected entities are replaced with `[REDACTED]` before the response leaves the server.

### Observability

Full-stack instrumentation for debugging, auditing, and real-time monitoring:

| Tool | Purpose |
|------|---------|
| **OpenTelemetry** | Distributed tracing across the full request lifecycle (API → NeMo → Agent → DB → Presidio) |
| **Prometheus** | Time-series metrics collection — tracks API throughput, agent latency, NeMo blocks, OPA denials, and Presidio redactions |
| **Grafana** | Pre-built dashboard for real-time visualization of all security and performance metrics |
| **MLflow** | Deep LLM diagnostics — captures exact system prompts, tool schemas, token counts, and raw model responses |

### Self-Healing

The platform includes an autonomous circuit breaker. If the primary model (`gpt-4o`) fails due to a timeout, rate limit, or API outage, the orchestrator intercepts the error and seamlessly routes the request to a fallback model (`gpt-4o-mini`). The failover event is recorded in Prometheus and surfaced on the Grafana dashboard.

No manual intervention required.

---

## Project Structure

```
FinGuard/
├── api/
│   └── main.py                  # FastAPI gateway — all endpoints, middleware, telemetry
├── agents/
│   ├── fraud_detector/          # LangGraph ReAct agent with OPA-secured tools
│   ├── kyc_agent/               # Stateful conversational verification agent
│   ├── support_agent/           # Policy-aware ticket resolution agent
│   └── loan_analyst/            # PDF ingestion and loan assessment agent
├── gateway/
│   ├── nemo_rails/              # Colang flows + LLM classifier for input interception
│   ├── opa_policies/            # Rego policy definitions + Python evaluation client
│   ├── presidio/                # PII scanner with custom FinGuard entity recognizers
│   └── guardrails_schemas/      # Pydantic strict output validation models
├── mlops/
│   ├── telemetry.py             # OpenTelemetry provider, tracer factory, span helpers
│   ├── metrics.py               # Prometheus metric definitions (counters, histograms, gauges)
│   ├── mlflow_tracker.py        # MLflow LangChain autologging initialization
│   └── self_healing.py          # Circuit breaker with automatic model fallback
├── infrastructure/
│   ├── docker-compose.yml       # Prometheus + Grafana container orchestration
│   ├── prometheus/              # Scrape configuration
│   └── grafana/                 # Dashboard JSON + provisioning configs
├── shared/
│   ├── token_issuer.py          # JWT minting and verification for agent identity
│   ├── models.py                # Shared Pydantic models
│   └── simulated_db.py          # In-memory FinTech database with REST endpoints
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key
- Docker (optional, for Grafana/Prometheus dashboards)

### Installation

```bash
git clone https://github.com/munal5923/FinGuard-AgentOps.git
cd FinGuard-AgentOps

python3 -m venv fin_venv
source fin_venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-key-here
```

### Running the API

```bash
uvicorn api.main:app --reload
```

The API is available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger documentation.

### Running the Observability Stack

```bash
cd infrastructure
sudo docker compose up -d
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | `http://localhost:3000` | `admin` / `finguard` |
| Prometheus | `http://localhost:9090` | — |

Navigate to **Dashboards → Security & Operations → FinGuard Security & Observability** to view live metrics.

### Viewing LLM Traces

```bash
./fin_venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://localhost:5000` to inspect raw LLM prompts, tool schemas, token usage, and agent reasoning chains.

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **API Framework** | FastAPI, Uvicorn |
| **AI Orchestration** | LangChain, LangGraph, OpenAI GPT-4o |
| **Input Security** | NVIDIA NeMo Guardrails (Colang + LLM classifier) |
| **Authorization** | Open Policy Agent (OPA), PyJWT |
| **Output Security** | Microsoft Presidio, Pydantic |
| **Tracing** | OpenTelemetry |
| **Metrics** | Prometheus, Grafana |
| **LLM Diagnostics** | MLflow |
| **Infrastructure** | Docker Compose, Kubernetes (Helm) |

---

## Roadmap

FinGuard is an ongoing research project. The following phases are currently implemented or in development:

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Specialized LangGraph multi-agent architecture | ✅ Complete |
| **Phase 2** | Database tooling and simulated FinTech environment | ✅ Complete |
| **Phase 3** | Defense-in-depth security gateway (NeMo, OPA, Presidio, Pydantic) | ✅ Complete |
| **Phase 4** | MLOps observability and self-healing (OpenTelemetry, Prometheus, Grafana, MLflow) | ✅ Complete |
| **Phase 5** | ASMO Orchestrator & Kubernetes Deployment (Helm charts, HPA, Kill-Switch) | ✅ Complete |
| **Phase 6** | The Red Team Engine — Automated adversarial testing (PromptFoo) and quality metrics (DeepEval) | 🔧 In Progress |
| **Phase 7** | Cascading Failure Simulation — Research study measuring poisoned memory propagation across multi-agent workflows | 📋 Planned |

Contributions, feedback, and discussions are welcome. If you are researching LLM security in production systems, feel free to open an issue or reach out.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
