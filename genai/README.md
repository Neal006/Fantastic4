# LUMIN.AI — GenAI Explanation Layer for Solar Plant Risk Assessment

> **Real-time ML processing with automated ticket generation, AI-powered explanations, and conversational Q&A for solar plant operators.**

Built for **HACKaMINeD 2026**

---

## Table of Contents

1. [What is LUMIN.AI?](#what-is-luminai)
2. [Quick Start (5 Minutes)](#quick-start-5-minutes)
3. [Live Simulation Dashboard](#live-simulation-dashboard)
4. [System Architecture](#system-architecture)
5. [Component Deep Dive](#component-deep-dive)
6. [API Reference](#api-reference)
7. [Test Data & Testing Guide](#test-data--testing-guide)
8. [Prompt Engineering — Iteration History](#prompt-engineering--iteration-history)
9. [LLM Ablation Study & Model Selection](#llm-ablation-study--model-selection)
10. [Hallucination Prevention](#hallucination-prevention)
11. [Integrating Your ML Model](#integrating-your-ml-model)
12. [Troubleshooting](#troubleshooting)
13. [Hackathon Criteria Coverage](#hackathon-criteria-coverage)
14. [Performance Characteristics](#performance-characteristics)
15. [Next Steps & Roadmap](#next-steps--roadmap)

---

## What is LUMIN.AI?

LUMIN.AI is a **GenAI-powered explanation layer** that sits between your ML risk-prediction model and solar plant operators. It solves a critical problem: **ML models output numbers, but operators need actionable guidance**.

### The Problem

Your ML model outputs:
```json
{
  "inverter_id": "INV-P1-L2-0",
  "risk_score": 0.89,
  "shap_values": {
    "inverters[0].temp": +0.35,
    "sensors[0].ambient_temp": +0.15,
    "inverters[0].alarm_code": +0.12
  }
}
```

**Operators ask**: "What does this mean? What should I do?"

### The Solution

LUMIN.AI transforms that into:
```
CRITICAL SHUTDOWN RISK (89%)

Summary:
Inverter INV-P1-L2-0 is experiencing severe overheating with
temperature at 78.6C, significantly above the safe operating
threshold. Combined with high ambient temperature (47.3C) and
alarm code 4003, immediate intervention is required to prevent
equipment damage.

Key Risk Factors:
- Temperature (HIGH IMPACT): 78.6C exceeds thermal protection
  threshold. Cooling system may be compromised.
- Ambient Temperature (MEDIUM IMPACT): 47.3C environmental
  temperature reduces cooling efficiency.
- Alarm Code 4003 (MEDIUM IMPACT): Over-temperature protection
  triggered. System is derating power output.

Recommended Actions:
1. Immediately inspect cooling fans and air filters
2. Check for blocked ventilation or debris accumulation
3. Verify ambient temperature sensors are functioning
4. Review alarm code 4003 in manual Section 7.3
5. Consider temporary shutdown if temperature exceeds 80C

Urgency: IMMEDIATE
```

Plus a **professional PDF maintenance ticket** and a **conversational Q&A** interface.

---

## Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
cd genai
pip install -r requirements.txt
```

**Expected time**: 1-2 minutes

### Step 2: Configure API Key

1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```

2. Get a free Groq API key:
   - Go to https://console.groq.com
   - Sign up (free)
   - Create API key

3. Edit `.env` and add your key:
   ```
   LLM_API_KEY=gsk_your_actual_key_here
   ```

### Step 3: Start Backend

```bash
uvicorn app.main:app --reload --port 8000
```

**First run**: 2-5 minutes (parses PDF, builds embeddings, caches to disk)
**Subsequent runs**: 2-3 seconds (loads from cache)

**Expected output**:
```
[RAG] Loading embedding model 'all-MiniLM-L6-v2' ...
[RAG] Building vector store from PDF (first run) ...
[RAG] Encoding 32 chunks ...
[RAG] Vector store cached to disk.
[RAG] Ready - 32 chunks indexed.
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Verify**: Open http://localhost:8000/docs (Swagger UI should load)

### Step 4: Open Dashboard

**Option A** (Simplest):
- Double-click `simulation_dashboard.html` in the `genai` folder

**Option B** (If Option A shows errors):
- Open new terminal
- Run: `python serve_ui.py`
- Open: http://localhost:8080

### Step 5: Test

1. **See 12 colored inverter cards** (green/orange/red)
2. **Click red card** `INV-P1-L2-0` -> Wait 2-4 seconds -> See AI explanation
3. **Click "Generate PDF"** -> PDF downloads
4. **Type in chat**: "Which inverters have elevated risk?" -> Get AI response

---

## Live Simulation Dashboard

### Features

- **Real-time data simulation** — New inverter readings every 3 seconds
- **Live processing logs** — See ML analysis in real-time
- **Auto-ticket generation** — Automatic tickets when risk > 80%
- **Blinking alerts** — Visual + audio notifications for critical issues
- **Click-to-explain** — Click any log entry for AI explanation
- **Statistics tracking** — Total processed, critical alerts, tickets generated

### How to Use

1. Click **"Start Simulation"**
2. Watch logs appear in real-time
3. When a critical alert appears (red, risk > 80%):
   - Bell icon blinks
   - Auto-ticket generated
   - Alert appears in panel
4. **Click any log entry** to view AI explanation
5. Download PDF tickets from the alerts panel

---

## System Architecture

### High-Level Flow

```
+-------------------------------------------------------------+
|  1. ML MODEL                                                 |
|     Analyzes sensor data -> Outputs risk score + SHAP        |
+--------------------------+----------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|  2. LUMIN.AI (GenAI Layer)                                   |
|     +---------------------------------------------+         |
|     |  a) Guardrails: Validate SHAP features      |         |
|     +---------------------------------------------+         |
|     +---------------------------------------------+         |
|     |  b) RAG: Search inverter manual for context  |         |
|     +---------------------------------------------+         |
|     +---------------------------------------------+         |
|     |  c) LLM: Generate plain-English explanation  |         |
|     +---------------------------------------------+         |
|     +---------------------------------------------+         |
|     |  d) Guardrails: Validate output, add sources |         |
|     +---------------------------------------------+         |
+--------------------------+----------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|  3. OPERATOR                                                 |
|     Receives: Summary + Actions + PDF Ticket + Chat          |
+-------------------------------------------------------------+
```

### Key Components

| Component | What It Does | Technology |
|-----------|--------------|------------|
| **FastAPI Server** | HTTP API that routes requests | Python, FastAPI |
| **Explainer** | Converts ML output to plain English | LLM + Prompt Engineering |
| **RAG Pipeline** | Searches inverter manual for context | PyMuPDF, SentenceTransformers, FAISS |
| **LLM Client** | Talks to Groq/OpenAI API | OpenAI SDK |
| **Guardrails** | Prevents hallucinations | Input/output validation |
| **Agent** | Auto-generates maintenance tickets | Multi-step workflow |
| **Conversation** | Multi-turn chat with memory | Session management |
| **Ticket Generator** | Creates professional PDFs | ReportLab |

### Data Flow Example

```
Operator clicks inverter "INV-P1-L2-0"
    |
UI sends: GET /explanation/INV-P1-L2-0
    |
Explainer fetches ML prediction:
    - Risk score: 0.89
    - SHAP: {temp: +0.35, alarm: +0.12, ...}
    - Raw: {temp: 78.6C, power: 4100W, ...}
    |
Guardrail validates SHAP features (removes invalid ones)
    |
RAG searches inverter manual:
    Query: "overheating thermal protection alarm"
    Returns: ["Section 7.3: Thermal Protection...",
              "Error Code 4003: Over-temperature...", ...]
    |
Explainer formats prompt:
    System: "You are LUMIN.AI. STRICT RULES:
             Only reference provided data..."
    User: "Risk: 0.89, SHAP: temp +0.35,
           Raw: temp 78.6C, Manual: [context]"
    |
LLM (Groq/Llama 3.3) generates JSON:
    {
      "summary": "Inverter at critical shutdown risk...",
      "key_factors": [...],
      "recommended_actions": [...],
      "urgency": "immediate"
    }
    |
Guardrail validates output:
    - Parse JSON
    - Check all cited features exist in input
    - Add disclaimer
    |
Return to operator:
    - Summary paragraph
    - Risk factors with SHAP analysis
    - Numbered action list
    - Urgency badge
    - Grounded sources
```

### File Structure

```
genai/
+-- app/
|   +-- __init__.py
|   +-- main.py              # FastAPI server (all endpoints)
|   +-- config.py            # Environment configuration
|   +-- models.py            # Pydantic data models
|   +-- llm.py               # LLM client (Groq/OpenAI)
|   +-- rag.py               # RAG pipeline (PDF -> embeddings -> search)
|   +-- prompts.py           # All prompt templates
|   +-- guardrails.py        # Hallucination prevention
|   +-- explainer.py         # Risk -> plain-English explanation
|   +-- agent.py             # Agentic ticket generation
|   +-- conversation.py      # Multi-turn chat memory
|   +-- ticket.py            # PDF maintenance ticket generator
|   +-- synthetic_data.py    # Mock ML backend (12 test inverters)
|
+-- comparative_analysis/
|   +-- config.py            # Model configurations for ablation study
|   +-- model_clients.py     # LLM client wrappers (Groq, Gemini, HuggingFace)
|   +-- evaluate.py          # Auto-evaluation on 5 weighted metrics
|   +-- run_ablation.py      # Run all 27 test cases
|   +-- generate_report.py   # Generate graphs and markdown report
|   +-- graphs/              # Generated comparison charts (7 graphs)
|   +-- results/             # Raw ablation results (JSON)
|   +-- README.md            # Detailed ablation study report
|   +-- ABLATION_REPORT.md   # Auto-generated summary report
|
+-- simulation_dashboard.html # Live monitoring UI
+-- dashboard.html            # Standalone inverter dashboard
+-- serve_ui.py               # Simple HTTP server for UI
+-- test_endpoints.py         # Automated test suite
+-- requirements.txt          # Python dependencies
+-- .env.example              # Config template
+-- .env                      # Your config (API keys, gitignored)
+-- 2cdb179b-...ef.pdf        # Inverter technical manual (for RAG)
+-- prompt_engineering.md     # Prompt iteration history
+-- SUPREME_README.md         # This file
```

---

## Component Deep Dive

### 1. FastAPI Server (`app/main.py`)

**What it does**: HTTP API gateway that routes requests.

**Key endpoints**:
- `GET /health` — System status
- `GET /explanation/{inverter_id}` — AI explanation
- `POST /chat` — Multi-turn Q&A
- `POST /agent/maintenance-ticket/{inverter_id}` — Generate ticket
- `GET /agent/risk-report/{plant_id}` — Plant-wide report
- `GET /inverters` — List all inverters
- `GET /predictions` — Raw ML predictions

**Startup sequence**:
1. Load config from `.env`
2. Initialize LLM client
3. Initialize RAG (parse PDF, build embeddings)
4. Start accepting requests

### 2. LLM Client (`app/llm.py`)

**What it does**: Thin wrapper around OpenAI-compatible APIs.

**Supports**:
- Groq (recommended — free, fast)
- OpenAI
- Together AI
- Any OpenAI-compatible endpoint

**Configuration** (`.env`):
```
LLM_API_KEY=gsk_xxxxx
LLM_MODEL=llama-3.3-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1
```

**Methods**:
- `generate(system, user)` — Single-turn completion
- `generate_with_history(system, messages)` — Multi-turn with context
- `check_connection()` — Health check

### 3. RAG Pipeline (`app/rag.py`)

**What it does**: Retrieval-Augmented Generation — grounds LLM in inverter manual.

**Architecture**:
```
Inverter Manual PDF (34 MB)
    |
PyMuPDF Parser (extracts text)
    |
Text Chunking (800 words, 200 overlap)
    |
SentenceTransformer Embeddings (all-MiniLM-L6-v2)
    |
FAISS Vector Store (cached to disk)
    |
Cosine Similarity Search
    |
Top-K Relevant Chunks
```

**First run**: 2-5 minutes (builds index, caches to `vector_store/`)
**Subsequent runs**: Instant (loads from cache)

**Example**:
```python
query = "overheating thermal protection"
chunks = rag.retrieve(query, top_k=3)
# Returns:
# [
#   {"text": "Section 7.3: Thermal Protection...", "score": 0.87},
#   {"text": "Error Code 4003: Over-temperature...", "score": 0.82},
#   ...
# ]
```

### 4. Prompts (`app/prompts.py`)

**What it does**: Structured prompts for different tasks.

**Prompt types**:
- `SYSTEM_PROMPT_EXPLANATION` — Risk analysis
- `SYSTEM_PROMPT_CHAT` — Conversational Q&A
- `SYSTEM_PROMPT_TICKET` — Maintenance ticket generation
- `SYSTEM_PROMPT_RISK_REPORT` — Plant-wide reports

**Key features**:
- Explicit STRICT RULES to prevent hallucinations
- JSON schema enforcement
- Grounding in provided data only
- Technical language for operators

**Example** (simplified):
```
System:
  You are LUMIN.AI, an expert solar-plant diagnostic assistant.

  STRICT RULES:
  1. ONLY reference data values explicitly provided
  2. ONLY discuss features in the SHAP analysis
  3. If data insufficient, say so
  4. Ground recommendations in manual context
  5. Return ONLY valid JSON

  JSON SCHEMA: {
    "summary": "<2-3 sentence summary>",
    "key_factors": [...],
    "recommended_actions": [...],
    "urgency": "immediate | within_24h | scheduled | routine"
  }

User:
  INVERTER: {id}   PLANT: {plant}   BLOCK: {block}
  Risk Score: {score}   Risk Class: {class}
  TOP SHAP FEATURES: ...
  RAW SENSOR VALUES: ...
  RELEVANT MANUAL CONTEXT: ...
```

See [Prompt Engineering — Iteration History](#prompt-engineering--iteration-history) for full evolution.

### 5. Guardrails (`app/guardrails.py`)

**What it does**: Prevents hallucinations — ensures LLM never fabricates data.

**4 layers**:

**Layer 1 — Input Validation**:
```python
# Remove SHAP features not in dataset schema
valid_shap = validate_shap_features(prediction.shap_values)
```

**Layer 2 — Prompt Rules**:
```
STRICT RULES:
1. ONLY reference provided data
2. NEVER fabricate sensor readings
```

**Layer 3 — Output Validation**:
```python
# Check every cited feature exists in input
for factor in response['key_factors']:
    if factor['feature'] not in provided_shap:
        flag_violation()
```

**Layer 4 — Disclaimer**:
```
"All referenced values come directly from sensor telemetry.
No values have been fabricated."
```

### 6. Explainer (`app/explainer.py`)

**What it does**: Converts ML predictions into operator-friendly narratives.

**Flow**:
1. Fetch prediction (risk + SHAP + raw features)
2. Validate SHAP features (guardrail)
3. RAG retrieval (search manual for context)
4. Format prompt with all data
5. Call LLM
6. Parse JSON response
7. Validate output (guardrail)
8. Return structured explanation

### 7. Agent (`app/agent.py`)

**What it does**: Autonomous multi-step workflow for ticket generation.

**Agentic pipeline**:
1. Autonomously retrieve prediction data
2. Validate SHAP features
3. RAG: Fetch troubleshooting context
4. Format data for LLM
5. LLM generates ticket content (JSON)
6. Parse JSON (with fallback)
7. Generate professional A4 PDF
8. Return ticket ID + PDF path

**Why "agentic"?**
- No human intervention required
- Agent decides what data to fetch
- Agent queries RAG autonomously
- Agent drafts complete ticket end-to-end

### 8. Conversation (`app/conversation.py`)

**What it does**: Multi-turn chat with session-based memory.

**Architecture**:
```python
sessions = {
  "session-uuid-123": [
    {"role": "user", "content": "Which inverters have risk?"},
    {"role": "assistant", "content": "INV-P1-L2-0 and..."},
    {"role": "user", "content": "Tell me more about the first one"},
    {"role": "assistant", "content": "INV-P1-L2-0 is overheating..."}
  ]
}
```

**Features**:
- Session IDs for continuity
- Rolling window (last 20 turns)
- Context passed to LLM on every turn

### 9. Ticket Generator (`app/ticket.py`)

**What it does**: Professional A4 maintenance tickets via ReportLab.

**Layout**:
- Header with branding
- Metadata table (ticket ID, priority, risk score)
- Issue title and description
- Root cause analysis (SHAP-based)
- Recommended actions (numbered)
- Estimated downtime
- Parts needed
- Safety notes
- Escalation flag (for critical issues)
- Disclaimer

**Output**: `genai/tickets/TKT-YYYYMMDD-XXXXXX.pdf`

### 10. Synthetic Data (`app/synthetic_data.py`)

**What it does**: Mock ML backend for testing.

**Data structure**:
- 3 plants (Plant 1, 2, 3)
- 2 blocks per plant (Block A, B)
- 2 inverters per block (12 total)
- Diverse scenarios (normal, degradation, shutdown, various failure modes)
- Realistic SHAP values and raw sensor readings

**When to replace**: When your ML model API is ready.

---

## API Reference

### Core Endpoints

#### `GET /health`

Health check — returns system status.

**Response**:
```json
{
  "status": "healthy",
  "llm_connected": true,
  "vector_store_loaded": true,
  "inverters_monitored": 12,
  "timestamp": "2026-03-05T15:30:00Z"
}
```

#### `GET /explanation/{inverter_id}`

Generate AI explanation for a single inverter.

**Parameters**:
- `inverter_id` (path): Inverter identifier (e.g., `INV-P1-L2-0`)

**Response**:
```json
{
  "inverter_id": "INV-P1-L2-0",
  "plant_id": "plant_1",
  "risk_score": 0.89,
  "risk_class": "shutdown_risk",
  "summary": "Inverter at critical shutdown risk...",
  "key_factors": [
    {
      "feature": "inverters[0].temp",
      "impact": "high",
      "explanation": "Temperature at 78.6C exceeds..."
    }
  ],
  "recommended_actions": [
    "Immediately inspect cooling system...",
    "Check air filters..."
  ],
  "urgency": "immediate",
  "generated_at": "2026-03-05T15:30:00Z",
  "grounded_sources": [
    "ML model prediction (SHAP analysis)",
    "Inverter sensor telemetry",
    "Inverter technical manual (RAG)"
  ],
  "disclaimer": "This analysis is AI-generated..."
}
```

### Chat Endpoints

#### `POST /chat`

Multi-turn conversational Q&A.

**Request**:
```json
{
  "session_id": "optional-session-uuid",
  "message": "Which inverters have elevated risk?"
}
```

**Response**:
```json
{
  "session_id": "session-uuid-123",
  "response": "Based on current predictions, the following inverters...",
  "sources_used": [
    "ML prediction data",
    "Inverter technical manual (RAG)"
  ],
  "data_referenced": null
}
```

### Agent Endpoints

#### `POST /agent/maintenance-ticket/{inverter_id}`

Generate maintenance ticket (JSON + PDF).

**Response**:
```json
{
  "ticket_id": "TKT-20260305-A3F2B1",
  "inverter_id": "INV-P1-L2-0",
  "pdf_path": "tickets/TKT-20260305-A3F2B1.pdf",
  "ticket_data": {
    "title": "Critical Overheating - Immediate Action Required",
    "priority": "P1-Critical",
    "description": "...",
    "root_cause_analysis": "...",
    "recommended_actions": ["..."],
    "estimated_downtime": "2-4 hours",
    "parts_needed": ["Replacement air filter", "Thermal paste"],
    "safety_notes": ["De-energize before maintenance"],
    "escalation_needed": true
  },
  "generated_at": "2026-03-05T15:30:00.000Z"
}
```

#### `GET /agent/maintenance-ticket/{inverter_id}/pdf`

Download maintenance ticket as PDF.

**Response**: PDF file (`application/pdf`)

#### `GET /agent/risk-report/{plant_id}`

Generate plant-wide narrative risk report.

**Response**:
```json
{
  "plant_id": "plant_1",
  "report": "# Plant 1 Risk Report\n\n## Executive Summary\n...",
  "generated_at": "2026-03-05T15:30:00.000Z"
}
```

### Utility Endpoints

#### `GET /inverters`

List all monitored inverters grouped by plant/block.

**Response**:
```json
{
  "plant_1": {
    "name": "Plant 1 - Celestical",
    "blocks": {
      "Block A": ["INV-P1-L1-0", "INV-P1-L1-1"],
      "Block B": ["INV-P1-L2-0", "INV-P1-L2-1"]
    }
  }
}
```

#### `GET /predictions`

Get raw ML predictions for all inverters.

**Query Parameters**:
- `plant_id` (optional): Filter by plant

**Response**:
```json
[
  {
    "inverter_id": "INV-P1-L2-0",
    "plant_id": "plant_1",
    "risk_score": 0.89,
    "risk_class": "shutdown_risk",
    "shap_values": {"...": "..."},
    "raw_features": {"...": "..."},
    "timestamp": "2026-03-05T15:30:00Z"
  }
]
```

#### `GET /predictions/{inverter_id}`

Get raw ML prediction for a single inverter.

**Full API documentation**: http://localhost:8000/docs (Swagger UI)

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/health

# Get explanation
curl http://localhost:8000/explanation/INV-P1-L2-0

# List all inverters
curl http://localhost:8000/inverters

# Get all predictions
curl http://localhost:8000/predictions

# Chat
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Which inverters have risk?\"}"

# Generate ticket
curl -X POST http://localhost:8000/agent/maintenance-ticket/INV-P1-L2-0

# Download ticket PDF
curl http://localhost:8000/agent/maintenance-ticket/INV-P1-L2-0/pdf --output ticket.pdf

# Plant risk report
curl http://localhost:8000/agent/risk-report/plant_1
```

---

## Test Data & Testing Guide

### Synthetic Inverters (12 Total)

| Inverter | Plant | Block | Risk | Scenario |
|----------|-------|-------|------|----------|
| INV-P1-L1-0 | Plant 1 | A | 12% | Normal operation |
| INV-P1-L1-1 | Plant 1 | A | 65% | String degradation |
| INV-P1-L2-0 | Plant 1 | B | **89%** | **Overheating (shutdown risk)** |
| INV-P1-L2-1 | Plant 1 | B | 8% | Normal operation |
| INV-P2-L1-0 | Plant 2 | A | 72% | Alarm triggered |
| INV-P2-L1-1 | Plant 2 | A | 15% | Normal operation |
| INV-P2-L2-0 | Plant 2 | B | 58% | Low power output |
| INV-P2-L2-1 | Plant 2 | B | **91%** | **Grid fault (shutdown risk)** |
| INV-P3-L1-0 | Plant 3 | A | 5% | Normal operation |
| INV-P3-L1-1 | Plant 3 | A | 11% | Normal operation |
| INV-P3-L2-0 | Plant 3 | B | 45% | Partial shading |
| INV-P3-L2-1 | Plant 3 | B | 52% | Communication issue |

Each has realistic SHAP values and raw sensor readings.

### Automated Test Suite

```bash
python test_endpoints.py
```

This tests all endpoints automatically. Expected output:
```
LUMIN.AI - Endpoint Test Suite

============================================================
  1. Health Check
============================================================
  [200] GET /health

============================================================
  2. Utility - List Inverters & Predictions
============================================================
  [200] GET /inverters
  [200] GET /predictions
  ...
```

### Manual UI Testing

#### Test 1: Normal Operation Inverter
1. Click `INV-P1-L1-0` (green card)
2. Should show: Risk 12%, Urgency ROUTINE, "Operating normally..."

#### Test 2: Degradation Risk Inverter
1. Click `INV-P1-L1-1` (orange card)
2. Should show: Risk 65%, Urgency WITHIN 24H, key factor `inverters[1].pv3_current`

#### Test 3: Shutdown Risk — Overheating
1. Click `INV-P1-L2-0` (red card)
2. Should show: Risk 89%, Urgency IMMEDIATE, Temperature 78.6C, alarm code 4003

#### Test 4: Shutdown Risk — Grid Fault
1. Click `INV-P2-L2-1` (red card)
2. Should show: Risk 91%, Urgency IMMEDIATE, Grid frequency 51.8 Hz

#### Test 5: PDF Ticket Generation
1. Select any red inverter, click "Generate PDF"
2. PDF should contain: Ticket ID, Priority badge, Root cause analysis, Actions, Parts, Safety notes

#### Test 6: Multi-Turn Chat
1. Type: "Which inverters have elevated risk right now?"
2. Type: "Tell me more about the overheating inverter"
3. Type: "Which inverters in Block B have elevated risk?"
4. Chat should remember context from previous messages.

---

## Prompt Engineering — Iteration History

All prompts live in `app/prompts.py`. This section documents the design process.

### Iteration 1 — Baseline (Naive)

**Prompt**:
```
System: You are a helpful assistant that explains solar inverter risk scores.
User:   The inverter {id} has a risk score of {score}. Explain why.
```

**Evaluation**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Accuracy | 2/5 | Frequently hallucinated sensor values not in the input |
| Structure | 2/5 | Free-form paragraphs; hard to scan on-site |
| Actionability | 1/5 | Vague advice like "check the system" |
| Guardrail compliance | 1/5 | No rules; LLM invented data freely |

**Key Problems**:
1. **Hallucination** — The model fabricated readings (e.g., "current is 2.3 A") that were never in the input.
2. **No structure** — Operators couldn't quickly find the action items.
3. **No SHAP grounding** — The model ignored feature importances entirely and guessed at root causes.

### Iteration 2 — Structured Output + Guardrails (Current Production)

| Change | Rationale |
|--------|-----------|
| Added explicit **STRICT RULES** | Directly addresses hallucination by forbidding fabricated values |
| Required **JSON output schema** | Forces machine-parseable structure for programmatic validation |
| Listed **exact SHAP features** | LLM must reference only these features, preventing invented factors |
| Included **raw sensor values** | Ground-truth numbers so LLM doesn't need to guess |
| Added **RAG manual context** | Grounds troubleshooting advice in the real inverter manual |
| Specified **urgency taxonomy** | `immediate / within_24h / scheduled / routine` — operators know how fast to act |
| Added **estimated_impact** field | Communicates consequence of inaction |

**Evaluation**:

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Accuracy | 4/5 | References only provided values; rare edge-case slips caught by output guardrails |
| Structure | 5/5 | JSON is deterministic; frontend can render cards/tables |
| Actionability | 4/5 | Concrete steps grounded in the manual (e.g., "check air filter per Section 7.3") |
| Guardrail compliance | 4/5 | Programmatic validation catches the remaining hallucinations |

**Remaining Risks & Mitigations**:
- **JSON parse failures** — `parse_llm_json()` handles markdown fences with graceful fallback
- **Feature name mismatch** — `validate_explanation_output()` cross-checks every cited feature against SHAP input; mismatches flagged as `_guardrail_warnings`
- **Urgency hallucination** — validated against an allow-list in guardrails

### Chat Prompt Design

- **System prompt** injects the full plant overview so the LLM always has ground-truth data
- **User prompt** injects per-turn data context and RAG manual excerpts
- Multi-turn history is passed as previous messages, giving context memory without re-sending all data each turn

### Ticket Prompt Design

- Requires parts lists, safety notes, and escalation flags
- Output validated and rendered into a ReportLab PDF with professional layout

### Future Iteration Ideas

1. **Few-shot examples** — Add 2-3 gold-standard examples in the system prompt
2. **Chain-of-thought** — Ask the LLM to reason step-by-step before producing JSON
3. **Self-critique loop** — Second LLM call reviews the first output for hallucinations
4. **Dynamic prompt selection** — Different templates per risk class (shutdown emphasises safety; degradation emphasises monitoring)

---

## LLM Ablation Study & Model Selection

> **Objective:** Determine the best free-tier LLM for LUMIN.AI's solar-plant diagnostic pipeline by comparing models across real production prompts, then justify the final selection with data.

### Models Evaluated

| Provider | Model | Parameters | Free Tier |
|----------|-------|------------|-----------|
| **Groq** | Llama 3.3 70B Versatile | 70B | Yes |
| **Google** | Gemini 2.5 Flash | -- | Yes |
| **HuggingFace** | Qwen 2.5 72B Instruct | 72B | Yes |

All three are large, instruction-tuned models. Each was tested on the **exact same prompts** from `app/prompts.py` across 3 tasks and 3 risk scenarios (**9 test cases per model, 27 total API calls**).

### Evaluation Method

Each model received identical inputs: system prompt + user prompt with real inverter sensor data and SHAP values from `app/synthetic_data.py`. Responses were auto-evaluated on 5 weighted metrics:

| Metric | Weight |
|--------|--------|
| JSON Validity | 25% |
| Hallucination Prevention | 25% |
| Technical Completeness | 25% |
| Response Quality | 15% |
| Urgency Accuracy | 10% |

Latency and token usage were recorded for every call. Full methodology is in `comparative_analysis/evaluate.py`.

### Overall Rankings

| Rank | Model | Overall Score | Avg Latency | JSON Valid | Hallucination Prevention | Urgency | Completeness | Quality |
|------|-------|:------------:|:-----------:|:----------:|:-------------:|:-------:|:------------:|:-------:|
| **1** | **Llama 3.3 70B (Groq)** | **91.7%** | **1.0s** | 67% | 100% | 100% | 100% | 100% |
| **2** | **Qwen 2.5 72B (HuggingFace)** | **91.7%** | **21.2s** | 67% | 100% | 100% | 100% | 100% |
| **3** | Gemini 2.5 Flash (Google) | 79.4% | 8.8s | 44% | 100% | 89% | 81% | 94% |

> **Groq and Qwen tie on accuracy (91.7%)**, but Groq is **21x faster**.

![Overall Scores](comparative_analysis/graphs/01_overall_scores.png)

### Radar Comparison

The radar chart shows each model's normalized score across all 5 evaluation dimensions. Groq and Qwen nearly overlap on every axis except latency, while Gemini shows clear dips in JSON validity and completeness.

![Radar Comparison](comparative_analysis/graphs/02_radar_comparison.png)

### Latency Comparison

| Model | Avg Latency | Min | Max |
|-------|:-----------:|:---:|:---:|
| **Groq Llama 3.3 70B** | **1.0s** | 0.5s | 1.9s |
| Gemini 2.5 Flash | 8.8s | 2.8s | 11.8s |
| Qwen 2.5 72B (HF) | 21.2s | 6.1s | 50.3s |

![Latency Comparison](comparative_analysis/graphs/03_latency_comparison.png)

### Per-Task Breakdown

#### Explanation Task (structured JSON diagnostic)

| Model | Score | Avg Latency |
|-------|:-----:|:-----------:|
| **Groq** | **100%** | **1.2s** |
| **Qwen** | **100%** | **28.0s** |
| Gemini | 82.4% | 10.5s |

#### Ticket Generation Task (maintenance ticket JSON)

| Model | Score | Avg Latency |
|-------|:-----:|:-----------:|
| **Groq** | **100%** | **1.3s** |
| **Qwen** | **100%** | **26.8s** |
| Gemini | 83.3% | 10.1s |

#### Chat Task (conversational Q&A)

| Model | Score | Avg Latency |
|-------|:-----:|:-----------:|
| **Groq** | **75%** | **0.6s** |
| **Qwen** | **75%** | **8.8s** |
| Gemini | 72.5% | 6.0s |

![Per-Task Scores](comparative_analysis/graphs/04_per_task_scores.png)

### Score vs Latency — The Efficiency Frontier

The ideal model is in the **top-left** corner (high score, low latency). Groq is clearly the most efficient.

![Score vs Latency](comparative_analysis/graphs/05_score_vs_latency.png)

### Detailed Metric Heatmap

![Metric Heatmap](comparative_analysis/graphs/06_metric_heatmap.png)

### Token Usage

Gemini tends to produce longer outputs. Groq is the most token-efficient.

![Token Usage](comparative_analysis/graphs/07_token_usage.png)

### Chat Response Comparison

Below are **actual chat responses** from each model for the same operator question across three risk scenarios.

#### Scenario 1: Normal Operation (Risk: 0.12, Class: no_risk)

**Question:** *"What is the current risk level of INV-P1-L1-0 and what actions should I take?"*

| Model | Latency | Score | Response Summary |
|-------|---------|-------|-----------------|
| **Groq** | **0.5s** | 0.75 | Concise: cites risk 0.12, no_risk, negative SHAP values. Defers to manual. |
| Gemini | 11.8s | 0.75 | Too brief: 2 lines, cites risk and class only. |
| Qwen | 6.1s | 0.75 | Most detailed: lists SHAP factors, recommends continued monitoring. |

#### Scenario 2: Degradation Risk (Risk: 0.65, Class: degradation_risk)

**Question:** *"What is the current risk level of INV-P1-L1-1 and what actions should I take?"*

| Model | Latency | Score | Response Summary |
|-------|---------|-------|-----------------|
| **Groq** | **0.7s** | 0.75 | Cites all SHAP features correctly. Defers to manual for actions. |
| Gemini | 3.5s | 0.71 | Minimal 2-line answer, scored lowest. |
| Qwen | 14.2s | 0.75 | Most actionable advice with numbered steps. |

#### Scenario 3: Shutdown Risk (Risk: 0.89, Class: shutdown_risk)

**Question:** *"What is the current risk level of INV-P1-L2-0 and what actions should I take?"*

| Model | Latency | Score | Response Summary |
|-------|---------|-------|-----------------|
| **Groq** | **0.7s** | 0.75 | All 3 critical SHAP factors with exact values. |
| Gemini | 2.8s | 0.71 | 2-line answer, lacks actionable detail. |
| Qwen | 6.1s | 0.75 | Best recommendations with numbered steps. |

#### Chat Comparison Summary

| Metric | Groq Llama 3.3 70B | Gemini 2.5 Flash | Qwen 2.5 72B |
|--------|:-------------------:|:----------------:|:-------------:|
| **Avg Chat Score** | **75.0%** | 72.5% | **75.0%** |
| **Avg Chat Latency** | **0.6s** | 6.0s | 8.8s |
| **Cites SHAP values** | Yes (exact) | Sometimes | Yes (rounded) |
| **Provides actions** | Defers to manual | Minimal | Yes (detailed) |
| **Hallucination-free** | 100% | 100% | 100% |
| **Tone** | Precise, technical | Too brief | Helpful, verbose |

### Why We Chose Groq (Llama 3.3 70B Versatile)

#### 1. Best-in-Class Latency (21x faster)

| Model | Avg Latency | Relative Speed |
|-------|:-----------:|:--------------:|
| **Groq** | **1.0s** | **1x (baseline)** |
| Gemini | 8.8s | 8.8x slower |
| Qwen | 21.2s | 21.2x slower |

For a **real-time diagnostic assistant** where plant operators need immediate answers during critical situations, sub-second response times are non-negotiable.

#### 2. Tied for Highest Accuracy (91.7%)

- **100%** on explanation generation
- **100%** on ticket generation
- **75%** on chat (identical to Qwen)
- **100%** hallucination prevention
- **100%** urgency accuracy
- **100%** technical completeness

#### 3. Zero Hallucinations

Never fabricates sensor readings or invents SHAP features not in the input.

#### 4. Superior JSON Compliance

For structured tasks, Groq produces valid JSON with all required fields. Gemini failed JSON parsing in 5 out of 9 test cases (44% validity vs Groq's 67%).

#### 5. Token Efficiency

- **Groq:** ~248 tokens/response avg
- **Gemini:** ~444 tokens/response avg
- **Qwen:** ~424 tokens/response avg

#### 6. Production Reliability

- Consistent sub-2s latency (no cold starts)
- OpenAI-compatible API (drop-in replacement)
- Generous free tier for development
- No quota exhaustion issues during testing

#### 7. Gemini & Qwen — Why Not?

| Model | Why Not Selected |
|-------|-----------------|
| **Gemini 2.5 Flash** | Lower accuracy (79.4%), poor JSON validity (44%), quota issues on free tier, 8.8x slower |
| **Qwen 2.5 72B** | Same accuracy as Groq but **21x slower** (21.2s avg). HuggingFace free tier has cold starts up to 50s. |

### Final Verdict

> **Groq Llama 3.3 70B Versatile** delivers the best combination of accuracy, speed, and reliability for LUMIN.AI's solar-plant diagnostic pipeline. It is the only model that achieves top-tier quality (91.7%) with production-grade latency (1.0s avg).

### Reproducing the Ablation Study

```bash
cd genai

# 1. Test API keys
python -m comparative_analysis.test_api_keys

# 2. Run ablation study (~3-5 minutes)
python -m comparative_analysis.run_ablation

# 3. Generate graphs and report
python -m comparative_analysis.generate_report
```

Results are saved to `comparative_analysis/results/` and graphs to `comparative_analysis/graphs/`.

---

## Hallucination Prevention

LUMIN.AI implements a **4-layer guardrail system** to ensure the LLM never fabricates data.

| Layer | Mechanism | Location |
|-------|-----------|----------|
| **1. Input Validation** | Only valid SHAP features pass through | `app/guardrails.py` |
| **2. Prompt Rules** | "ONLY reference provided data. NEVER fabricate." | `app/prompts.py` |
| **3. Output Validation** | Every cited feature checked against input SHAP | `app/guardrails.py` |
| **4. Disclaimer** | "All values from sensor telemetry. Nothing fabricated." | Appended to every response |

**Result**: 100% hallucination prevention across all 27 ablation test cases for all 3 models.

---

## Integrating Your ML Model

### Current State (Synthetic Data)

Right now, `app/synthetic_data.py` provides mock predictions:
```python
def get_prediction(inverter_id: str) -> InverterPrediction:
    return SYNTHETIC_PREDICTIONS.get(inverter_id)
```

### Required ML API Response Format

Your ML model should return predictions in this format:

```json
{
  "inverter_id": "INV-P1-L2-0",
  "plant_id": "plant_1",
  "block": "Block B",
  "timestamp": "2026-03-05T15:30:00Z",
  "risk_score": 0.89,
  "risk_class": "shutdown_risk",
  "shap_values": {
    "inverters[0].temp": 0.35,
    "sensors[0].ambient_temp": 0.15,
    "inverters[0].alarm_code": 0.12
  },
  "raw_features": {
    "inverters[0].temp": 78.6,
    "inverters[0].alarm_code": 4003,
    "sensors[0].ambient_temp": 47.3
  }
}
```

**Required fields**: `inverter_id`, `plant_id`, `risk_score`, `risk_class`, `shap_values`, `raw_features`
**Optional**: `timestamp`, `block`, `logger_mac`

### Integration Steps

1. **Add ML API settings to `.env`**:
   ```env
   ML_API_URL=http://your-ml-server:5000/api
   ```

2. **Update `app/synthetic_data.py`** to call ML API instead of returning hardcoded data

3. **Update `app/main.py`** endpoints to use async ML calls

4. **Test**:
   ```bash
   curl http://localhost:8000/explanation/INV-P1-L2-0
   ```

### Integration Checklist

- [ ] ML model API is running and accessible
- [ ] ML API returns data in correct format (matches `InverterPrediction` model)
- [ ] Added ML API URL to `.env`
- [ ] Updated `synthetic_data.py` or created `ml_client.py`
- [ ] Updated `main.py` to use async ML calls
- [ ] Tested `/explanation/{inverter_id}` endpoint
- [ ] Tested `/predictions` endpoint
- [ ] Tested chat functionality
- [ ] Tested ticket generation
- [ ] Verified SHAP features match dataset schema
- [ ] Added error handling for ML API failures

---

## Troubleshooting

### Backend Won't Start

| Cause | Fix |
|-------|-----|
| Missing dependencies | `pip install -r requirements.txt` |
| Port 8000 in use | `uvicorn app.main:app --reload --port 8001` |
| Import errors | Make sure you're in the `genai/` folder |

### LLM Not Connected

| Cause | Fix |
|-------|-----|
| Missing API key | Add `LLM_API_KEY=gsk_your_key_here` to `.env` |
| Invalid API key | Verify at https://console.groq.com |
| Network issues | `curl https://api.groq.com/openai/v1/models` |

### Rate Limit Exceeded

- Wait 10 minutes for reset
- Upgrade Groq: https://console.groq.com/settings/billing
- Switch to OpenAI in `.env`: `LLM_API_KEY=sk-xxxxx`, `LLM_BASE_URL=https://api.openai.com/v1`, `LLM_MODEL=gpt-4o-mini`

### RAG Not Loading

| Cause | Fix |
|-------|-----|
| PDF not found | Verify `2cdb179b-...ef.pdf` exists in `genai/` |
| Corrupted cache | Delete `vector_store/` folder and restart |
| Insufficient memory | First run needs ~2GB RAM for embedding model |

### Explanations Are Generic

- Check ML API response includes `shap_values`
- Verify RAG initialized (check backend logs)
- Ensure inverter manual PDF exists

### PDF Generation Fails

| Cause | Fix |
|-------|-----|
| Directory not writable | `mkdir tickets` |
| ReportLab not installed | `pip install reportlab` |
| Invalid ticket data | Check backend logs for JSON parse errors |

### Chat Not Remembering Context

- Verify `session_id` is being sent from UI
- Backend restart clears all sessions (in-memory only)
- Ensure same `session_id` used for follow-up messages

---

## Hackathon Criteria Coverage

### Required Criteria

| Criterion | Implementation | Location |
|-----------|----------------|----------|
| **Automated Narrative Generation** | Risk score + SHAP -> plain-English summary + actions | `app/explainer.py` |
| **Retrieval-Augmented Q&A (RAG)** | Natural language questions grounded in manual + data | `app/rag.py`, `app/main.py` `/chat` |
| **Prompt Design** | 2 iterations documented with rationale | `prompt_engineering.md`, this README |

### Bonus Criteria (All Implemented)

| Criterion | Implementation | Location |
|-----------|----------------|----------|
| **Agentic Workflow** | Auto-retrieves data -> runs assessment -> drafts ticket | `app/agent.py` |
| **Multi-turn Conversation** | Session-based context memory | `app/conversation.py` |
| **Hallucination Guardrails** | 4-layer validation (input/prompt/output/disclaimer) | `app/guardrails.py` |
| **Multi-class Output** | no_risk / degradation_risk / shutdown_risk | `app/models.py` |
| **Comparative Analysis** | 3-model ablation study with 27 test cases, 7 graphs | `comparative_analysis/` |

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Backend startup (first) | 2-5 min | PDF parsing + embedding (one-time) |
| Backend startup (cached) | 2-3 sec | Loads from cache |
| Health check | <100ms | Simple status check |
| Explanation | 2-4 sec | RAG retrieval + LLM inference |
| Chat turn | 2-3 sec | Context + RAG + LLM |
| PDF ticket | 3-5 sec | LLM + ReportLab rendering |
| Risk report | 4-6 sec | Multi-inverter analysis |

**Optimization tips**:
- Use Groq for fastest LLM inference (~500 tokens/sec)
- RAG cache persists across restarts
- Batch predictions for plant reports

---

## Next Steps & Roadmap

### For Hackathon Demo

1. Start backend: `uvicorn app.main:app --reload --port 8000`
2. Open `simulation_dashboard.html`
3. Click "Start Simulation"
4. Show overheating inverter explanation
5. Generate PDF ticket
6. Ask chat: "Which inverters in Block B have risk?"
7. Highlight bonus features (agentic, multi-turn, guardrails, ablation study)

### For Production Deployment

1. **Connect Real ML Model** (see [Integrating Your ML Model](#integrating-your-ml-model))
2. **Add Authentication**: JWT tokens, API keys
3. **Deploy Backend**: Docker + Kubernetes / Cloud Run
4. **Deploy Frontend**: Production React/Next.js app
5. **Add Monitoring**: Logging, metrics, alerts (Prometheus, Grafana)
6. **Add Database**: PostgreSQL for prediction history, tickets
7. **Add Caching**: Redis for frequently accessed predictions
8. **Scale RAG**: Dedicated vector database (Pinecone, Weaviate)
9. **Add Rate Limiting**: Protect against abuse
10. **HTTPS**: SSL certificates

### For Enhancement

1. **Few-shot Prompting**: Add example explanations to prompts
2. **Chain-of-Thought**: Ask LLM to reason before generating JSON
3. **Self-Critique Loop**: Second LLM reviews first output
4. **Dynamic Prompts**: Different templates per risk class
5. **Historical Analysis**: Trend analysis over time
6. **Predictive Maintenance**: Forecast future failures
7. **Mobile App**: Native iOS/Android operator app
8. **Email Alerts**: Auto-send tickets to maintenance team
9. **Dashboard Analytics**: Plant-wide KPIs and trends
10. **Multi-language**: Support for regional languages

---

## Summary

**LUMIN.AI** is a complete GenAI explanation layer that:

1. **Takes ML predictions** (risk scores + SHAP values)
2. **Searches inverter manual** for relevant context (RAG)
3. **Generates plain-English explanations** via LLM (Groq Llama 3.3 70B)
4. **Prevents hallucinations** with 4-layer guardrails
5. **Creates PDF maintenance tickets** automatically (agentic workflow)
6. **Answers operator questions** via multi-turn chat
7. **Provides plant-wide reports** for management
8. **Backed by data** — 3-model ablation study with 27 test cases proves model selection

**Delivered**:
- Fully functional backend API (FastAPI)
- Interactive web dashboard with live simulation
- 12 test inverters with diverse scenarios
- Automated test suite
- Comprehensive ablation study with 7 comparative graphs
- Complete documentation (architecture, testing, prompt engineering)

---

**Built for HACKaMINeD 2026**
