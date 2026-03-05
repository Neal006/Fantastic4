# SolarGuard AI - Complete System Guide

> **GenAI Explanation Layer for Solar Plant Risk Assessment**  
> Real-time ML processing with automated ticket generation and AI-powered explanations.

---

## 📋 Quick Navigation

1. [Quick Start](#quick-start)
2. [Live Simulation Dashboard](#live-simulation-dashboard)
3. [System Architecture](#system-architecture)
4. [ML Model Integration](#ml-model-integration)
5. [API Reference](#api-reference)
6. [Troubleshooting](#troubleshooting)

---

## What is SolarGuard AI?

SolarGuard AI is a **GenAI-powered explanation layer** that sits between your ML risk-prediction model and solar plant operators. It solves a critical problem: **ML models output numbers, but operators need actionable guidance**.

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
SolarGuard AI transforms that into:
```
🔴 CRITICAL SHUTDOWN RISK (89%)

Summary:
Inverter INV-P1-L2-0 is experiencing severe overheating with 
temperature at 78.6°C, significantly above the safe operating 
threshold. Combined with high ambient temperature (47.3°C) and 
alarm code 4003, immediate intervention is required to prevent 
equipment damage.

Key Risk Factors:
• Temperature (HIGH IMPACT): 78.6°C exceeds thermal protection 
  threshold. Cooling system may be compromised.
• Ambient Temperature (MEDIUM IMPACT): 47.3°C environmental 
  temperature reduces cooling efficiency.
• Alarm Code 4003 (MEDIUM IMPACT): Over-temperature protection 
  triggered. System is derating power output.

Recommended Actions:
1. Immediately inspect cooling fans and air filters
2. Check for blocked ventilation or debris accumulation
3. Verify ambient temperature sensors are functioning
4. Review alarm code 4003 in manual Section 7.3
5. Consider temporary shutdown if temperature exceeds 80°C

Urgency: IMMEDIATE
```

Plus a **professional PDF maintenance ticket** and **conversational Q&A** interface.

---

## How It Works (System Architecture)

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. ML MODEL                                                 │
│     Analyzes sensor data → Outputs risk score + SHAP        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  2. SOLARGUARD AI (GenAI Layer)                             │
│     ┌─────────────────────────────────────────────┐         │
│     │  a) Guardrails: Validate SHAP features      │         │
│     └─────────────────────────────────────────────┘         │
│     ┌─────────────────────────────────────────────┐         │
│     │  b) RAG: Search inverter manual for context │         │
│     └─────────────────────────────────────────────┘         │
│     ┌─────────────────────────────────────────────┐         │
│     │  c) LLM: Generate plain-English explanation │         │
│     └─────────────────────────────────────────────┘         │
│     ┌─────────────────────────────────────────────┐         │
│     │  d) Guardrails: Validate output, add sources│         │
│     └─────────────────────────────────────────────┘         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  3. OPERATOR                                                 │
│     Receives: Summary + Actions + PDF Ticket + Chat         │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | What It Does | Technology |
|-----------|--------------|------------|
| **FastAPI Server** | HTTP API that routes requests | Python, FastAPI |
| **Explainer** | Converts ML output → plain English | LLM + Prompt Engineering |
| **RAG Pipeline** | Searches inverter manual for context | PyMuPDF, SentenceTransformers, FAISS |
| **LLM Client** | Talks to Groq/OpenAI API | OpenAI SDK |
| **Guardrails** | Prevents hallucinations | Input/output validation |
| **Agent** | Auto-generates maintenance tickets | Multi-step workflow |
| **Conversation** | Multi-turn chat with memory | Session management |
| **Ticket Generator** | Creates professional PDFs | ReportLab |

### Data Flow Example

```
Operator clicks inverter "INV-P1-L2-0"
    ↓
UI sends: GET /explanation/INV-P1-L2-0
    ↓
Explainer fetches ML prediction:
    - Risk score: 0.89
    - SHAP: {temp: +0.35, alarm: +0.12, ...}
    - Raw: {temp: 78.6°C, power: 4100W, ...}
    ↓
Guardrail validates SHAP features (removes invalid ones)
    ↓
RAG searches inverter manual:
    Query: "overheating thermal protection alarm"
    Returns: ["Section 7.3: Thermal Protection...", 
              "Error Code 4003: Over-temperature...", ...]
    ↓
Explainer formats prompt:
    System: "You are SolarGuard AI. STRICT RULES: 
             Only reference provided data..."
    User: "Risk: 0.89, SHAP: temp +0.35, 
           Raw: temp 78.6°C, Manual: [context]"
    ↓
LLM (Groq/Llama 3.3) generates JSON:
    {
      "summary": "Inverter at critical shutdown risk...",
      "key_factors": [...],
      "recommended_actions": [...],
      "urgency": "immediate"
    }
    ↓
Guardrail validates output:
    - Parse JSON
    - Check all cited features exist in input
    - Add disclaimer
    ↓
Return to operator:
    - Summary paragraph
    - Risk factors with SHAP analysis
    - Numbered action list
    - Urgency badge
    - Grounded sources
```

---

## What You Have Right Now

### File Structure

```
genai/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI server (all endpoints)
│   ├── config.py            # Environment configuration
│   ├── models.py            # Pydantic data models
│   ├── llm.py               # LLM client (Groq/OpenAI)
│   ├── rag.py               # RAG pipeline (PDF → embeddings → search)
│   ├── prompts.py           # All prompt templates
│   ├── guardrails.py        # Hallucination prevention
│   ├── explainer.py         # Risk → plain-English explanation
│   ├── agent.py             # Agentic ticket generation
│   ├── conversation.py      # Multi-turn chat memory
│   ├── ticket.py            # PDF maintenance ticket generator
│   └── synthetic_data.py    # Mock ML backend (12 test inverters)
│
├── ui/
│   └── index.html           # Original UI (needs server)
│
├── dashboard.html           # Standalone UI (works directly)
├── serve_ui.py              # Simple HTTP server for UI
├── test_endpoints.py        # Automated test suite
├── requirements.txt         # Python dependencies
├── .env.example             # Config template
├── .env                     # Your config (API keys)
├── .gitignore
│
├── 2cdb179b-9321-4b69-871a-ff5f5df3b3ef.pdf  # Inverter manual
│
├── README.md                # Original quick-start guide
├── ARCHITECTURE.md          # Detailed system design
├── TESTING_GUIDE.md         # Step-by-step testing
├── prompt_engineering.md    # Prompt iteration history
└── COMPLETE_README.md       # This file (master guide)
```

### Test Data (Synthetic)

You have **12 synthetic inverters** across 3 plants with diverse failure scenarios:

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

---

## Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
cd c:\Users\Priyanshu\OneDrive\Desktop\Hackamine\genai
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
[RAG] Loading embedding model 'all-MiniLM-L6-v2' …
[RAG] Building vector store from PDF (first run) …
[RAG] Encoding 32 chunks …
[RAG] Vector store cached to disk.
[RAG] Ready – 32 chunks indexed.
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

✅ **Verify**: Open http://localhost:8000/docs (Swagger UI should load)

### Step 4: Open Dashboard

**Option A** (Simplest):
- Double-click `dashboard.html` in the `genai` folder

**Option B** (If Option A shows errors):
- Open new terminal
- Run: `python serve_ui.py`
- Open: http://localhost:8080

### Step 5: Test

1. **See 12 colored inverter cards** (green/orange/red)
2. **Click red card** `INV-P1-L2-0` → Wait 2-4 seconds → See AI explanation
3. **Click "Generate PDF"** → PDF downloads
4. **Type in chat**: "Which inverters have elevated risk?" → Get AI response

---

## Testing the System

### Automated Test Suite

```bash
python test_endpoints.py
```

This tests all endpoints automatically. Expected output:
```
🔆  SolarGuard AI – Endpoint Test Suite

============================================================
  1. Health Check
============================================================
  ✓ [200] GET /health

============================================================
  2. Utility – List Inverters & Predictions
============================================================
  ✓ [200] GET /inverters
  ✓ [200] GET /predictions
  ...
```

### Manual Testing via UI

#### Test 1: Normal Operation Inverter
1. Click `INV-P1-L1-0` (green card)
2. Should show:
   - Risk score: 12%
   - Urgency: ROUTINE
   - Summary: "Operating normally..."
   - Actions: Routine maintenance

#### Test 2: Degradation Risk Inverter
1. Click `INV-P1-L1-1` (orange card)
2. Should show:
   - Risk score: 65%
   - Urgency: WITHIN 24H
   - Key factor: `inverters[1].pv3_current` (string degradation)
   - Actions: Inspect string 3, check for panel issues

#### Test 3: Shutdown Risk Inverter (Overheating)
1. Click `INV-P1-L2-0` (red card)
2. Should show:
   - Risk score: 89%
   - Urgency: IMMEDIATE
   - Key factors: Temperature (78.6°C), alarm code 4003
   - Actions: Inspect cooling, check filters, verify ambient temp

#### Test 4: Shutdown Risk Inverter (Grid Fault)
1. Click `INV-P2-L2-1` (red card)
2. Should show:
   - Risk score: 91%
   - Urgency: IMMEDIATE
   - Key factors: Grid frequency (51.8 Hz), voltage deviation
   - Actions: Check grid connection, verify protection settings

#### Test 5: PDF Ticket Generation
1. Select any red inverter
2. Click "Generate PDF"
3. PDF should download with:
   - Professional A4 layout
   - Ticket ID (e.g., TKT-20260305-A3F2B1)
   - Priority badge (P1-Critical for red inverters)
   - Root cause analysis
   - Numbered action steps
   - Parts needed
   - Safety notes

#### Test 6: Multi-Turn Chat
1. Type: "Which inverters have elevated risk right now?"
2. Should list all orange/red inverters with details
3. Type: "Tell me more about the overheating inverter"
4. Should provide detailed analysis of INV-P1-L2-0
5. Type: "Which inverters in Block B have elevated risk?"
6. Should filter by Block B only

**Notice**: Chat remembers context from previous messages.

### Manual Testing via API

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
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Which inverters have risk?\"}"

# Generate ticket
curl -X POST http://localhost:8000/agent/maintenance-ticket/INV-P1-L2-0

# Download ticket PDF
curl http://localhost:8000/agent/maintenance-ticket/INV-P1-L2-0/pdf --output ticket.pdf

# Plant risk report
curl http://localhost:8000/agent/risk-report/plant_1
```

---

## Understanding the Components

### 1. FastAPI Server (`app/main.py`)

**What it does**: HTTP API gateway that routes requests.

**Key endpoints**:
- `GET /health` → System status
- `GET /explanation/{inverter_id}` → AI explanation
- `POST /chat` → Multi-turn Q&A
- `POST /agent/maintenance-ticket/{inverter_id}` → Generate ticket
- `GET /agent/risk-report/{plant_id}` → Plant-wide report
- `GET /inverters` → List all inverters
- `GET /predictions` → Raw ML predictions

**Startup sequence**:
1. Load config from `.env`
2. Initialize LLM client
3. Initialize RAG (parse PDF, build embeddings)
4. Start accepting requests

### 2. LLM Client (`app/llm.py`)

**What it does**: Thin wrapper around OpenAI-compatible APIs.

**Supports**:
- Groq (recommended - free, fast)
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
- `generate(system, user)` → Single-turn completion
- `generate_with_history(system, messages)` → Multi-turn with context
- `check_connection()` → Health check

### 3. RAG Pipeline (`app/rag.py`)

**What it does**: Retrieval-Augmented Generation - grounds LLM in inverter manual.

**Architecture**:
```
Inverter Manual PDF (34 MB)
    ↓
PyMuPDF Parser (extracts text)
    ↓
Text Chunking (800 words, 200 overlap)
    ↓
SentenceTransformer Embeddings (all-MiniLM-L6-v2)
    ↓
FAISS Vector Store (cached to disk)
    ↓
Cosine Similarity Search
    ↓
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
- `SYSTEM_PROMPT_EXPLANATION` → Risk analysis
- `SYSTEM_PROMPT_CHAT` → Conversational Q&A
- `SYSTEM_PROMPT_TICKET` → Maintenance ticket generation
- `SYSTEM_PROMPT_RISK_REPORT` → Plant-wide reports

**Key features**:
- Explicit STRICT RULES to prevent hallucinations
- JSON schema enforcement
- Grounding in provided data only
- Technical language for operators

**Example** (simplified):
```
System:
  You are SolarGuard AI, an expert solar-plant diagnostic assistant.
  
  STRICT RULES:
  1. ONLY reference data values explicitly provided
  2. ONLY discuss features in the SHAP analysis
  3. If data insufficient, say so
  4. Return ONLY valid JSON
  
  JSON SCHEMA: {
    "summary": "<2-3 sentence summary>",
    "key_factors": [...],
    "recommended_actions": [...],
    "urgency": "immediate | within_24h | scheduled | routine"
  }

User:
  INVERTER: INV-P1-L2-0
  Risk Score: 0.89
  SHAP: temp +0.35, alarm +0.12, ...
  Raw Values: temp 78.6°C, alarm 4003, ...
  Manual Context: [Section 7.3: Thermal Protection...]
```

See `prompt_engineering.md` for full iteration history.

### 5. Guardrails (`app/guardrails.py`)

**What it does**: Prevents hallucinations - ensures LLM never fabricates data.

**4 layers**:

**Layer 1 - Input Validation**:
```python
# Remove SHAP features not in dataset schema
valid_shap = validate_shap_features(prediction.shap_values)
```

**Layer 2 - Prompt Rules**:
```
STRICT RULES:
1. ONLY reference provided data
2. NEVER fabricate sensor readings
```

**Layer 3 - Output Validation**:
```python
# Check every cited feature exists in input
for factor in response['key_factors']:
    if factor['feature'] not in provided_shap:
        flag_violation()
```

**Layer 4 - Disclaimer**:
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
- Realistic SHAP values
- Raw sensor readings

**When to replace**: When your ML model API is ready.

---

## Integrating Your ML Model

### Current State (Synthetic Data)

Right now, `app/synthetic_data.py` provides mock predictions:

```python
def get_prediction(inverter_id: str) -> InverterPrediction:
    # Returns hardcoded test data
    return SYNTHETIC_PREDICTIONS.get(inverter_id)
```

### Integration Steps

#### Step 1: Define Your ML API Contract

Your ML model should expose an API endpoint that returns predictions in this format:

```json
{
  "inverter_id": "INV-P1-L2-0",
  "plant_id": "plant_1",
  "block": "Block B",
  "logger_mac": "ICR2-LT2-Celestical-10000.73",
  "inverter_index": 0,
  "timestamp": "2026-03-05T15:30:00Z",
  "risk_score": 0.89,
  "risk_class": "shutdown_risk",
  "shap_values": {
    "inverters[0].temp": 0.35,
    "sensors[0].ambient_temp": 0.15,
    "inverters[0].alarm_code": 0.12,
    "inverters[0].power": 0.10,
    "inverters[0].limit_percent": 0.08
  },
  "raw_features": {
    "inverters[0].temp": 78.6,
    "inverters[0].alarm_code": 4003,
    "inverters[0].op_state": 2,
    "inverters[0].power": 4100.0,
    "inverters[0].pv1_current": 8.1,
    "inverters[0].pv1_voltage": 36.5,
    "inverters[0].kwh_today": 18.7,
    "inverters[0].limit_percent": 42,
    "meters[0].pf": 0.94,
    "meters[0].freq": 50.02,
    "sensors[0].ambient_temp": 47.3
  }
}
```

**Required fields**:
- `inverter_id` (string)
- `plant_id` (string)
- `risk_score` (float, 0-1)
- `risk_class` (string: "no_risk" | "degradation_risk" | "shutdown_risk")
- `shap_values` (dict: feature_name → importance)
- `raw_features` (dict: feature_name → actual value)

**Optional but recommended**:
- `timestamp` (ISO 8601 datetime)
- `block` (string)
- `logger_mac` (string)

#### Step 2: Update Configuration

Add ML API settings to `.env`:

```env
# ML Model API
ML_API_URL=http://your-ml-server:5000/api
ML_API_KEY=your_ml_api_key_if_needed
```

Update `app/config.py`:

```python
# Add to config.py
ML_API_URL = os.getenv("ML_API_URL", "http://localhost:5000/api")
ML_API_KEY = os.getenv("ML_API_KEY", "")
```

#### Step 3: Replace Synthetic Data Module

**Option A - Minimal Changes** (recommended):

Edit `app/synthetic_data.py`:

```python
import httpx
from app.config import ML_API_URL, ML_API_KEY
from app.models import InverterPrediction

# Keep PLANTS dictionary for metadata
PLANTS = { ... }  # Keep existing

async def get_prediction(inverter_id: str) -> InverterPrediction | None:
    """Fetch prediction from ML API."""
    try:
        headers = {}
        if ML_API_KEY:
            headers["Authorization"] = f"Bearer {ML_API_KEY}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ML_API_URL}/predictions/{inverter_id}",
                headers=headers,
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return InverterPrediction(**data)
    except Exception as e:
        print(f"[ERROR] Failed to fetch prediction for {inverter_id}: {e}")
        return None

async def get_all_predictions() -> list[InverterPrediction]:
    """Fetch all predictions from ML API."""
    try:
        headers = {}
        if ML_API_KEY:
            headers["Authorization"] = f"Bearer {ML_API_KEY}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ML_API_URL}/predictions",
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return [InverterPrediction(**p) for p in data]
    except Exception as e:
        print(f"[ERROR] Failed to fetch predictions: {e}")
        return []

async def get_plant_predictions(plant_id: str) -> list[InverterPrediction]:
    """Fetch predictions for a specific plant."""
    all_preds = await get_all_predictions()
    return [p for p in all_preds if p.plant_id == plant_id]

def get_all_inverter_ids() -> list[str]:
    """Return list of monitored inverter IDs."""
    # This should come from your ML API or database
    # For now, extract from PLANTS structure
    ids = []
    for pdata in PLANTS.values():
        for ldata in pdata["loggers"].values():
            ids.extend(ldata["inverters"])
    return ids

def get_plant_overview() -> str:
    """Return text summary for chat context."""
    # This can stay as-is or be enhanced with live data
    return "..."  # Keep existing implementation
```

**Option B - New Module**:

Create `app/ml_client.py`:

```python
import httpx
from typing import Optional, List
from app.config import ML_API_URL, ML_API_KEY
from app.models import InverterPrediction

class MLClient:
    def __init__(self):
        self.base_url = ML_API_URL
        self.api_key = ML_API_KEY
    
    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def get_prediction(self, inverter_id: str) -> Optional[InverterPrediction]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/predictions/{inverter_id}",
                headers=self._headers(),
                timeout=10.0
            )
            response.raise_for_status()
            return InverterPrediction(**response.json())
    
    async def get_all_predictions(self) -> List[InverterPrediction]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/predictions",
                headers=self._headers(),
                timeout=30.0
            )
            response.raise_for_status()
            return [InverterPrediction(**p) for p in response.json()]
    
    async def get_plant_predictions(self, plant_id: str) -> List[InverterPrediction]:
        all_preds = await self.get_all_predictions()
        return [p for p in all_preds if p.plant_id == plant_id]
```

Then update `app/main.py`:

```python
# Replace imports
from app.ml_client import MLClient

# Replace
# from app.synthetic_data import get_prediction, ...

# Initialize
ml_client = MLClient()

# Update endpoints to use ml_client
@app.get("/explanation/{inverter_id}")
async def get_explanation(inverter_id: str):
    prediction = await ml_client.get_prediction(inverter_id)
    if not prediction:
        raise HTTPException(404, f"No prediction for {inverter_id}")
    # ... rest stays the same
```

#### Step 4: Update Main App to Use Async

Since ML API calls are async, update `app/main.py` endpoints:

```python
# Change from:
@app.get("/explanation/{inverter_id}")
async def get_explanation(inverter_id: str):
    try:
        return explainer.explain(inverter_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

# To:
@app.get("/explanation/{inverter_id}")
async def get_explanation(inverter_id: str):
    try:
        prediction = await get_prediction(inverter_id)  # Now async
        if not prediction:
            raise HTTPException(404, f"No prediction for {inverter_id}")
        return explainer.explain_from_prediction(prediction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```

And update `app/explainer.py`:

```python
# Add new method
def explain_from_prediction(self, prediction: InverterPrediction) -> ExplanationResponse:
    """Generate explanation from a prediction object."""
    # Same logic as explain(), but takes prediction directly
    # instead of calling get_prediction()
    ...
```

#### Step 5: Test Integration

1. **Start your ML model API**
2. **Verify it returns correct format**:
   ```bash
   curl http://your-ml-server:5000/api/predictions/INV-P1-L2-0
   ```
3. **Update `.env` with ML API URL**
4. **Restart SolarGuard AI backend**
5. **Test explanation endpoint**:
   ```bash
   curl http://localhost:8000/explanation/INV-P1-L2-0
   ```

#### Step 6: Handle Edge Cases

Add error handling for common issues:

```python
async def get_prediction(inverter_id: str) -> InverterPrediction | None:
    try:
        # ... API call ...
    except httpx.TimeoutException:
        print(f"[ERROR] ML API timeout for {inverter_id}")
        return None
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"[WARN] No prediction found for {inverter_id}")
        else:
            print(f"[ERROR] ML API error {e.response.status_code}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return None
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

## API Reference

### Core Endpoints

#### `GET /health`

Health check - returns system status.

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
- `inverter_id` (path): Inverter identifier

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
      "explanation": "Temperature at 78.6°C exceeds..."
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
  "pdf_path": "c:\\...\\tickets\\TKT-20260305-A3F2B1.pdf",
  "ticket_data": {
    "title": "Critical Overheating - Immediate Action Required",
    "priority": "P1-Critical",
    "description": "...",
    "root_cause_analysis": "...",
    "recommended_actions": [...],
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

**Response**: PDF file (application/pdf)

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
  },
  ...
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
    "shap_values": {...},
    "raw_features": {...},
    "timestamp": "2026-03-05T15:30:00Z"
  },
  ...
]
```

#### `GET /predictions/{inverter_id}`

Get raw ML prediction for a single inverter.

**Response**: Same as single item from `/predictions`

---

## Troubleshooting

### Backend Won't Start

**Symptom**: Error when running `uvicorn app.main:app`

**Causes & Fixes**:

1. **Missing dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Port 8000 already in use**
   ```bash
   # Use different port
   uvicorn app.main:app --reload --port 8001
   ```

3. **Import errors**
   ```bash
   # Make sure you're in genai/ folder
   cd c:\Users\Priyanshu\OneDrive\Desktop\Hackamine\genai
   ```

### LLM Not Connected

**Symptom**: UI shows "LLM Offline" or explanations fail

**Causes & Fixes**:

1. **Missing API key**
   - Edit `.env`
   - Add: `LLM_API_KEY=gsk_your_key_here`
   - Get key: https://console.groq.com
   - Restart backend

2. **Invalid API key**
   - Verify key is correct
   - Check Groq console for quota/limits

3. **Network issues**
   - Check internet connection
   - Try: `curl https://api.groq.com/openai/v1/models`

### RAG Not Loading

**Symptom**: "vector_store_loaded": false or RAG errors

**Causes & Fixes**:

1. **PDF not found**
   - Verify: `genai/2cdb179b-9321-4b69-871a-ff5f5df3b3ef.pdf` exists
   - Check path in `.env`: `INVERTER_MANUAL_PATH`

2. **Corrupted cache**
   ```bash
   # Delete cache and rebuild
   rmdir /s vector_store
   # Restart backend (will rebuild)
   ```

3. **Insufficient memory**
   - First run needs ~2GB RAM for embedding model
   - Close other applications

### Explanations Are Generic

**Symptom**: Explanations don't mention specific SHAP features

**Causes & Fixes**:

1. **SHAP features not being passed**
   - Check ML API response includes `shap_values`
   - Verify format matches `InverterPrediction` model

2. **RAG not retrieving relevant context**
   - Check backend logs for RAG queries
   - Verify manual chunks are relevant

3. **Prompt needs tuning**
   - Edit `app/prompts.py`
   - Document changes in `prompt_engineering.md`

### PDF Generation Fails

**Symptom**: Ticket generation returns error

**Causes & Fixes**:

1. **Directory not writable**
   ```bash
   mkdir tickets
   ```

2. **ReportLab not installed**
   ```bash
   pip install reportlab
   ```

3. **Invalid ticket data**
   - Check backend logs for JSON parse errors
   - LLM may have returned malformed JSON

### UI Shows JSON Instead of Dashboard

**Symptom**: Browser displays raw JSON text

**Causes & Fixes**:

1. **Opening wrong URL**
   - Don't go to `localhost:8000` (that's the API)
   - Use `dashboard.html` or `localhost:8080` (UI server)

2. **File opened as file://**
   - Use `python serve_ui.py` instead
   - Or double-click `dashboard.html`

3. **Browser caching**
   - Hard refresh: Ctrl+Shift+R (Windows) / Cmd+Shift+R (Mac)
   - Clear browser cache

### Chat Not Remembering Context

**Symptom**: Each message treated as new conversation

**Causes & Fixes**:

1. **Session ID not being sent**
   - UI should automatically include `session_id` from previous response
   - Check browser console for errors

2. **Backend restarted**
   - Session memory is in-memory only
   - Restart clears all sessions

3. **Different session IDs**
   - Verify UI is using same `session_id` for follow-up messages

---

## Hackathon Criteria Coverage

### ✅ Required Criteria

| Criterion | Implementation | Location |
|-----------|----------------|----------|
| **Automated Narrative Generation** | Risk score + SHAP → plain-English summary + actions | `app/explainer.py` |
| **Retrieval-Augmented Q&A (RAG)** | Natural language questions grounded in manual + data | `app/rag.py`, `app/main.py` `/chat` |
| **Prompt Design** | 2 iterations documented with rationale | `prompt_engineering.md` |

### ✅ Bonus Criteria (All Implemented)

| Criterion | Implementation | Location |
|-----------|----------------|----------|
| **Agentic Workflow** | Auto-retrieves data → runs assessment → drafts ticket | `app/agent.py` |
| **Multi-turn Conversation** | Session-based context memory | `app/conversation.py` |
| **Hallucination Guardrails** | 4-layer validation (input/prompt/output/disclaimer) | `app/guardrails.py` |
| **Multi-class Output** | no_risk / degradation_risk / shutdown_risk | `app/models.py` |

### Documentation

| Document | Purpose |
|----------|---------|
| `COMPLETE_README.md` | This file - master guide |
| `ARCHITECTURE.md` | Detailed system design |
| `TESTING_GUIDE.md` | Step-by-step testing instructions |
| `prompt_engineering.md` | Prompt iteration history (hackathon requirement) |
| `README.md` | Quick-start guide |

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

## Next Steps

### For Hackathon Demo

1. ✅ System is complete and functional
2. ✅ Test all features (use `test_endpoints.py`)
3. ✅ Prepare demo script:
   - Show overheating inverter explanation
   - Generate PDF ticket
   - Ask chat: "Which inverters in Block B have risk?"
   - Show plant risk report
4. ✅ Highlight bonus features (agentic, multi-turn, guardrails)

### For Production Deployment

1. **Connect Real ML Model** (see [Integrating Your ML Model](#integrating-your-ml-model))
2. **Add Authentication**: JWT tokens, API keys
3. **Deploy Backend**: Docker + Kubernetes / Cloud Run
4. **Deploy Frontend**: Build production React/Next.js app
5. **Add Monitoring**: Logging, metrics, alerts (Prometheus, Grafana)
6. **Add Database**: PostgreSQL for prediction history, tickets
7. **Add Caching**: Redis for frequently accessed predictions
8. **Scale RAG**: Move to dedicated vector database (Pinecone, Weaviate)
9. **Add Rate Limiting**: Protect against abuse
10. **HTTPS**: SSL certificates for production

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

**SolarGuard AI** is a complete GenAI explanation layer that:

1. **Takes ML predictions** (risk scores + SHAP values)
2. **Searches inverter manual** for relevant context (RAG)
3. **Generates plain-English explanations** via LLM
4. **Prevents hallucinations** with 4-layer guardrails
5. **Creates PDF maintenance tickets** automatically
6. **Answers operator questions** via multi-turn chat
7. **Provides plant-wide reports** for management

**You have**:
- ✅ Fully functional backend API (FastAPI)
- ✅ Interactive web dashboard
- ✅ 12 test inverters with diverse scenarios
- ✅ Automated test suite
- ✅ Complete documentation

**To integrate your ML model**:
1. Ensure ML API returns predictions in correct format
2. Update `app/synthetic_data.py` to call ML API
3. Add ML API URL to `.env`
4. Test integration

**For questions**: Check `ARCHITECTURE.md` for deep dives, `TESTING_GUIDE.md` for testing, or `prompt_engineering.md` for prompt design rationale.

---

**You're ready for the hackathon demo! 🚀**
