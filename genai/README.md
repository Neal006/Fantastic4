# LUMIN.AI - GenAI Explanation Layer

> Converts ML risk predictions into human-readable explanations and operational guidance for solar plant operators.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
# Copy template
copy .env.example .env

# Edit .env and add your Groq API key
LLM_API_KEY=gsk_your_key_here
```

Get free key: https://console.groq.com

### 3. Start Backend
```bash
uvicorn app.main:app --reload --port 8000
```

**First run**: 2-5 minutes (builds RAG index)  
**Subsequent runs**: 2-3 seconds (loads from cache)

### 4. Open Dashboard
```bash
# Open in browser
start simulation_dashboard.html
```

---

## 🎯 What It Does

**Problem**: ML models output numbers, operators need actionable guidance.

**Solution**: LUMIN.AI transforms ML predictions into:
- Plain-English risk summaries
- SHAP-based factor analysis
- Prioritized action recommendations
- Auto-generated maintenance tickets (PDF)
- Multi-turn conversational Q&A

---

## 🎬 Live Simulation Dashboard

### Features
- **Real-time data simulation** - New inverter readings every 3 seconds
- **Live processing logs** - See ML analysis in real-time
- **Auto-ticket generation** - Automatic tickets when risk > 80%
- **Blinking alerts** - Visual + audio notifications for critical issues
- **Click-to-explain** - Click any log entry for AI explanation
- **Statistics tracking** - Total processed, critical alerts, tickets generated

### How to Use
1. Click **"Start Simulation"**
2. Watch logs appear in real-time
3. When critical alert (red, risk > 80%):
   - Bell icon blinks 🔔
   - Auto-ticket generated
   - Alert appears in panel
4. **Click any log entry** to view AI explanation
5. Download PDF tickets from alerts panel

---

## 📊 System Architecture

```
Inverter Data → ML Model → LUMIN.AI → Operator
                              ↓
                    [Guardrails + RAG + LLM]
                              ↓
                    Explanation + Ticket + Chat
```

### Components
- **FastAPI Backend** - HTTP API server
- **LLM Client** - Groq/OpenAI integration
- **RAG Pipeline** - Searches inverter manual for context
- **Explainer** - Risk → plain English
- **Agent** - Auto-generates tickets
- **Guardrails** - Prevents hallucinations (4 layers)
- **Conversation** - Multi-turn chat with memory

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System status |
| `/explanation/{inverter_id}` | GET | AI explanation |
| `/chat` | POST | Multi-turn Q&A |
| `/agent/maintenance-ticket/{inverter_id}` | POST | Generate ticket |
| `/agent/maintenance-ticket/{inverter_id}/pdf` | GET | Download PDF |
| `/predictions` | GET | Raw ML predictions |
| `/inverters` | GET | List all inverters |

**API Docs**: http://localhost:8000/docs

---

## 🧪 Test Data

12 synthetic inverters across 3 plants:
- **Normal**: INV-P1-L1-0, INV-P1-L2-1, INV-P2-L1-1, INV-P3-L1-0, INV-P3-L1-1
- **Degradation**: INV-P1-L1-1, INV-P2-L1-0, INV-P2-L2-0, INV-P3-L2-0, INV-P3-L2-1
- **Critical**: INV-P1-L2-0 (overheating), INV-P2-L2-1 (grid fault)

---

## 🔗 Integrating Your ML Model

### Current State
Using `app/synthetic_data.py` for mock predictions.

### Integration Steps

1. **Ensure ML API returns this format**:
```json
{
  "inverter_id": "INV-P1-L2-0",
  "risk_score": 0.89,
  "risk_class": "shutdown_risk",
  "shap_values": {"feature": importance, ...},
  "raw_features": {"feature": value, ...}
}
```

2. **Update `app/synthetic_data.py`**:
```python
import httpx
from app.config import ML_API_URL

async def get_prediction(inverter_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ML_API_URL}/predictions/{inverter_id}")
        return InverterPrediction(**response.json())
```

3. **Add to `.env`**:
```env
ML_API_URL=http://your-ml-server:5000/api
```

4. **Test**:
```bash
curl http://localhost:8000/explanation/INV-P1-L2-0
```

---

## 🛡️ Hallucination Prevention

**4-Layer Guardrails**:
1. **Input validation** - Only valid SHAP features pass through
2. **Prompt rules** - "ONLY reference provided data. NEVER fabricate."
3. **Output validation** - Every cited feature checked against input
4. **Disclaimer** - "All values from sensor telemetry. Nothing fabricated."

---

## 🎓 Hackathon Criteria

### ✅ Required
- **Automated Narrative Generation** - Risk → plain English
- **RAG Q&A** - Natural language questions grounded in manual
- **Prompt Design** - 2 iterations documented (`prompt_engineering.md`)

### ✅ Bonus (All Implemented)
- **Agentic Workflow** - Auto-retrieves data → drafts ticket
- **Multi-turn Conversation** - Session-based context memory
- **Hallucination Guardrails** - 4-layer validation
- **Multi-class Output** - no_risk / degradation_risk / shutdown_risk

---

## 🔧 Troubleshooting

### Backend Won't Start
```bash
# Check dependencies
pip install -r requirements.txt

# Check port
uvicorn app.main:app --reload --port 8001
```

### LLM Not Connected
```bash
# Verify API key in .env
LLM_API_KEY=gsk_xxxxx

# Test connection
curl http://localhost:8000/health
```

### Rate Limit Exceeded
**Solutions**:
- Wait 10 minutes for reset
- Upgrade Groq account: https://console.groq.com/settings/billing
- Switch to OpenAI in `.env`:
  ```env
  LLM_API_KEY=sk-xxxxx
  LLM_BASE_URL=https://api.openai.com/v1
  LLM_MODEL=gpt-4o-mini
  ```

### Explanations Are Generic
- Check SHAP features in ML response
- Verify RAG initialized (check backend logs)
- Ensure PDF exists: `2cdb179b-9321-4b69-871a-ff5f5df3b3ef.pdf`

---

## 📁 File Structure

```
genai/
├── app/
│   ├── main.py              # FastAPI server
│   ├── llm.py               # LLM client
│   ├── rag.py               # RAG pipeline
│   ├── explainer.py         # Explanation engine
│   ├── agent.py             # Agentic workflow
│   ├── prompts.py           # Prompt templates
│   ├── guardrails.py        # Hallucination prevention
│   ├── conversation.py      # Multi-turn chat
│   ├── ticket.py            # PDF generator
│   └── synthetic_data.py    # Mock ML backend
│
├── simulation_dashboard.html # Live monitoring UI
├── requirements.txt          # Dependencies
├── .env                      # Configuration
├── prompt_engineering.md     # Prompt iterations
└── README.md                 # This file
```

---

## 🎯 For Demo

1. Start backend: `uvicorn app.main:app --reload --port 8000`
2. Open `simulation_dashboard.html`
3. Click "Start Simulation"
4. Wait for critical alert (bell blinks)
5. Click log entry → Show AI explanation
6. Download auto-generated PDF ticket

**Key talking points**:
- Real-time ML processing simulation
- Automatic ticket generation for critical issues
- AI explanations grounded in SHAP + manual (RAG)
- Hallucination prevention with 4-layer guardrails
- Multi-turn conversational interface

---

## 📚 Documentation

- **`prompt_engineering.md`** - Prompt iteration history (hackathon requirement)
- **API Docs** - http://localhost:8000/docs (Swagger UI)

---

## 🚀 Next Steps

**For Production**:
1. Connect real ML model API
2. Add authentication (JWT)
3. Deploy backend (Docker + cloud)
4. Add database for prediction history
5. Implement caching (Redis)
6. Add monitoring (logs, metrics, alerts)

**For Enhancement**:
1. Few-shot prompting with examples
2. Chain-of-thought reasoning
3. Historical trend analysis
4. Predictive maintenance forecasting
5. Multi-language support

---

**Built for HACKaMINeD 2026** 🏆
