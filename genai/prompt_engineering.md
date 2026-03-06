# Prompt Engineering – Iteration History

## Overview

This document records the prompt design process for **SolarGuard AI**, showing
how each iteration was evaluated and improved. All prompts live in
`app/prompts.py`.

---

## Iteration 1 – Baseline (Naive)

### Prompt

```
System: You are a helpful assistant that explains solar inverter risk scores.
User:   The inverter {id} has a risk score of {score}. Explain why.
```

### Observations

| Criterion              | Rating | Notes                                                     |
|------------------------|--------|-----------------------------------------------------------|
| Accuracy               | ★★☆☆☆ | Frequently hallucinated sensor values not in the input.    |
| Structure              | ★★☆☆☆ | Free-form paragraphs; hard to scan on-site.                |
| Actionability          | ★☆☆☆☆ | Vague advice like "check the system".                      |
| Guardrail compliance   | ★☆☆☆☆ | No rules → LLM invented data freely.                      |

### Key Problems

1. **Hallucination** – The model fabricated readings (e.g., "current is 2.3 A")
   that were never in the input.
2. **No structure** – Operators couldn't quickly find the action items.
3. **No SHAP grounding** – The model ignored feature importances entirely and
   guessed at root causes.

---

## Iteration 2 – Structured Output + Guardrails (Current)

### Changes & Rationale

| Change                               | Why                                                                                           |
|--------------------------------------|-----------------------------------------------------------------------------------------------|
| Added explicit **STRICT RULES**      | Directly addresses hallucination by forbidding fabricated values.                              |
| Required **JSON output schema**      | Forces machine-parseable structure so the API can validate fields programmatically.            |
| Listed **exact SHAP features**       | The LLM must reference only these features, preventing it from inventing contributing factors. |
| Included **raw sensor values**       | Gives the LLM ground-truth numbers so it doesn't need to guess.                               |
| Added **RAG manual context**         | Grounds troubleshooting advice in the real inverter manual.                                    |
| Specified **urgency taxonomy**       | `immediate / within_24h / scheduled / routine` – operators know exactly how fast to act.       |
| Added **estimated_impact** field     | Communicates consequence of inaction, motivating timely response.                              |

### Prompt (abbreviated)

```
System:
  You are SolarGuard AI …
  STRICT RULES:
    1. ONLY reference data values explicitly provided …
    2. ONLY discuss features in the SHAP analysis …
    3. If data insufficient, say so …
    4. Ground recommendations in manual context …
    5. Return ONLY valid JSON …
  JSON SCHEMA: { summary, key_factors, recommended_actions, urgency, estimated_impact }

User:
  INVERTER: {id}   PLANT: {plant}   BLOCK: {block}
  Risk Score: {score}   Risk Class: {class}
  TOP SHAP FEATURES: …
  RAW SENSOR VALUES: …
  RELEVANT MANUAL CONTEXT: …
```

### Observations

| Criterion              | Rating | Notes                                                            |
|------------------------|--------|------------------------------------------------------------------|
| Accuracy               | ★★★★☆ | References only provided values; rare edge-case slips caught by  |
|                        |        | output guardrails.                                               |
| Structure              | ★★★★★ | JSON is deterministic; frontend can render cards/tables.          |
| Actionability          | ★★★★☆ | Concrete steps grounded in the manual (e.g., "check air filter   |
|                        |        | per Section 7.3").                                               |
| Guardrail compliance   | ★★★★☆ | Programmatic validation catches the remaining hallucinations.    |

### Remaining Risks & Mitigations

- **JSON parse failures** → `parse_llm_json()` handles markdown fences and
  provides a graceful fallback.
- **Feature name mismatch** → `validate_explanation_output()` cross-checks
  every cited feature against the SHAP input; mismatches are flagged as
  `_guardrail_warnings`.
- **Urgency hallucination** → validated against an allow-list in guardrails.

---

## Chat Prompt Design

The chat prompt follows the same guardrail philosophy:

- **System prompt** injects the full plant overview so the LLM always has
  ground-truth data available.
- **User prompt** injects per-turn data context and RAG manual excerpts.
- Multi-turn history is passed as previous messages, giving context memory
  without re-sending all data each turn.

---

## Ticket Prompt Design

Maintenance tickets require an even stricter tone:

- The prompt explicitly requests parts lists, safety notes, and escalation
  flags.
- Output is validated and rendered into a ReportLab PDF with a professional
  layout.

---

## Future Iteration Ideas

1. **Few-shot examples** – Add 2–3 gold-standard explanation examples in the
   system prompt to improve consistency.
2. **Chain-of-thought** – Ask the LLM to reason step-by-step before
   producing JSON (hidden reasoning + final JSON).
3. **Self-critique loop** – Have a second LLM call review the first output
   for hallucinations before returning to the user.
4. **Dynamic prompt selection** – Use different prompt templates per risk
   class (shutdown prompts emphasise safety; degradation prompts emphasise
   monitoring).
