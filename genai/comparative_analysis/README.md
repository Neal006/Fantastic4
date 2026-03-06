# LUMIN.AI – LLM Ablation Study & Comparative Analysis

> **Objective:** Determine the best free-tier LLM for LUMIN.AI's solar-plant diagnostic pipeline by comparing models across real production prompts, then justify the final selection with data.

---

## Models Evaluated

| Provider | Model | Parameters | Free Tier |
|----------|-------|------------|-----------|
| **Groq** | Llama 3.3 70B Versatile | 70B | Yes |
| **Google** | Gemini 2.5 Flash | — | Yes |
| **HuggingFace** | Qwen 2.5 72B Instruct | 72B | Yes |

All three are large, instruction-tuned models comparable in capability. Each was tested on the **exact same prompts** used in production (`app/prompts.py`) across 3 tasks and 3 risk scenarios (9 test cases per model, **27 total API calls**).

---

## Method (Brief)

Each model received identical inputs: system prompt + user prompt with real inverter sensor data and SHAP values from `app/synthetic_data.py`. Responses were auto-evaluated on 5 weighted metrics:

| Metric | Weight |
|--------|--------|
| JSON Validity | 25% |
| Hallucination Prevention | 25% |
| Technical Completeness | 25% |
| Response Quality | 15% |
| Urgency Accuracy | 10% |

Latency and token usage were recorded for every call. Full methodology is in `evaluate.py`.

---

## Results

### Overall Rankings

| Rank | Model | Overall Score | Avg Latency | JSON Valid | Hallucination Prevention | Urgency | Completeness | Quality |
|------|-------|:------------:|:-----------:|:----------:|:-------------:|:-------:|:------------:|:-------:|
| **1** | **Llama 3.3 70B (Groq)** | **91.7%** | **1.0s** | 67% | 100% | 100% | 100% | 100% |
| **2** | **Qwen 2.5 72B (HuggingFace)** | **91.7%** | **21.2s** | 67% | 100% | 100% | 100% | 100% |
| **3** | Gemini 2.5 Flash (Google) | 79.4% | 8.8s | 44% | 100% | 89% | 81% | 94% |

> **Groq and Qwen tie on accuracy (91.7%)**, but Groq is **21x faster**.

![Overall Scores](graphs/01_overall_scores.png)

---

### Radar Comparison — All Metrics at a Glance

The radar chart below shows each model's normalized score across all 5 evaluation dimensions. Groq and Qwen nearly overlap on every axis except latency, while Gemini shows clear dips in JSON validity and completeness.

![Radar Comparison](graphs/02_radar_comparison.png)

---

### Latency Comparison

Response speed is critical for a real-time diagnostic assistant. Groq's inference infrastructure delivers sub-second responses, while HuggingFace averages 21 seconds.

| Model | Avg Latency | Min | Max |
|-------|:-----------:|:---:|:---:|
| **Groq Llama 3.3 70B** | **1.0s** | 0.5s | 1.9s |
| Gemini 2.5 Flash | 8.8s | 2.8s | 11.8s |
| Qwen 2.5 72B (HF) | 21.2s | 6.1s | 50.3s |

![Latency Comparison](graphs/03_latency_comparison.png)

---

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

![Per-Task Scores](graphs/04_per_task_scores.png)

---

### Score vs Latency — The Efficiency Frontier

This scatter plot shows quality vs speed. The ideal model is in the **top-left** corner (high score, low latency). Groq is clearly the most efficient.

![Score vs Latency](graphs/05_score_vs_latency.png)

---

### Detailed Metric Heatmap

The heatmap below shows per-metric scores for each model. Groq and Qwen are nearly identical in accuracy, but Gemini falls behind on JSON validity and completeness.

![Metric Heatmap](graphs/06_metric_heatmap.png)

---

### Token Usage

Token consumption affects cost and rate limits. Gemini tends to produce longer outputs (higher token count), while Groq is the most token-efficient.

![Token Usage](graphs/07_token_usage.png)

---

## Chat Response Comparison

Below are the **actual chat responses** from each model for the same operator question across three risk scenarios. This shows how each model handles conversational queries in practice.

### Scenario 1: Normal Operation (Risk: 0.12, Class: no_risk)

**Question:** *"What is the current risk level of INV-P1-L1-0 and what actions should I take?"*

---

#### Groq Llama 3.3 70B — 0.5s, Score: 0.75

> According to the data, INV-P1-L1-0 has a risk level of 0.12, which is classified as "no_risk". The SHAP values indicate that the inverter's power, temperature, and ambient temperature are contributing negatively to the risk, but no specific actions are recommended based on the provided data. I don't have the inverter manual available to provide guidance on specific actions.

#### Gemini 2.5 Flash — 11.8s, Score: 0.75

> For INV-P1-L1-0:
> * The risk is 0.12, according to the data context.
> * The class is `no_risk`, according to the data context.
>
> I don't have maintenance advice or actions available right now.

#### Qwen 2.5 72B — 6.1s, Score: 0.75

> The current status of INV-P1-L1-0 in Plant 1, Block A, is as follows:
>
> - Risk Level: 0.12
> - Risk Class: No Risk
>
> Based on the provided data, the inverter is currently classified as having no risk. The factors contributing to this low risk include:
> - Power: -0.040
> - Temperature: -0.030
> - Ambient Temperature: -0.020
>
> Since the inverter is classified as no risk, no immediate action is required. Continue with regular monitoring and maintenance schedules. If you notice any changes or if the risk level increases, reassess the situation accordingly.
>
> I don't have that data available right now for specific maintenance steps, but you can refer to the inverter manual for routine maintenance procedures.

**Analysis:** All three models scored equally (0.75). Groq was concise and accurate in **0.5s**. Gemini was too brief (11.8s for a 2-line answer). Qwen was the most detailed but took **6.1s**.

---

### Scenario 2: Degradation Risk (Risk: 0.65, Class: degradation_risk)

**Question:** *"What is the current risk level of INV-P1-L1-1 and what actions should I take?"*

---

#### Groq Llama 3.3 70B — 0.7s, Score: 0.75

> The current status of INV-P1-L1-1 is a degradation risk with a risk value of 0.65, according to the provided data. The key factors contributing to this risk are inverters[1].pv3_current, smu[1].string3, and inverters[1].power, with SHAP values of +0.180, +0.140, and +0.120, respectively.
>
> I don't have the inverter manual available to provide specific guidance on actions to take for a degradation risk with this risk value.

#### Gemini 2.5 Flash — 3.5s, Score: 0.71

> According to the data, INV-P1-L1-1 has a risk of 0.65 and is classified as degradation_risk.
>
> I don't have that data available right now for what actions to take.

#### Qwen 2.5 72B — 14.2s, Score: 0.75

> The current status of INV-P1-L1-1 in Plant 1, Block A, shows a risk level of 0.65, classified as degradation risk. The key factors contributing to this risk are the PV3 current (+0.180), string 3 of the SMU (+0.140), and the power output (+0.120).
>
> Given this information, it is recommended to:
>
> 1. Monitor the PV3 current and string 3 of the SMU closely, as they are the primary contributors to the risk.
> 2. Check the power output to ensure it remains within expected parameters.
> 3. Consider scheduling a maintenance check to investigate potential issues with these components.
>
> If any of these values continue to degrade or show abnormal behavior, further action may be necessary.
>
> I don't have specific manual guidance available, but these steps should help mitigate the risk.

**Analysis:** Groq correctly cited all SHAP features in **0.7s**. Gemini gave a minimal 2-line answer and scored lowest (0.71) due to brevity. Qwen provided the most actionable advice but took **14.2s**.

---

### Scenario 3: Shutdown Risk (Risk: 0.89, Class: shutdown_risk)

**Question:** *"What is the current risk level of INV-P1-L2-0 and what actions should I take?"*

---

#### Groq Llama 3.3 70B — 0.7s, Score: 0.75

> According to the data, INV-P1-L2-0 has a risk level of 0.89 and is classified as a shutdown risk. The key factors contributing to this risk are:
> - Inverters[0].temp with a SHAP value of +0.350,
> - sensors[0].ambient_temp with a SHAP value of +0.150, and
> - inverters[0].alarm_code with a SHAP value of +0.120.
>
> I don't have the inverter manual available to provide specific guidance on what actions to take for a shutdown risk. I recommend consulting the inverter manual for shutdown risk procedures.

#### Gemini 2.5 Flash — 2.8s, Score: 0.71

> INV-P1-L2-0 has a risk of 0.89 and is classified as `shutdown_risk`.
>
> I don't have maintenance recommendations available right now.

#### Qwen 2.5 72B — 6.1s, Score: 0.75

> The current status of INV-P1-L2-0 in Plant 1, Block B, shows a high shutdown risk with a risk score of 0.89. The main factors contributing to this risk are:
>
> - Inverter temperature (+0.350)
> - Ambient temperature (+0.150)
> - Alarm code (+0.120)
>
> Given the high risk, it is advisable to:
>
> 1. Check the inverter temperature and ambient temperature to ensure they are within safe operating limits.
> 2. Review any active alarm codes and address them promptly.
> 3. Consider performing a maintenance check on the inverter to prevent a potential shutdown.
>
> If the temperatures or alarm codes indicate an issue, take immediate action to resolve them. If you need further guidance, refer to the inverter manual or contact support.

**Analysis:** Groq identified all three critical SHAP factors with exact values in **0.7s**. Gemini again gave a 2-line answer that lacks actionable detail. Qwen gave the best recommendations but at **6.1s** latency.

---

### Chat Comparison Summary

| Metric | Groq Llama 3.3 70B | Gemini 2.5 Flash | Qwen 2.5 72B |
|--------|:-------------------:|:----------------:|:-------------:|
| **Avg Chat Score** | **75.0%** | 72.5% | **75.0%** |
| **Avg Chat Latency** | **0.6s** | 6.0s | 8.8s |
| **Cites SHAP values** | Yes (exact) | Sometimes | Yes (rounded) |
| **Provides actions** | Defers to manual | Minimal | Yes (detailed) |
| **Hallucination-free** | 100% | 100% | 100% |
| **Tone** | Precise, technical | Too brief | Helpful, verbose |

---

## Why We Chose Groq (Llama 3.3 70B Versatile)

After evaluating all three models across 27 test cases, **Groq's Llama 3.3 70B Versatile** is the clear choice for LUMIN.AI. Here's why:

### 1. Best-in-Class Latency (21x faster)

| Model | Avg Latency | Relative Speed |
|-------|:-----------:|:--------------:|
| **Groq** | **1.0s** | **1x (baseline)** |
| Gemini | 8.8s | 8.8x slower |
| Qwen | 21.2s | 21.2x slower |

For a **real-time diagnostic assistant** where plant operators need immediate answers during critical situations (shutdown risk), sub-second response times are non-negotiable. A 21-second wait from HuggingFace is unacceptable in production.

### 2. Tied for Highest Accuracy (91.7%)

Groq matches Qwen's quality score across all tasks:
- **100%** on explanation generation
- **100%** on ticket generation
- **75%** on chat (identical to Qwen)
- **100%** hallucination prevention
- **100%** urgency accuracy
- **100%** technical completeness

### 3. Zero Hallucinations

All three models achieved 100% hallucination prevention, but Groq does this while being the fastest — it never fabricates sensor readings or invents SHAP features that weren't in the input.

### 4. Superior JSON Compliance

For structured tasks (explanation + ticket), Groq produces valid JSON with all required fields in **every single test case**. Gemini failed JSON parsing in 5 out of 9 test cases (44% validity vs Groq's 67%).

### 5. Token Efficiency

Groq uses fewer output tokens on average, which matters for free-tier rate limits:
- **Groq:** ~248 tokens/response avg
- **Gemini:** ~444 tokens/response avg
- **Qwen:** ~424 tokens/response avg

### 6. Production Reliability

Groq's API provides:
- Consistent sub-2s latency (no cold starts)
- OpenAI-compatible API (drop-in replacement)
- Generous free tier for development
- No quota exhaustion issues during testing

### 7. Gemini & Qwen — Why Not?

| Model | Why Not Selected |
|-------|-----------------|
| **Gemini 2.5 Flash** | Lower accuracy (79.4%), poor JSON validity (44%), quota issues on free tier, 8.8x slower |
| **Qwen 2.5 72B** | Same accuracy as Groq but **21x slower** (21.2s avg). Unacceptable for real-time diagnostics. HuggingFace free tier has cold starts up to 50s. |

### Final Verdict

> **Groq Llama 3.3 70B Versatile** delivers the best combination of accuracy, speed, and reliability for LUMIN.AI's solar-plant diagnostic pipeline. It is the only model that achieves top-tier quality (91.7%) with production-grade latency (1.0s avg).

---

## How to Reproduce

```bash
cd genai

# 1. Test API keys
python -m comparative_analysis.test_api_keys

# 2. Run ablation study (~3-5 minutes)
python -m comparative_analysis.run_ablation

# 3. Generate graphs and report
python -m comparative_analysis.generate_report
```

Results are saved to `results/` and graphs to `graphs/`.
