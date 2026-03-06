"""
Test script for Google Gemini models (free tier)
Compares Gemini 1.5 Flash with Groq Llama 3.3 70B for LUMIN.AI use case
"""

import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠ google-generativeai not installed. Run: pip install google-generativeai")

from openai import OpenAI

# ---------------------------
# CONFIG
# ---------------------------

GOOGLE_API_KEY = ""
GROQ_API_KEY = ""

# Try multiple Gemini models (in order of likelihood to have free quota)
# Note: Must include 'models/' prefix
GEMINI_MODELS = [
    "models/gemini-2.0-flash",      # Fast, good for production
    "models/gemini-2.5-flash",      # Latest flash model
    "models/gemma-3-27b-it",        # 27B comparable to Groq 70B
]

GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------
# TEST PROMPT (from LUMIN.AI)
# ---------------------------

SYSTEM_PROMPT = """You are **LUMIN.AI**, an expert solar-plant diagnostic assistant.
You convert ML model predictions into actionable explanations for solar plant operators.

STRICT RULES:
1. ONLY reference data values explicitly provided in the user message.
2. Return ONLY valid JSON – no markdown fences, no extra text.

JSON SCHEMA:
{
  "summary": "<2-3 sentence plain-English summary>",
  "key_factors": [
    {"feature": "<exact feature name>", "impact": "high | medium | low", "explanation": "<what this means>"}
  ],
  "recommended_actions": ["<action 1>", "<action 2>"],
  "urgency": "immediate | within_24h | scheduled | routine"
}"""

USER_PROMPT = """Analyze this inverter prediction and generate an operational explanation.

INVERTER : INV-P1-L2-0
PLANT    : plant_1 (Plant 1 - Celestical)
BLOCK    : Block B
TIMESTAMP: 2026-03-05T15:30:00Z

ML PREDICTION
  Risk Score         : 0.8900  (0 = safe, 1 = critical)
  Risk Classification: shutdown_risk

TOP SHAP FEATURES (ranked by absolute importance):
  • inverters[0].temp: +0.3500  (raw value: 78.6)
  • sensors[0].ambient_temp: +0.1500  (raw value: 47.3)
  • inverters[0].alarm_code: +0.1200  (raw value: 4003)
  • inverters[0].power: +0.1000  (raw value: 4100.0)
  • inverters[0].limit_percent: +0.0800  (raw value: 42)

RAW SENSOR VALUES:
  • inverters[0].power: 4100.0
  • inverters[0].temp: 78.6
  • inverters[0].alarm_code: 4003
  • inverters[0].op_state: 2
  • inverters[0].pv1_current: 8.1
  • inverters[0].pv1_voltage: 36.5
  • inverters[0].kwh_today: 18.7
  • inverters[0].limit_percent: 42
  • meters[0].pf: 0.94
  • meters[0].freq: 50.02
  • sensors[0].ambient_temp: 47.3

Return the JSON explanation now."""


# ---------------------------
# TEST GEMINI
# ---------------------------

def test_gemini():
    if not GEMINI_AVAILABLE:
        return None
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    for model_id in GEMINI_MODELS:
        print(f"\n{'='*70}")
        print(f"  [GOOGLE GEMINI] {model_id}")
        print(f"{'='*70}")
        
        try:
            model = genai.GenerativeModel(
                model_name=model_id,
                system_instruction=SYSTEM_PROMPT,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                ),
            )
            
            start_time = time.time()
            response = model.generate_content(USER_PROMPT)
            latency = time.time() - start_time
            
            print(f"  ✓ Success!")
            print(f"  Latency: {latency:.2f}s")
            
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                print(f"  Input tokens: {getattr(response.usage_metadata, 'prompt_token_count', '?')}")
                print(f"  Output tokens: {getattr(response.usage_metadata, 'candidates_token_count', '?')}")
            
            print(f"\n  Response:\n")
            print(response.text[:800])
            if len(response.text) > 800:
                print(f"\n  ... (truncated, total {len(response.text)} chars)")
            
            return {"model": model_id, "response": response.text, "latency": latency}
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                print(f"  ✗ Quota exceeded: {error_str[:150]}")
                continue
            else:
                print(f"  ✗ Error: {error_str[:200]}")
                continue
    
    print(f"\n  ✗ All Gemini models exhausted quota or failed")
    return None


# ---------------------------
# TEST GROQ (baseline)
# ---------------------------

def test_groq():
    print(f"\n{'='*70}")
    print(f"  [GROQ] {GROQ_MODEL} (baseline)")
    print(f"{'='*70}")
    
    try:
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        latency = time.time() - start_time
        
        text = response.choices[0].message.content
        usage = response.usage
        
        print(f"  ✓ Success!")
        print(f"  Latency: {latency:.2f}s")
        print(f"  Input tokens: {usage.prompt_tokens}")
        print(f"  Output tokens: {usage.completion_tokens}")
        
        print(f"\n  Response:\n")
        print(text[:800])
        if len(text) > 800:
            print(f"\n  ... (truncated, total {len(text)} chars)")
        
        return {"model": GROQ_MODEL, "response": text, "latency": latency}
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:200]}")
        return None


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  LUMIN.AI – Gemini vs Groq Comparison Test")
    print("="*70)
    
    gemini_result = test_gemini()
    groq_result = test_groq()
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    
    if gemini_result and groq_result:
        print(f"\n  ✓ Both models responded successfully")
        print(f"\n  Gemini {gemini_result['model']}: {gemini_result['latency']:.2f}s")
        print(f"  Groq {GROQ_MODEL}: {groq_result['latency']:.2f}s")
        
        if gemini_result['latency'] < groq_result['latency']:
            print(f"\n  → Gemini is {groq_result['latency']/gemini_result['latency']:.1f}x faster")
        else:
            print(f"\n  → Groq is {gemini_result['latency']/groq_result['latency']:.1f}x faster")
    elif groq_result:
        print(f"\n  ✓ Groq working: {groq_result['latency']:.2f}s")
        print(f"  ✗ Gemini: All models quota-exhausted or unavailable")
    elif gemini_result:
        print(f"\n  ✓ Gemini working: {gemini_result['latency']:.2f}s")
        print(f"  ✗ Groq: Failed")
    else:
        print(f"\n  ✗ Both models failed")
    
    print(f"\n{'='*70}\n")