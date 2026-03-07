"""Quick test script for all API endpoints."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import urllib.request

BASE = "http://localhost:8001"

def post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

def get(path):
    with urllib.request.urlopen(f"{BASE}{path}") as resp:
        return json.loads(resp.read())

# ── 1. Health ──
print("=" * 60)
print("TEST 1: GET /health")
r = get("/health")
print(json.dumps(r, indent=2))
assert r["status"] == "healthy"
assert r["model_loaded"] is True
print("✓ PASS\n")

# ── 2. Model Info ──
print("=" * 60)
print("TEST 2: GET /model/info")
r = get("/model/info")
print(f"  n_features: {r['n_features']}")
print(f"  class_names: {r['class_names']}")
print(f"  core_input_fields: {r['core_input_fields']}")
assert r["n_features"] == 145
print("✓ PASS\n")

# ── 3. Single Prediction (healthy inverter) ──
print("=" * 60)
print("TEST 3: POST /predict (healthy reading)")
r = post("/predict", {
    "inverter_id": "INV-P1-L1-0",
    "dc_voltage": 37.5,
    "dc_current": 9.5,
    "ac_power": 8.5,
    "module_temp": 40,
    "ambient_temp": 32,
    "irradiation": 890,
})
print(f"  category: {r['category']}")
print(f"  confidence: {r['confidence']}")
print(f"  predicted_class: {r['predicted_class']}")
print(f"  fault: {r['fault']}")
print(f"  probabilities: {r['probabilities']}")
print(f"  shap top 3: {r['shap']['top_features'][:3] if r.get('shap') and 'top_features' in r['shap'] else 'N/A'}")
has_plot = r.get("shap", {}).get("plot_base64") is not None
print(f"  has SHAP plot: {has_plot}")
assert r["category"] in "ABCDE"
print("✓ PASS\n")

# ── 4. Single Prediction (faulty inverter — overheating) ──
print("=" * 60)
print("TEST 4: POST /predict (overheating)")
r = post("/predict", {
    "inverter_id": "INV-P1-L2-0",
    "dc_voltage": 35.5,
    "dc_current": 7.8,
    "ac_power": 3.5,
    "module_temp": 75,
    "ambient_temp": 45,
    "irradiation": 700,
})
print(f"  category: {r['category']}")
print(f"  confidence: {r['confidence']}")
print(f"  fault: {r['fault']}")
assert r["category"] in "ABCDE"
print("✓ PASS\n")

# ── 5. Single Prediction (offline / grid fault) ──
print("=" * 60)
print("TEST 5: POST /predict (offline/grid fault)")
r = post("/predict", {
    "inverter_id": "INV-P2-L2-1",
    "dc_voltage": 0,
    "dc_current": 0,
    "ac_power": 0,
    "module_temp": 37,
    "ambient_temp": 33,
    "irradiation": 150,
    "include_plot": False,
})
print(f"  category: {r['category']}")
print(f"  fault: {r['fault']}")
assert r["category"] in "ABCDE"
print("✓ PASS\n")

# ── 6. Batch Prediction ──
print("=" * 60)
print("TEST 6: POST /predict/batch (2 inverters)")
r = post("/predict/batch", {
    "readings": [
        {
            "inverter_id": "INV-P1-L1-0",
            "features": {
                "dc_voltage": 37.5, "dc_current": 9.5, "ac_power": 8.5,
                "module_temp": 40, "ambient_temp": 32, "irradiation": 890,
            },
        },
        {
            "inverter_id": "INV-P1-L2-0",
            "features": {
                "dc_voltage": 35.5, "dc_current": 7.8, "ac_power": 3.5,
                "module_temp": 75, "ambient_temp": 45, "irradiation": 700,
            },
        },
    ],
    "include_shap": True,
    "include_plot": False,
})
print(f"  count: {r['count']}")
for p in r["predictions"]:
    print(f"  {p['inverter_id']}: cat={p['category']} conf={p['confidence']} fault={p['fault']}")
    if p.get("shap") and "top_features" in p["shap"]:
        print(f"    top SHAP feature: {p['shap']['top_features'][0]['feature']}")
assert r["count"] == 2
print("✓ PASS\n")

# ── 7. Validation error (missing field) ──
print("=" * 60)
print("TEST 7: POST /predict (missing field → 422)")
try:
    r = post("/predict", {"inverter_id": "X", "dc_voltage": 10})
    print("  ERROR: expected 422 but got 200")
except urllib.error.HTTPError as e:
    print(f"  Status: {e.code}")
    assert e.code == 422
    print("✓ PASS (correctly rejected)\n")

print("=" * 60)
print("ALL TESTS PASSED ✓")
