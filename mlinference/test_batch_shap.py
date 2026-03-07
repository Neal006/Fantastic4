"""Test batch predictions with SHAP."""

import requests
import json

data = {
    'readings': [
        {
            'inverter_id': 'INV-1',
            'features': {
                'dc_voltage': 37.5,
                'dc_current': 9.5,
                'ac_power': 8.5,
                'module_temp': 40,
                'ambient_temp': 32,
                'irradiation': 890
            }
        },
        {
            'inverter_id': 'INV-2',
            'features': {
                'dc_voltage': 35.5,
                'dc_current': 7.8,
                'ac_power': 3.5,
                'module_temp': 75,
                'ambient_temp': 45,
                'irradiation': 700
            }
        }
    ],
    'mode': 'manual',
    'include_shap': True,
    'include_plot': False
}

r = requests.post('http://localhost:8001/predict/batch', json=data)
result = r.json()

print(f"Count: {result['count']}")
print("\nPredictions:")
for p in result['predictions']:
    has_shap = 'shap' in p and 'top_features' in p.get('shap', {})
    top_feature = p.get('shap', {}).get('top_features', [{}])[0].get('feature', 'N/A') if has_shap else 'N/A'
    print(f"  {p['inverter_id']}:")
    print(f"    Category: {p['category']}")
    print(f"    Confidence: {p['confidence']}")
    print(f"    Has SHAP: {has_shap}")
    print(f"    Top feature: {top_feature}")
    if has_shap:
        print(f"    Top 3 features:")
        for feat in p['shap']['top_features'][:3]:
            print(f"      - {feat['feature']}: {feat['shap_value']:.4f}")
