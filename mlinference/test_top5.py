"""Test that SHAP returns top 5 features."""

import requests
import json

data = {
    'inverter_id': 'INV-TEST',
    'dc_voltage': 37.5,
    'dc_current': 9.5,
    'ac_power': 8.5,
    'module_temp': 40,
    'ambient_temp': 32,
    'irradiation': 890,
    'include_shap': True,
    'include_plot': False
}

r = requests.post('http://localhost:8001/predict', json=data)
result = r.json()

top_features = result.get('shap', {}).get('top_features', [])
print(f'Top features count: {len(top_features)}')
print('\nTop 5 features:')
for i, feat in enumerate(top_features[:5], 1):
    print(f'{i}. {feat["feature"]}: {feat["shap_value"]:.4f}')
