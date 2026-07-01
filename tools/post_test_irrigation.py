import json
import sys

try:
    import requests
except ImportError:
    print('requests not installed. Run: pip install requests')
    sys.exit(1)

URL = 'http://localhost:8000/api/solve'

payload = {
    "problem_id": "irrigation-coloring",
    "mode": "water_source_allocation",
    "inputs": {
        "field_count": 16,
        "aqueduct_supply_limit": 2500,
        # include both flow_rates and per_field_demands to mirror frontend mappings
        "flow_rates": [max(10, 40 + 4 * 35 + (i % 5) * 12) for i in range(16)],
        "priority_values": [70.0 for _ in range(16)],
        "seed": 0
    }
}

print('Posting sample payload to', URL)
print(json.dumps(payload, indent=2))

resp = requests.post(URL, json=payload)
print('\nStatus:', resp.status_code)
try:
    print('Response JSON:\n', json.dumps(resp.json(), indent=2))
except Exception:
    print('Response text:\n', resp.text)
