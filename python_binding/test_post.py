import json, urllib.request, sys

url = 'http://127.0.0.1:8000/api/solve'
payload = {
    "problem_id": "irrigation-coloring",
    "inputs": {"field_count": 4, "conflict_density": 40, "average_duration": 4, "aqueduct_supply_limit": 2500, "priority_weighting": 70},
    "mode": "window_minimization",
    "mode_id": "window_minimization"
}
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode('utf-8')
        print(body)
except Exception as e:
    print('ERROR', e)
    sys.exit(1)
