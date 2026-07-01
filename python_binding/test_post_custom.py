import json, urllib.request, sys

url = 'http://127.0.0.1:8000/api/solve'
payload = {
    "problem_id": "irrigation-coloring",
    "inputs": {
        "field_count": 6,
        "conflict_density": 30,
        "average_duration": 2,
        "aqueduct_supply_limit": 1800,
        "priority_weighting": 80,
        "per_field_demands": [180, 160, 200, 150, 190, 170],
        "conflict_matrix": [[0,1,0,0,1,0],[1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,1,0,1,0],[1,0,0,1,0,1],[0,0,0,0,1,0]],
        "sources": [{"id":"river","capacity_gpm":1800}],
        "slot_length_hours": 1,
        "time_horizon_hours": 24,
        "per_field_priorities": [100,80,70,90,60,75],
        "hydraulic_model": {"pipe_loss_factor": 0.002},
        "seed": 42,
        "timeout_ms": 5000
    },
    "mode": "water_source_allocation",
    "mode_id": "water_source_allocation"
}
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode('utf-8')
        print(body)
except Exception as e:
    print('ERROR', e)
    sys.exit(1)
