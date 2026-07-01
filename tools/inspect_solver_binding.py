import json
import solver_py

inputs = {
  'field_count':16,
  'aqueduct_supply_limit':2500,
  'flow_rates':[max(10,40+4*35+(i%5)*12) for i in range(16)],
  'priority_values':[70.0]*16,
  'seed':0
}
res = solver_py.solve_irrigation_schedule(inputs, 'water_source_allocation')
print('TYPE0', type(res[0]), 'TYPE1', type(res[1]))
try:
    sched = list(res[0])
except Exception:
    sched = res[0]
print('RES0 (schedule) len=', len(sched) if hasattr(sched,'__len__') else None)
print('RES0 snippet=', sched[:16])
print('RES1 keys=', list(res[1].keys()))
print('METRICS JSON:')
print(json.dumps(dict(res[1]), indent=2))
