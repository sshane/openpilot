import matplotlib.pyplot as plt
import numpy as np
import math

res = 50

printed = False
def _print(msg):
  global printed
  if not printed:
    print(msg)
    printed = True


x = np.array([-2.6822, -1.7882, -0.8941, -0.447, -0.2235, 0.0, 0.2235, 0.447, 0.8941, 1.7882, 2.6822]) * 2.2369
y = [0.35, 0.3, 0.125, 0.09375, 0.075, 0, -0.09, -0.09375, -0.125, -0.3, -0.35]


v_ego_start = 25
v_ego_end = 21

v_lead_start = 25
v_lead_end = 26
time = 2

v_egos = np.linspace(v_ego_start, v_ego_end, res)
# a_lead = 10
v_leads = np.linspace(v_lead_start, v_lead_end, res)


TR = 1.8

v_rels = [v_lead - v_ego for v_lead, v_ego in zip(v_leads, v_egos)]
rel_accels = [(v_rel - v_rels[0]) / time for v_rel in v_rels]
rel_accels2 = [(v_rels[0] - v_rel) / time for v_rel in v_rels]



v_ego_change = v_egos[-1] - v_egos[0]
v_lead_change = v_leads[-1] - v_leads[0]
print('v_ego_change: {}'.format(v_ego_change))
print('v_lead_change: {}'.format(v_lead_change))

rel_accels3 = []
for v_rel in v_rels:
  p1 = (v_rel - v_rels[0]) / time
  diff_mod = False
  if v_ego_change == 0 or v_lead_change == 0:
    _print('zero')
    lead_factor = v_lead_change / (v_lead_change - v_ego_change)
    # rel_accels3.append(math.copysign(p1, v_lead_change - v_ego_change) * lead_factor)
    # continue  # this or just setting lead_factor to 0 works
  elif (v_ego_change < 0) != (v_lead_change < 0):  # one is negative and one is positive, or ^ = XOR
    lead_factor = v_lead_change / (v_lead_change - v_ego_change)
    if v_ego_change > 0 > v_lead_change:
      p1 = -p1
    diff_mod = True
  elif v_ego_change * v_lead_change > 0:
    _print('here2')
    lead_factor = v_lead_change / (v_lead_change + v_ego_change)
    if v_ego_change > 0 and v_lead_change > 0:
      if v_ego_change < v_lead_change:
        p1 = -p1
    elif v_ego_change > v_lead_change:
      p1 = -p1

  if not diff_mod:
    rel_accels3.append((-p1 * abs(lead_factor)) + (p1 * (1 - abs(lead_factor))))
    # if lead_factor < 0:
    #   rel_accels3.append((p1 * abs(lead_factor)) + (-p1 * (1 - abs(lead_factor))))
    # else:
    #   rel_accels3.append((-p1 * abs(lead_factor)) + (p1 * (1 - abs(lead_factor))))
  else:
    rel_accels3.append(math.copysign(p1, v_lead_change - v_ego_change) * lead_factor)

print('lead_factor: {}'.format(lead_factor))
calc_TRs = [np.interp(accel, x, y) + TR for accel in rel_accels]
calc_TRs2 = [np.interp(accel, x, y) + TR for accel in rel_accels2]
calc_mods = np.interp(rel_accels3, x, y)
if v_lead_end > v_ego_end and np.mean(calc_mods) >= 0:
  print('modding')
  x = [0, 2, 4, 8]
  y = [1.0, -0.25, -0.65, -0.95]
  v_rel_mod = np.interp(v_lead_end - v_ego_end, x, y)
  calc_mods *= v_rel_mod

calc_TRs3 = calc_mods + TR
# plt.plot(rel_accels, label='rel_accel')
plt.figure(1)
plt.plot(np.linspace(0, 1, res), calc_TRs, label='TRs')
plt.plot(np.linspace(0, 1, res), calc_TRs2, label='TRs2')
plt.plot(np.linspace(0, 1, res), calc_TRs3, label='new TRs')
plt.legend()

plt.figure(2)
plt.plot(np.linspace(0, res, res), v_egos, label='v_ego')
plt.plot(np.linspace(0, res, res), v_leads, label='v_lead')
plt.legend()
plt.show()
