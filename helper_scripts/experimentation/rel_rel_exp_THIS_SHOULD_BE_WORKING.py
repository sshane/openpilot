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
# y = [0.35, 0.3, 0.125, 0.09375, 0.075, 0, -0.09, -0.09375, -0.125, -0.3, -0.35]
y = [0.35, 0.3, 0.125, 0.09375, 0.075, 0, -0.075, -0.09375, -0.125, -0.3, -0.35]

# TR_mods.append(interp(rel_accel, x, y))

# >>> a = [(15 - 18), (14 - 17)]
# >>> (a[-1] - a[0]) / 1
v_ego_start = 20
v_ego_end = 21

v_lead_start = 20
v_lead_end = 19
time = 2

v_egos = np.linspace(v_ego_start, v_ego_end, res)
# a_lead = 10
v_leads = np.linspace(v_lead_start, v_lead_end, res)


TR = 1.8

v_rels = [v_lead - v_ego for v_lead, v_ego in zip(v_leads, v_egos)]
rel_accels = [(v_rel - v_rels[0]) / time for v_rel in v_rels]
rel_accels2 = [(v_rels[0] - v_rel) / time for v_rel in v_rels]

# v_ego_change = v_egos[-1] - v_egos[0]
# v_lead_change = v_leads[-1] - v_leads[0]
# change = v_lead_change - v_ego_change

v_ego_change = v_egos[-1] - v_egos[0]
v_lead_change = v_leads[-1] - v_leads[0]
print('v_ego_change: {}'.format(v_ego_change))
print('v_lead_change: {}'.format(v_lead_change))
# print(v_lead_change / v_ego_change)

rel_accels3 = []
for v_rel in v_rels:
  p1 = (v_rel - v_rels[0]) / time
  # if v_ego_change < 0 and v_lead_change > 0 or v_ego_change > 0 and v_lead_change < 0:
  # if v_ego_change < 0 < v_lead_change or v_ego_change > 0 > v_lead_change:
  if v_ego_change == 0 or v_lead_change == 0:
    _print('zero')
    if (v_ego_change == 0 and v_lead_change != 0) or v_lead_change == 0 == v_ego_change:
      lead_factor = 1
    elif v_lead_change == 0 and v_ego_change != 0:
      lead_factor = 0
    else:
      raise Exception('Uncovered case!')
  elif (v_ego_change < 0) != (v_lead_change < 0):  # one is negative and one is positive, or ^ = XOR
    _print('here1')
    # lead_factor = 1 - (v_lead_change / v_ego_change)
    # lead_factor = (v_ego_change - v_lead_change) / v_ego_change
    # lead_factor = v_ego_change / (v_ego_change + v_lead_change)
    # lead_factor = v_lead_change/(v_lead_change + v_ego_change)
    if v_ego_change < 0 and v_lead_change > 0:
      print('HA!')
      lead_factor = v_lead_change / (v_lead_change - v_ego_change)
      rel_accels3.append(math.copysign(p1, v_lead_change - v_ego_change) * lead_factor)
      continue
    else:
      print('HO!')
      lead_factor = v_lead_change / (v_lead_change - v_ego_change)
      rel_accels3.append(math.copysign(-p1, v_lead_change - v_ego_change) * lead_factor)
      continue
    # else:
    #   lead_factor = v_ego_change / abs(v_lead_change - v_ego_change)
    # rel_accels3.append(math.copysign(p1, v_lead_change + v_ego_change) * lead_factor)
    # continue
    # rel_accels3.append(p1 * lead_factor + (p1 * (1 - lead_factor)))
    if v_ego_change > v_lead_change:
      rel_accels3.append((p1 * abs(lead_factor)) + (-p1 * (1 - abs(lead_factor))))
    elif v_ego_change > v_lead_change:
      rel_accels3.append((-p1 * abs(lead_factor)) + (p1 * (1 - abs(lead_factor))))
    else:
      rel_accels3.append((-p1 * abs(lead_factor)) + (p1 * (1 - abs(lead_factor))))
    continue
  elif v_ego_change > 0 and v_lead_change > 0:
    _print('here2')
    if v_ego_change > v_lead_change:
      lead_factor = v_lead_change/abs(v_lead_change + v_ego_change)
      rel_accels3.append((-p1 * abs(lead_factor)) + (p1 * (1 - abs(lead_factor))))
      continue
    else:
      # lead_factor = v_ego_change/abs(v_lead_change + v_ego_change)
      # lead_factor = (v_ego_change / v_lead_change)
      lead_factor = v_lead_change/abs(v_lead_change + v_ego_change)
      rel_accels3.append((p1 * abs(lead_factor)) + (-p1 * (1 - abs(lead_factor))))  # TODO: test this
      continue
  elif v_ego_change < 0 and v_lead_change < 0:
    _print('here3')
    lead_factor = v_lead_change/abs(v_lead_change + v_ego_change)
    if abs(v_ego_change) < abs(v_lead_change):
      print('switched sign')
      lead_factor = -lead_factor
    # if abs(v_ego_change) > abs(v_lead_change):
    #   print('jere')
    #   rel_accels3.append((p1 * abs(lead_factor)) + (-p1 * (1 - abs(lead_factor))))  # TODO: test this
    #   continue
    # else:
    #   rel_accels3.append((-p1 * abs(lead_factor)) + (p1 * (1 - abs(lead_factor))))  # TODO: test this
    #   continue


  if lead_factor < 0:
    rel_accels3.append((p1 * abs(lead_factor)) + (-p1 * (1 - abs(lead_factor))))
  else:
    rel_accels3.append((-p1 * abs(lead_factor)) + (p1 * (1 - abs(lead_factor))))

# rel_accels3 = []
# for v_rel in v_rels:
#   p1 = (v_rel - v_rels[0]) / time
#   # p2 = (v_rels[0] - v_rel) / time
#
#   if v_ego_change == 0 and v_lead_change != 0:
#     print('case1')
#     lead_factor = 1
#     # rel_accels3.append(math.copysign(p1, v_lead_change))
#   elif v_lead_change == 0 and v_ego_change != 0:
#     print('case2')
#     lead_factor = 1
#     # rel_accels3.append(-math.copysign(p1, v_ego_change))
#
#   elif v_lead_change == v_ego_change:
#     print('equal')
#     lead_factor = 0.5
#     # rel_accels3.append((p1 + p2) / 2)
#   elif abs(v_ego_change) > abs(v_lead_change):
#     lead_factor = v_lead_change / v_ego_change
#
#   elif v_ego_change < 0 and v_lead_change > 0:
#     print('here1')
#     lead_factor = v_lead_change / (v_lead_change - v_ego_change)
#     # rel_accels3.append(p2 * lead_factor + p1 * (1 - lead_factor))
#   elif v_ego_change < 0 and v_lead_change < 0:
#     print('here2')
#     lead_factor = v_ego_change / v_lead_change  # or (v_lead_change - v_ego_change) / v_lead_change
#     # rel_accels3.append(p2 * lead_factor + p1 * (1 - lead_factor))
#   elif v_ego_change > 0 and v_lead_change > 0:
#     print('here3')
#     # lead_factor = v_lead_change / v_ego_change
#     lead_factor = 1 - (v_ego_change / v_lead_change)
#     # rel_accels3.append(p2 * lead_factor + p1 * (1 - lead_factor))
#   elif v_ego_change > 0 and v_lead_change < 0:
#     print('here4')
#     lead_factor = v_ego_change / (v_ego_change - v_lead_change)
#     # rel_accels3.append(p1 * lead_factor + p2 * (1 - lead_factor))
#   else:
#     raise Exception('Uncovered case!')
#
#   # sign = math.copysign(1, v_lead_change - v_ego_change)
#   rel_accels3.append(math.copysign(p1, v_lead_change - v_ego_change) * lead_factor)
#   # lead_factor *= sign
#   # rel_accels3.append(p1 * lead_factor - (p1 * (1 - lead_factor)))
#   # rel_accels3.append(p1 * lead_factor - (p1 * (1 - lead_factor)))
#   # rel_accels3.append(p1 * lead_factor + p2 * (1 - lead_factor))
#
#
#   # if change < 0:  # cars getting closer
#   #   # lead_factor = abs((v_lead_change - v_ego_change) / v_lead_change)
#   #   lead_factor = v_ego_change/v_lead_change
#   #   # if v_lead_change < 0 or v_ego_change < 0:
#   #   #   lead_factor = (v_ego_change - v_lead_change) / v_lead_change
#   #   # else:
#   #   #   lead_factor = v_lead_change / (v_lead_change - v_ego_change)
#   #   rel_accels3.append(p2 * lead_factor + p1 * (1 - lead_factor))
#   # elif change > 0:  # cars getting farther
#   #   print('not here')
#   #   # lead_factor = v_lead_change / (v_lead_change - v_ego_change)
#   #   # lead_factor = (v_lead_change - v_ego_change) / v_lead_change
#   #   lead_factor = v_lead_change/v_ego_change
#   #   rel_accels3.append(p2 * lead_factor + p1 * (1 - lead_factor))
#   # else:
#   #   print('equal')
#   #   lead_factor = 0.5
#   #   rel_accels3.append((p1 + p2) / 2)
#
#
#   # print(p1, p2, v_rels[len(v_rels) // 2])
#   # if p1 > p2:
#   #   rel_accels3.append(p1)
#   # else:
#   #   rel_accels3.append(p2)

print('lead_factor: {}'.format(lead_factor))

calc_TRs = [np.interp(accel, x, y) + TR for accel in rel_accels]
calc_TRs2 = [np.interp(accel, x, y) + TR for accel in rel_accels2]
calc_TRs3 = [np.interp(accel, x, y) + TR for accel in rel_accels3]

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
