import matplotlib.pyplot as plt
import ast
import numpy as np

with open('/Git/rel_accel_mod', 'r') as f:
  data = [ast.literal_eval(line) for line in f.read().split('\n')[:-1]]

# data = data[980:1100]
print(data[0].keys())
v_lead = [line['v_lead'] for line in data]
v_ego = [line['v_ego'] for line in data]
plt.figure(0)
plt.plot(v_lead, label='v_lead')
plt.plot(v_ego, label='v_ego')
plt.legend()

plt.figure(1)
a_leads = []
for line in data:
  a_l = line['a_lead']
  # print(a_l)
  mods_x = [0, .75, 1.5]
  mods_y = [3, 1.5, 1]
  if a_l < 0:
    print('here!')
    a_l *= np.interp(abs(a_l), mods_x, mods_y)
  # print(a_l)
  # print('---')
  a_leads.append(a_l)

rel_accel_mod = [line['rel_accel_mod'] for line in data]
# a_lead_a_ego = [line['a_lead']**2 - line['a_ego'] for a_l, line in zip(a_lead, data)]
a_lead_a_ego = [a_l - line['a_ego'] for a_l, line in zip(a_leads, data)]
plt.plot(rel_accel_mod, label='rel_accel_mod')
plt.plot(a_lead_a_ego, label='a_lead_a_ego')
plt.legend()

plt.show()
