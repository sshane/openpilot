import matplotlib.pyplot as plt
import os
import numpy as np
import json
import ast

from common.numpy_fast import clip

DT_CTRL = 0.01

eager_file = 'C:/Users/Shane Smiskol/eager-new.txt'
with open(eager_file, 'r') as f:
  data = f.read()

data = [ast.literal_eval(l) for l in data.split('\n') if len(l) > 1]

sequences = [[]]
for idx, line in enumerate(data):
  if line['enabled']:  # and line['v_ego'] > .25 and (line['apply_accel'] < 0.5 and line['v_ego'] < 8.9 or line['v_ego'] > 8.9):
    line['apply_accel'] *= 3  # this is what we want a_ego to match
    line['eager_accel'] *= 3  # this is the actual accel sent to the car
    sequences[-1].append(line)
  elif len(sequences[-1]) != 0 and len(data) - 1 != idx:
    sequences.append([])
del data

# todo: sequences = [seq for seq in sequences if len(seq) > 5 * 100]

print('Samples: {}'.format(sum([len(s) for s in sequences])))
print('Sequences: {}'.format(len(sequences)))


# 34, 35, 36  these sequences have eager accel disabled

def plot_seq(idx=33, title=''):
  seq = sequences[idx]
  apply_accel, eager_accel, a_ego, v_ego = zip(*[(l['apply_accel'], l['eager_accel'], l['a_ego'], l['v_ego']) for l in seq])

  RC_orig = 0.5
  alpha_orig = 1. - DT_CTRL / (RC_orig + DT_CTRL)

  RC_1 = 0.5  # fast average
  alpha_1 = 1. - DT_CTRL / (RC_1 + DT_CTRL)

  # Variables for new eager accel
  _delayed_output = 0
  _delayed_derivative = 0

  # Variables to visualize (not required in practice)
  _delayed_outputs = []
  _new_accels = []


  eags = [0]
  accel_with_deriv = []
  accel_with_sorta_smooth_jerk = []
  derivatives = []
  jerks = []
  sorta_smooth_jerks = []
  less_smooth_derivative_2 = []
  jerk_TC = round(0.25 * 100)
  for idx, line in enumerate(seq):  # todo: left off at trying to plot derivative of accel (jerk)
    _delayed_output = _delayed_output * alpha_orig + line['apply_accel'] * (1. - alpha_orig)
    _derivative = line['apply_accel'] - _delayed_output
    _delayed_derivative = _delayed_derivative * alpha_orig + _derivative * (1. - alpha_orig)

    _new_accels.append(line['apply_accel'] + (_derivative - _delayed_derivative))

    # Visualize
    _delayed_outputs.append(_delayed_output)


    # todo: edit: accel_with_sorta_smooth_jerk seems promising
    eags.append(eags[-1] * alpha_1 + line['apply_accel'] * (1. - alpha_1))

    less_smooth_derivative_2.append((line['apply_accel'] - eags[-1]))  # todo: ideally use two delayed output variables
    if idx > jerk_TC:
      derivatives.append((line['apply_accel'] - seq[idx - jerk_TC]['apply_accel']))
      jerks.append(derivatives[-1] - derivatives[idx - jerk_TC])
      sorta_smooth_jerks.append(less_smooth_derivative_2[-1] - less_smooth_derivative_2[idx - jerk_TC])
    else:
      jerks.append(0)
      derivatives.append(0)
      sorta_smooth_jerks.append(0)
    accel_with_deriv.append(line['apply_accel'] + derivatives[-1] / 10)
    accel_with_sorta_smooth_jerk.append(line['apply_accel'] + sorta_smooth_jerks[-1] / 2)
    # calc_eager_accels.append(line['apply_accel'] - (eag - line['apply_accel']) * 0.5)

  plt.figure()
  plt.plot(apply_accel, label='original desired accel')
  plt.plot(a_ego, label='a_ego')
  # plt.plot(_new_accels, label='new_accels')
  # plt.plot(eager_accel, label='current eager accel')
  plt.legend()
  plt.title(title)
  plt.figure()
  plt.plot(v_ego, label='v_ego')
  plt.legend()
  plt.title(title)
  # plt.plot(eags, label='exp. average')
  # plt.plot(derivatives, label='reg derivative')
  # plt.plot(jerks, label='jerk of reg deriv')
  # plt.plot(accel_with_sorta_smooth_jerk, label='acc with sorta smooth jerk')
  # plt.plot(accel_with_deriv, label='acc with true derivative')



  # calc_eager_accels = []
  # eag = 0
  # for line in seq:
  #   eag = eag * alpha + line['apply_accel'] * (1. - alpha)
  #   calc_eager_accels.append(line['apply_accel'] - (eag - line['apply_accel']) * 0.5)

  # plt.plot(apply_accel, label='original desired accel')
  # plt.plot(calc_eager_accels, label='calc. accel sent to car')
  # plt.plot(a_ego, label='a_ego')
  plt.show()


# plot_seq(10)


# 0 to 6 are good seqs with new data
plot_seq(2, title='eager 1')  # eager
plot_seq(3, title='eager 2')  # eager
# plot_seq(4)  # not eager (todo: think this is bad)
plot_seq(5, title='not eager 1')  # not eager
plot_seq(6, title='not eager 2')  # not eager

# with open('C:/Users/Shane Smiskol/apply_accel_test', 'w') as f:
#   f.write(json.dumps(apply_accel))
