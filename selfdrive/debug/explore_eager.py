import matplotlib.pyplot as plt
import os
import numpy as np
import json
import ast

from common.numpy_fast import clip

DT_CTRL = 0.01

eager_file = 'C:/Users/Shane Smiskol/eager.txt'
with open(eager_file, 'r') as f:
  data = f.read()

data = [ast.literal_eval(l) for l in data.split('\n') if len(l) > 1]

sequences = [[]]
for idx, line in enumerate(data):
  if line['enabled'] and line['v_ego'] > .25 and (line['apply_accel'] < 0.5 and line['v_ego'] < 8.9 or line['v_ego'] > 8.9):
    line['apply_accel'] *= 3  # this is what we want a_ego to match
    line['eager_accel'] *= 3  # this is the actual accel sent to the car
    sequences[-1].append(line)
  elif len(sequences[-1]) != 0 and len(data) - 1 != idx:
    sequences.append([])
del data

sequences = [seq for seq in sequences if len(seq) > 5 * 100]

print('Samples: {}'.format(sum([len(s) for s in sequences])))
print('Sequences: {}'.format(len(sequences)))


# 34, 35, 36  these sequences have eager accel disabled

def plot_seq(idx=33):
  seq = sequences[idx]
  apply_accel, eager_accel, a_ego = zip(*[(l['apply_accel'], l['eager_accel'], l['a_ego']) for l in seq])

  RC_orig = 0.25
  alpha_orig = 1. - DT_CTRL / (RC_orig + DT_CTRL)

  RC_1 = 0.5  # fast average
  RC_2 = 0.75  # slow average
  alpha_1 = 1. - DT_CTRL / (RC_1 + DT_CTRL)
  alpha_2 = 1. - DT_CTRL / (RC_2 + DT_CTRL)

  eags = [0]
  eag_eags = [0]
  smooth_derivs = []
  smooth_eager_accel = []
  accel_with_deriv = []
  accel_with_jerk = []
  accel_with_sorta_smooth_jerk = []
  derivatives = []
  jerks = []
  sorta_smooth_jerks = []
  less_smooth_derivative_2 = []
  _delayed_output = 0
  eager_accel_new = []
  jerk_TC = round(0.5 * 100)
  for idx, line in enumerate(seq):  # todo: left off at trying to plot derivative of accel (jerk)
    # todo: edit: accel_with_sorta_smooth_jerk seems promising
    eags.append(eags[-1] * alpha_1 + line['apply_accel'] * (1. - alpha_1))
    eag_eags.append(eag_eags[-1] * alpha_2 + eags[-1] * (1 - alpha_2))
    accel_derivative = eags[-1] - eag_eags[-1]
    smooth_derivs.append(accel_derivative)
    _delayed_output = _delayed_output * alpha_orig + line['apply_accel'] * (1. - alpha_orig)
    eager_accel_new.append(line['apply_accel'] - (_delayed_output - line['apply_accel']) * 1.4)

    # smooth_eager_accel.append(clip(line['apply_accel'] + accel_derivative, -3, 3))
    less_smooth_derivative_2.append((line['apply_accel'] - eags[-1]))  # todo: ideally use two delayed output variables
    if idx > jerk_TC:
      derivatives.append((line['apply_accel'] - seq[idx - jerk_TC]['apply_accel']))
      jerks.append(derivatives[-1] - derivatives[idx - jerk_TC])
      sorta_smooth_jerks.append(less_smooth_derivative_2[-1] - less_smooth_derivative_2[idx - jerk_TC])
    else:
      jerks.append(0)
      derivatives.append(0)
      sorta_smooth_jerks.append(0)
    # accel_with_deriv.append(line['apply_accel'] + derivatives[-1] / 10)
    accel_with_jerk.append(line['apply_accel'] + jerks[-1] / 4)
    accel_with_sorta_smooth_jerk.append(line['apply_accel'] + sorta_smooth_jerks[-1] / 2)
    # calc_eager_accels.append(line['apply_accel'] - (eag - line['apply_accel']) * 0.5)

  plt.clf()
  plt.plot(apply_accel, label='original desired accel')
  # plt.plot(smooth_eager_accel, label='smooth eager accel')
  plt.plot(eager_accel, label='current eager accel')
  plt.plot(eager_accel_new, label='calc eager accel')
  # plt.plot(eags, label='exp. average')
  # plt.plot(eag_eags[11000:-1], label='exp. exp. average')
  # plt.plot(smooth_derivs, label='eag derivative')
  # plt.plot(derivatives, label='reg derivative')
  # plt.plot(jerks, label='jerk of reg deriv')
  # plt.plot(accel_with_jerk, label='acc with jerk')
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
  plt.legend()
  plt.show()
plot_seq()
