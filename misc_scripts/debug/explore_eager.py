import matplotlib.pyplot as plt
import os
import numpy as np
import json
import ast
from selfdrive.config import Conversions as CV

from common.numpy_fast import clip

DT_CTRL = 0.01

# eager_file = 'C:/Users/Shane Smiskol/eager-new.txt'  # to compare with and without eagerness
eager_file = 'C:/Users/Shane Smiskol/eager.txt'
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

def get_alpha_jerk(speed):
  RC = np.interp(speed * CV.MS_TO_MPH, [0, 10, 80], [.01, .1, 1])
  return 1. - DT_CTRL / (RC + DT_CTRL)

def get_alpha_eager(speed):
  RC = np.interp(speed * CV.MS_TO_MPH, [0, 10, 80], [.01, .1, 1])
  return 1. - DT_CTRL / (RC + DT_CTRL)


def plot_seq(idx=33, title=''):
  seq = sequences[idx]
  apply_accel, eager_accel, a_ego, v_ego = zip(*[(l['apply_accel'], l['eager_accel'], l['a_ego'], l['v_ego']) for l in seq])

  # RC_jerk = 0.5
  # alpha_jerk = 1. - DT_CTRL / (RC_jerk + DT_CTRL)
  # RC_eager = 0.5
  # alpha_eager = 1. - DT_CTRL / (RC_eager + DT_CTRL)

  # Variables for new eager accel (using jerk)
  _delayed_output_jerk = 0
  _delayed_derivative_jerk = 0

  # For original eager accel
  _delayed_output_eager = 0

  # Variables to visualize (not required in practice)
  # _delayed_outputs_jerk = []
  _new_accels_jerk = []

  # For original eager accel
  # _delayed_outputs_eager = []
  _eager_accels = []


  # eags = [0]
  # accel_with_deriv = []
  # accel_with_sorta_smooth_jerk = []
  # derivatives = []
  # jerks = []
  # sorta_smooth_jerks = []
  # less_smooth_derivative_2 = []
  # original_eager_accel = []
  # jerk_TC = round(0.25 * 100)
  for idx, line in enumerate(seq):  # todo: left off at trying to plot derivative of accel (jerk)
    alpha_jerk = get_alpha_jerk(line['v_ego'])
    _delayed_output_jerk = _delayed_output_jerk * alpha_jerk + line['apply_accel'] * (1. - alpha_jerk)
    alpha_eager = get_alpha_eager(line['v_ego'])
    _delayed_output_eager = _delayed_output_eager * alpha_eager + line['apply_accel'] * (1. - alpha_eager)

    _derivative_jerk = line['apply_accel'] - _delayed_output_jerk
    _delayed_derivative_jerk = _delayed_derivative_jerk * alpha_jerk + _derivative_jerk * (1. - alpha_jerk)

    # Visualize
    _new_accels_jerk.append(line['apply_accel'] - (_delayed_derivative_jerk - _derivative_jerk))
    _eager_accels.append(line['apply_accel'] + (line['apply_accel'] - _delayed_output_eager))

    # # _delayed_outputs_jerk.append(_delayed_output_jerk)
    #
    # original_eager_accel.append(line['apply_accel'] - (_delayed_output - line['apply_accel']))
    #
    #
    # # todo: edit: accel_with_sorta_smooth_jerk seems promising
    # eags.append(eags[-1] * alpha_1 + line['apply_accel'] * (1. - alpha_1))
    #
    # less_smooth_derivative_2.append((line['apply_accel'] - eags[-1]))  # todo: ideally use two delayed output variables
    # if idx > jerk_TC:
    #   derivatives.append((line['apply_accel'] - seq[idx - jerk_TC]['apply_accel']))
    #   jerks.append(derivatives[-1] - derivatives[idx - jerk_TC])
    #   sorta_smooth_jerks.append(less_smooth_derivative_2[-1] - less_smooth_derivative_2[idx - jerk_TC])
    # else:
    #   jerks.append(0)
    #   derivatives.append(0)
    #   sorta_smooth_jerks.append(0)
    # accel_with_deriv.append(line['apply_accel'] + derivatives[-1] / 10)
    # accel_with_sorta_smooth_jerk.append(line['apply_accel'] + sorta_smooth_jerks[-1] / 2)
    # # calc_eager_accels.append(line['apply_accel'] - (eag - line['apply_accel']) * 0.5)

  plt.figure()
  plt.plot(apply_accel, label='original desired accel')
  # plt.plot(a_ego, label='a_ego')
  plt.plot(_new_accels_jerk, label='eager accel using jerk')
  # plt.plot(eager_accel, label='current eager accel')
  # plt.plot(eags, label='exp. average')
  # plt.plot(derivatives, label='reg derivative')
  # plt.plot(jerks, label='jerk of reg deriv')
  # plt.plot(accel_with_sorta_smooth_jerk, label='acc with sorta smooth jerk')
  # plt.plot(_eager_accels, label='original eager accel')
  # plt.plot(accel_with_deriv, label='acc with true derivative')

  plt.legend()
  plt.title(title)

  # plt.figure()
  # plt.plot(v_ego, label='v_ego')
  # plt.legend()
  # plt.title(title)



  # calc_eager_accels = []
  # eag = 0
  # for line in seq:
  #   eag = eag * alpha + line['apply_accel'] * (1. - alpha)
  #   calc_eager_accels.append(line['apply_accel'] - (eag - line['apply_accel']) * 0.5)

  # plt.plot(apply_accel, label='original desired accel')
  # plt.plot(calc_eager_accels, label='calc. accel sent to car')
  # plt.plot(a_ego, label='a_ego')
  plt.show()


plot_seq(14)

# # to compare with and without eagerness
# # 0 to 6 are good seqs with new data
# plot_seq(2, title='eager 1')  # eager
# plot_seq(3, title='eager 2')  # eager
# # plot_seq(4)  # not eager (todo: think this is bad)
# plot_seq(5, title='not eager 1')  # not eager
# plot_seq(6, title='not eager 2')  # not eager
#
# # with open('C:/Users/Shane Smiskol/apply_accel_test', 'w') as f:
# #   f.write(json.dumps(apply_accel))
