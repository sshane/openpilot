#!/usr/bin/env python3
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, GaussianNoise, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adagrad, Adadelta, Adam

from sklearn.model_selection import train_test_split

from opendbc.can.parser import CANParser
from tools.lib.logreader import MultiLogIterator
from tools.lib.route import Route
from cereal import car
from common.filter_simple import FirstOrderFilter
from selfdrive.controls.lib.drive_helpers import CONTROL_N
from selfdrive.modeld.constants import T_IDXS

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm   # type: ignore
from scipy.optimize import curve_fit

from selfdrive.config import Conversions as CV
from common.realtime import DT_CTRL
import pickle

dir_name = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_name)

MIN_SAMPLES = int(30 / DT_CTRL)

T_IDXS = np.array(T_IDXS[:CONTROL_N])
T_FRAMES = [int(round(t * 100.)) for t in T_IDXS]

FUTURE_TIME = 0.15
FUTURE_FRAMES = round(FUTURE_TIME / DT_CTRL)
# T_FRAME_IDX_FUTURE = np.argmin([abs(i - FUTURE_TIME) for i in T_IDXS])  # find closest index


def tokenize(data, seq_length):
  seq = []
  for i in range(len(data) - seq_length + 1):
    token = data[i:i + seq_length]
    if len(token) == seq_length:
      seq.append(token)
  return seq


def split_list(l, n, enforce_len=True):
  output = []
  for i in range(0, len(l), n):
    output.append(l[i:i + n])
  return [i for i in output if (len(i) == n and enforce_len) or not enforce_len]


def get_lrs():
  rlog_dir = 'rlogs'
  route_dirs = [f for f in os.listdir(os.path.join(rlog_dir)) if '.ini' not in f and f != 'exclude']
  lrs = []
  for route in route_dirs:
    route_files = []
    for segment in os.listdir(os.path.join(rlog_dir, route)):
      if segment == 'exclude' or '.ini' in segment:
        continue
      route_files.append(os.path.join(rlog_dir, route, segment))
    lrs.append(MultiLogIterator(route_files))
  return lrs


def load_processed(file_name):
  with open(file_name, 'rb') as f:
    return pickle.load(f)


def show_pred():
  seq = data_tokenized_test[np.random.randint(len(data_tokenized_test))]
  idx = np.random.randint(len(seq))
  _aEgos = [seq[t_frame]['aEgo'] for t_frame in T_FRAMES]
  _accels = seq[0]['accels']
  _accel_cmds = [seq[t_frame]['accel_cmd'] for t_frame in T_FRAMES]
  pred = model.predict([[seq[0]['vEgo']] + _aEgos])
  plt.clf()
  plt.plot(_aEgos, label='aEgo')
  plt.plot(pred[0], label='prediction')
  plt.plot(_accel_cmds, label='accel cmds')
  # plt.plot(_accels, label='lat_plan.accels')
  plt.xticks(range(len(T_IDXS)), np.round(T_IDXS, 2))
  plt.xlabel('seconds')
  plt.legend()
  plt.show()
  plt.pause(1)
  plt.show()


def load_and_process_rlogs(lr):
  data = [[]]

  log_msgs = {
    'carState': {
      'vEgo': None,
      'aEgo': None,
      'gasPressed': None,
    },
    'controlsState': {
      'enabled': False,
    },
    'longitudinalPlan': {
      'jerks': None,
      'accels': None,
      'speeds': None,
    }
  }
  last_time = 0
  can_updated = False

  can_signals = [
    ("ACCEL_CMD", "ACC_CONTROL", 0),
  ]
  cp = CANParser("toyota_nodsu_hybrid_pt_generated", can_signals, enforce_checks=False)  # todo: auto load dbc from logs

  all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)

  for msg in tqdm(all_msgs):
    for log_msg in log_msgs:
      if msg.which() == log_msg:
        for log_signal in log_msgs[log_msg]:
          log_msgs[log_msg][log_signal] = getattr(getattr(msg, log_msg), log_signal)

    if msg.which() != 'sendcan':
      continue

    cp_updated = cp.update_string(msg.as_builder().to_bytes())
    for u in cp_updated:
      if u == 0x343:  # ACC_CONTROL
        can_updated = True

    accel_cmd = float(cp.vl['ACC_CONTROL']['ACCEL_CMD'])

    sample_ok = log_msgs['carState']['vEgo'] is not None and can_updated
    sample_ok = sample_ok and log_msgs['controlsState']['enabled'] and not log_msgs['carState']['gasPressed']
    # sample_ok = sample_ok and log_msgs['carState']['vEgo'] >= 0.

    # creates uninterupted sections of engaged data
    if sample_ok and abs(msg.logMonoTime - last_time) * 1e-9 < 1 / 20:  # also split if there's a break in time
      to_append = {'accel_cmd': accel_cmd, 'time': msg.logMonoTime * 1e-9}
      for log_signals in log_msgs.values():
        to_append = {**to_append, **log_signals}
      for i in to_append:
        if 'capnp' in str(type(to_append[i])):
          to_append[i] = list(to_append[i])
      data[-1].append(to_append)
    elif len(data[-1]):  # if last list has items in it, append new empty section
      data.append([])
    last_time = msg.logMonoTime

  del all_msgs
  data = [sec for sec in data if len(sec) > (5 / DT_CTRL)]  # long enough sections

  return data


if __name__ == "__main__":
  # # r = Route(sys.argv[1])
  # # lr = MultiLogIterator(r.log_paths(), wraparound=False)
  # use_dir = os.path.join(dir_name, 'rlogs')
  # route_files = [os.path.join(use_dir, f) for f in os.listdir(use_dir) if f != 'exclude' and '.ini' not in f]
  # print(route_files)
  # lr = MultiLogIterator(route_files)

  if os.path.exists('processed_data'):
    print('exists')
    data = load_processed('processed_data')
    data += load_processed('processed_data')
  else:
    data = []
    for lr in get_lrs():  # each logreader is for a full route. sorting by time doesn't work for different routes
      print('processing {}'.format('--'.join(lr._log_paths[0].split('--')[:-2])))
      data += load_and_process_rlogs(lr)
    with open('processed_data', 'wb') as f:  # now dump
      pickle.dump(data, f)
  print('Max seq. len: {}'.format(max([len(line) for line in data])))
  print('Seq. lens: {}'.format([len(line) for line in data]))

  data_tokenized = []
  data_tokenized_test = []
  for seq in data:
    data_tokenized += tokenize(seq, max(T_FRAMES) + 1 + FUTURE_FRAMES)
    data_tokenized_test += tokenize(seq, round(5. / DT_CTRL))
  del data

  # data_sequences = []
  # idx = 0
  # for seq in data_tokenized:
  #   if idx > 1:
  #     data_sequences.append(seq)
  #     idx = 0
  #   idx += 1
  data_sequences = data_tokenized
  del data_tokenized

  print(f'Tokenized sequences: {len(data_sequences)}')
  if PLOT := False:
    r = np.random.randint(25000)
    plt.plot([i['aEgo'] for i in data_sequences[r]], label='aEgo')
    plt.plot([i['accel_cmd'] for i in data_sequences[r]], label='acc_command')
    # plt.plot([i['jerks'][2] for i in data_sequences[r]], label='jerks')
    plt.plot([i['accels'][0] for i in data_sequences[r]], label='accels')
    plt.legend()
    plt.show()
    input()

  x_train = []
  y_train = []
  for seq in data_sequences:
    # given current and future accel, output accel cmd we used to get there
    # x_train.append([seq[0]['vEgo'], seq[FUTURE_FRAMES]['vEgo'], seq[0]['aEgo'], seq[FUTURE_FRAMES]['aEgo']])
    # x_train.append([seq[0]['vEgo'], seq[FUTURE_FRAMES]['vEgo']])
    # x_train.append([seq[0]['vEgo'], seq[FUTURE_FRAMES]['vEgo']])
    keep_seq = 0.25 if any([line['vEgo'] < 0.2 for line in seq]) else 1.0
    if np.random.uniform(0, 1) > keep_seq:
      continue
    accels = [seq[t_frame + FUTURE_FRAMES]['aEgo'] for t_frame in T_FRAMES]
    speeds = [seq[t_frame + FUTURE_FRAMES]['vEgo'] for t_frame in T_FRAMES]
    x_train.append([seq[0]['vEgo'], seq[0]['aEgo'], speeds[0]] + accels)
    accel_cmds = [seq[t_frame]['accel_cmd'] for t_frame in T_FRAMES]  # this is offset left by FUTURE_FRAMES
    y_train.append(accel_cmds)

  x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=42)

  x_train = np.array(x_train)
  y_train = np.array(y_train)
  print('x_train shape: {}, y_train shape: {}'.format(x_train.shape, y_train.shape))

  model = Sequential()
  # model.add(GaussianNoise(0.1, input_shape=(3,)))
  model.add(Dense(32, input_shape=(len(T_FRAMES)+3,), activation=LeakyReLU()))
  # model.add(Dropout(0.05))
  model.add(Dense(32, activation=LeakyReLU()))
  # model.add(Dropout(0.05))
  model.add(Dense(16, activation=LeakyReLU()))
  # model.add(Dropout(0.1))
  model.add(Dense(len(T_FRAMES), activation='linear'))

  opt = Adam(lr=0.001, amsgrad=True)
  # opt = Adagrad(lr=0.001)
  # opt = Adadelta(lr=1.)

  model.compile(opt, loss='mse', metrics='mae')
  epochs = [2, 6, 8, 2]
  batch_sizes = [256, 64, 32, 16]
  for epoch, batch_size in zip(epochs, batch_sizes):
    try:
      # model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, batch_size=batch_size)
      model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size)
    except KeyboardInterrupt:
      pass

  def show_random_preds(num=20):
    rand_idxs = np.random.randint(len(x_test), size=num)
    _x = np.take(x_test, rand_idxs, axis=0)
    _y = np.take(y_test, rand_idxs, axis=0)
    _y_pred = model.predict(_x)

    plt.clf()
    plt.plot(_y, label='ground truth')
    plt.plot(_y_pred, label='prediction')
    plt.legend()
    plt.show()
    plt.pause(0.01)

  def show_pred_seqs():
    while True:
      rand_idx = np.random.randint(len(data_tokenized_test))
      seq = data_tokenized_test[rand_idx]
      if seq[-1]['vEgo'] < 2:
        break
    preds = []
    for idx, line in enumerate(seq):
      if idx + FUTURE_FRAMES >= len(seq):
        break
      fut_speed = np.interp(FUTURE_TIME, T_IDXS, seq[idx]['speeds'])
      fut_accel = np.interp(FUTURE_TIME, T_IDXS, seq[idx]['accels'])
      # preds.append(model.predict([[seq[idx]['accels'][0], seq[idx + FUTURE_FRAMES]['accels'][2]]])[0])
      # preds.append(model.predict([[seq[idx]['vEgo'], seq[idx]['speeds'][2], seq[idx]['aEgo'], seq[idx]['accels'][2]]])[0])
      # preds.append(model.predict([[seq[idx]['vEgo'], fut_speed]])[0])
      pred_accels = model.predict([[seq[idx]['vEgo']] + seq[idx]['accels']])[0]
      # preds.append(np.interp(FUTURE_TIME, T_IDXS, pred_accels))
      preds.append(pred_accels[0])
      # preds.append(model.predict([[seq[idx]['vEgo'], seq[idx]['speeds'][2]]])[0])

    plt.figure(0)
    plt.clf()
    plt.plot([line['accel_cmd'] for line in seq], label='accel_cmd')
    plt.plot(preds, label='prediction')
    plt.plot([line['aEgo'] for line in seq], label='aEgo')
    plt.plot([line['accels'][0] for line in seq], label='accels[0]')
    # plt.plot([np.interp(FUTURE_TIME, T_IDXS, line['accels']) for line in seq], label='future accel')
    plt.legend()

    plt.figure(1)
    plt.clf()
    plt.plot([line['vEgo'] for line in seq], label='vEgo')
    plt.plot([line['speeds'][0] for line in seq], label='speeds[0]')
    plt.plot([np.interp(FUTURE_TIME, T_IDXS, line['speeds']) for line in seq], label='future speed')
    plt.legend()

    plt.show()
    plt.pause(0.05)

  def pred_animation():
    for seq in data_sequences:
      aEgos = [seq[t_frame]['aEgo'] for t_frame in T_FRAMES]
      vEgos = [seq[t_frame]['vEgo'] for t_frame in T_FRAMES]
      pred = model.predict([[vEgos[0]] + seq[0]['accels']])[0]
      accel_cmds = [seq[t_frame]['accel_cmd'] for t_frame in T_FRAMES]
      plt.clf()
      plt.plot(accel_cmds, label='accel_cmds')
      plt.plot(pred, label='prediction')
      plt.plot(aEgos, label='aEgo')
      plt.plot(seq[0]['accels'], label='accels')
      plt.legend()
      plt.show()
      plt.pause(1 / 100)

  # pred_animation()
  show_pred()
  raise Exception


  while 1:
    # show_preds()
    show_pred_seqs()
    e = input()
    if e == 'exit':
      break


  # data_speeds = np.array([line['v_ego'] for line in data])
  # data_angles = np.array([line['angle_steers'] for line in data])
  # data_torque = np.array([line['torque'] for line in data])
  #
  # params, covs = curve_fit(_fit_kf, np.array([data_speeds, data_angles]), np.array(data_torque) / MAX_TORQUE,
  #                          # maxfev=800
  #                          )
  # fit_kf = params[0]
  # print('FOUND KF: {}'.format(fit_kf))
  #
  # print()
  #
  # std_func = []
  # fitted_func = []
  # for line in data:
  #   std_func.append(abs(feedforward(line['v_ego'], line['angle_steers'], old_kf) * MAX_TORQUE - line['torque']))
  #   fitted_func.append(abs(feedforward(line['v_ego'], line['angle_steers'], fit_kf) * MAX_TORQUE - line['torque']))
  # print('Function comparison on input data')
  # print('Torque MAE, current vs. fitted: {}, {}'.format(round(np.mean(std_func), 3), round(np.mean(fitted_func), 3)))
  # print('Torque STD, current vs. fitted: {}, {}'.format(round(np.std(std_func), 3), round(np.std(fitted_func), 3)))
  #
  # if SPEED_DATA_ANALYSIS := True:  # analyzes how torque needed changes based on speed
  #   if PLOT_ANGLE_DIST := False:
  #     sns.distplot([line['angle_steers'] for line in data if abs(line['angle_steers']) < 30], bins=200)
  #     raise Exception
  #
  #   res = 100
  #   color = 'blue'
  #
  #   _angles = [
  #     [5, 10],
  #     [10, 20],
  #     [10, 15],
  #     [15, 20],
  #     [20, 25],
  #     [20, 30],
  #     [30, 45],
  #   ]
  #
  #   for idx, angle_range in enumerate(_angles):
  #     angle_range_str = '{} deg'.format('-'.join(map(str, angle_range)))
  #     temp_data = [line for line in data if angle_range[0] <= abs(line['angle_steers']) <= angle_range[1]]
  #     if not len(temp_data):
  #       continue
  #     print(f'{angle_range} samples: {len(temp_data)}')
  #     plt.figure()
  #     speeds, torque = zip(*[[line['v_ego'], line['torque']] for line in temp_data])
  #     plt.scatter(np.array(speeds) * CV.MS_TO_MPH, torque, label=angle_range_str, color=color, s=0.05)
  #
  #     _x_ff = np.linspace(0, max(speeds), res)
  #     _y_ff = [feedforward(_i, np.mean(angle_range), old_kf) * MAX_TORQUE for _i in _x_ff]
  #     plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='orange', label='standard ff model at {} deg'.format(np.mean(angle_range)))
  #
  #     _y_ff = [feedforward(_i, np.mean(angle_range), fit_kf) * MAX_TORQUE for _i in _x_ff]
  #     plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='purple', label='new fitted ff function')
  #
  #     plt.legend()
  #     plt.xlabel('speed (mph)')
  #     plt.ylabel('torque')
  #     plt.savefig('plots/{}.png'.format(angle_range_str))
  #
  # if ANGLE_DATA_ANALYSIS := True:  # analyzes how angle changes need of torque (RESULT: seems to be relatively linear, can be tuned by k_f)
  #   if PLOT_ANGLE_DIST := False:
  #     sns.distplot([line['angle_steers'] for line in data if abs(line['angle_steers']) < 30], bins=200)
  #     raise Exception
  #
  #   res = 100
  #
  #   _speeds = np.r_[[
  #     [0, 10],
  #     [10, 20],
  #     [20, 30],
  #     [30, 40],
  #     [40, 50],
  #     [50, 60],
  #     [60, 70],
  #   ]] * CV.MPH_TO_MS
  #   color = 'blue'
  #
  #   for idx, speed_range in enumerate(_speeds):
  #     speed_range_str = '{} mph'.format('-'.join([str(round(i * CV.MS_TO_MPH, 1)) for i in speed_range]))
  #     temp_data = [line for line in data if speed_range[0] <= line['v_ego'] <= speed_range[1]]
  #     if not len(temp_data):
  #       continue
  #     print(f'{speed_range_str} samples: {len(temp_data)}')
  #     plt.figure()
  #     angles, torque, speeds = zip(*[[line['angle_steers'], line['torque'], line['v_ego']] for line in temp_data])
  #     plt.scatter(angles, torque, label=speed_range_str, color=color, s=0.05)
  #
  #     _x_ff = np.linspace(0, max(angles), res)
  #     _y_ff = [feedforward(np.mean(speed_range), _i, old_kf) * MAX_TORQUE for _i in _x_ff]
  #     plt.plot(_x_ff, _y_ff, color='orange', label='standard ff model at {} mph'.format(np.round(np.mean(speed_range) * CV.MS_TO_MPH, 1)))
  #
  #     _y_ff = [feedforward(np.mean(speed_range), _i, fit_kf) * MAX_TORQUE for _i in _x_ff]
  #     plt.plot(_x_ff, _y_ff, color='purple', label='new fitted ff function')
  #
  #     plt.legend()
  #     plt.xlabel('angle (deg)')
  #     plt.ylabel('torque')
  #     plt.savefig('plots/{}.png'.format(speed_range_str))

  # if PLOT_3D := False:
  #   X_test = np.linspace(0, max(data_speeds), 20)
  #   Y_test = np.linspace(0, max(data_angles), 20)
  #
  #   Z_test = np.zeros((len(X_test), len(Y_test)))
  #   for i in range(len(X_test)):
  #     for j in range(len(Y_test)):
  #       Z_test[i][j] = feedforward(X_test[i], Y_test[j], fit_kf)
  #
  #   X_test, Y_test = np.meshgrid(X_test, Y_test)
  #
  #   fig = plt.figure()
  #   ax = plt.axes(projection='3d')
  #
  #   surf = ax.plot_surface(X_test * CV.MS_TO_MPH, Y_test, Z_test, cmap=cm.magma,
  #                          linewidth=0, antialiased=False)
  #   fig.colorbar(surf, shrink=0.5, aspect=5)
  #
  #   ax.set_xlabel('speed (mph)')
  #   ax.set_ylabel('angle')
  #   ax.set_zlabel('feedforward')
  #   plt.title('New fitted polynomial feedforward function')


# Compares poly with old ff speed function
# x = np.linspace(0, 30, 100)
# y = x ** 2
# _c1, _c2, _c3 = 0.34365576041121065, 12.845373070976711, 51.63304088261174
# y_poly = _c1 * x ** 2 + _c2 * x + _c3
# plt.plot(x, y_poly, label='poly')
# plt.plot(x, y, label='v_ego**2')
# plt.legend()
# plt.show()
