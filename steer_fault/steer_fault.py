#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm   # type: ignore
from scipy.optimize import curve_fit
from selfdrive.car.toyota.values import STEER_THRESHOLD

from common.realtime import DT_CTRL
from selfdrive.config import Conversions as CV
from selfdrive.car.toyota.values import SteerLimitParams as TOYOTA_PARAMS
from selfdrive.car.subaru.carcontroller import CarControllerParams as SUBARU_PARAMS
from tools.lib.route import Route
import seaborn as sns
from tools.lib.logreader import MultiLogIterator
import pickle
import binascii
from tools.lib.auth_config import set_token

MIN_SAMPLES = 60 * 100

# with open('/data/jwt', 'r') as f:
#   set_token(f.read().strip())
if len(sys.argv) > 1:
  set_token(sys.argv[1])


def to_signed(n, bits):
  if n >= (1 << max((bits - 1), 0)):
    n = n - (1 << max(bits, 0))
  return n


def hex_to_binary(hexdata):
  return (bin(int(binascii.hexlify(hexdata), 16))[2:]).zfill(len(hexdata) * 8)  # adds leading/trailing zeros so data matches up with 8x8 array on cabana


def steer_fault(use_dir, plot=False):
  CAR_MAKE = 'toyota'
  MAX_TORQUE = TOYOTA_PARAMS.STEER_MAX if CAR_MAKE == 'toyota' else SUBARU_PARAMS().STEER_MAX

  steer_delay = None

  files = sorted(os.listdir(use_dir))
  files = [f for f in files if '.ini' not in f]
  # print(files)

  routes = [[files[0]]]  # this mess ensures we process each route's segments independantly since sorting will join samples from random routes
  for rt in files[1:]:  # todo: clean up
    rt_name = '--'.join(rt.split('--')[:2])
    if rt_name != '--'.join(routes[-1][-1].split('--')[:2]):
      routes.append([rt])
    else:
      routes[-1].append(rt)

  print(list(map(len, routes)))
  lrs = []
  for _routes in routes:
    lrs.append(MultiLogIterator([os.path.join(use_dir, i) for i in _routes], wraparound=False))

  data = [[]]
  for lr in lrs:
    CS, PP = None, None
    # engaged, steering_pressed = False, False
    # torque_cmd, angle_steers = None, None
    can = {'torque_cmd': None, 'steering_pressed': False, 'angle_steers': None, 'lka_state': None}
    last_time = 0

    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)

    for msg in tqdm(all_msgs):
      if msg.which() == 'carParams':
        if steer_delay is None:
          steer_delay = round(msg.carParams.steerActuatorDelay / DT_CTRL)

      elif msg.which() == 'carState':
        CS = msg.carState

      elif msg.which() == 'pathPlan':
        PP = msg.pathPlan

      if msg.which() != 'can':
        continue

      for m in msg.can:
        if m.address == 0x2e4 and m.src == 128:  # STEERING_LKA
          # can['engaged'] = bool(m.dat[0] & 1)
          can['torque_cmd'] = to_signed((m.dat[1] << 8) | m.dat[2], 16)
        elif m.address == 0x260 and m.src == 0:  # STEER_TORQUE_SENSOR
          can['steering_pressed'] = abs(to_signed((m.dat[1] << 8) | m.dat[2], 16)) > STEER_THRESHOLD
          can['driver_torque'] = to_signed((m.dat[1] << 8) | m.dat[2], 16)
        elif m.address == 0x25 and m.src == 0:  # STEER_ANGLE_SENSOR
          steer_angle = to_signed(int(bin(m.dat[0])[2:].zfill(8)[4:] + bin(m.dat[1])[2:].zfill(8), 2), 12) * 1.5
          steer_fraction = to_signed(int(bin(m.dat[4])[2:].zfill(8)[:4], 2), 4) * 0.1
          can['angle_steers'] = steer_angle + steer_fraction
        elif m.address == 0x262 and m.src == 0:
          can['lka_state'] = int(bin(m.dat[3])[2:].zfill(8)[:7], 2)

      if (None not in [CS, PP, *can.values()] and CS.cruiseState.enabled and  # creates uninterupted sections of engaged data
              abs(msg.logMonoTime - last_time) * 1e-9 < 1 / 20):  # also split if there's a break in time

        data[-1].append({'can': can.copy(), 'v_ego': CS.vEgo, 'angle_steers_des': PP.angleSteers,
                         'angle_offset': PP.angleOffset, 'time': msg.logMonoTime * 1e-9, 'angle_rate': CS.steeringRate})

      elif len(data[-1]):  # if last list has items in it, append new empty section
        data.append([])

      last_time = msg.logMonoTime

  del all_msgs

  print('Max seq. len: {}'.format(max([len(line) for line in data])))

  if WRITE_DATA := True:  # todo: temp, for debugging
    with open('steer_fault/data', 'wb') as f:
      pickle.dump(data, f)

  print(f'Samples (before filtering): {len(data)}')



  print(f'Samples (after filtering):  {len(data)}\n')

  # assert len(data) > MIN_SAMPLES, 'too few valid samples found in route'




if __name__ == "__main__":
  # r = Route("14431dbeedbf3558%7C2020-11-10--22-24-34")
  # lr = MultiLogIterator(r.log_paths(), wraparound=False)
  use_dir = '/data/openpilot/steer_fault/rlogs/shane/good'
  # lr = MultiLogIterator([os.path.join(use_dir, i) for i in os.listdir(use_dir)], wraparound=False)
  steer_fault(use_dir, plot="--plot" in sys.argv)
