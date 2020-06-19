# import matplotlib
# matplotlib.use('Qt5Agg')
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import time

os.chdir(os.getcwd())

data = 'lane_speed2'

# good ones: 134000, 107823 + 900, 98708, 98708+9000, 117734
start = 117734

with open(data, 'r') as f:
  data = f.read().split('\n')[:-1][start:start+30000]

data_parsed = []
for idx, line in enumerate(data):
  if 'nan' in line:
    continue
  line = line.replace('builder', 'reader').replace('<capnp list reader ', '').replace('>', '')
  line = ast.literal_eval(line)
  if len(line['d_poly']) == 0:
    continue
  # if abs(line['v_ego'] * 2.2369 - 57) < 0.2235 and len(line['live_tracks']) > 1:
  if len(line['live_tracks']) > 2 and np.mean([trk['vRel'] for trk in line['live_tracks'] if trk['vRel'] > -2]) > -5 and line['v_ego'] > 5:
    print(line['v_ego'] * 2.2369)
    print(idx)
    print()
  data_parsed.append(line)
data = data_parsed

# dPoly = [line['d_poly'] for line in data]
max_dist = 200
preprocessed = []

for idx, line in enumerate(data):
  preprocessed.append({})
  preprocessed[-1]['title'] = 'v_ego: {} mph, idx: {}'.format(line['v_ego'] * 2.2369, idx)

  dPoly = line['d_poly']

  x = np.linspace(0, max_dist, 100)
  y = np.polyval(dPoly, x)
  preprocessed[-1]['x'] = x
  preprocessed[-1]['y'] = y

  preprocessed[-1]['live_tracks'] = line['live_tracks']

  # for track in line['live_tracks']:
  #   plt.plot(track['dRel'], track['yRel'], 'bo')

  # plt.plot(x, y, label='desired path')
  # plt.legend()
  # plt.xlabel('longitudinal position (m)')
  # plt.ylabel('lateral position (m)')
  # plt.xlim(0, max_dist)
  ylim = [min(min(y), -7.5), max(max(y), 7.5)]
  preprocessed[-1]['ylim'] = ylim
  # plt.ylim(*ylim)
  # plt.pause(0.01)

preprocessed = preprocessed[::12]
plt.clf()
plt.pause(0.01)
input('press any key to continue')
for line in preprocessed:
  t = time.time()
  plt.clf()
  plt.title(line['title'])

  for track in line['live_tracks']:
    if track['vRel'] > -5:
      plt.plot(track['dRel'], track['yRel'], 'bo')

  plt.plot(line['x'], line['y'], label='desired path')
  plt.plot(line['x'], line['y'] + 3.7 / 2, 'r--', label='lane line')
  plt.plot(line['x'], line['y'] - 3.7 / 2, 'r--', label='lane line')
  plt.show()

  plt.legend()
  plt.xlabel('longitudinal position (m)')
  plt.ylabel('lateral position (m)')
  plt.xlim(0, max_dist)
  plt.ylim(*line['ylim'])
  plt.pause(0.001)
  print(time.time() - t)
  # time.sleep(0.01)
  # print(dPoly)
  # input()
