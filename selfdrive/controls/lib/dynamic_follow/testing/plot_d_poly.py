# import matplotlib
# matplotlib.use('Qt5Agg')
import os
import ast
import matplotlib.pyplot as plt
import numpy as np
import time

os.chdir(os.getcwd())

data = 'lane_speed'

with open(data, 'r') as f:
  data = f.read().split('\n')[:-1][10400:12000]

data_parsed = []
for idx, line in enumerate(data):
  if 'nan' in line:
    continue
  line = line.replace('builder', 'reader').replace('<capnp list reader ', '').replace('>', '')
  line = ast.literal_eval(line)
  if len(line['d_poly']) == 0:
    continue
  data_parsed.append(line)
data = data_parsed

# dPoly = [line['d_poly'] for line in data]
max_dist = 91
preprocessed = []

for idx, line in enumerate(data):
  preprocessed.append({})
  preprocessed[-1]['title'] = 'v_ego: {} mph, idx: {}'.format(line['v_ego'] * 2.2369, idx)

  dPoly = line['d_poly']

  x = np.linspace(0, max_dist, 100)
  y = np.polyval(dPoly, x)
  preprocessed[-1]['x'] = x
  preprocessed[-1]['y'] = y

  # for track in line['live_tracks']:
  #   plt.plot(track['dRel'], track['yRel'], 'bo')

  # plt.plot(x, y, label='desired path')
  # plt.legend()
  # plt.xlabel('longitudinal position (m)')
  # plt.ylabel('lateral position (m)')
  # plt.xlim(0, max_dist)
  ylim = [min(min(y), -20), max(max(y), 20)]
  preprocessed[-1]['ylim'] = ylim
  # plt.ylim(*ylim)
  # plt.pause(0.01)

preprocessed = preprocessed[::3]

for line in preprocessed:
  t = time.time()
  plt.clf()
  plt.title(line['title'])

  # for track in line['live_tracks']:
  #   plt.plot(track['dRel'], track['yRel'], 'bo')

  plt.plot(line['x'], line['y'], label='desired path')
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
