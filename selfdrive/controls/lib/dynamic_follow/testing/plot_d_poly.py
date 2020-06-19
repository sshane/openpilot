import os
import ast
import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.getcwd())

data = 'lane_speed'

with open(data, 'r') as f:
  data = f.read().split('\n')[:-1][75000:80000]

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

for line in data:
  plt.clf()
  plt.title('v_ego: {} mph'.format(line['v_ego'] * 2.2369))
  dPoly = line['d_poly']
  max_dist = 182
  x = np.linspace(0, max_dist, 100)
  y = np.polyval(dPoly, x)

  tracks = []
  for track in line['live_tracks']:
    print(track['dRel'])
    plt.plot(track['dRel'], track['yRel'], 'bo')

  plt.plot(x, y, label='desired path')
  plt.legend()
  plt.xlabel('longitudinal position (m)')
  plt.ylabel('lateral position (m)')
  plt.xlim(0, max_dist)
  plt.ylim(-10, 10)
  plt.pause(0.01)
  # print(dPoly)
  # input()
