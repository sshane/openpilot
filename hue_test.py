import numpy as np
import matplotlib.pyplot as plt


# lane_pos = std::abs(path->poly[3]);  // get redder when line is closer to car
def get_hue(x):
  lane_pos = 1.2
  dists = [1.4, 1.0]
  hues = [133, 0]
  hue = (lane_pos - dists[0]) * (hues[1] - hues[0]) / (dists[1] - dists[0]) + hues[0]
  hue = min(133, max(0, hue)) / 360
  return hue

x = np.linspace(1, 1.4, 100)
y = [get_hue(i) for i in x]

plt.plot(x, y)
