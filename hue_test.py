import numpy as np
import matplotlib.pyplot as plt


# lane_pos = std::abs(path->poly[3]);  // get redder when line is closer to car
def get_hue(x):
  lane_pos = x
  dists = [1.4, 1.0]
  hues = [133, 0]
  hue = (lane_pos - dists[0]) * (hues[1] - hues[0]) / (dists[1] - dists[0]) + hues[0]
  # hue = min(133, max(0, hue)) / 360
  return hue

x = np.linspace(0.75, 1.6, 100)
y = [get_hue(i) for i in x]

poly = np.polyfit(x, y, 1)
y_poly = np.polyval(poly, x)
print(poly)
print(np.allclose(y, y_poly))

y = [min(133, max(0, i)) / 360 for i in y]
y_poly = [min(133, max(0, i)) / 360 for i in y_poly]

plt.plot(x, y, label='original function')
plt.plot(x, y_poly, label='from poly')
plt.show()
