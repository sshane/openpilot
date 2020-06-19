import matplotlib.pyplot as plt
import numpy as np
import time


def circle(_x, r):
  # _y = -np.sqrt(_x ** 2 - r ** 2)
  _y = np.sqrt(abs(r ** 2 - _x ** 2))
  return _y

angle_x = [-10, 10]
angle_y = [0, 2]

for angle in np.linspace(0, 700, 200):
  # angle = np.interp(angle, angle_x, angle_y)
  print(angle)

  x = np.linspace(0.0, np.pi, 100)
  y = np.array([(circle(_x, angle)) for _x in x])

  y = y - angle

  plt.clf()
  plt.plot(x, y)
  # plt.xlim([0, 0.5])
  plt.ylim([-1, 1])
  plt.show()
  plt.pause(0.01)
