import matplotlib.pyplot as plt
import math
import numpy as np
import time

def circle(x, angle):
  k = angle
  r = 1. / k * 100 if k != 0 else np.inf
  return r - np.sqrt(r * r - x * x)

alin = np.linspace(-80, 80, 100) * 0.01

for a in alin:
  xlin = np.linspace(0, 100, 1000)
  print(a)
  # y = [-(i * a) ** 2 / (1000 * (a * 2)) for i in xlin]
  y = [circle(i, abs(a)) for i in xlin]
  if a < 0:
    y = -np.array(y)

  plt.clf()
  plt.ylim(-100, 100)
  plt.plot(xlin, y)
  plt.pause(.01)
  plt.show()
  # time.sleep(.01)
