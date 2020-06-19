import matplotlib.pyplot as plt
import math
import numpy as np
import time



alin = np.linspace(-80, 80, 100)

for a in alin:
  xlin = np.linspace(0, 100, 1000)
  print(a)
  # x = [i for i in range(100)]
  # x = [i*math.cos(a) for i in xlin]
  y = [-(i * a) ** 2 / (1000 * (a * 2)) for i in xlin]

  plt.clf()
  plt.ylim(-500, 500)
  plt.plot(xlin, y)
  plt.pause(.01)
  plt.show()
  # time.sleep(.01)
