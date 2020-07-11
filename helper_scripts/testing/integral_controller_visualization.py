import matplotlib.pyplot as plt
import numpy as np
import time
import random

class IController:
  def __init__(self):
    self.i = 0
    self.k_i = 4.5
    self.i_rate = 1 / 20

  def update(self, setpoint, measurement):
    error = setpoint - measurement
    self.i += error * self.k_i * self.i_rate
    return self.i

  def reset(self):
    self.i = 0


class PController:
  def __init__(self):
    self.k_p = 0.5

  def update(self, setpoint, measurement):
    error = setpoint - measurement
    return error * self.k_p


setpoint = 0.75
measurement = 0.5
i = IController()
# i = PController()
ms = []
plt.figure(0)
plt.plot(0, 0)
plt.pause(0.05)
input()
t_start = time.time()
while 1:  # this simulates controlling lane position with camera offset
  if time.time() - t_start > 5:
    if random.randint(0, 1) == 1:
      setpoint += 0.1
    else:
      setpoint -= 0.1
    t_start = time.time()
  t = time.time()
  measurement = i.update(setpoint, measurement)
  ms.append(measurement)
  plt.clf()
  plt.plot(np.linspace(0, len(ms), len(ms)), [setpoint for _ in range(len(ms))], label='setpoint')
  plt.plot(ms, label='measurement')
  plt.legend()
  plt.show()
  plt.pause(0.01)

  t_elapsed = time.time() - t
  if 1/20 - t_elapsed > 0:
    time.sleep(1/20 - t_elapsed)
  else:
    print('lagging by {}'.format(abs(1/20 - t_elapsed)))
