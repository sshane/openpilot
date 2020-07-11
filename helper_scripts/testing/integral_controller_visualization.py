import matplotlib.pyplot as plt
import numpy as np
import time

class IController:
  def __init__(self):
    self.i = 0
    self.k_i = 1.5
    self.i_rate = 1 / 5

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


setpoint = 0.7
measurement = 0.5
i = IController()
# i = PController()
ms = []
plt.figure(0)
plt.plot(0, 0)
plt.pause(0.05)
input()
while 1:  # this simulates controlling lane position with camera offset
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
  if 1/5 - t_elapsed > 0:
    time.sleep(1/5 - t_elapsed)
  else:
    print('lagging by {}'.format(abs(1/5 - t_elapsed)))
