import matplotlib.pyplot as plt
import numpy as np
import math

def points_from_curv(_x):
  a = 100
  b = 2
  print(_x)
  _y = (_x ** 2) / (a ** 2)
  _y = _y - 1
  _y = _y * (b ** 2)
  print(_y)
  sign = 1 if _y > 0 else -1
  _y = math.sqrt(abs(_y)) * sign

  # _y = math.sqrt(( (_x**2 / a**2) - 1 ) * (b**2))
  # x = curvature * math.cos(curvature)
  # y = curvature * math.cos(_x)
  return _y

def circle_y(theta, r):
  return r * np.sin(theta)

def circle_x(theta, r):
  return r * np.cos(theta)


angle = 10

x = np.linspace(0.0, 10, 100)
y = [(circle_y(_x, angle)+1) * angle for _x in x]
x = [circle_x(_x, angle) for _x in x]

x = x[1:]
y = y[1:]

x, y = zip(*[(_x, _y) for _x, _y in zip(x, y) if _y <= 1 and _x >= 0])
# x, y = zip(*[(_x, _y) for _x, _y in zip(x, y)])

plt.plot(x, y)
# plt.xlim([-.5, 2])
# plt.ylim([-.5, 2])
plt.show()
