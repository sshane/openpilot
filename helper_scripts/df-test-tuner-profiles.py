import matplotlib.pyplot as plt
import numpy as np
from selfdrive.config import Conversions as CV

x_vel_relaxed = [0.0, 4.166741231209736, 8.333258768790266, 12.500000000000002, 16.666741231209738, 20.833258768790266, 25.85853614889048, 30.52299570508232, 50.00000000000001, 70.0, 75.0, 80.0, 90.00000000000001]
y_dist_relaxed = [1.385, 1.394, 1.406, 1.421, 1.444, 1.474, 1.516, 1.534, 1.546, 1.568, 1.579, 1.593, 1.614]
plt.plot(np.array(x_vel_relaxed), y_dist_relaxed, label='relaxed')


y_dist_relaxed_new = []
for x_vel, y_dist in zip(x_vel_relaxed, y_dist_relaxed):
  x = [22, 28, 45, 60]
  y = [1., 1.01, 1.03, 1.025]
  y_dist *= ((np.interp(x_vel, x, y) - 1) / 2) + 1
  y_dist_relaxed_new.append(y_dist)

plt.plot(np.array(x_vel_relaxed), np.round(y_dist_relaxed_new, 3), label='new relaxed')


# x_vel_traffic = [0.0, 1.892, 3.7432, 5.8632, 8.0727, 10.7301, 14.343, 17.6275, 22.4049, 28.6752, 34.8858, 40.35]
# y_dist_traffic = [1.3781, 1.3791, 1.3457, 1.3134, 1.3145, 1.318, 1.3485, 1.257, 1.144, 0.979, 0.9461, 0.9156]
# plt.plot(np.array(x_vel_traffic) * CV.MS_TO_MPH, y_dist_traffic, label='traffic')


x_vel_roadtrip = [0.0, 4.166741231209736, 8.333258768790266, 12.500000000000002, 16.666741231209738, 20.833258768790266, 25.85853614889048, 30.52299570508232, 50.00000000000001, 70.0, 75.0, 80.0, 90.00000000000001]
y_dist_roadtrip = [1.3978, 1.4132, 1.4318, 1.4536, 1.485, 1.5229, 1.5819, 1.6203, 1.7238, 1.8231, 1.8379, 1.8495, 1.8535]  # TRs
plt.plot(np.array(x_vel_roadtrip), y_dist_roadtrip, label='roadtrip')


y_dist_roadtrip_new = []
for x_vel, y_dist in zip(x_vel_roadtrip, y_dist_roadtrip):
  x = [16, 22, 30, 45, 60, 90]
  y = [1., 1.015, 1.046666, 1.075, 1.045, 1.08]
  y_dist *= ((np.interp(x_vel, x, y) - 1) / 2) + 1
  y_dist_roadtrip_new.append(y_dist)

plt.plot(np.array(x_vel_roadtrip), np.round(y_dist_roadtrip_new, 3), label='new roadtrip')



# plt.plot([min(x), max(x)], [0, 0], 'r--')
# plt.plot([0, 0], [min(y), max(y)], 'r--')

plt.xlabel('mph')
plt.ylabel('sec')

# poly = np.polyfit(x, y, 6)
# x = np.linspace(min(x), max(x), 100)
# y = np.polyval(poly, x)
# plt.plot(x, y, label='poly fit')

# to_round = True
# if to_round:
#   x = np.round(x, 4)
#   y = np.round(y, 5)
#   print('x = {}'.format(x.tolist()))
#   print('y = {}'.format(y.tolist()))

plt.legend()
plt.show()
