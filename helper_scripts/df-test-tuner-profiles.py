import matplotlib.pyplot as plt
import numpy as np
from selfdrive.config import Conversions as CV

x_vel = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]  # velocities
y_dist = [1.385, 1.394, 1.406, 1.421, 1.444, 1.474, 1.516, 1.534, 1.546, 1.568, 1.579, 1.593, 1.614]
# plt.plot(np.array(x_vel) * CV.MS_TO_MPH, y_dist, label='relaxed profile')

traffic_x_vel = [0.0, 1.892, 3.7432, 5.8632, 8.0727, 10.7301, 14.343, 17.6275, 22.4049, 28.6752, 34.8858, 40.35]
traffic_y_dist = [1.3781, 1.3791, 1.3802, 1.3825, 1.3984, 1.4249, 1.4194, 1.3162, 1.1916, 1.0145, 0.9855, 0.9562]
# plt.plot(np.array(traffic_x_vel) * CV.MS_TO_MPH, traffic_y_dist, label='traffic profile')

ft = np.array(traffic_x_vel) * np.array(traffic_y_dist) * 3.28084
print(ft.tolist())
plt.plot(np.array(traffic_x_vel) * CV.MS_TO_MPH, ft, 'o-', label='traffic profile')
# plt.plot(np.array(x_vel) * CV.MS_TO_MPH, np.array(x_vel) * np.array(y_dist), 'o-', label='relaxed profile')


traffic_x_vel = [0.0, 1.892, 3.7432, 5.8632, 8.0727, 10.7301, 14.343, 17.6275, 22.4049, 28.6752, 34.8858, 40.35]
traffic_y_dist = [1.3781, 1.3791, 1.3802 * 0.95, 1.3825 * 0.9, 1.3984 * 0.88, 1.4249 * 0.85, 1.4194 * 0.9, 1.3162 * 0.91, 1.1916 * 0.92, 1.0145 * 0.93, 0.9855 * 0.92, 0.9562 * 0.915]

ft = np.array(traffic_x_vel) * np.array(traffic_y_dist) * 3.28084
print(ft.tolist())
plt.plot(np.array(traffic_x_vel) * CV.MS_TO_MPH, ft, 'o-', label='new traffic profile')

# plt.plot([min(x), max(x)], [0, 0], 'r--')
# plt.plot([0, 0], [min(y), max(y)], 'r--')

plt.xlabel('mph')
plt.ylabel('feet')

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
