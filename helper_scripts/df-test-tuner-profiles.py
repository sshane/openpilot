import matplotlib.pyplot as plt
import numpy as np
from selfdrive.config import Conversions as CV

x_vel = [0.0, 1.892, 3.7432, 5.8632, 8.0727, 10.7301, 14.343, 17.6275, 22.4049, 28.6752, 34.8858, 40.35]
y_dist_new = [1.3781, 1.3791, 1.3112, 1.2442, 1.2306, 1.2112, 1.2775, 1.1977, 1.0963, 0.9435, 0.9067, 0.8749]  # avg. 7.3 ft closer from 18 to 90 mph
plt.plot(np.array(x_vel) * CV.MS_TO_MPH, y_dist_new, label='new traffic')

y_dist_old = [1.3781, 1.3791, 1.3802, 1.3825, 1.3984, 1.4249, 1.4194, 1.3162, 1.1916, 1.0145, 0.9855, 0.9562]  # original
plt.plot(np.array(x_vel) * CV.MS_TO_MPH, y_dist_old, label='old traffic')

y_dist = np.mean(np.array([y_dist_old, y_dist_new]).T, axis=1)
plt.plot(np.array(x_vel) * CV.MS_TO_MPH, y_dist, label='avg. traffic')
print(y_dist.tolist())
# ft = np.array(traffic_x_vel) * np.array(traffic_y_dist) * 3.28084
# print(ft.tolist())
# plt.plot(np.array(traffic_x_vel) * CV.MS_TO_MPH, ft, 'o-', label='traffic profile')
# plt.plot(np.array(x_vel) * CV.MS_TO_MPH, np.array(x_vel) * np.array(y_dist), 'o-', label='relaxed profile')


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
