import matplotlib.pyplot as plt
import numpy as np
from selfdrive.config import Conversions as CV
# x = np.array([-4.4795, -2.8122, -1.5727, -1.1129, -0.6611, -0.2692, 0.0, 0.1466, 0.5144, 0.6903, 0.9302]) * CV.MS_TO_MPH
# y = [0.265, 0.1877, 0.0984, 0.0574, 0.034, 0.024, 0.0, -0.009, -0.042, -0.053, -0.059]
#
# plt.plot(x, y, label='orig. lead accel mod')





x = [-6, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 6]
y = [0.35, 0.3, 0.125, 0.075, 0.06, 0, -0.06, -0.075, -0.125, -0.3, -0.35]
plt.plot(x, y, label='new rel vel mod')

plt.plot([min(x), max(x)], [0, 0], 'r--')
plt.plot([0, 0], [min(y), max(y)], 'r--')

# poly = np.polyfit(x, y, 8, rcond=0.2)
# x = np.linspace(min(x), max(x), 100)
# y = np.polyval(poly, x)
# plt.plot(x, y, label='poly fit')
