import matplotlib.pyplot as plt
import numpy as np
from selfdrive.config import Conversions as CV

x = np.array([-10.020356120257695, -6.29071224051539, -3.5180297065139587, -2.4894863994273444, -1.4788385826771655, -0.6021832498210451, 0.0, 0.32793486041517544, 1.1506800286327845, 1.544157122405154, 2.0807981388690053])  # lead acceleration values
y = [0.24, 0.16, 0.092, 0.0515, 0.0305, 0.022, 0.0, -0.009*1.7, -0.042, -0.053, -0.059]  # modification values
plt.plot(x, y, 'bo-', label='current a_lead mod')

plt.plot([min(x), max(x)], [0, 0], 'r--')
plt.plot([0, 0], [min(y), max(y)], 'r--')

# poly = np.polyfit(x, y, 6)
# x = np.linspace(min(x), max(x), 100)
# y = np.polyval(poly, x)
# plt.plot(x, y, label='poly fit')
# x = np.array(x) * CV.MPH_TO_MS

to_round = True
if to_round:
  x = np.round(x, 6)
  y = np.round(y, 5)
  print('x = {}'.format(x.tolist()))
  print('y = {}'.format(y.tolist()))

plt.legend()
plt.show()
