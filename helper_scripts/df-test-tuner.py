import matplotlib.pyplot as plt
import numpy as np
from selfdrive.config import Conversions as CV


x = [-44.80315, -35.091043, -25.045857, -17.592385, -11.066571, -6.831827, -4.975841, -3.3655, -1.7689, -0.715, 0.0, 1.25, 3.060576, 4.245705, 6.110415, 10.0]
y = [0.623232, 0.49487999999999993, 0.40656, 0.322272, 0.239136, 0.12268799999999999, 0.104832, 0.08073599999999999, 0.048864, 0.0072, 0, -0.0443*1.275, -0.066*1.2, -0.1425*1.1, -0.2218*1.05, -0.315]
plt.plot(x, y, 'bo-', label='new rel vel mod')

plt.plot([min(x), max(x)], [0, 0], 'r--')
plt.plot([0, 0], [min(y), max(y)], 'r--')

# poly = np.polyfit(x, y, 6)
# x = np.linspace(min(x), max(x), 100)
# y = np.polyval(poly, x)
# plt.plot(x, y, label='poly fit')

x = np.array(x) * CV.MPH_TO_MS

to_round = True
if to_round:
  x = np.round(x, 4)
  y = np.round(y, 5)
  print('x = {}'.format(x.tolist()))
  print('y = {}'.format(y.tolist()))

plt.legend()
plt.show()
