import matplotlib.pyplot as plt
import numpy as np
from selfdrive.config import Conversions as CV

x = [-5.999910522548318, -4.000089477451683, -2.0000447387258413, -0.9999105225483179, -0.49995526127415896, 0.0, 0.49995526127415896, 0.9999105225483179, 2.0000447387258413, 4.000089477451683, 5.999910522548318]
y = [0.35, 0.3, 0.125, 0.09375, 0.075, 0, -0.09, -0.09375, -0.125, -0.3, -0.35]
plt.plot(x, y, 'bo-', label='current a_rel_accel mod')

plt.plot([min(x), max(x)], [0, 0], 'r--')
plt.plot([0, 0], [min(y), max(y)], 'r--')


y_new = []
for y_ in y:
  if y_ > 0:
    y_ = (y_ * 0.95) - 0.008

  y_new.append(y_)
plt.plot(x, y_new, 'go-', label='new a_rel_accel mod')
y = y_new

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
