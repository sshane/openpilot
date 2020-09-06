import numpy as np

x = [0, .075]
y = [0, -148]

i = np.linspace(0, .075, 100)
u = np.interp(i, x, y)

# plt.plot(i, u)

poly = np.polyfit(i, u, 1)