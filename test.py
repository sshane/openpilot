import numpy as np

x = [0, .15]
y = [0, -148]

i = np.linspace(0, .15, 100)
u = np.interp(i, x, y)

# plt.plot(i, u)

poly = np.polyfit(i, u, 1)