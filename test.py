import numpy as np

x = [0, 3.75]
y = [148, 0]

i = np.linspace(0, .075, 100)
u = np.interp(i, x, y)

# plt.plot(i, u)

poly = np.polyfit(i, u, 1)
print(poly.tolist())
