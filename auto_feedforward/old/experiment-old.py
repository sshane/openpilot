import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import scipy
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from mpl_toolkits.mplot3d import Axes3D
from auto_feedforward.load_data import get_data
import seaborn as sns

MS_TO_MPH = 2.23694
MPH_TO_MS = 1 / MS_TO_MPH

k_f = 0.00007818594
steerRatio = 17.8
wheelbase = 2.70

MAX_TORQUE = 1500


def get_feedforward(v_ego, angle_steers_des):
  # steer_feedforward = v_ego ** 2 * angle_steers_des * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  steer_feedforward = angle_steers_des * v_ego ** 2  # proportional to realigning tire momentum (~ lateral accel)
  return steer_feedforward


# RANDOM = False
# if not RANDOM:
#   speeds = [13.4, 13.05, 10.7, 15.64, 19.48, 11.89, 11.92, 9.71, 10.15, 10.4, 10.64, 10.88, 10.99, 11.06, 11.31, 11.23, 31.29, 30.55, ]
#   angles = [6, 3, 10.5, -9, -10.5, 4.5, 7.5, -19.5, -27, -19.5, 4.5, 19.5, 24, 25.5, 15, -43.5, -0.01, -3, ]
#   torque = [315, 270, 539, -495, -594, 796, 707, -652, -488, -268, 448, 707, 917, 537, 312, -990, -25, -343]
# else:
#   samples = 100
#   speeds = np.linspace(.5, 80, samples)
#   angles = np.linspace(.5, 45, samples)
#   torque = [np.random.uniform(a / max(angles) * 1200 + (a / max(angles) * 1200) * 0.2, a / max(angles) * 1200 - (a / max(angles) * 1200) * 0.2) for a in angles]


# speeds, angles, torque = get_data(os.getcwd() + '/ff_data')
data = get_data(os.getcwd() + '/ff_data')


print(len(data))

data = [line for line in data if 3 <= abs(line['angle_steers']) <= 35]
data = [line for line in data if abs(line['v_ego']) > 5 * MPH_TO_MS]
for line in data:
  line['torque'] = line['output_steer'] * MAX_TORQUE

print(len(data))
speeds, angles, torque = zip(*[[line['v_ego'], line['angle_steers'], line['torque']] for line in data])

angles = np.abs(np.array(angles))
torque = [abs(torq) / MAX_TORQUE for torq in torque]
feedfs = [torq / get_feedforward(spd, ang) for ang, spd, torq in zip(angles, speeds, torque)]
del torque

speeds, angles, feedfs = zip(*[[spd, ang, ff] for spd, ang, ff in zip(speeds, angles, feedfs) if ff < .0008])
print(len(speeds))

# feedfs = torque

assert all([i > 0 for i in feedfs]), 'A feedforward sample is negative?'


# ff = [get_feedforward(spd, 25) * k_f for spd in np.linspace(0, 20, 50)]
# plt.plot(np.linspace(0, 20, 50) * MS_TO_MPH, ff, label='op ff')
#
# avgs = {i: [] for i in np.arange(1, 20+1, 1)}
# for ff, spd in zip(feedfs, speeds):
#   for i in avgs:
#     if spd <= i:
#       avgs[i].append(ff)
#       break
#
# avgs = {i: np.mean(avgs[i]) if len(avgs[i]) else 0 for i in avgs}
# plt.plot(np.arange(1, 20+1, 1), [avgs[i] for i in range(1, len(avgs)+1)], label='gathered')
#
# plt.legend()
# plt.figure()
# sns.distplot(feedfs)
# raise Exception

# plt.plot(speeds, torque, label='torque')
# plt.plot(speeds, feedfs, label='feedforward')

# Prepare data for lineral regression
# X = np.array(speeds).reshape(-1, 1)  # for just speeds
X = np.array((speeds, angles)).T
feedfs = np.array(feedfs)

fig = plt.figure()
ax = plt.axes(projection='3d')

feedf_scale = [min(feedfs), max(feedfs)]
scale_to = [-1, 1]
feedfs = np.interp(feedfs, feedf_scale, scale_to)

use_polyfit = True
if use_polyfit:
  NORMALIZED = True
  degree = 3
  polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression(normalize=False))
  polyreg.fit(X, feedfs)
  print(f'\nregression score: {polyreg.score(X, feedfs)}')
else:
  NORMALIZED = True
  model = Sequential()
  model.add(Dense(32, activation='tanh', input_shape=X.shape[1:]))
  model.add(Dense(1))
  model.compile('adam', loss='mse', metrics=['mae'])
  model.fit(X, feedfs, epochs=10, batch_size=24)


X_test = np.linspace(0, max(speeds), 20)
Y_test = np.linspace(0, max(angles), 20)


Z_test = np.zeros((len(X_test), len(Y_test)))
for i in range(len(X_test)):
  for j in range(len(Y_test)):
    if use_polyfit:
      Z_test[i][j] = polyreg.predict([[X_test[i], Y_test[j]]])[0]
    else:
      Z_test[i][j] = model.predict_on_batch(np.array([[X_test[i], Y_test[j]]]))[0][0]
    if NORMALIZED:
      Z_test[i][j] = np.interp(Z_test[i][j], scale_to, feedf_scale)

X_test, Y_test = np.meshgrid(X_test, Y_test)

# ax.contour3D(X, Y, Z)
PLOT_DATA = True
if PLOT_DATA:
  if not NORMALIZED:
    ax.scatter3D(speeds, angles, feedfs, c='red', s=1)
  else:
    ax.scatter3D(speeds, angles, np.interp(feedfs, scale_to, feedf_scale), c='red', s=1)

PLOT_PRED = True
if PLOT_PRED:
  surf = ax.plot_surface(X_test, Y_test, Z_test, cmap=cm.magma,
                         linewidth=0, antialiased=False)
  fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('speed')
ax.set_ylabel('angle')
ax.set_zlabel('feedforward')
if use_polyfit:
  plt.title(f'Feedforward model ({degree} deg. poly)')
else:
  plt.title('Feedforward keras model')


plt.show()
