import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential, load_model
from auto_feedforward.load_data import get_data
import seaborn as sns

MS_TO_MPH = 2.23694
MPH_TO_MS = 1 / MS_TO_MPH

k_f = 0.0000795769068
steerRatio = 17.8
wheelbase = 2.70

MAX_TORQUE = 1500


def get_feedforward(v_ego, angle_steers, angle_offset=0):
  steer_feedforward = (angle_steers - angle_offset)
  steer_feedforward *= v_ego ** 2  # + np.interp(v_ego, [0, 70 * MPH_TO_MS], [75, 0])
  return steer_feedforward


def _custom_feedforward(_X, _k_f, _c1, _c2, _c3):  # power is to be fit, X is v_ego, then angle_steers
  v_ego, angle_steers = _X.copy()
  steer_feedforward = angle_steers * (_c3 * v_ego ** 2 + _c2 * v_ego + _c1)
  return steer_feedforward * _k_f


def custom_feedforward(v_ego, angle_steers, *args):  # helper function to easily use fitting ff function
  _X = np.array((v_ego, angle_steers)).T
  return _custom_feedforward(_X, *args)


# speeds, angles, torque = get_data(os.getcwd() + '/ff_data')
data = get_data(os.getcwd() + '/ff_data')
print(f'Samples: {len(data)}')

print('Max angle: {}'.format(max([abs(i['angle_steers']) for i in data])))
print('Top speed: {} mph'.format(max([i['v_ego'] for i in data]) * MS_TO_MPH))

# Data filtering
data = [line for line in data if .0001 <= abs(line['angle_steers']) <= 60]
# data = [line for line in data if abs(line['torque']) >= 25]
data = [line for line in data if abs(line['v_ego']) > 1 * MPH_TO_MS]
# data = [line for line in data if np.sign(line['angle_steers']) == np.sign(line['torque'])]
data = [line for line in data if abs(line['angle_steers'] - line['angle_steers_des']) < .5]

# Data preprocessing
for line in data:
  line['angle_steers'] = abs(line['angle_steers'])
  line['angle_steers_des'] = abs(line['angle_steers_des'])
  line['torque'] = abs(line['torque'])
  line['feedforward'] = (line['torque'] / MAX_TORQUE) / get_feedforward(line['v_ego'], line['angle_steers'])
  # line['feedforward'] = (line['torque'] / MAX_TORQUE) / get_feedforward(line['v_ego'], line['angle_steers'], line['angle_offset'])
  # line['feedforward'] = (line['torque'] / MAX_TORQUE) / get_feedforward(line['v_ego'], line['angle_steers'], -line['angle_offset'])

# data = [line for line in data if line['feedforward'] < .001]  # todo: uncomment
print(f'Samples: {len(data)}')
speeds = np.array([line['v_ego'] for line in data])
angles = np.array([line['angle_steers'] for line in data])
torque = np.array([line['torque'] for line in data])

params, covs = curve_fit(_custom_feedforward, np.array([speeds, angles]), np.array(torque) / MAX_TORQUE, maxfev=1000)
print('FOUND PARAMS: {}'.format(params))

std_func = []
fitted_func = []
for line in data:
  std_func.append(abs(get_feedforward(line['v_ego'], line['angle_steers']) * k_f * MAX_TORQUE - line['torque']))
  fitted_func.append(abs(custom_feedforward(line['v_ego'], line['angle_steers'], *params) * MAX_TORQUE - line['torque']))

print('Torque MAE: {} (standard) - {} (fitted)'.format(np.mean(std_func), np.mean(fitted_func)))


if SPEED_DATA_ANALYSIS := True:  # analyzes how torque needed changes based on speed
  if PLOT_ANGLE_DIST := False:
    sns.distplot([line['angle_steers'] for line in data if abs(line['angle_steers']) < 30], bins=200)
    raise Exception

  res = 100

  _angles = [
    [[5, 10], 'blue'],
    [[10, 20], 'blue'],
    [[10, 15], 'blue'],
    [[15, 20], 'blue'],
    [[20, 25], 'blue'],
    # [[20, 30], 'blue'],

  ]

  for idx, (angle_range, color) in enumerate(_angles):
    temp_data = [line for line in data if angle_range[0] <= abs(line['angle_steers'] - line['angle_offset']) <= angle_range[1]]
    print(f'{angle_range} samples: {len(temp_data)}')
    plt.figure(idx)
    speeds, torque = zip(*[[line['v_ego'], line['torque']] for line in temp_data])
    plt.scatter(np.array(speeds) * MS_TO_MPH, torque, label='{} deg'.format('-'.join(map(str, angle_range))), color=color, s=0.05)

    _x_ff = np.linspace(0, max(speeds), res)
    _y_ff = [get_feedforward(_i, np.mean(angle_range)) * k_f * MAX_TORQUE for _i in _x_ff]
    plt.plot(_x_ff * MS_TO_MPH, _y_ff, color='orange', label='standard ff model at {} deg'.format(np.mean(angle_range)))

    _y_ff = [custom_feedforward(_i, np.mean(angle_range), *params) * MAX_TORQUE for _i in _x_ff]
    plt.plot(_x_ff * MS_TO_MPH, _y_ff, color='purple', label='new fitted ff function')

    plt.legend()
    plt.xlabel('speed (mph)')
    plt.ylabel('torque')

  # plt.scatter(speeds, feedfs, s=0.1)

  # sns.distplot(ffs)


if ANGLE_DATA_ANALYSIS := False:  # analyzes how angle changes need of torque (RESULT: seems to be relatively linear, can be tuned by k_f)
  if PLOT_ANGLE_DIST := False:
    sns.distplot([line['angle_steers'] for line in data if abs(line['angle_steers']) < 30], bins=200)
    raise Exception

  res = 100

  _speeds = np.r_[[
    [0, 10],
    [10, 20],
    [20, 30],
    [30, 40],
    [40, 50],
    [50, 60],
    # [60, 70],
  ]] * MPH_TO_MS
  color = 'blue'

  for idx, speed_range in enumerate(_speeds):
    temp_data = [line for line in data if speed_range[0] <= line['v_ego'] <= speed_range[1]]
    # print(f'{speed_range} samples: {len(temp_data)}')
    plt.figure(idx)
    angles, torque, speeds = zip(*[[line['angle_steers'], line['torque'], line['v_ego']] for line in temp_data])
    plt.scatter(angles, torque, label='{} mph'.format('-'.join([str(round(i * MS_TO_MPH, 1)) for i in speed_range])), color=color, s=0.05)

    _x_ff = np.linspace(0, max(angles), res)
    _y_ff = [get_feedforward(np.mean(speed_range), _i) * k_f * MAX_TORQUE for _i in _x_ff]
    plt.plot(_x_ff, _y_ff, color='orange', label='standard ff model at {} mph'.format(np.round(np.mean(speed_range) * MS_TO_MPH, 1)))

    _y_ff = [custom_feedforward(np.mean(speed_range), _i, *params) * MAX_TORQUE for _i in _x_ff]
    plt.plot(_x_ff, _y_ff, color='purple', label='new fitted ff function')

    plt.legend()
    plt.xlabel('angle (deg)')
    plt.ylabel('torque')

  # plt.scatter(speeds, feedfs, s=0.1)

  # sns.distplot(ffs)

if _2D_MODEL :=False:
  # feedf_scale = [min(feedfs), max(feedfs)]  # todo: the following is a 2d model (speed in, ff out)
  # scale_to = [0, 1]
  # feedfs = np.interp(feedfs, feedf_scale, scale_to)

  poly = np.polyfit(speeds, feedfs, 3)

  X = np.linspace(0, max(speeds), 50)

  # plt.plot(X, reg.coef_ * X + reg.intercept_, label='fitted')
  plt.scatter(speeds, feedfs, s=.1)
  plt.plot(X, np.polyval(poly, X), color='orange', label='fitted')
  plt.legend()
  raise Exception


feedfs = [line['feedforward'] for line in data]

USE_TORQUE_AS_FF = False
if USE_TORQUE_AS_FF:
  feedfs = torque
del torque

# Tests
assert all([i > 0 for i in feedfs]), 'A feedforward sample is zero or negative'
assert all([i >= 0 for i in angles]), 'An angle sample is negative'


print(f'Samples: {len(speeds)}')

PLOT_FF_DIST = False
if PLOT_FF_DIST:
  plt.figure()
  sns.distplot(feedfs)






X = np.array((speeds, angles)).T
feedfs = np.array(feedfs)

fig = plt.figure()
ax = plt.axes(projection='3d')

NORMALIZED = False
if NORMALIZED:
  feedf_scale = [min(feedfs), max(feedfs)]
  scale_to = [0, 1]
  feedfs = np.interp(feedfs, feedf_scale, scale_to)

use_polyfit = False
if use_polyfit:
  degree = 3
  polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression(normalize=False))
  polyreg.fit(X, feedfs)
  print(f'\nregression score: {polyreg.score(X, feedfs)}')
else:
  LOAD_MODEL = False
  if LOAD_MODEL:
    model = load_model('ff_model.h5')
  else:
    model = Sequential()
    model.add(Dense(12, activation='tanh', input_shape=X.shape[1:]))
    # model.add(Dropout(1/12))
    model.add(Dense(1))
    model.compile('adam', loss='mse', metrics=['mae'])
    model.fit(X, feedfs, epochs=7, batch_size=32)


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
    ax.scatter3D(speeds, angles, feedfs, c='red', s=.1)
  else:
    ax.scatter3D(speeds, angles, np.interp(feedfs, scale_to, feedf_scale), c='red', s=.1)

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


PLOT_MODEL_2D = True
if PLOT_MODEL_2D:
  # plt.clf()
  plt.figure()
  for spd in [5, 25, 45]:
    x = np.linspace(0, 75, 100)
    y = model.predict([[spd * 0.4470392589877243, i] for i in x])
    plt.plot(x, y, label='{} mph'.format(spd))
  plt.ylabel('torque output')
  plt.xlabel('angle')
  plt.legend()
  plt.show()
