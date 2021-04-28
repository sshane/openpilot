import matplotlib.pyplot as plt
import numpy as np
import time
from selfdrive.config import Conversions as CV
from scipy.optimize import curve_fit


def std_feedforward(angle, speed):
  """
  What latcontrol_pid uses and is technically correct (~lateral accel)
  """
  return speed ** 2 * angle


def acc_feedforward(angle, speed):
  """
  Fitted from data from 2017 Corolla. Much more accurate at low speeds
  (Torque almost drops to 0 at low speeds with std feedforward)
  """
  # todo: this is a bit out of date, it was fitted assuming 0.12s of delay
  _c1, _c2, _c3 = 0.35189607550172824, 7.506201251644202, 69.226826411091
  return (_c1 * speed ** 2 + _c2 * speed + _c3) * angle


def convert_torque_corolla(speed):
  """Returns a multiplier to convert universal torque to vehicle-specific torque"""
  # todo: can we fit a function that outputs this without calculating both feedforwards?
  speed = np.array([max(s, 5 * CV.MPH_TO_MS) for s in speed])
  mult = acc_feedforward(1, speed) / std_feedforward(1, speed)
  return mult


fitting = True
def convert_torque_corolla_fitted(speed, _c1, _c2, _c3):
  # this is almost 1 to 1 with above and only ~0.001 seconds slower after 1 hour
  # edit: it can either be accurate from 1-5 mph or 5-70 mph, not really both
  # but if we clip it below x (5?) mph then it doesn't really matter
  if not fitting:
    speed = [max(s, 5 * CV.MPH_TO_MS) for s in speed]
  return np.exp(_c2 * -(np.log(speed) - _c1)) + _c3


# Plot how multiplier changes with speed
spds = np.linspace(1, 70, 6000*60) * CV.MPH_TO_MS  # *60 is 1 hour
t = time.time()
mults = convert_torque_corolla(spds)
t = time.time() - t
print(f'Took {t} seconds to execute using ff functions')

params = curve_fit(convert_torque_corolla_fitted, spds, mults)[0].tolist()
# params = [2.264884864988784, 1.9174723981343285, 0.7601190196820092]  # 1-70 mph
# params = [2.4175284500713237, 1.7501977566285476, 0.5355805302921479]  # 5-70 mph
print(f'Params: {params}')
fitting = False

t = time.time()
mults_fitted = convert_torque_corolla_fitted(spds, *params)
t = time.time() - t
print(f'Took {t} seconds to execute fitted function')

plt.title('output of torque conversion function')
plt.plot(spds * CV.MS_TO_MPH, mults, label='torque multiplier')
plt.plot(spds * CV.MS_TO_MPH, mults_fitted, label='torque multiplier (fitted)')
plt.xlabel('speed (mph)')
plt.legend()


# Plot comparison between std ff and more accurate fitted ff
plt.figure()
deg = 10
plt.title(f'torque response at {deg} degrees')
torques = std_feedforward(deg, spds)
plt.plot(spds * CV.MS_TO_MPH, torques, label='standard feedforward (~lateral accel)')

torques = acc_feedforward(deg, spds)
plt.plot(spds * CV.MS_TO_MPH, torques, label='custom-fit \'17 Corolla feedforward')
plt.xlabel('speed (mph)')
plt.ylabel('torque')
plt.legend()
