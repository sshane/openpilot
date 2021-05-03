import matplotlib.pyplot as plt
import numpy as np
from selfdrive.config import Conversions as CV


def std_feedforward(angle, speed, mph=False):
  """
  What latcontrol_pid uses and is technically correct (~lateral accel)
  """
  speed = CV.MPH_TO_MS * speed if mph else speed
  return speed ** 2 * angle


def acc_feedforward(angle, speed, mph=False):
  """
  Fitted from data from 2017 Corolla. Much more accurate at low speeds
  (Torque almost drops to 0 at low speeds with std feedforward)
  """
  speed = CV.MPH_TO_MS * speed if mph else speed
  # todo: this is a bit out of date, it was fitted assuming 0.12s of delay
  _c1, _c2, _c3 = 0.35189607550172824, 7.506201251644202, 69.226826411091
  return (_c1 * speed ** 2 + _c2 * speed + _c3) * angle


def convert_torque_corolla(speed, mph=False):
  """Returns a multiplier to convert universal torque to vehicle-specific torque"""
  # todo: can we fit a function that outputs this without calculating both feedforwards?
  speed = CV.MPH_TO_MS * speed if mph else speed
  mult = acc_feedforward(1, speed) / std_feedforward(1, speed)
  return mult


def convert_torque_corolla_new(speed, mph=False):
  """Returns a multiplier to convert universal torque to vehicle-specific torque"""
  # todo: can we fit a function that outputs this without calculating both feedforwards?
  speed = CV.MPH_TO_MS * speed if mph else speed
  weight = np.interp(speed, [0, 20 * CV.MPH_TO_MS], [0.5, 0])
  speed = speed * (1 - weight) + 20 * CV.MPH_TO_MS * weight
  mult = acc_feedforward(1, speed) / std_feedforward(1, speed)
  return mult


# Plot how multiplier changes with speed
spds = np.linspace(0.01, 70, 1000)
mults = convert_torque_corolla(spds, mph=True)
mults_new = [convert_torque_corolla_new(spd, mph=True) for spd in spds]
plt.title('output of torque conversion function')
# plt.plot(spds, mults, label='torque multiplier')
plt.plot(spds, mults_new, label='new torque multiplier')
plt.xlabel('speed (mph)')
plt.legend()


# Plot comparison between std ff and more accurate fitted ff
plt.figure()
deg = 10
plt.title(f'torque response at {deg} degrees')
torques = std_feedforward(deg, spds, mph=True)
plt.plot(spds, torques, label='standard feedforward (~lateral accel)')

torques = acc_feedforward(deg, spds, mph=True)
plt.plot(spds, torques, label='custom-fit \'17 Corolla feedforward')

torques = std_feedforward(deg, spds, mph=True)
plt.plot(spds, torques * mults, label='standard ff transformed')

plt.plot(spds, torques * mults_new, label='standard ff transformed new')

plt.xlabel('speed (mph)')
plt.ylabel('torque')
plt.legend()
