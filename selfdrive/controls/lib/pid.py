import numpy as np
from common.numpy_fast import clip, interp
from common.op_params import opParams
from selfdrive.config import Conversions as CV


def apply_deadzone(error, deadzone):
  if error > deadzone:
    error -= deadzone
  elif error < - deadzone:
    error += deadzone
  else:
    error = 0.
  return error

class LatPIDController():
  def __init__(self, k_p, k_i, k_d, k_f=1., pos_limit=None, neg_limit=None, rate=100, sat_limit=0.8, convert=None):
    self._k_p = k_p  # proportional gain
    self._k_i = k_i  # integral gain
    self._k_d = k_d  # derivative gain
    self.k_f = k_f  # feedforward gain

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.sat_count_rate = 1.0 / rate
    self.i_unwind_rate = 0.3 / rate
    self.i_rate = 1.0 / rate
    self.sat_limit = sat_limit
    self.convert = convert

    self.reset()

  def convert_pid_gains(self, setpoint, measurement, error, pid):
    _c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _c10 = [0.010969681918287285, -1.317303952625088, 0.003090297042995128, 0.0006174063280014462, -0.11408303431617882, -0.004223466361442405,
                                                         0.5459159299542625, -0.4208442440715394, -0.016940226962729534, 1.1968717444641879]

    if abs(setpoint) < abs(measurement) or setpoint * measurement < 0:
      mod = _c1 * abs(error)
      mod += _c3 * abs(setpoint)
      mod += _c2
    else:
      mod = _c4 * abs(error)
      mod += _c6 * abs(setpoint)
      mod += _c5

    new_pid = pid + pid * mod

    weight = np.interp(abs(error), [0, _c7 * abs(measurement) + _c8], [1, 0])

    pid = (new_pid * weight) + ((1 - weight) * pid)
    pid *= (_c9 * self.speed + _c10)
    return float(pid)

  @property
  def k_p(self):
    return interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def k_i(self):
    return interp(self.speed, self._k_i[0], self._k_i[1])

  @property
  def k_d(self):
    return interp(self.speed, self._k_d[0], self._k_d[1])

  def _check_saturation(self, control, check_saturation, error):
    saturated = (control < self.neg_limit) or (control > self.pos_limit)

    if saturated and check_saturation and abs(error) > 0.1:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate

    self.sat_count = clip(self.sat_count, 0.0, 1.0)

    return self.sat_count > self.sat_limit

  def reset(self):
    self.p = 0.0
    self.i = 0.0
    self.f = 0.0
    self.sat_count = 0.0
    self.saturated = False
    self.control = 0
    self.errors = []

  def update(self, setpoint, measurement, speed=0.0, check_saturation=True, override=False, feedforward=0., deadzone=0., freeze_integrator=False):
    self.speed = speed

    error = float(apply_deadzone(setpoint - measurement, deadzone))
    self.p = error * self.k_p
    self.f = feedforward * self.k_f

    d = 0
    if len(self.errors) >= 5:  # makes sure list is long enough
      d = (error - self.errors[-5]) / 5  # get deriv in terms of 100hz (tune scale doesn't change)
      d *= self.k_d

    if override:
      self.i -= self.i_unwind_rate * float(np.sign(self.i))
    else:
      i = self.i + error * self.k_i * self.i_rate
      control = self.p + self.f + i + d

      if self.convert is not None:
        control = self.convert(control, speed=self.speed)

      # Update when changing i will move the control away from the limits
      # or when i will move towards the sign of the error
      if ((error >= 0 and (control <= self.pos_limit or i < 0.0)) or
          (error <= 0 and (control >= self.neg_limit or i > 0.0))) and \
         not freeze_integrator:
        self.i = i

    new_pid = self.convert_pid_gains(setpoint, measurement, error, self.p + self.i + d)

    control = new_pid + d
    if self.convert is not None:
      control = self.convert(control, speed=self.speed)

    self.saturated = self._check_saturation(control, check_saturation, error)

    self.errors.append(float(error))
    while len(self.errors) > 5:
      self.errors.pop(0)

    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control


class LongPIDController:
  def __init__(self, k_p, k_i, k_d, k_f=1., pos_limit=None, neg_limit=None, rate=100, sat_limit=0.8, convert=None):
    self.op_params = opParams()
    self._k_p = k_p  # proportional gain
    self._k_i = k_i  # integral gain
    self._k_d = k_d  # derivative gain
    self.k_f = k_f  # feedforward gain

    self.max_accel_d = 0.4 * CV.MPH_TO_MS

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.sat_count_rate = 1.0 / rate
    self.i_unwind_rate = 0.3 / rate
    self.rate = 1.0 / rate
    self.sat_limit = sat_limit
    self.convert = convert

    self.reset()

  @property
  def k_p(self):
    return interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def k_i(self):
    return interp(self.speed, self._k_i[0], self._k_i[1])

  @property
  def k_d(self):
    return interp(self.speed, self._k_d[0], self._k_d[1])

  def _check_saturation(self, control, check_saturation, error):
    saturated = (control < self.neg_limit) or (control > self.pos_limit)

    if saturated and check_saturation and abs(error) > 0.1:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate

    self.sat_count = clip(self.sat_count, 0.0, 1.0)

    return self.sat_count > self.sat_limit

  def reset(self):
    self.p = 0.0
    self.id = 0.0
    self.f = 0.0
    self.sat_count = 0.0
    self.saturated = False
    self.control = 0
    self.last_setpoint = 0.0
    self.last_error = 0.0

  def update(self, setpoint, measurement, speed=0.0, check_saturation=True, override=False, feedforward=0., deadzone=0., freeze_integrator=False):
    self.speed = speed

    error = float(apply_deadzone(setpoint - measurement, deadzone))

    self.p = error * self.k_p
    self.f = feedforward * self.k_f

    if override:
      self.id -= self.i_unwind_rate * float(np.sign(self.id))
    else:
      i = self.id + error * self.k_i * self.rate
      control = self.p + self.f + i

      if self.convert is not None:
        control = self.convert(control, speed=self.speed)

      # Update when changing i will move the control away from the limits
      # or when i will move towards the sign of the error
      if ((error >= 0 and (control <= self.pos_limit or i < 0.0)) or \
          (error <= 0 and (control >= self.neg_limit or i > 0.0))) and \
              not freeze_integrator:
        self.id = i

    if self.op_params.get('enable_long_derivative'):
      if abs(setpoint - self.last_setpoint) / self.rate < self.max_accel_d:  # if setpoint isn't changing much
        d = self.k_d * (error - self.last_error)
        if (self.id > 0 and self.id + d >= 0) or (self.id < 0 and self.id + d <= 0):  # if changing integral doesn't make it cross zero
          self.id += d

    control = self.p + self.f + self.id
    if self.convert is not None:
      control = self.convert(control, speed=self.speed)

    self.saturated = self._check_saturation(control, check_saturation, error)

    self.last_setpoint = float(setpoint)
    self.last_error = float(error)

    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control
