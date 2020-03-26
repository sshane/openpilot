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

class PIDController():
  def __init__(self, k_p, k_i, k_d, k_f=1., pos_limit=None, neg_limit=None, rate=100, sat_limit=0.8, convert=None):
    self.op_params = opParams()
    self.error_idx = -1
    self.get_live_params()
    self._k_p = k_p  # proportional gain
    self._k_i = k_i  # integral gain
    self._k_d = k_d  # derivative gain
    self.k_f = k_f  # feedforward gain

    self.p = 0.0
    self.i = 0.0
    self.d = 0.0

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.sat_count_rate = 1.0 / rate
    self.i_unwind_rate = 0.3 / rate
    self.rate = 1.0 / rate
    self.sat_limit = sat_limit
    self.convert = convert

    self.past_errors = []
    self.last_setpoint = 0.0

    self.reset()

  def get_live_params(self):
    self.enable_derivative = self.op_params.get('enable_derivative', True)
    self.error_idx = self.op_params.get('error_idx', -1)
    self.derivative = self.op_params.get('derivative', 0.16)
    self.max_accel_d = self.op_params.get('max_accel_d', 1.0) * CV.MPH_TO_MS

  @property
  def k_p(self):
    return interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def k_i(self):
    return interp(self.speed, self._k_i[0], self._k_i[1])

  @property
  def k_d(self):
    return self.derivative

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
    self.d = 0.0
    self.f = 0.0
    self.sat_count = 0.0
    self.saturated = False
    self.control = 0

  def set_d(self, error):
    if len(self.past_errors) >= -self.error_idx and self.enable_derivative:
      last_error = self.past_errors[self.error_idx]
      rate = -self.error_idx / 100
      self.d = self.k_d * ((error - last_error) / rate)
    else:  # wait until we gather enough data to allow index get
      self.d = 0.0

  def update(self, setpoint, measurement, speed=0.0, check_saturation=True, override=False, feedforward=0., deadzone=0., freeze_integrator=False):
    self.get_live_params()
    self.speed = speed

    error = float(apply_deadzone(setpoint - measurement, deadzone))
    self.p = error * self.k_p
    self.f = feedforward * self.k_f

    if override:
      self.i -= self.i_unwind_rate * float(np.sign(self.i))
      self.d = 0.0
    else:
      i = self.i + error * self.k_i * self.rate
      self.set_d(error)
      control = self.p + self.f + i  # don't add d here

      if self.convert is not None:
        control = self.convert(control, speed=self.speed)

      # Update when changing i will move the control away from the limits
      # or when i will move towards the sign of the error
      if ((error >= 0 and (control <= self.pos_limit or i < 0.0)) or \
          (error <= 0 and (control >= self.neg_limit or i > 0.0))) and \
         not freeze_integrator:
        self.i = i

    control = self.p + self.f + self.i
    # with open('/data/accel_pid', 'a') as f:
    #   f.write('{}\n'.format(abs(setpoint - self.last_setpoint) / self.rate))
    if abs(setpoint - self.last_setpoint) / self.rate < self.max_accel_d:  # if cruising with minimal setpoint change
      control += self.d  # then use derivative
    if self.convert is not None:
      control = self.convert(control, speed=self.speed)

    self.saturated = self._check_saturation(control, check_saturation, error)

    self.past_errors.append(error)
    while len(self.past_errors) > 200:  # keep last 2 seconds
      del self.past_errors[0]
    self.last_setpoint = setpoint

    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control
