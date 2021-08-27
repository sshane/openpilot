import importlib
import math

from common.filter_simple import FirstOrderFilter
from common.realtime import DT_CTRL
from selfdrive.controls.lib.pid import LatPIDController
from selfdrive.controls.lib.drive_helpers import get_steer_max
from cereal import log


class LatControlPID():
  def __init__(self, CP):
    self.pid = LatPIDController((CP.lateralTuning.pid.kpBP, CP.lateralTuning.pid.kpV),
                                (CP.lateralTuning.pid.kiBP, CP.lateralTuning.pid.kiV),
                                (CP.lateralTuning.pid.kdBP, CP.lateralTuning.pid.kdV),
                                k_f=CP.lateralTuning.pid.kf, pos_limit=1.0, sat_limit=CP.steerLimitTimer)
    self.new_kf_tuned = True  # CP.lateralTuning.pid.newKfTuned
    self.kf_filter = FirstOrderFilter(CP.lateralTuning.pid.kf, 10, DT_CTRL)

    self.CarControllerParams = importlib.import_module('selfdrive.car.{}.values'.format(CP.carName)).CarControllerParams
    assert self.CarControllerParams, 'Missing CarControllerParams!'
    assert self.CarControllerParams.STEER_MAX != 0, 'Can\'t be 0'

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, CP, VM, params, desired_curvature, desired_curvature_rate):
    pid_log = log.ControlsState.LateralPIDState.new_message()
    pid_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    pid_log.steeringRateDeg = float(CS.steeringRateDeg)

    angle_steers_des_no_offset = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo))
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg

    if CS.vEgo < 0.3 or not active:
      output_steer = 0.0
      pid_log.active = False
      self.pid.reset()

      if abs(CS.steeringRateDeg) < 20 and 5 < abs(CS.steeringAngleDeg) < 90 and CS.vEgo > 5:
        torque = CS.steeringTorqueEps / self.CarControllerParams.STEER_MAX
        predicted_kf = torque / (CS.steeringAngleDeg * CS.vEgo ** 2)
        self.pid.k_f = self.kf_filter.update(predicted_kf)

        print('PREDICTED KF: {}'.format(self.kf_filter.x))
    else:
      steers_max = get_steer_max(CP, CS.vEgo)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max

      # TODO: feedforward something based on lat_plan.rateSteers
      steer_feedforward = angle_steers_des_no_offset  # offset does not contribute to resistive torque
      if self.new_kf_tuned:
        _c1, _c2, _c3 = 0.35189607550172824, 7.506201251644202, 69.226826411091
        steer_feedforward *= _c1 * CS.vEgo ** 2 + _c2 * CS.vEgo + _c3
      else:
        steer_feedforward *= CS.vEgo ** 2  # proportional to realigning tire momentum (~ lateral accel)

      deadzone = 0.0

      check_saturation = (CS.vEgo > 10) and not CS.steeringRateLimited and not CS.steeringPressed
      output_steer = self.pid.update(angle_steers_des, CS.steeringAngleDeg, check_saturation=check_saturation, override=CS.steeringPressed,
                                     feedforward=steer_feedforward, speed=CS.vEgo, deadzone=deadzone)
      pid_log.active = True
      pid_log.p = self.pid.p
      pid_log.i = self.pid.i
      pid_log.f = self.pid.f
      pid_log.output = output_steer
      pid_log.saturated = bool(self.pid.saturated)

    return output_steer, angle_steers_des, pid_log
