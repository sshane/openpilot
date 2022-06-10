from cereal import car
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL
from selfdrive.controls.lib.pid import PIDController
from selfdrive.controls.lib.drive_helpers import CONTROL_N
from selfdrive.modeld.constants import T_IDXS
from common.op_params import opParams
import numpy as np

LongCtrlState = car.CarControl.Actuators.LongControlState

# As per ISO 15622:2018 for all speeds
ACCEL_MIN_ISO = -3.5  # m/s^2
ACCEL_MAX_ISO = 2.0  # m/s^2


wb = np.load('/data/openpilot/accelnet/models/accelnetv7_weights.npz', allow_pickle=True)
w, b = wb['wb']


def accel_predict(x):
  x = np.array(x, dtype=np.float32)
  l0 = np.dot(x, w[0]) + b[0]
  l0 = np.where(l0 > 0, l0, l0 * 0.3)
  l1 = np.dot(l0, w[1]) + b[1]
  l1 = np.where(l1 > 0, l1, l1 * 0.3)
  l2 = np.dot(l1, w[2]) + b[2]
  l2 = np.where(l2 > 0, l2, l2 * 0.3)
  l3 = np.dot(l2, w[3]) + b[3]
  return l3


def long_control_state_trans(CP, active, long_control_state, v_ego, v_target_future,
                             brake_pressed, cruise_standstill):
  """Update longitudinal control state machine"""
  stopping_condition = (v_ego < 2.0 and cruise_standstill) or \
                       (v_ego < CP.vEgoStopping and
                        (v_target_future < CP.vEgoStopping or brake_pressed))

  starting_condition = v_target_future > CP.vEgoStarting and not cruise_standstill

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if active:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.pid:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition:
        long_control_state = LongCtrlState.pid

  return long_control_state


class LongControl():
  def __init__(self, CP):
    self.long_control_state = LongCtrlState.off  # initialized to off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             (CP.longitudinalTuning.kdBP, CP.longitudinalTuning.kdV),
                             rate=1/DT_CTRL,
                             sat_limit=0.8,
                             derivative_period=0.5)
    self.v_pid = 0.0
    self.last_output_accel = 0.0

  def reset(self, v_pid):
    """Reset PID controller and change setpoint"""
    self.pid.reset()
    self.v_pid = v_pid

  def update(self, active, CS, CP, long_plan, accel_limits, extras):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    # Interp control trajectory
    # TODO estimate car specific lag, use .15s for now
    speeds = long_plan.speeds
    if len(speeds) == CONTROL_N:
      # v_target_lower = interp(CP.longitudinalActuatorDelayLowerBound, T_IDXS[:CONTROL_N], speeds)
      # a_target_lower = 2 * (v_target_lower - speeds[0])/CP.longitudinalActuatorDelayLowerBound - long_plan.accels[0]

      # v_target_upper = interp(CP.longitudinalActuatorDelayUpperBound, T_IDXS[:CONTROL_N], speeds)
      # a_target_upper = 2 * (v_target_upper - speeds[0])/CP.longitudinalActuatorDelayUpperBound - long_plan.accels[0]
      a_target_cur = long_plan.accels[0]
      a_target_fut = interp(CP.longitudinalActuatorDelayUpperBound, T_IDXS[:CONTROL_N], long_plan.accels)
      # a_target = accel_predict([a_target_cur, a_target_fut])[0]

      # Output is the acceleration requests that will accurately get us to the requested accelerations.
      # Requested accel to accurate accel command is same time stamp, so offsetting output prediction is needed
      # ie. predicted requested accel at idx 0 will get us to accels[0] (in 0.15s)
      accel_pred = accel_predict([CS.vEgo, CS.aEgo, speeds[0]] + list(long_plan.accels))
      a_target = float(interp(CP.longitudinalActuatorDelayUpperBound, T_IDXS[:CONTROL_N], accel_pred))

      v_target = speeds[0]
      v_target_future = speeds[-1]
    else:
      v_target = 0.0
      v_target_future = 0.0
      a_target = 0.0

    # TODO: This check is not complete and needs to be enforced by MPC
    a_target = clip(a_target, ACCEL_MIN_ISO, ACCEL_MAX_ISO)

    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    # Update state machine
    output_accel = self.last_output_accel
    self.long_control_state = long_control_state_trans(CP, active, self.long_control_state, CS.vEgo,
                                                       v_target_future, CS.brakePressed,
                                                       CS.cruiseState.standstill)

    if self.long_control_state == LongCtrlState.off or CS.gasPressed:
      self.reset(CS.vEgo)
      output_accel = 0.

    # tracking objects and driving
    elif self.long_control_state == LongCtrlState.pid:
      self.v_pid = v_target

      # Toyota starts braking more when it thinks you want to stop
      # Freeze the integrator so we don't accelerate to compensate, and don't allow positive acceleration
      prevent_overshoot = not CP.stoppingControl and CS.vEgo < 1.5 and v_target_future < 0.7 and v_target_future < self.v_pid
      deadzone = interp(CS.vEgo, CP.longitudinalTuning.deadzoneBP, CP.longitudinalTuning.deadzoneV)
      freeze_integrator = prevent_overshoot

      output_accel = self.pid.update(self.v_pid, CS.vEgo, speed=CS.vEgo, deadzone=deadzone, feedforward=a_target, freeze_integrator=freeze_integrator)

      if prevent_overshoot:
        output_accel = min(output_accel, 0.0)

    # Intention is to stop, switch to a different brake control until we stop
    elif self.long_control_state == LongCtrlState.stopping:
      # Keep applying brakes until the car is stopped
      if not CS.standstill or output_accel > CP.stopAccel:
        output_accel -= CP.stoppingDecelRate * DT_CTRL
      output_accel = clip(output_accel, accel_limits[0], accel_limits[1])
      self.reset(CS.vEgo)

    self.last_output_accel = output_accel
    final_accel = clip(output_accel, accel_limits[0], accel_limits[1])

    return final_accel
