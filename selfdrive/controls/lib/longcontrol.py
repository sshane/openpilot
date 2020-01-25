from cereal import log
from common.numpy_fast import clip, interp
from selfdrive.controls.lib.pid_long import PIController
from common.travis_checker import travis
from selfdrive.controls.lib.dynamic_lane_speed import DynamicLaneSpeed
from selfdrive.controls.lib.dynamic_gas import DynamicGas

LongCtrlState = log.ControlsState.LongControlState

STOPPING_EGO_SPEED = 0.5
MIN_CAN_SPEED = 0.3  # TODO: parametrize this in car interface
STOPPING_TARGET_SPEED = MIN_CAN_SPEED + 0.01
STARTING_TARGET_SPEED = 0.5
BRAKE_THRESHOLD_TO_PID = 0.2

STOPPING_BRAKE_RATE = 0.2  # brake_travel/s while trying to stop
STARTING_BRAKE_RATE = 0.8  # brake_travel/s while releasing on restart
BRAKE_STOPPING_TARGET = 0.5  # apply at least this amount of brake to maintain the vehicle stationary

_MAX_SPEED_ERROR_BP = [0., 30.]  # speed breakpoints
_MAX_SPEED_ERROR_V = [1.5, .8]  # max positive v_pid error VS actual speed; this avoids controls windup due to slow pedal resp

RATE = 100.0


def long_control_state_trans(active, long_control_state, v_ego, v_target, v_pid,
                             output_gb, brake_pressed, cruise_standstill):
  """Update longitudinal control state machine"""
  stopping_condition = (v_ego < 2.0 and cruise_standstill) or \
                       (v_ego < STOPPING_EGO_SPEED and \
                        ((v_pid < STOPPING_TARGET_SPEED and v_target < STOPPING_TARGET_SPEED) or
                        brake_pressed))

  starting_condition = v_target > STARTING_TARGET_SPEED and not cruise_standstill

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
        long_control_state = LongCtrlState.starting

    elif long_control_state == LongCtrlState.starting:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif output_gb >= -BRAKE_THRESHOLD_TO_PID:
        long_control_state = LongCtrlState.pid

  return long_control_state


class LongControl():
  def __init__(self, CP, compute_gb, candidate):
    self.long_control_state = LongCtrlState.off  # initialized to off
    self.pid = PIController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                            (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                            rate=RATE,
                            sat_limit=0.8,
                            convert=compute_gb)
    self.v_pid = 0.0
    self.last_output_gb = 0.0

    self.dynamic_gas = DynamicGas(CP, candidate)

    self.gas_pressed = False
    self.lead_data = {'v_rel': None, 'a_lead': None, 'x_lead': None, 'status': False}
    self.track_data = []
    self.mpc_TR = 1.8
    self.blinker_status = False
    self.dynamic_lane_speed = DynamicLaneSpeed()

  def reset(self, v_pid):
    """Reset PID controller and change setpoint"""
    self.pid.reset()
    self.v_pid = v_pid

  def handle_passable(self, passable, v_ego):
    CS = passable['car_state']
    self.blinker_status = CS.leftBlinker or CS.rightBlinker
    self.gas_pressed = CS.gasPressed
    self.lead_data['v_rel'] = passable['lead_one'].vRel
    self.lead_data['a_lead'] = passable['lead_one'].aLeadK
    self.lead_data['x_lead'] = passable['lead_one'].dRel
    self.lead_data['status'] = passable['has_lead']  # this fixes radarstate always reporting a lead, thanks to arne
    self.mpc_TR = passable['mpc_TR']
    self.track_data = []
    for track in passable['live_tracks']:
      self.track_data.append({'v_lead': v_ego + track.vRel, 'y_rel': track.yRel, 'x_lead': track.dRel})

  def update(self, active, v_ego, brake_pressed, standstill, cruise_standstill, v_cruise, v_target, v_target_future, a_target, CP, passable):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    # Actuation limits
    if not travis:
      self.handle_passable(passable, v_ego)
      gas_max = self.dynamic_gas.update(v_ego, self.lead_data, self.mpc_TR, self.blinker_status)
      # v_target, v_target_future, a_target = self.dynamic_lane_speed.update(v_target, v_target_future, v_cruise, a_target, v_ego, self.track_data, self.lead_data)
    else:
      gas_max = interp(v_ego, CP.gasMaxBP, CP.gasMaxV)
    brake_max = interp(v_ego, CP.brakeMaxBP, CP.brakeMaxV)

    # Update state machine
    output_gb = self.last_output_gb
    self.long_control_state = long_control_state_trans(active, self.long_control_state, v_ego,
                                                       v_target_future, self.v_pid, output_gb,
                                                       brake_pressed, cruise_standstill)

    v_ego_pid = max(v_ego, MIN_CAN_SPEED)  # Without this we get jumps, CAN bus reports 0 when speed < 0.3

    if self.long_control_state == LongCtrlState.off or (self.gas_pressed and not travis):
      self.v_pid = v_ego_pid
      self.pid.reset()
      output_gb = 0.

    # tracking objects and driving
    elif self.long_control_state == LongCtrlState.pid:
      self.v_pid = v_target
      self.pid.pos_limit = gas_max
      self.pid.neg_limit = - brake_max

      # Toyota starts braking more when it thinks you want to stop
      # Freeze the integrator so we don't accelerate to compensate, and don't allow positive acceleration
      prevent_overshoot = not CP.stoppingControl and v_ego < 1.5 and v_target_future < 0.7
      deadzone = interp(v_ego_pid, CP.longitudinalTuning.deadzoneBP, CP.longitudinalTuning.deadzoneV)

      output_gb = self.pid.update(self.v_pid, v_ego_pid, speed=v_ego_pid, deadzone=deadzone, feedforward=a_target, freeze_integrator=prevent_overshoot)

      if prevent_overshoot:
        output_gb = min(output_gb, 0.0)

    # Intention is to stop, switch to a different brake control until we stop
    elif self.long_control_state == LongCtrlState.stopping:
      # Keep applying brakes until the car is stopped
      if not standstill or output_gb > -BRAKE_STOPPING_TARGET:
        output_gb -= STOPPING_BRAKE_RATE / RATE
      output_gb = clip(output_gb, -brake_max, gas_max)

      self.v_pid = v_ego
      self.pid.reset()

    # Intention is to move again, release brake fast before handing control to PID
    elif self.long_control_state == LongCtrlState.starting:
      if output_gb < -0.2:
        output_gb += STARTING_BRAKE_RATE / RATE
      self.v_pid = v_ego
      self.pid.reset()

    self.last_output_gb = output_gb
    final_gas = clip(output_gb, 0., gas_max)
    final_brake = -clip(output_gb, -brake_max, 0.)

    return final_gas, final_brake
