from cereal import car
from common.numpy_fast import clip, interp
from selfdrive.car import apply_toyota_steer_torque_limits, create_gas_command, make_can_msg
from selfdrive.car.toyota.toyotacan import create_steer_command, create_ui_command, \
                                           create_accel_command, create_acc_cancel_command, \
                                           create_fcw_command
from selfdrive.car.toyota.values import Ecu, CAR, STATIC_MSGS, NO_STOP_TIMER_CAR, CarControllerParams, MIN_ACC_SPEED
from opendbc.can.packer import CANPacker
from common.op_params import opParams
from selfdrive.config import Conversions as CV
from selfdrive.accel_to_gas import predict as accel_to_gas

VisualAlert = car.CarControl.HUDControl.VisualAlert


def accel_hysteresis(accel, accel_steady, enabled):

  # for small accel oscillations within ACCEL_HYST_GAP, don't change the accel command
  if not enabled:
    # send 0 when disabled, otherwise acc faults
    accel_steady = 0.
  elif accel > accel_steady + CarControllerParams.ACCEL_HYST_GAP:
    accel_steady = accel - CarControllerParams.ACCEL_HYST_GAP
  elif accel < accel_steady - CarControllerParams.ACCEL_HYST_GAP:
    accel_steady = accel + CarControllerParams.ACCEL_HYST_GAP
  accel = accel_steady

  return accel, accel_steady


def coast_accel(speed, which_func):  # given a speed, output coasting acceleration
  if which_func == 0:
    points = [[0.0, 0.538], [1.697, 0.28],
              [2.853, -0.199], [3.443, -0.249],
              [MIN_ACC_SPEED, -0.145]]
  else:
    points = [[0.0, 0.03], [.166, .424], [.335, .568],
              [.731, .440], [1.886, 0.262], [2.809, -0.207],
              [3.443, -0.249], [MIN_ACC_SPEED, -0.145]]
  return interp(speed, *zip(*points))


def compute_gb_pedal(accel, speed, which_func):
  # return accel_to_gas([accel, speed])[0]

  if which_func == 0:  # this is the above model converted to two polynomials
    _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [0.003237731717735036, -0.014032122419520062, 0.06717810220003029, 0.06629322939776298, -0.0006271460492818756, 0.0003429579347678683, 0.0019324020352106985, 0.0005829182414089772, 0.0002115616066200471, 0.0003269658627676601, 0.001992360262648108, 0.0035529270807654876]
  elif which_func == 1:  # this is the bottom function fitted on data
    # this is using future accel with current speed, gas, etc
    _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.05852890417685879, -0.0547687121098424, 0.17805761080500854, 0.02495029020857692, -1.3833659932240724e-05, -0.01051744191918636, -0.00010315551743123573, 0.05453987977699605, 0.0025468819577489994, -0.0018799608421947848, -0.010975243196134687, 0.020412221507199665]
  elif which_func == 2:  # bad
    # this is offsetting speed as well as accel
    _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.034972355520499945, -0.07258554000950052, 0.16766449798907357, 0.03796260968937626, 0.0006959184578795798, -0.010998968796341645, -0.002003721891231214, 0.049407329593411174, 0.002330180120906204, -0.0012535103221882832, -0.007723171310009469, 0.01441869761201456]
  elif which_func == 3:
    # This is same as 1 but reducing accel offset from 5 to 0 mph
    _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.05105606659218529, -0.0576261315554768, 0.15081380210023376, 0.05133068734696987, 5.9146813042508826e-05, -0.010139940821540159, -0.0003663558989772673, 0.05160370603505373, 0.0016194893168806697, -0.000803750169959028, -0.0006693327099676753, 0.009085497532847294]
  elif which_func == 4:
    # this is applying the accel filter on func 3: 3.0 > line['a_ego'] > coast_accel(line['v_ego']) - 0.2
    _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [0.022149049850234834, -0.11598085079184534, 0.16992234903369172, 0.04655053410459134, 5.010745172143593e-05, -0.00463810139475851, -0.0002985399523045501, 0.016845335025805565, 0.0018906355335290609, -0.0010547338746394025, -0.0038444819348291545, 0.011862218030372108]
  elif which_func == 5:
    _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [0.029256062736492627, -0.12102238822617944, 0.15981781977529405, 0.05496682933138211, 0.0008371370146672613, -0.0067242068166888155, -0.0023957293554218305, 0.02042199676500777, 0.0018610460523360629, -0.0006628198431291858, -0.002073832055454375, 0.007970801443623363]
  else:
    _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.05340517679465475, -0.053495524853591506, 0.1692496880860915, 0.02378568771700229, 5.9774819503946536e-05, -0.009988274638231051, -0.0003203880816858484, 0.051321083586716484, 0.0023280402254177005, -0.0018446446967463183, -0.008536402106750801, 0.020362858606493128]
  # reverse engineered the model, almost identical output. super cool how it uses the inputs in a linear function as coefficients for the opposite input poly
  speed_part = (_e5 * accel + _e6) * speed ** 2 + (_e7 * accel + _e8) * speed
  accel_part = ((_e1 * speed + _e2) * accel ** 5 + (_e3 * speed + _e4) * accel ** 4 + _a3 * accel ** 3 + _a4 * accel ** 2 + _a5 * accel)
  return speed_part + accel_part + _offset

  # # _c1, _c2, _c3, _c4 = [0.04412016647510183, 0.018224465923095633, 0.09983653162564889, 0.08837909527049172]
  # # return (desired_accel * _c1 + (_c4 * (speed * _c2 + 1))) * (speed * _c3 + 1)
  # if which_func == 0:
  #   _c1, _c2, _c3, _c4  = [0.014834278942078814, -0.019486618189634007, -0.04866680885268496, 0.18130227709359556]  # fit on both engaged and disengaged
  # elif which_func == 1:
  #   _c1, _c2, _c3, _c4  = [0.015545494731421215, -0.011431576758264202, -0.056374605760840496, 0.1797404798536819]  # just fit on engaged
  # else:
  #   _c1, _c2, _c3, _c4, _c5  = [0.0004504646112499155, 0.010911174152383137, 0.020950462773718394, 0.0971672107576878, -0.007383724106218966]
  #   return (_c1 * speed ** 2 + _c2 * speed + _c5) + (_c3 * desired_accel ** 2 + _c4 * desired_accel)
  # return (_c1 * speed + _c2) + (_c3 * desired_accel ** 2 + _c4 * desired_accel)


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.last_steer = 0
    self.accel_steady = 0.
    self.alert_active = False
    self.last_standstill = False
    self.standstill_req = False
    self.op_params = opParams()
    self.standstill_hack = self.op_params.get('standstill_hack')

    self.steer_rate_limited = False

    self.fake_ecus = set()
    if CP.enableCamera:
      self.fake_ecus.add(Ecu.fwdCamera)
    if CP.enableDsu:
      self.fake_ecus.add(Ecu.dsu)

    self.packer = CANPacker(dbc_name)

  def update(self, enabled, CS, frame, actuators, pcm_cancel_cmd, hud_alert,
             left_line, right_line, lead, left_lane_depart, right_lane_depart):

    # *** compute control surfaces ***

    # gas and brake
    apply_gas = 0.
    apply_accel = actuators.gas - actuators.brake

    if CS.CP.enableGasInterceptor and enabled and CS.out.vEgo < MIN_ACC_SPEED:
      # converts desired acceleration to gas percentage for pedal
      # +0.06 offset to reduce ABS pump usage when applying very small gas
      # if self.op_params.get('apply_accel') is not None:
      #   apply_accel = self.op_params.get('apply_accel')
      print(round(apply_accel * 3, 2), round(CS.out.aEgo, 2))
      if apply_accel * CarControllerParams.ACCEL_SCALE > coast_accel(CS.out.vEgo, self.op_params.get('coast_function')):
        apply_gas = clip(compute_gb_pedal(apply_accel * CarControllerParams.ACCEL_SCALE, CS.out.vEgo, self.op_params.get('ff_function')), 0., 1.)
      apply_accel = 0.06 - actuators.brake

    apply_accel, self.accel_steady = accel_hysteresis(apply_accel, self.accel_steady, enabled)
    apply_accel = clip(apply_accel * CarControllerParams.ACCEL_SCALE, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX)

    if enabled and self.op_params.get('apply_gas') is not None:
      apply_gas = self.op_params.get('apply_gas')
      apply_accel = 0


    # steer torque
    new_steer = int(round(actuators.steer * CarControllerParams.STEER_MAX))
    apply_steer = apply_toyota_steer_torque_limits(new_steer, self.last_steer, CS.out.steeringTorqueEps, CarControllerParams)
    self.steer_rate_limited = new_steer != apply_steer

    # Cut steering while we're in a known fault state (2s)
    if not enabled or CS.steer_state in [9, 25] or abs(CS.out.steeringRateDeg) > 100:
      apply_steer = 0
      apply_steer_req = 0
    else:
      apply_steer_req = 1

    if not enabled and CS.pcm_acc_status:
      # send pcm acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
      pcm_cancel_cmd = 1

    # on entering standstill, send standstill request
    if CS.out.standstill and not self.last_standstill and CS.CP.carFingerprint not in NO_STOP_TIMER_CAR and not self.standstill_hack:
      self.standstill_req = True
    if CS.pcm_acc_status != 8:
      # pcm entered standstill or it's disabled
      self.standstill_req = False

    self.last_steer = apply_steer
    self.last_accel = apply_accel
    self.last_standstill = CS.out.standstill

    can_sends = []

    #*** control msgs ***
    #print("steer {0} {1} {2} {3}".format(apply_steer, min_lim, max_lim, CS.steer_torque_motor)

    # toyota can trace shows this message at 42Hz, with counter adding alternatively 1 and 2;
    # sending it at 100Hz seem to allow a higher rate limit, as the rate limit seems imposed
    # on consecutive messages
    if Ecu.fwdCamera in self.fake_ecus:
      can_sends.append(create_steer_command(self.packer, apply_steer, apply_steer_req, frame))

      # LTA mode. Set ret.steerControlType = car.CarParams.SteerControlType.angle and whitelist 0x191 in the panda
      # if frame % 2 == 0:
      #   can_sends.append(create_steer_command(self.packer, 0, 0, frame // 2))
      #   can_sends.append(create_lta_steer_command(self.packer, actuators.steeringAngleDeg, apply_steer_req, frame // 2))

    # we can spam can to cancel the system even if we are using lat only control
    if (frame % 3 == 0 and CS.CP.openpilotLongitudinalControl) or (pcm_cancel_cmd and Ecu.fwdCamera in self.fake_ecus):
      lead = lead or CS.out.vEgo < 12.    # at low speed we always assume the lead is present do ACC can be engaged

      # Lexus IS uses a different cancellation message
      if pcm_cancel_cmd and CS.CP.carFingerprint == CAR.LEXUS_IS:
        can_sends.append(create_acc_cancel_command(self.packer))
      elif CS.CP.openpilotLongitudinalControl:
        can_sends.append(create_accel_command(self.packer, apply_accel, pcm_cancel_cmd, self.standstill_req, lead))
      else:
        can_sends.append(create_accel_command(self.packer, 0, pcm_cancel_cmd, False, lead))

    if (frame % 2 == 0) and (CS.CP.enableGasInterceptor):
      # send exactly zero if apply_gas is zero. Interceptor will send the max between read value and apply_gas.
      # This prevents unexpected pedal range rescaling
      can_sends.append(create_gas_command(self.packer, apply_gas, frame//2))

    # ui mesg is at 100Hz but we send asap if:
    # - there is something to display
    # - there is something to stop displaying
    fcw_alert = hud_alert == VisualAlert.fcw
    steer_alert = hud_alert == VisualAlert.steerRequired

    send_ui = False
    if ((fcw_alert or steer_alert) and not self.alert_active) or \
       (not (fcw_alert or steer_alert) and self.alert_active):
      send_ui = True
      self.alert_active = not self.alert_active
    elif pcm_cancel_cmd:
      # forcing the pcm to disengage causes a bad fault sound so play a good sound instead
      send_ui = True

    if (frame % 100 == 0 or send_ui) and Ecu.fwdCamera in self.fake_ecus:
      can_sends.append(create_ui_command(self.packer, steer_alert, pcm_cancel_cmd, left_line, right_line, left_lane_depart, right_lane_depart))

    if frame % 100 == 0 and Ecu.dsu in self.fake_ecus:
      can_sends.append(create_fcw_command(self.packer, fcw_alert))

    #*** static msgs ***

    for (addr, ecu, cars, bus, fr_step, vl) in STATIC_MSGS:
      if frame % fr_step == 0 and ecu in self.fake_ecus and CS.CP.carFingerprint in cars:
        can_sends.append(make_can_msg(addr, vl, bus))

    return can_sends
