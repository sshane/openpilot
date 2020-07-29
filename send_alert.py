from selfdrive.controls.lib.alertmanager import AlertManager
from cereal import car, log, messaging

EventName = car.CarEvent.EventName

AM = AlertManager()
frame = 0
while True:
  input()
  AM.add_custom(frame, EventName.laneSpeedKeeping, False, extra_text_1='RIGHT', extra_text_2='Oncoming traffic in left lane')
  AM.process_alerts(frame)

  print(AM.alert_text_1)
  print(AM.alert_text_2)
  print(AM.alert_type)

  # dat = messaging.new_message('controlsState')
  # dat.valid = True
  # controlsState = dat.controlsState
  # controlsState.alertText1 = AM.alert_text_1
  # controlsState.alertText2 = AM.alert_text_2
  # controlsState.alertSize = AM.alert_size
  # controlsState.alertStatus = AM.alert_status
  # controlsState.alertBlinkingRate = AM.alert_rate
  # controlsState.alertType = AM.alert_type
  # controlsState.alertSound = AM.audible_alert
  # controlsState.enabled = self.enabled
  # controlsState.active = self.active
  # controlsState.vEgo = CS.vEgo
  # controlsState.vEgoRaw = CS.vEgoRaw
  # controlsState.angleSteers = CS.steeringAngle
  # controlsState.curvature = self.VM.calc_curvature(steer_angle_rad, CS.vEgo)
  # controlsState.steerOverride = CS.steeringPressed
  # controlsState.state = self.state
  # controlsState.engageable = not self.events.any(ET.NO_ENTRY)
  # controlsState.longControlState = self.LoC.long_control_state
  # controlsState.vPid = float(self.LoC.v_pid)
  # controlsState.vCruise = float(self.v_cruise_kph)
  # controlsState.upAccelCmd = float(self.LoC.pid.p)
  # controlsState.uiAccelCmd = float(self.LoC.pid.id)
  # controlsState.ufAccelCmd = float(self.LoC.pid.f)
  # controlsState.angleSteersDes = float(self.LaC.angle_steers_des)
  # controlsState.vTargetLead = float(v_acc)
  # controlsState.aTarget = float(a_acc)
  # controlsState.jerkFactor = float(self.sm['plan'].jerkFactor)
  # controlsState.gpsPlannerActive = self.sm['plan'].gpsPlannerActive
  # controlsState.vCurvature = self.sm['plan'].vCurvature
  # controlsState.decelForModel = self.sm['plan'].longitudinalPlanSource == LongitudinalPlanSource.model
  # controlsState.cumLagMs = -self.rk.remaining * 1000.
  # controlsState.startMonoTime = int(start_time * 1e9)
  # controlsState.mapValid = self.sm['plan'].mapValid
  # controlsState.forceDecel = bool(force_decel)
  # controlsState.canErrorCounter = self.can_error_counter
  # self.pm.send('controlsState', dat)


  frame += 500
