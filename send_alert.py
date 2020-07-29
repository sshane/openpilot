from selfdrive.controls.lib.alertmanager import AlertManager
from cereal import car, log

EventName = car.CarEvent.EventName

AM = AlertManager()
frame = 0
while True:
  input()
  AM.add_custom(frame, EventName.laneSpeedKeeping, False, extra_text_1='RIGHT', extra_text_2='Oncoming traffic in left lane')
  AM.process_alerts(frame)
  frame += 500
