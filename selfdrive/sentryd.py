#!/usr/bin/env python3
import numpy as np

from cereal import messaging
from common.filter_simple import FirstOrderFilter
from common.params import Params
from common.realtime import sec_since_boot, DT_CTRL
from opendbc.can.parser import CANParser

# ****** Sentry mode states ******
# Enabled: parameter is set allowing operation
# Armed: watching for car movement
# Tripped: movement tripped sentry mode, recording and alarming
# Car active: any action that signifies a user is present and interacting with their car

# ****** Sentry mode behavior ******
# Sentry arms immediately when car is locked with keyfob
# - Normal inactive timeout proceeds if driver locks car internally
# If car is left unlocked (or falsely detected so), sentry becomes armed after INACTIVE_TIME
# Locking, unlocking, or starting your car will disarm sentry mode and reset timer


MAX_TIME_ONROAD = 5 * 60.  # after this is reached, car stops recording, disregarding movement
MOVEMENT_TIME = 60.  # each movement resets onroad timer to this
MIN_TIME_ONROAD = MOVEMENT_TIME + 5.
INACTIVE_TIME = 2. * 60.  # car needs to be inactive for this time before sentry mode is enabled

DEBUG = False

signals = [
  ("LOCK_STATUS_CHANGED", "DOOR_LOCKS", 0),
  ("LOCK_STATUS", "DOOR_LOCKS", 1),  # 1 is unlocked
  ("LOCKED_VIA_KEYFOB", "DOOR_LOCKS", 0),
]


class SentryMode:
  def __init__(self):
    self.sm = messaging.SubMaster(['deviceState', 'sensorEvents'], poll=['sensorEvents'])
    self.pm = messaging.PubMaster(['sentryState'])

    # TODO: Only toyota supported for now. detect car type and switch DBC/signals
    self.cp = CANParser("toyota_nodsu_pt_generated", signals, bus=0, enforce_checks=False)
    self.can_sock = messaging.sub_sock('can', timeout=100)

    self.prev_accel = np.zeros(3)
    self.initialized = False

    self.params = Params()
    self.sentry_enabled = self.params.get_bool("SentryMode")
    self.last_read_ts = sec_since_boot()

    self.car_locked = False
    self.sentry_tripped = False
    self.sentry_armed = False
    self.sentry_tripped_ts = 0.
    self.car_active_ts = sec_since_boot()  # start at active
    self.movement_ts = 0.
    self.accel_filters = [FirstOrderFilter(0, 0.5, DT_CTRL) for _ in range(3)]

  def sprint(self, *args, **kwargs):  # slow print
    if DEBUG:
      if self.sm.frame % (100 / 20.) == 0:  # 20 hz
        print(*args, **kwargs)

  def update(self):
    # Update CAN
    can_strs = messaging.drain_sock_raw(self.can_sock, wait_for_one=True)
    self.cp.update_strings(can_strs)

    # Update car locked status
    self.car_locked = self.cp.vl["DOOR_LOCKS"]["LOCK_STATUS"] == 0
    self.sprint("car locked: {}".format(self.car_locked))

    # Update parameter
    now_ts = sec_since_boot()
    if now_ts - self.last_read_ts > 15.:
      self.sentry_enabled = self.params.get_bool("SentryMode")
      self.last_read_ts = float(now_ts)

    # Handle sensors
    for sensor in self.sm['sensorEvents']:
      if sensor.which() == 'acceleration':
        accels = sensor.acceleration.v
        if len(accels) == 3:  # sometimes is empty, in that case don't update
          if self.initialized:  # prevent initial jump # TODO: can remove since we start at an active car state?
            for idx, v in enumerate(accels):
              self.accel_filters[idx].update(accels[idx] - self.prev_accel[idx])
          self.initialized = True
          self.prev_accel = list(accels)

    self.update_sentry_tripped(now_ts)
    # self.sprint(f"{self.sentry_tripped=}")

  def is_sentry_armed(self, now_ts):
    """Returns if sentry is actively monitoring for movements/can be alarmed"""
    # Handle car interaction, reset interaction timeout
    car_active = self.sm['deviceState'].started
    car_active |= bool(self.cp.vl["DOOR_LOCKS"]["LOCK_STATUS_CHANGED"])
    if bool(self.cp.vl["DOOR_LOCKS"]["LOCK_STATUS_CHANGED"]):
      self.sprint('lock status changed!')
    if car_active:
      self.car_active_ts = float(now_ts)

    car_inactive_long_enough = now_ts - self.car_active_ts > INACTIVE_TIME
    car_locked_with_fob = self.car_locked and bool(self.cp.vl["DOOR_LOCKS"]["LOCKED_VIA_KEYFOB"])
    return self.sentry_enabled and (car_inactive_long_enough or car_locked_with_fob)

  def update_sentry_tripped(self, now_ts):
    movement = any([abs(a_filter.x) > .01 for a_filter in self.accel_filters])
    if movement:
      self.movement_ts = float(now_ts)

    # For as long as we see movement, extend timer by MOVEMENT_TIME.
    tripped_long_enough = now_ts - self.movement_ts > MOVEMENT_TIME
    tripped_long_enough &= now_ts - self.sentry_tripped_ts > MIN_TIME_ONROAD  # minimum of
    tripped_long_enough |= now_ts - self.sentry_tripped_ts > MAX_TIME_ONROAD  # maximum of

    sentry_tripped = False
    self.sentry_armed = self.is_sentry_armed(now_ts)
    self.sprint(f"{now_ts - self.sentry_tripped_ts=} > {MIN_TIME_ONROAD=}")
    if self.sentry_armed:
      if movement:  # trip if armed and there's movement
        sentry_tripped = True
      elif self.sentry_tripped and not tripped_long_enough:  # trip for long enough
        sentry_tripped = True

    # set when we first tripped
    if sentry_tripped and not self.sentry_tripped:
      self.sentry_tripped_ts = sec_since_boot()
    self.sentry_tripped = sentry_tripped

  def publish(self):
    sentry_state = messaging.new_message('sentryState')
    sentry_state.sentryState.started = bool(self.sentry_tripped)
    sentry_state.sentryState.armed = bool(self.sentry_armed)

    self.pm.send('sentryState', sentry_state)

  def start(self):
    while 1:
      self.sm.update()
      self.update()
      self.publish()


def main():
  sentry_mode = SentryMode()
  sentry_mode.start()


if __name__ == "__main__":
  main()
