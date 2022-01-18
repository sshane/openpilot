#!/usr/bin/env python3
import numpy as np

from common.realtime import sec_since_boot, DT_CTRL
from cereal import messaging
from common.filter_simple import FirstOrderFilter
from common.params import Params
from opendbc.can.parser import CANParser

MAX_TIME_ONROAD = 5 * 60.
MOVEMENT_TIME = 1. * 60  # normal time allowed is one minute
INTERACTION_TIME = 1. * 30  # car needs to be inactive for this time before sentry mode is enabled


# Sentry mode state legend:
# Enabled: parameter is set allowing operation
# Armed: watching for car movement
# Tripped: movement tripped sentry mode, recording and alarming
# Car active: any action that signifies a user is present and interacting with their car


class SentryMode:
  def __init__(self):
    self.sm = messaging.SubMaster(['deviceState', 'sensorEvents'], poll=['sensorEvents'])
    self.pm = messaging.PubMaster(['sentryState'])

    self.cp = CANParser("toyota_nodsu_pt_generated", [("DOOR_LOCK_FEEDBACK_LIGHT", "CENTRAL_GATEWAY_UNIT", 0)], bus=0, enforce_checks=False)
    self.can_sock = messaging.sub_sock('can', timeout=100)

    self.prev_accel = np.zeros(3)
    self.initialized = False

    self.params = Params()
    self.sentry_enabled = self.params.get_bool("SentryMode")
    self.last_read_ts = sec_since_boot()

    self.sentry_tripped = False
    self.sentry_tripped_ts = 0.
    self.car_active_ts = sec_since_boot()  # start at active
    self.movement_ts = 0.
    self.accel_filters = [FirstOrderFilter(0, 0.5, DT_CTRL) for _ in range(3)]

  def update(self):
    # Update CAN
    can_strs = messaging.drain_sock_raw(self.can_sock, wait_for_one=True)
    self.cp.update_strings(can_strs)
    print("LOCK LIGHT 1: {}".format(bool(self.cp.vl["CENTRAL_GATEWAY_UNIT"]["DOOR_LOCK_FEEDBACK_LIGHT"])))

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
    print(f"{self.sentry_tripped=}")

  def is_sentry_armed(self, now_ts):
    """Returns if sentry is actively monitoring for movements/can be alarmed"""
    # Handle car interaction, reset interaction timeout
    car_active = self.sm['deviceState'].started
    # FIXME: why doesn't this work anymore?
    car_active = car_active or bool(self.cp.vl["CENTRAL_GATEWAY_UNIT"]["DOOR_LOCK_FEEDBACK_LIGHT"])
    print("LOCK LIGHT 2: {}".format(bool(self.cp.vl["CENTRAL_GATEWAY_UNIT"]["DOOR_LOCK_FEEDBACK_LIGHT"])))
    if car_active:
      self.car_active_ts = float(now_ts)

    car_inactive_long_enough = now_ts - self.car_active_ts > INTERACTION_TIME  # needs to be inactive for long enough
    return car_inactive_long_enough

  def update_sentry_tripped(self, now_ts):
    movement = any([abs(a_filter.x) > .01 for a_filter in self.accel_filters])
    if movement:
      self.movement_ts = float(now_ts)

    # Maximum allowed time onroad without movement is 1 minute. Any movement resets time allowed, maximum time is 5 minutes
    tripped_long_enough = now_ts - self.movement_ts > MOVEMENT_TIME
    tripped_long_enough = tripped_long_enough or now_ts - self.sentry_tripped_ts > MAX_TIME_ONROAD  # or total time reached

    sentry_tripped = False
    sentry_armed = self.is_sentry_armed(now_ts)
    print(f"{sentry_armed=}, {movement=}")
    print(f"{tripped_long_enough=}")
    print(f"{now_ts - self.sentry_tripped_ts=}")
    if sentry_armed and self.sentry_enabled:
      if movement:  # trip if armed, enabled, and there's movement
        sentry_tripped = True
      elif self.sentry_tripped and not tripped_long_enough:  # trip for long enough
        sentry_tripped = True

    # set when we first tripped
    if sentry_tripped and not self.sentry_tripped:
      self.sentry_tripped_ts = sec_since_boot()
    self.sentry_tripped = sentry_tripped

  def publish(self):
    sentry_state = messaging.new_message('sentryState')
    sentry_state.sentryState.started = self.sentry_tripped

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
