#!/usr/bin/env python3
import time
import numpy as np

from common.realtime import sec_since_boot, DT_CTRL
from cereal import messaging
from common.filter_simple import FirstOrderFilter
from common.params import Params

MAX_TIME_ONROAD = 5 * 60.
MOVEMENT_TIME = 1. * 60  # normal time allowed is one minute
OFFROAD_TIME = 1. * 30  # needs to be offroad for this time before sentry mode is active


class SentryMode:
  def __init__(self):
    self.sm = messaging.SubMaster(['deviceState', 'sensorEvents'], poll=['sensorEvents'])
    self.pm = messaging.PubMaster(['sentryState'])

    self.params = Params()
    self.sentry_enabled = self.params.get_bool("SentryMode")
    self.last_read_ts = sec_since_boot()

    self.prev_accel = np.zeros(3)
    self.initialized = False
    self.started = False
    self.prev_started = False
    self.started_ts = 0.
    self.movement_ts = 0.
    self.accel_filters = [FirstOrderFilter(0, 0.5, DT_CTRL) for _ in range(3)]

  def update(self):
    # Update parameter
    now_ts = sec_since_boot()
    if now_ts - self.last_read_ts > 30.:
      self.sentry_enabled = self.params.get_bool("SentryMode")
      self.last_read_ts = float(now_ts)

    for sensor in self.sm['sensorEvents']:
      if sensor.which() == 'acceleration':
        accels = sensor.acceleration.v
        if len(accels) == 3:  # sometimes is empty, in that case don't update
          if self.initialized:
            for idx, v in enumerate(accels):
              self.accel_filters[idx].update(accels[idx] - self.prev_accel[idx])
          self.initialized = True
          self.prev_accel = list(accels)

    self.started = True  # self.get_started(now_ts)
    print(f"{self.started=}")

    if self.started and not self.prev_started:
      self.started_ts = sec_since_boot()

    self.prev_started = self.started

  def get_started(self, now_ts):
    offroad = not self.sm['deviceState'].started
    offroad_long_enough = now_ts - (self.sm['deviceState'].offMonoTime / 1e9) > OFFROAD_TIME  # needs to be offroad for 30 sec

    movement = any([abs(a_filter.x) > .01 for a_filter in self.accel_filters])
    if movement:
      self.movement_ts = float(now_ts)
    # print([a_filter.x for a_filter in self.accel_filters])

    # Maximum allowed time onroad without movement is 1 minute. Any movement resets time allowed, maximum time is 5 minutes
    # TODO: can remove started_ts if time is the same
    onroad_long_enough = (now_ts - self.started_ts > MOVEMENT_TIME and now_ts - self.movement_ts > MOVEMENT_TIME) or now_ts - self.started_ts > MAX_TIME_ONROAD

    started = False
    print(f"{offroad=}, {offroad_long_enough=}, {movement=}")
    print(f"{onroad_long_enough=}")
    print(f"{now_ts - self.started_ts=}")
    if offroad and self.sentry_enabled:  # car's ignitions needs to be off (not started by user)
      if offroad_long_enough and movement:
        started = True
      elif self.started and not onroad_long_enough:
        started = True

    return started

  def publish(self):
    sentry_state = messaging.new_message('sentryState')
    sentry_state.sentryState.started = self.started

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
