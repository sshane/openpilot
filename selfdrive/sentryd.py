#!/usr/bin/env python3
import time
import numpy as np

from common.realtime import sec_since_boot, DT_CTRL
from cereal import log, messaging
from common.filter_simple import FirstOrderFilter


class SentryMode:
  def __init__(self):
    self.sm = messaging.SubMaster(['deviceState', 'sensorEvents'], poll=['sensorEvents'])
    self.pm = messaging.PubMaster(['sentryState'])

    self.prev_accel = np.zeros(3)
    self.initialized = False
    self.started = False
    self.prev_started = False
    self.started_ts = 0.
    self.accel_filters = [FirstOrderFilter(0, 0.5, DT_CTRL) for _ in range(3)]

  def update(self):
    for sensor in self.sm['sensorEvents']:
      if sensor.which() == 'acceleration':
        accels = sensor.acceleration.v
        if len(accels) == 3:  # sometimes is empty, in that case don't update
          if self.initialized:
            for idx, v in enumerate(accels):
              self.accel_filters[idx].update(accels[idx] - self.prev_accel[idx])
          self.initialized = True
          self.prev_accel = list(accels)

    self.started = self.get_started()
    print(self.started)

    if self.started and not self.prev_started:
      self.started_ts = sec_since_boot()

    self.prev_started = self.started

  def get_started(self):
    now_ts = sec_since_boot()
    offroad = not self.sm['deviceState'].started
    offroad_long_enough = now_ts - self.sm['deviceState'].offMonoTime > 5.  # needs to be offroad for 30 sec
    print(now_ts - self.sm['deviceState'].offMonoTime, self.sm['deviceState'].offMonoTime)

    movement = any([abs(a_filter.x) > .01 for a_filter in self.accel_filters])
    print([a_filter.x for a_filter in self.accel_filters])

    onroad_long_enough = self.started and (now_ts - self.started_ts)

    started = False
    print(f"{offroad=}, {offroad_long_enough=}, {movement=}")
    print(f"{onroad_long_enough=}")
    if offroad and offroad_long_enough and movement:
      started = True
    elif self.started and not onroad_long_enough:
      started = True

    return started

  def publish(self):
    sentry_state = messaging.new_message('sentryState')
    sentry_state.sentryState.started = self.started

    self.pm.send('lateralPlan', sentry_state)

  def start(self):
    while 1:
      self.sm.update()
      self.update()


def main():
  sentry_mode = SentryMode()
  sentry_mode.start()


if __name__ == "__main__":
  main()
