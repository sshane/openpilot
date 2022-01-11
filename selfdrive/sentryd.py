#!/usr/bin/env python3
import time
import numpy as np

from common.realtime import sec_since_boot, DT_CTRL
from cereal import log, messaging
from common.filter_simple import FirstOrderFilter


class SentryMode:
  def __init__(self):
    self.sm = messaging.SubMaster(['deviceState', 'sensorEvents'], poll=['sensorEvents'])

    self.prev_accel = np.zeros(3)
    self.initialized = False
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
    self.started

  @property
  def started(self):
    now_ts = sec_since_boot()
    started = not self.sm['deviceState'].started
    started = started and (now_ts - self.sm['deviceState'].offMonoTime) > 30.
    started = started or self.sm['deviceState'].startedSentry

    movement = any([abs(a_filter.x) > .01 for a_filter in self.accel_filters])
    print(movement)
    print([a_filter.x for a_filter in self.accel_filters])
    started = started and movement

    return started

  def start(self):
    while 1:
      self.sm.update()
      self.update()


def main():
  sentry_mode = SentryMode()
  sentry_mode.start()


if __name__ == "__main__":
  main()
