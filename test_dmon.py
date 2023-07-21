import time
import os
from cereal.messaging import SubMaster
from common.realtime import DT_MDL
from common.params import Params


def all_dead(managerState):
  # return if all processes that should be dead are dead
  return all([not p.running or p.shouldBeRunning for p in managerState.processes])


if __name__ == "__main__":
  params = Params()
  params.put_bool("FakeIgnition", False)

  sm = SubMaster(["driverStateV2", "managerState", "deviceState", "navModel"])
  occurrences = 0
  loops = 0

  while 1:
    params.put_bool("FakeIgnition", True)
    sm.update(0)
    dmon_frame = None
    navmodel_frame = None

    # print('Waiting for driverStateV2')
    st = time.monotonic()
    timeout = 30  # s

    # successful if we get 100 messages from dmonitoringmodeld (2s)
    can_break = {'driverStateV2': False, 'navModel': False}
    while time.monotonic() - st < timeout:
      sm.update(0)
      time.sleep(DT_MDL)
      if sm.updated["driverStateV2"]:
        if dmon_frame is None:
          dmon_frame = sm.rcv_frame['driverStateV2']

        if (sm.rcv_frame['driverStateV2'] - dmon_frame) > (2 / DT_MDL):
          can_break['driverStateV2'] = True

      if sm.updated["navModel"]:
        if navmodel_frame is None:
          navmodel_frame = sm.rcv_frame['navModel']

        if (sm.rcv_frame['navModel'] - navmodel_frame) > (2 / DT_MDL):
          can_break['navModel'] = True

      # can break if both navmodeld and dmonitoringmodeld started with enough frames
      if all(can_break.values()):
        print('Got navModel and driverStateV2! Exiting', sm.rcv_frame['driverStateV2'],
              dmon_frame, sm.rcv_frame['navModel'], navmodel_frame)
        time.sleep(1)
        break

    else:
      occurrences += 1
      print(f'WARNING: timed out in 15s waiting for 40 messages from both procs, occurrences: {occurrences}, '
            f'got driverStateV2: {can_break["driverStateV2"]}, got navModel: {can_break["navModel"]}, '
            f'driverStateV2 frames: {(sm.rcv_frame["driverStateV2"], dmon_frame)}, navModel frames: {(sm.rcv_frame["navModel"], navmodel_frame)}')
      print('CurrentRoute:', params.get('CurrentRoute'))

      if os.path.exists('/data/hang_dmon'):
        print('Hanging')
        while 1:
          time.sleep(1)

    # TODO: is there a better way? we can't check managerState immediately since it takes a while to get the ignition
    # wait for thermald to pick up ignition, then an update from managerState, and THEN it should be safe to check procs
    params.put_bool("FakeIgnition", False)
    while sm['deviceState'].started:
      sm.update(0)
      time.sleep(0.05)

    while not sm.updated['managerState']:
      sm.update(0)
      time.sleep(0.05)

    st = time.monotonic()
    while time.monotonic() - st < timeout:
      sm.update(0)
      time.sleep(0.1)
      if all_dead(sm['managerState']):
        # print('all dead')
        break
    else:
      print('WARNING: timed out waiting for processes to die!', time.monotonic() - st)
      time.sleep(5)

    loops += 1
    if loops % 120 == 0:
      print('Tries so far:', loops, 'occurrences:', occurrences)
