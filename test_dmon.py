import time
from cereal.messaging import SubMaster
from common.params import Params


if __name__ == "__main__":
  params = Params()
  params.put_bool("FakeIgnition", False)

  sm = SubMaster(["driverStateV2"])


  while 1:
    params.put_bool("FakeIgnition", True)
    sm.update(0)
    dmon_frame = 0  # sm.rcv_frame['driverStateV2']

    print('Waiting for driverStateV2')
    st = time.monotonic()
    timeout = 15  # s
    while (sm.rcv_frame['driverStateV2'] - dmon_frame) < 5 * 20 and time.monotonic() - st < timeout:
      sm.update(0)
      time.sleep(0.05)

    if sm.rcv_frame['driverStateV2'] == dmon_frame:
      print('WARNING: never saw frame from driverStateV2 in 15 seconds', sm.rcv_frame['driverStateV2'], dmon_frame)
    else:
      print('Got driverStateV2! Exiting', sm.rcv_frame['driverStateV2'], dmon_frame)
      time.sleep(1)

    params.put_bool("FakeIgnition", False)
    time.sleep(5)

