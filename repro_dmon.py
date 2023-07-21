import random
import time
from selfdrive.manager.process_config import managed_processes
from common.params import Params

INIT_PARAMS = ["DmModelInit", "NavModelInit"]
PROCS = ["dmonitoringmodeld", "navmodeld"]


def get_init_params(_params):
  return [_params.get_bool(_p) for _p in INIT_PARAMS]


if __name__ == "__main__":
  params = Params()

  while 1:
    print('\nStarting')
    for p in INIT_PARAMS:
      params.put_bool(p, False)

    st = time.monotonic()
    procs = PROCS.copy()
    random.shuffle(procs)
    for proc in procs:
      managed_processes[proc].start()
      time.sleep(random.uniform(0.0, 1.0))  # 0 to 1000 ms offset

    while time.monotonic() - st < 10:
      print('not alive')
      time.sleep(0.1)
      if all(get_init_params(params)):
        print('both done init')
        time.sleep(1)
        break
    else:
      pd = dict(zip(INIT_PARAMS, get_init_params(params)))
      print('WARNING: timed out waiting for model procs to initialize!', pd)
      while 1:
        time.sleep(1)

    for proc in procs:
      print('Stopping', proc)
      managed_processes[proc].stop()
