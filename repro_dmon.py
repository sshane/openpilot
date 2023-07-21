import time
from selfdrive.manager.process_config import managed_processes
from common.params import Params

INIT_PARAMS = ["DmModelInit", "NavModelInit"]


def get_init_params(_params):
  return [_params.get_bool(_p) for _p in INIT_PARAMS]


if __name__ == "__main__":
  params = Params()

  while 1:
    print('Starting')
    for p in INIT_PARAMS:
      params.put_bool(p, False)

    st = time.monotonic()
    managed_processes["dmonitoringmodeld"].start()
    managed_processes["navmodeld"].start()

    while time.monotonic() - st < 10:
      time.sleep(0.1)
      if all(get_init_params(params)):
        break
    else:
      pd = dict(zip(INIT_PARAMS, get_init_params(params)))
      print('WARNING: timed out waiting for model procs to initialize!', pd)

    managed_processes["dmonitoringmodeld"].stop()
    managed_processes["navmodeld"].stop()










