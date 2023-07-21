import os
import random
import time
from selfdrive.manager.process_config import managed_processes

PROCS = ["dmonitoringmodeld", "navmodeld"]
INIT_PARAMS = ["DmModelInit", "NavModelInit"]
BASE_DIR = '/data/tmp/'


def get_init_params(_random_str):
  return [os.path.exists(BASE_DIR + _p + _random_str) for _p in INIT_PARAMS]


def kill_procs(_procs):
  for proc in PROCS:
    print('Stopping', proc)
    managed_processes[proc].stop()


def get_random_str():
  return str(random.randint(1, int(1e16)))


if __name__ == "__main__":
  # params = Params()

  random_str = get_random_str()

  try:
    while 1:
      print('\nStarting')
      for p in INIT_PARAMS:
        try:
          os.remove(BASE_DIR + p + random_str)
        except:
          ...

      st = time.monotonic()
      procs = PROCS.copy()
      random.shuffle(procs)
      random_str = get_random_str()
      os.environ['PARAM_SUFFIX'] = random_str
      for proc in procs:
        managed_processes[proc].start()
        time.sleep(random.uniform(0.0, 1.0))  # 0 to 1000 ms offset

      while time.monotonic() - st < 10:
        print('not alive')
        time.sleep(0.1)
        if all(get_init_params(random_str)):
          print('both done init')
          time.sleep(1)
          break
      else:
        pd = dict(zip(INIT_PARAMS, get_init_params(random_str)))
        print('WARNING: timed out waiting for model procs to initialize!', pd)
        while 1:
          time.sleep(1)

      kill_procs(procs)

  finally:
    kill_procs(PROCS)
