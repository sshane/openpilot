from threading import Thread
import time
from common.op_params import opParams
import numpy as np

op_params = opParams()


def thread_test_write(op_p):
  for _ in range(1000):
    time.sleep(0.01)
    op_p.put('test_param', np.random.randint(0, 100))


def thread_test_read(op_p):
  for _ in range(1000):
    time.sleep(0.01)
    if op_p.get('test_param') is None:
      print('error reading!')


threads = 10
for i in range(threads):
  Thread(target=thread_test_write, args=(op_params,)).start()
  Thread(target=thread_test_read, args=(op_params,)).start()

print('started all threads...')
