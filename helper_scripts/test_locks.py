from threading import Thread
import time
from multiprocessing import Process, Queue
from common.op_params import opParams
import numpy as np

op_params = opParams()


def thread_test_write(op_p):
  for _ in range(500):
    time.sleep(0.01)
    op_p.write_db('camera_offset', np.random.randint(0, 100))


def thread_test_read(op_p):
  for _ in range(500):
    time.sleep(0.01)
    p = op_p.get('camera_offset')
    if p is None:
      print('error reading!')


# queue = Queue()

for i in range(2):
  # Thread(target=thread_test_write, args=(op_params,)).start()
  Process(target=thread_test_write, args=(op_params,)).start()

for i in range(8):
  # Thread(target=thread_test_read, args=(op_params,)).start()
  Process(target=thread_test_read, args=(op_params,)).start()
print('started all threads...')
