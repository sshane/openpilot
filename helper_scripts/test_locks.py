from threading import Thread
import time
from common.op_params import opParams

op_params = opParams()


def thread_test(op_p):
  for _ in range(1000):
    time.sleep(0.01)
    op_p.put('test_param', 15)


threads = 10
for i in range(threads):
  Thread(target=thread_test, args=(op_params,)).start()

print('started all threads...')
