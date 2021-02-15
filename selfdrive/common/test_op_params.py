from common.op_params import opParams
import threading
import time
import random

threads = 5

def start():
  op_params = opParams()
  time.sleep(random.uniform(0.00001, 0.0001))
  op_params.get('camera_offset')
  op_params.put('camera_offset', 0.1)
  op_params.get('camera_offset')
  return op_params


for i in range(threads):
  threading.Thread(target=start).start()
