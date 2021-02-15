from common.op_params import opParams
import threading

threads = 10

def start():
  op_params = opParams()
  op_params.get('camera_offset')
  # op_params.put('camera_offset', 0.1)
  op_params.get('camera_offset')
  return op_params


for i in range(threads):
  threading.Thread(target=start).start()
