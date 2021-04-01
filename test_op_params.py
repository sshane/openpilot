import time
import random
from common.op_params import opParams


op = opParams()

profiles = ['traffic', 'relaxed', 'stock', 'auto']
i = 0

while 1:
  time.sleep(random.uniform(.01, .05))
  v = profiles[i]
  i += 1
  op.put('dynamic_follow', v, old=True)
  print('Put dynamic_follow with {}'.format(v), flush=True)

