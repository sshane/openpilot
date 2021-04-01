import time
import random
from common.op_params import opParams


op = opParams()

profiles = ['traffic', 'relaxed', 'stock', 'auto']
i = 0

while 1:
  time.sleep(random.uniform(.001, .01))
  v = profiles[i % 4]
  i += 1
  op.put('dynamic_follow', v, old=False)
  print('Put dynamic_follow with {}'.format(v), flush=True)

