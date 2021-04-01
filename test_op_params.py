import time
import random
from common.op_params import opParams


op = opParams()

profiles = ['traffic', 'relaxed', 'stock', 'auto']

while 1:
  time.sleep(random.uniform(.25, 1))
  v = random.choice(profiles)
  op.put('dynamic_follow', v, old=True)
  print('Put dynamic_follow with {}'.format(v), flush=True)

