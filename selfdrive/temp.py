import sys
import time
from cereal.messaging import SubMaster

sm = SubMaster(['logMessage'])

while 1:
  sm.update(0)
  print(sm['logMessage'])
  time.sleep(1)
# print("PRINT")