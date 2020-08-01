import sys
import time
from cereal.messaging import SubMaster

sm = SubMaster(['logMessage'])

while 1:
  while not sm.updated['logMessage']:
    sm.update(0)
  print(sm['logMessage'])
# print("PRINT")