import sys
import time
from cereal.messaging import SubMaster

sm = SubMaster(['logMessage'])

last = ''
while 1:
  while sm['logMessage'] == last:
    sm.update(0)
  last = sm['logMessage']
  print(last)
# print("PRINT")