import sys
import ast
import json
import time
from cereal.messaging import SubMaster

sm = SubMaster(['logMessage'])

last = ''
msg = ''
while 1:
  while msg == last:
    sm.update(0)
    try:
      msg = json.loads(sm['logMessage'])['msg']
    except:
      continue
  last = str(msg)
  print(last)
# print("PRINT")