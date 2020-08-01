import sys
import ast
import time
from cereal.messaging import SubMaster

sm = SubMaster(['logMessage'])

last = ''
msg = ''
while 1:
  while msg == last:
    try:
      msg = ast.literal_eval(sm['logMessage'])['msg']
    except:
      continue
    sm.update(0)
  last = str(msg)
  print(last)
# print("PRINT")