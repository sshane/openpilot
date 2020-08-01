import sys
import ast
import time
from cereal.messaging import SubMaster

sm = SubMaster(['logMessage'])

last = ''
while 1:
  try:
    msg = ast.literal_eval(sm['logMessage'])['msg']
  except:
    continue
  while msg == last:
    sm.update(0)
  last = str(msg)
  print(last)
# print("PRINT")