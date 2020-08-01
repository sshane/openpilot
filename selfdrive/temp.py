import sys
import ast
import time
from cereal.messaging import SubMaster

sm = SubMaster(['logMessage'])

last = ''
while 1:
  msg = ast.literal_eval(sm['logMessage'])['msg']
  while msg == last:
    sm.update(0)
  last = str(msg)
  print(last)
# print("PRINT")