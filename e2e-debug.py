import time
import cereal.messaging as messaging

sm = messaging.SubMaster(['modelV2'])

while True:
  sm.update(0)
  print(sm['modelV2'].position.x)

