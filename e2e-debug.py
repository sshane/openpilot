import time
import cereal.messaging as messaging

sm = messaging.SubMaster(['modelV2'])
print(sm['modelV2'].position.t)

while True:
  sm.update(0)
  print(sm['modelV2'].velocity.x[0] * 2.2369)

