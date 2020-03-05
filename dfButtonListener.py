import time
import cereal.messaging as messaging

sm = messaging.SubMaster(['smiskolData'])
while True:
  sm.update(0)
  print(sm['smiskolData'].dfButtonTouched)
  time.sleep(1)