import time
import cereal.messaging as messaging

sm = messaging.SubMaster(['smiskolData'])
while True:
  sm.update(0)
  print(sm['smiskolData'].dfButtonStatus)
  time.sleep(1)