import time
import cereal.messaging as messaging

sm = messaging.SubMaster(['dynamicFollowButton'])
while True:
  sm.update(0)
  print(sm['dynamicFollowButton'].status)
  time.sleep(1)