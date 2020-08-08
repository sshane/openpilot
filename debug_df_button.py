import time
import cereal.messaging as messaging

sm = messaging.SubMaster(['dynamicFollowButton', 'dynamicFollowData'])
while 1:
  sm.update(0)
  print('status: {}'.format(sm['dynamicFollowButton'].status))
  time.sleep(1)
