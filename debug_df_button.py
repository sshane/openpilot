import time
import cereal.messaging as messaging

sm = messaging.SubMaster(['dynamicFollowButton', 'dynamicFollowData'])
while 1:
  sm.update(0)
  print('status: {}'.format(sm['dynamicFollowButton'].status))
  print('updated: {}'.format(sm.updated['dynamicFollowButton']))
  time.sleep(1)
