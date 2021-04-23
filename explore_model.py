import cereal.messaging as messaging


sm = messaging.SubMaster(['modelV2'])

while 1:
  sm.update(0)

  print('left lane line:')
  print(list(sm['modelV2'].lanelines[1].y))

  print('right lane line:')
  print(list(sm['modelV2'].lanelines[2].y))

  input()

