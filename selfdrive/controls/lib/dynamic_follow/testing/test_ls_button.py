from cereal.messaging import SubMaster

sm = SubMaster(['laneSpeedButton'])

while 1:
  sm.update(0)
  print('status: {}'.format(sm['laneSpeedButton'].status))
  input('press enter')
