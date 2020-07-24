from cereal.messaging import SubMaster
sm = SubMaster(['modelLongButton'])
while 1:
  input()
  sm.update(0)
  print(sm['modelLongButton'].enabled)
