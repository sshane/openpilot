from cereal.messaging import SubMaster

sm = SubMaster(['laneSpeed'])

while True:
  sm.update(0)
  print(sm['laneSpeed'].leftAvgSpeed)
  print(sm['laneSpeed'].middleAvgSpeed)
  print(sm['laneSpeed'].rightAvgSpeed)
  print('---')