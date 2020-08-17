from cereal.messaging import PubMaster, new_message, SubMaster
from cereal import car

ret = car.CarParams.new_message()

ret.lateralTuning.init('pid')
ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]

print('which: {}'.format(ret.lateralTuning.which()))
print(ret.lateralTuning.pid.kdV)
print(ret.lateralTuning.pid.kdBP)

