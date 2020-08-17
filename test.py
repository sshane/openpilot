from cereal.messaging import PubMaster, new_message, SubMaster
from cereal import car

ret = car.CarParams.new_message()

ret.lateralTuning.init('pid')
ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]

print('which: {}'.format(ret.lateralTuning.which()))
ret.lateralTuning.init('lqr')
print('which: {}'.format(ret.lateralTuning.which()))
