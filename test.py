from cereal.messaging import PubMaster, new_message, SubMaster
from selfdrive.config import Conversions as CV
from cereal import car

ret = car.CarParams.new_message()

ret.lateralTuning.init('pid')
ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]

print('which: {}'.format(ret.lateralTuning.which()))
ret.lateralTuning.pid.kdBP = [40 * CV.MPH_TO_MS, 60 * CV.MPH_TO_MS, 78.2928 * CV.MPH_TO_MS]
ret.lateralTuning.pid.kdV = [0.045, 0.075, 0.085]
print(ret.lateralTuning.pid.kdV)
print(ret.lateralTuning.pid.kdBP)
