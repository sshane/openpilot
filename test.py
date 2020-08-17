# from cereal.messaging import PubMaster, new_message, SubMaster
from selfdrive.config import Conversions as CV
# from cereal import car
import numpy as np
import matplotlib.pyplot as plt

# ret = car.CarParams.new_message()

# ret.lateralTuning.init('pid')
# ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]

# print('which: {}'.format(ret.lateralTuning.which()))
kdBP = [40 * CV.MPH_TO_MS, 60 * CV.MPH_TO_MS, 78.2928 * CV.MPH_TO_MS]
kdV = [0.045, 0.0725, 0.085]

plt.scatter(kdBP, kdV, c='r', label='known good points')
plt.legend()
plt.xlim(0, 35)
plt.ylim(0, .1)

kdBP_new = [0, 6.7, 30]
kdV_new = [0.005, 0.016, 0.08]
plt.plot(kdBP_new, kdV_new, 'bo--', label='new')
plt.legend()
plt.show()

