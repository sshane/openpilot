import time
import cereal.messaging as messaging
import json

sm = messaging.SubMaster(['modelV2'])
print(sm['modelV2'].position.t)

v = []
try:
  while True:
    sm.update(0)
    time.sleep(1/3)
    if len(sm['modelV2'].velocity.x) == 0:
      continue
    v.append(list(sm['modelV2'].velocity.x))
    print(v[-1])
except:
  pass

with open('/data/v', 'w') as f:
  json.dump(v, f)

class ModelMpcHelper:
  def __init__(self):
    self.model_t = [i ** 2 / 102.4 for i in range(33)]  # the timesteps of the model predictions
    self.mpc_t = list(range(10))  # the timesteps of what the LongMpcModel class takes in, 1 sec intervels to 10
    self.model_t_idx = [sorted(range(len(self.model_t)), key=[abs(idx - t) for t in self.model_t].__getitem__)[0] for idx in self.mpc_t]  # matches 0 to 9 interval to idx from t

  def convert_data(self, modelV2):
    distances, speeds, accelerations = [], [], []
    if not sm.updated['modelV2'] or len(modelV2.position.x) == 0:
      return distances, speeds, accelerations

    for t in self.model_t_idx:
      speeds.append(modelV2.velocity.x[t])
      distances.append(modelV2.position.x[t])
      if self.model_t_idx.index(t) > 0:  # skip first since we can't calculate (and don't want to use v_ego)
        accelerations.append((speeds[-1] - speeds[-2]) / self.model_t[t])

    accelerations.insert(0, accelerations[1] - (accelerations[2] - accelerations[1]))  # extrapolate back first accel from second and third, less weight
    return distances, speeds, accelerations
