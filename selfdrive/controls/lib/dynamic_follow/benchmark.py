from selfdrive.controls.lib.dynamic_follow.auto_df import predict
import time
import numpy as np

samples = np.random.rand(2000, 800).astype(np.float32)

t = time.time()
for sample in samples:
  predict(sample)

print(time.time() - t)
print(predict(samples[0]).dtype)