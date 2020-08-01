from common.realtime import sec_since_boot
from numpy import clip
from common.numpy_fast import clip as clip_fast


t = sec_since_boot()
for _ in range(100000):
  clip(3.2, -3, 3)
print(sec_since_boot() - t)

t = sec_since_boot()
for _ in range(100000):
  clip_fast(3.2, -3, 3)
print(sec_since_boot() - t)
