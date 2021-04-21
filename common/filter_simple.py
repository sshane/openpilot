class FirstOrderFilter():
  # first order filter
  def __init__(self, x0, ts, dt):
    self.k = (dt / ts) / (1. + dt / ts)
    self.x = x0

  def update(self, x):
    self.x = (1. - self.k) * self.x + self.k * x
    return self.x


DT_DMON = 0.1
_DISTRACTED_FILTER_TS = 0.25  # 0.6Hz
filter = FirstOrderFilter(0., _DISTRACTED_FILTER_TS, DT_DMON)

