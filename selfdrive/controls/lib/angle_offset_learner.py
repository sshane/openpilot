import os
import json
from numpy import clip
from common.realtime import sec_since_boot
from selfdrive.config import Conversions as CV


# CurvatureLearner by Zorrobyte (version 4)
# modified to add speed and curve direction as learning factors and change to angle offset learner for lane hugging
# version 5 due to json incompatibilities

class AngleOffsetLearner:
  def __init__(self):
    self.offset_file = '/data/angle_offset_v5.json'
    rate = 1 / 20.  # pathplanner is 20 hz
    self.learning_rate = 7.5e-3 * rate
    self.write_frequency = 5  # in seconds

    self.directions = ['left', 'right']
    self.speed_bands = ['slow', 'medium', 'fast']
    self.angle_bands = ['center', 'inner', 'outer']
    self._load_offsets()

  def update(self, angle_steers, d_poly, v_ego):
    offset = 0
    angle_band, direction = self.pick_angle_band(angle_steers)

    if angle_band is not None:  # don't return an offset if not between a band
      speed_band = self.pick_speed_band(v_ego)  # will never be none
      learning_sign = 1 if angle_steers >= 0 else -1
      self.learned_offsets[direction][angle_band][speed_band] -= d_poly[3] * self.learning_rate * learning_sign  # the learning
      offset = self.learned_offsets[direction][angle_band][speed_band]

    if sec_since_boot() - self._last_write_time >= self.write_frequency:
      self._write_offsets()
    return clip(offset, -3, 3)

  def pick_speed_band(self, v_ego):
    if v_ego <= 30 * CV.MPH_TO_MS:
      return 'slow'
    if v_ego <= 55 * CV.MPH_TO_MS:
      return 'medium'
    return 'fast'

  def pick_angle_band(self, angle_steers):
    direction = 'left' if angle_steers > 0 else 'right'
    if abs(angle_steers) >= 0.1:
      if abs(angle_steers) < 2:  # between +=[.1, 2)
        return 'center', direction
      if abs(angle_steers) < 5.:  # between +=[2, 5)
        return 'inner', direction
      return 'outer', direction  # between +=[5, inf)
    return None, direction  # return none when below +-0.1, removes possibility of returning offset in this case

  def _load_offsets(self):
    self._last_write_time = 0
    try:
      with open(self.offset_file, 'r') as f:
        self.learned_offsets = json.load(f)
      return
    except:  # can't read file or doesn't exist
      self.learned_offsets = {d: {s: {a: 0 for a in self.angle_bands} for s in self.speed_bands} for d in self.directions}
      self._write_offsets()  # rewrite/create new file

  def _write_offsets(self):
    with open(self.offset_file, 'w') as f:
      f.write(json.dumps(self.learned_offsets, indent=2))
    os.chmod(self.offset_file, 0o777)
    self._last_write_time = sec_since_boot()
