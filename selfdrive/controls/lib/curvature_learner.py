from numpy import clip
import json
import csv
import os


# HOW TO
# import this module to where you want to use it, such as from ```selfdrive.controls.lib.curvature_learner import CurvatureLearner```
# create the object ```self.curvature_offset = CurvatureLearner(debug=False)```
# call the update method ```self.curvature_offset.update(angle_steers - angle_offset, self.LP.d_poly)```
# The learned curvature offsets will save and load automatically
# If you still need help, check out how I have it implemented in the devel_curvaturefactorlearner branch
# by Zorrobyte
# version 4
# Credit: https://github.com/zorrobyte/openpilot


class CurvatureLearner:
  def __init__(self, debug=False):
    self.offset = 0.
    self.learning_rate = 12000
    self.frame = 0
    self.debug = debug
    self.curvature_file = '/data/curvaturev4.json'
    self.initialize()

  def initialize(self):
    try:
      with open(self.curvature_file, "r") as f:
        self.learned_offsets = json.load(f)
    except (OSError, IOError):
      self.learned_offsets = {
        "center": 0.,
        "inner": 0.,
        "outer": 0.
      }
      self.write_data()

  def update(self, angle_steers=0., d_poly=None, v_ego=0.):
    if angle_steers > 0.1:
      if abs(angle_steers) < 2.:
        self.learned_offsets["center"] -= d_poly[3] / self.learning_rate
        self.offset = self.learned_offsets["center"]
      elif 2. < abs(angle_steers) < 5.:
        self.learned_offsets["inner"] -= d_poly[3] / self.learning_rate
        self.offset = self.learned_offsets["inner"]
      elif abs(angle_steers) > 5.:
        self.learned_offsets["outer"] -= d_poly[3] / self.learning_rate
        self.offset = self.learned_offsets["outer"]
    elif angle_steers < -0.1:
      if abs(angle_steers) < 2.:
        self.learned_offsets["center"] += d_poly[3] / self.learning_rate
        self.offset = self.learned_offsets["center"]
      elif 2. < abs(angle_steers) < 5.:
        self.learned_offsets["inner"] += d_poly[3] / self.learning_rate
        self.offset = self.learned_offsets["inner"]
      elif abs(angle_steers) > 5.:
        self.learned_offsets["outer"] += d_poly[3] / self.learning_rate
        self.offset = self.learned_offsets["outer"]

    self.offset = clip(self.offset, -0.3, 0.3)
    self.frame += 1

    if self.frame == 12000:  # every 2 mins
      self.write_data()
      self.frame = 0
    if self.debug:
      with open('/data/curvdebug.csv', 'a') as csv_file:
        csv_file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_file_writer.writerow([self.learned_offsets, v_ego])
    return self.offset

  def write_data(self):
    with open(self.curvature_file, "w") as f:
      json.dump(self.learned_offsets, f)
    os.chmod(self.curvature_file, 0o777)
