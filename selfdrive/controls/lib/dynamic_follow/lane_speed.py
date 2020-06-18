from common.op_params import opParams
from selfdrive.config import Conversions as CV
from common.numpy_fast import clip, interp
import numpy as np


class LaneSpeed:
  def __init__(self):
    self.op_params = opParams()
    self.use_lane_speed = self.op_params.get('use_lane_speed', default=True)

  def update(self, v_ego, lead, live_tracks):
    self.v_ego = v_ego
    self.lead = lead
    self.live_tracks = live_tracks
