# from common.op_params import opParams
from selfdrive.config import Conversions as CV
from common.numpy_fast import clip, interp
import numpy as np


class Lane:
  def __init__(self, name, pos):
    self.name = name
    self.pos = pos
    self.tracks = []

  def reset(self):
    self.tracks = []

  def add(self, track):
    self.tracks.append(track)


class LaneSpeed:
  def __init__(self):
    # self.op_params = opParams()
    self.use_lane_speed = True  # self.op_params.get('use_lane_speed', default=True)

    self.lane_width = 3.7  # in meters todo: update this based on what openpilot sees/current lane width

    self.lane_positions = [-self.lane_width, 0, self.lane_width]  # lateral position in meters from center of car to center of lane
    self.lane_names = ['left', 'middle', 'right']
    self.lanes = [Lane(name, pos) for name, pos in zip(self.lane_names, self.lane_positions)]

  def update(self, v_ego, lead, steer_angle, live_tracks):
    self.v_ego = v_ego
    self.lead = lead
    self.steer_angle = steer_angle
    self.live_tracks = live_tracks['tracks']
    self.reset_lanes()
    self.group_tracks()
    self.debug()

  def debug(self):
    for lane in self.lanes:
      print('Lane: {}'.format(lane.name))
      for tracks in lane.tracks:
        print(tracks)
      print()

  def reset_lanes(self):
    for lane in self.lanes:
      lane.reset()

  def group_tracks(self):
    """Groups tracks based on lateral position and lane width"""
    for track in self.live_tracks:
      yRel = track['yRel']
      lane_diffs = [{'diff': abs(lane.pos - yRel), 'lane': lane} for lane in self.lanes]
      closest_lane = min(lane_diffs, key=lambda x: x['diff'])
      closest_lane['lane'].add(track)


ls = LaneSpeed()
trks = {'tracks': [{'status': 0.0, 'yRel': 2.5999999046325684, 'dRel': 12.319999694824219, 'vRel': -4.525000095367432, 'trackID': 647, 'aRel': 1.0, 'stationary': False, 'oncoming': False}, {'status': 0.0, 'yRel': 2.0399999618530273, 'dRel': 12.119999885559082, 'vRel': -4.375, 'trackID': 650, 'aRel': 9.0, 'stationary': False, 'oncoming': False}, {'status': 0.0, 'yRel': 2.8399999141693115, 'dRel': 51.36000061035156, 'vRel': -4.949999809265137, 'trackID': 652, 'aRel': 12.0, 'stationary': False, 'oncoming': False}, {'status': 0.0, 'yRel': 4.440000057220459, 'dRel': 22.479999542236328, 'vRel': -4.974999904632568, 'trackID': 653, 'aRel': 5.0, 'stationary': False, 'oncoming': False}, {'status': 0.0, 'yRel': 5.440000057220459, 'dRel': 22.479999542236328, 'vRel': -4.900000095367432, 'trackID': 654, 'aRel': 12.0, 'stationary': False, 'oncoming': False}, {'status': 0.0, 'yRel': 6.800000190734863, 'dRel': 28.360000610351562, 'vRel': -4.875, 'trackID': 655, 'aRel': 13.0, 'stationary': False, 'oncoming': False}, {'status': 0.0, 'yRel': 2.2799999713897705, 'dRel': 51.439998626708984, 'vRel': -5.025000095367432, 'trackID': 656, 'aRel': 11.0, 'stationary': False, 'oncoming': False}, {'status': 0.0, 'yRel': -4.71999979019165, 'dRel': 51.84000015258789, 'vRel': -4.900000095367432, 'trackID': 657, 'aRel': 17.0, 'stationary': False, 'oncoming': False}, {'status': 0.0, 'yRel': 2.2799999713897705, 'dRel': 12.199999809265137, 'vRel': -4.300000190734863, 'trackID': 476, 'aRel': 0.0, 'stationary': False, 'oncoming': False}], 'live': True}
ls.update(10, 10, 0, trks)
