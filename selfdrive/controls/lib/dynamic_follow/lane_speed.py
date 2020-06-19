# from common.op_params import opParams
from selfdrive.config import Conversions as CV
from common.numpy_fast import clip, interp
import numpy as np
import matplotlib.pyplot as plt


class Lane:
  def __init__(self, name, pos):
    self.name = name
    self.pos = pos
    self.tracks = []

    self.fastest_count = False

  def fastest(self):
    """Increments this lane's fast count"""
    if self.fastest_count is False:
      self.fastest_count = 0
    else:
      self.fastest_count += 1

  def reset(self):
    self.tracks = []

  def reset_fastest(self):
    self.fastest_count = False

  def add(self, track):
    self.tracks.append(track)


class LaneSpeed:
  def __init__(self):
    # self.op_params = opParams()
    self.use_lane_speed = True  # self.op_params.get('use_lane_speed', default=True)

    self.lane_width = 3.7  # in meters todo: update this based on what openpilot sees/current lane width
    self.track_speed_margin = 0.15  # track has to be above X% of v_ego (excludes oncoming)
    self.faster_than_margin = 0.05  # avg of secondary lane has to be faster by X% to show alert
    self.min_fastest_time = 0.5 * 100  # how long should we wait for a specific lane to be faster than middle before alerting; 100 is 1 second
    self.max_steer_angle = 40

    self.lane_positions = [-self.lane_width, 0, self.lane_width]  # lateral position in meters from center of car to center of lane
    self.lane_names = ['left', 'middle', 'right']

    self.lanes = [Lane(name, pos) for name, pos in zip(self.lane_names, self.lane_positions)]

  def update(self, v_ego, lead, steer_angle, d_poly, live_tracks):
    # print('steer angle: {}'.format(steer_angle))
    self.v_ego = v_ego
    self.lead = lead
    self.steer_angle = steer_angle
    self.d_poly = d_poly
    self.live_tracks = live_tracks

    self.reset_lanes()
    if abs(steer_angle) < self.max_steer_angle:
      self.group_tracks()
      # self.debug()
      return self.evaluate_lanes()

  def evaluate_lanes(self):
    avg_lane_speeds = {}
    for lane in self.lanes:
      track_speeds = [track.vRel + self.v_ego for track in lane.tracks]
      track_speeds = [speed for speed in track_speeds if speed > self.v_ego * self.track_speed_margin]
      if len(track_speeds):  # filters out oncoming tracks and very slow tracks
        avg_lane_speeds[lane.name] = np.mean(track_speeds)

    # print('avg_lane_speeds: {}'.format(avg_lane_speeds))
    # print()

    if 'middle' not in avg_lane_speeds or len(avg_lane_speeds) == 0:
      # if no tracks in middle lane or no secondary lane, we have nothing to compare
      return

    # we have a middle and secondary lane to compare
    middle_speed = avg_lane_speeds['middle']
    fastest_name = max(avg_lane_speeds, key=lambda x: avg_lane_speeds[x])
    fastest_speed = avg_lane_speeds[fastest_name]

    # print('middle: {}'.format(middle_speed))
    # print('fastest: {}'.format(fastest_speed))

    if fastest_name == 'middle':  # already in fastest lane
      return

    # print('Fastest lane is {} at an average of {} m/s faster'.format(fastest, fastest_speed - middle_speed))
    fastest_percent = (fastest_speed / middle_speed) - 1

    if fastest_percent < self.faster_than_margin:  # fastest lane is not above margin, ignore
      # todo: could remove since we wait for a lane to be faster for a bit
      return

    # print('Fastest lane is {}% faster!'.format(round(fastest_percent*100, 2)))
    # if we are here, there's a faster lane available that's above our minimum margin

    self.get_lane(fastest_name).fastest()  # increment fastest lane
    self.get_lane(self.opposite_lane(fastest_name)).reset_fastest()  # reset slowest lane (opposite, never middle)

    if self.get_lane(fastest_name).fastest_count < self.min_fastest_time:
      # fastest lane hasn't been fastest long enough
      return

    self.get_lane(fastest_name).reset_fastest()  # reset once we show alert so we don't continually send same alert
    # if here, we've found a lane faster than our lane by a margin and it's been faster for long enough
    return self.get_lane(fastest_name).name

  def group_tracks(self):
    """Groups tracks based on lateral position and lane width"""
    # todo: factor in steer angle
    for track in self.live_tracks:
      lane_diffs = [{'diff': abs(lane.pos - track.yRel), 'lane': lane} for lane in self.lanes]
      closest_lane = min(lane_diffs, key=lambda x: x['diff'])
      closest_lane['lane'].add(track)

  def get_lane(self, name):
    """Returns lane by name"""
    for lane in self.lanes:
      if lane.name == name:
        return lane

  def opposite_lane(self, name):
    return {'left': 'right', 'right': 'left'}[name]

  def reset_lanes(self):
    for lane in self.lanes:
      lane.reset()

  def debug(self):
    for lane in self.lanes:
      print('Lane: {}'.format(lane.name))
      for track in lane.tracks:
        print(track.vRel, track.yRel, track.dRel)
      print()


DEBUG = False

if DEBUG:
  def circle_y(_x, _angle):  # fixme: not sure if this is correct
    return -(_x * _angle) ** 2 / (1000 * (_angle * 2))

  ls = LaneSpeed()

  class Track:
    def __init__(self, vRel, yRel, dRel):
      self.vRel = vRel
      self.yRel = yRel
      self.dRel = dRel


  keys = ['v_ego', 'a_ego', 'v_lead', 'lead_status', 'x_lead', 'y_lead', 'a_lead', 'a_rel', 'v_lat', 'steer_angle', 'steer_rate', 'track_data', 'time', 'gas', 'brake', 'car_gas', 'left_blinker', 'right_blinker', 'set_speed', 'new_accel', 'gyro']

  sample = [8.013258934020996, 0.14726917445659637, 8.45051383972168, True, 12.680000305175781, 0.19999998807907104, 0.7618321180343628, 0.0, 0.0, -0.30000001192092896, 0.0, {'tracks': [{'trackID': 13482, 'yRel': 0.1599999964237213, 'dRel': 12.680000305175781, 'vRel': 0.4000000059604645, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13652, 'yRel': -0.03999999910593033, 'dRel': 19.360000610351562, 'vRel': 0.5249999761581421, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13690, 'yRel': -0.20000000298023224, 'dRel': 22.639999389648438, 'vRel': 0.25, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13691, 'yRel': 4.440000057220459, 'dRel': 27.520000457763672, 'vRel': 8.824999809265137, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13692, 'yRel': 2.8399999141693115, 'dRel': 36.68000030517578, 'vRel': -5.099999904632568, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13694, 'yRel': 2.9600000381469727, 'dRel': 36.68000030517578, 'vRel': -5.074999809265137, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13698, 'yRel': -1.1200000047683716, 'dRel': 17.040000915527344, 'vRel': 0.32499998807907104, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13700, 'yRel': -0.20000000298023224, 'dRel': 25.31999969482422, 'vRel': 0.699999988079071, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13703, 'yRel': -0.11999999731779099, 'dRel': 19.84000015258789, 'vRel': 0.20000000298023224, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13704, 'yRel': 0.23999999463558197, 'dRel': 12.680000305175781, 'vRel': 0.4749999940395355, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13705, 'yRel': 0.03999999910593033, 'dRel': 25.360000610351562, 'vRel': 0.3499999940395355, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13706, 'yRel': -5.599999904632568, 'dRel': 116.4800033569336, 'vRel': 8.675000190734863, 'stationary': False, 'oncoming': False, 'status': 0.0}], 'live': True}, 1571441322.0375044, 26.91699981689453, 0.0, 0.07500000298023224, False, False, 21.94444465637207, 0.06090415226828825, [0.0047149658203125, -0.039764404296875, 0.029388427734375]]
  sample = dict(zip(keys, sample))
  trks = sample['track_data']['tracks']
  trks = [Track(trk['vRel'], trk['yRel'], trk['dRel']) for trk in trks]

  dRel = [t.dRel for t in trks]
  yRel = [t.yRel for t in trks]
  steerangle = sample['steer_angle']
  plt.scatter(dRel, yRel, label='tracks')
  x_path = np.linspace(0, 130, 100)
  y_path = circle_y(x_path, steerangle)
  plt.plot([0, 130], [0, 0])
  plt.plot(x_path, y_path)
  plt.legend()
  plt.show()

  ls.update(10, None, steerangle, None, trks)  # v_ego, lead, steer_angle, d_poly, live_tracks
  ls.update(10, None, steerangle, None, trks)  # v_ego, lead, steer_angle, d_poly, live_tracks
  ls.update(10, None, steerangle, None, trks)  # v_ego, lead, steer_angle, d_poly, live_tracks
  ls.update(10, None, steerangle, None, trks)  # v_ego, lead, steer_angle, d_poly, live_tracks
  ls.update(10, None, steerangle, None, trks)  # v_ego, lead, steer_angle, d_poly, live_tracks
  out = ls.update(10, None, steerangle, None, trks)  # v_ego, lead, steer_angle, d_poly, live_tracks
  print([(lane.name, lane.fastest_count) for lane in ls.lanes])
  print('out: {}'.format(out))
