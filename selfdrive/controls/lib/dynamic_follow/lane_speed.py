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

  def reset(self):
    self.tracks = []

  def add(self, track):
    self.tracks.append(track)


class LaneSpeed:
  def __init__(self):
    # self.op_params = opParams()
    self.use_lane_speed = True  # self.op_params.get('use_lane_speed', default=True)

    self.lane_width = 3.7  # in meters todo: update this based on what openpilot sees/current lane width
    self.track_speed_margin = 0.15  # track has to be above X% of v_ego (excludes oncoming)
    self.faster_than_margin = 0.05  # avg of secondary lane has to be faster by X% to show alert

    self.lane_positions = [-self.lane_width, 0, self.lane_width]  # lateral position in meters from center of car to center of lane
    self.lane_names = ['left', 'middle', 'right']

    self.lanes = [Lane(name, pos) for name, pos in zip(self.lane_names, self.lane_positions)]

  def update(self, v_ego, lead, steer_angle, d_poly, live_tracks):
    print('steer angle: {}'.format(steer_angle))
    self.v_ego = v_ego
    self.lead = lead
    self.steer_angle = steer_angle
    self.d_poly = d_poly
    self.live_tracks = live_tracks

    self.reset_lanes()
    if abs(steer_angle) < 10 or True:
      self.group_tracks()
      # self.debug()
      self.evaluate_lanes()
      return 0, 0, 0
    return 0, 0, 0

  def evaluate_lanes(self):
    avg_lane_speeds = {}
    for lane in self.lanes:
      track_speeds = [track.vRel + self.v_ego for track in lane.tracks]
      track_speeds = [speed for speed in track_speeds if speed > self.v_ego * self.track_speed_margin]
      if len(track_speeds):  # filters out oncoming tracks and very slow tracks
        avg_lane_speeds[lane.name] = np.mean(track_speeds)

    print('avg_lane_speeds: {}'.format(avg_lane_speeds))
    print()
    if 'middle' in avg_lane_speeds and len(avg_lane_speeds) > 1:
      # if we have a middle and secondary lane to compare
      middle_speed = avg_lane_speeds['middle']
      fastest = max(avg_lane_speeds, key=lambda x: avg_lane_speeds[x])
      fastest_speed = avg_lane_speeds[fastest]

      print('middle: {}'.format(middle_speed))
      print('fastest: {}'.format(fastest_speed))
      if fastest != 'middle':
        # If not in fastest lane already
        # self.faster_than_margin
        print('Fastest lane is {} at an average of {} m/s faster'.format(fastest, fastest_speed - middle_speed))
        fastest_percent = (fastest_speed / middle_speed) - 1
        if fastest_percent > self.faster_than_margin:
          print('Fastest lane is {}% faster! Send alert now'.format(round(fastest_percent*100, 2)))
        else:
          print('Fastest lane is not above margin, ignore')
      else:
        print('Already in fastest lane!')

    return None

  def group_tracks(self):
    """Groups tracks based on lateral position and lane width"""
    for track in self.live_tracks:
      lane_diffs = [{'diff': abs(lane.pos - track.yRel), 'lane': lane} for lane in self.lanes]
      closest_lane = min(lane_diffs, key=lambda x: x['diff'])
      closest_lane['lane'].add(track)

  def reset_lanes(self):
    for lane in self.lanes:
      lane.reset()

  def debug(self):
    for lane in self.lanes:
      print('Lane: {}'.format(lane.name))
      for track in lane.tracks:
        print(track.vRel, track.yRel, track.dRel)
      print()


ls = LaneSpeed()


class Track:
  def __init__(self, vRel, yRel, dRel):
    self.vRel = vRel
    self.yRel = yRel
    self.dRel = dRel


keys = ['v_ego', 'a_ego', 'v_lead', 'lead_status', 'x_lead', 'y_lead', 'a_lead', 'a_rel', 'v_lat', 'steer_angle', 'steer_rate', 'track_data', 'time', 'gas', 'brake', 'car_gas', 'left_blinker', 'right_blinker', 'set_speed', 'new_accel', 'gyro']
#
sample = [34.980491638183594, 0.17890378832817078, 33.02806854248047, True, 102.91999816894531, 0.2800000011920929, -0.20045480132102966, 0.0, 0.0, 0.800000011920929, 0.0, {'tracks': [{'trackID': 11433, 'yRel': 0.2800000011920929, 'dRel': 102.91999816894531, 'vRel': -1.9500000476837158, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 11487, 'yRel': 2.559999942779541, 'dRel': 53.47999954223633, 'vRel': -2.450000047683716, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 11491, 'yRel': 3.240000009536743, 'dRel': 136.60000610351562, 'vRel': -1.7000000476837158, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 11493, 'yRel': 3.1600000858306885, 'dRel': 30.360000610351562, 'vRel': -2.3499999046325684, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 11494, 'yRel': 3.1600000858306885, 'dRel': 30.360000610351562, 'vRel': -2.3499999046325684, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 11507, 'yRel': -2.0, 'dRel': 141.47999572753906, 'vRel': -4.449999809265137, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 11510, 'yRel': -6.400000095367432, 'dRel': 22.600000381469727, 'vRel': -5.275000095367432, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 11518, 'yRel': 2.559999942779541, 'dRel': 53.47999954223633, 'vRel': -2.450000047683716, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 11522, 'yRel': 2.640000104904175, 'dRel': 57.119998931884766, 'vRel': -2.325000047683716, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 11523, 'yRel': 3.0, 'dRel': 55.20000076293945, 'vRel': -2.7249999046325684, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 11526, 'yRel': -4.480000019073486, 'dRel': 161.63999938964844, 'vRel': -7.5, 'stationary': False, 'oncoming': False, 'status': 0.0}], 'live': True}, 1571440447.8279765, 46.759185791015625, 0.0, 0.17000000178813934, False, False, 31.11111068725586, 0.15986532878124438, [0.0007171630859375, -0.1064453125, 0.0576934814453125]]
sample = dict(zip(keys, sample))
trks = sample['track_data']['tracks']
trks = [Track(trk['vRel'], trk['yRel'], trk['dRel']) for trk in trks]

dRel = [t.dRel for t in trks]
yRel = [t.yRel for t in trks]
steer_angle = sample['steer_angle']
plt.scatter(dRel, yRel, label='tracks')
plt.plot([0, 160], [0, 0])
plt.legend()
plt.show()

ls.update(10, None, steer_angle, None, trks)  # v_ego, lead, steer_angle, d_poly, live_tracks
