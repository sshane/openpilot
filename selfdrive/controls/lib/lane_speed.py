from common.op_params import opParams

from selfdrive.config import Conversions as CV
# from common.numpy_fast import clip, interp
import numpy as np
import time
try:
  from common.realtime import sec_since_boot
  import cereal.messaging as messaging
except:
  pass
# try:
#   from common.realtime import sec_since_boot
# except ImportError:
#   import matplotlib.pyplot as plt
#   import time
#   sec_since_boot = time.time


def cluster(data, maxgap):
  data.sort(key=lambda _trk: _trk.dRel)
  groups = [[data[0]]]
  for x in data[1:]:
    if abs(x.dRel - groups[-1][-1].dRel) <= maxgap:
      groups[-1].append(x)
    else:
      groups.append([x])
  return groups


class Lane:
  def __init__(self, name, pos):
    self.name = name
    self.pos = pos
    self.bounds = []
    self.tracks = []
    self.oncoming_tracks = []

    self.avg_speed = None
    self.fastest_count = 0

  def set_fastest(self):
    """Increments this lane's fast count"""
    self.fastest_count += 1


LANE_SPEED_RATE = 1 / 20.

class LaneSpeed:
  def __init__(self):
    self.op_params = opParams()

    self._track_speed_margin = 0.05  # track has to be above X% of v_ego (excludes oncoming and stopped)
    self._faster_than_margin = 0.075  # avg of secondary lane has to be faster by X% to show alert
    self._min_enable_speed = 0  # 35 * CV.MPH_TO_MS
    self._min_fastest_time = 0.01 / LANE_SPEED_RATE  # 3 / LANE_SPEED_RATE  # how long should we wait for a specific lane to be faster than middle before alerting
    self._max_steer_angle = 100  # max supported steering angle
    # self._alert_length = 10  # in seconds
    self._extra_wait_time = 5  # in seconds, how long to wait after last alert finished before allowed to show next alert

    self.fastest_lane = 'none'  # always will be either left, right, or none as a string, never middle or NoneType
    self.last_fastest_lane = 'none'
    self._setup()

  def _setup(self):
    self.lane_width = 3.7  # in meters, just a starting point
    self.sm = messaging.SubMaster(['carState', 'liveTracks', 'pathPlan'])
    self.pm = messaging.PubMaster(['laneSpeed'])

    lane_positions = {'left': self.lane_width, 'middle': 0, 'right': -self.lane_width}  # lateral position in meters from center of car to center of lane
    lane_names = ['left', 'middle', 'right']
    self.lanes = {name: Lane(name, lane_positions[name]) for name in lane_names}

    self.last_alert_end_time = 0

  def start(self):
    while True:  # this loop can take up 0.049_ seconds without lagging
      while not self.op_params.get('lane_speed_alerts', default=True):  # check every 10 seconds if disabled, no need to run at full rate
        if self.fastest_lane != 'none':
          self.fastest_lane = 'none'
          self.send_status()
        time.sleep(10)
      print('main loop')
      input('>> ')

      t_start = sec_since_boot()
      self.sm.update(0)

      self.v_ego = 8  # self.sm['carState'].vEgo
      self.steer_angle = 0  # self.sm['carState'].steeringAngle
      self.d_poly = [-2.2093094e-05, 0.0016798804, 0.14078693, 0.24536133]  # np.array(list(self.sm['pathPlan'].dPoly))
      self.live_tracks = TEMP_LIVE_TRACKS  # self.sm['liveTracks']

      self.update_lane_bounds()
      self.update()
      self.send_status()
      print(self.fastest_lane)

      t_sleep = LANE_SPEED_RATE - (sec_since_boot() - t_start)
      if t_sleep > 0:
        time.sleep(t_sleep)
      else:  # don't sleep if lagging
        print('lane_speed lagging by: {} ms'.format(round(-t_sleep * 1000, 3)))

  def update(self):
    self.reset(reset_tracks=True, reset_avg_speed=True)

    # checks that we have dPoly, dPoly is not NaNs, and steer angle is less than max allowed
    if len(self.d_poly) and not np.isnan(self.d_poly[0]) and abs(self.steer_angle) < self._max_steer_angle:
      if self.v_ego > self._min_enable_speed:
        # self.filter_tracks()  # todo: will remove tracks very close to other tracks to make averaging more robust
        self.group_tracks()
        self.get_fastest_lane()
    else:  # should we reset state when not enabled?
      self.reset(reset_fastest=True)

  def update_lane_bounds(self):
    lane_width = self.sm['pathPlan'].laneWidth
    if isinstance(lane_width, float) and lane_width > 1:
      self.lane_width = min(lane_width, 4)  # LanePlanner uses 4 as max width for dPoly calculation

    self.lanes['left'].pos = self.lane_width  # update with new lane center positions
    self.lanes['right'].pos = -self.lane_width

    # and now update bounds
    self.lanes['left'].bounds = np.array([self.lanes['left'].pos * 1.5, self.lanes['left'].pos / 2])
    self.lanes['middle'].bounds = np.array([self.lanes['left'].pos / 2, self.lanes['right'].pos / 2])
    self.lanes['right'].bounds = np.array([self.lanes['right'].pos / 2, self.lanes['right'].pos * 1.5])

  # def filter_tracks(self):  # fixme: make cluster() return indexes of live_tracks instead
  #   print(type(self.live_tracks))
  #   clustered = cluster(self.live_tracks, 0.048)  # clusters tracks based on dRel
  #   clustered = [clstr for clstr in clustered if len(clstr) > 1]
  #   print([[trk.dRel for trk in clstr] for clstr in clustered])
  #   for clstr in clustered:
  #     pass
  #   # print(c)

  def group_tracks(self):
    """Groups tracks based on lateral position, dPoly offset, and lane width"""
    y_offsets = np.polyval(self.d_poly, [trk.dRel for trk in self.live_tracks])  # it's faster to calculate all at once
    for track, y_offset in zip(self.live_tracks, y_offsets):
      for lane_name in self.lanes:
        lane_bounds = self.lanes[lane_name].bounds + y_offset  # offset lane bounds based on our future lateral position (dPoly) and track's distance
        if lane_bounds[0] >= track.yRel >= lane_bounds[1]:  # track is in a lane
          self.lanes[lane_name].tracks.append(track)
          break  # skip to next track

  def lanes_with_avg_speeds(self):
    """Returns a dict of lane objects where avg_speed not None"""
    return {lane: self.lanes[lane] for lane in self.lanes if self.lanes[lane].avg_speed is not None}

  def get_fastest_lane(self):
    self.fastest_lane = 'none'
    for lane_name in self.lanes:
      lane = self.lanes[lane_name]
      track_speeds = [track.vRel + self.v_ego for track in lane.tracks]
      track_speeds = [speed for speed in track_speeds if speed > self.v_ego * self._track_speed_margin]
      if len(track_speeds):  # filters out oncoming tracks and very slow tracks
        lane.avg_speed = np.mean(track_speeds)  # todo: something with std?

    lanes_with_avg_speeds = self.lanes_with_avg_speeds()
    if 'middle' not in lanes_with_avg_speeds or len(lanes_with_avg_speeds) < 2:
      # if no tracks in middle lane or no secondary lane, we have nothing to compare
      self.reset(reset_fastest=True)  # reset fastest, sanity
      return

    fastest_lane = self.lanes[max(lanes_with_avg_speeds, key=lambda x: self.lanes[x].avg_speed)]
    if fastest_lane.name == 'middle':  # already in fastest lane
      self.reset(reset_fastest=True)
      return
    if (fastest_lane.avg_speed / self.lanes['middle'].avg_speed) - 1 < self._faster_than_margin:  # fastest lane is not above margin, ignore
      # todo: could remove since we wait for a lane to be faster for a bit
      return

    # if we are here, there's a faster lane available that's above our minimum margin
    fastest_lane.set_fastest()  # increment fastest lane
    self.lanes[self.opposite_lane(fastest_lane.name)].fastest_count = 0  # reset slowest lane (opposite, never middle)

    _f_time_x = [1, 4, 12]  # change the minimum time for fastest based on how many tracks are in fastest lane
    _f_time_y = [1.5, 1, 0.5]  # this is multiplied by base fastest time todo: probably need to tune this
    min_fastest_time = np.interp(len(fastest_lane.tracks), _f_time_x, _f_time_y)  # get multiplier
    min_fastest_time = int(min_fastest_time * self._min_fastest_time)  # now get final min_fastest_time

    if fastest_lane.fastest_count < min_fastest_time:
      return  # fastest lane hasn't been fastest long enough
    if sec_since_boot() - self.last_alert_end_time < self._extra_wait_time:
      return  # don't reset fastest lane count or show alert until last alert has gone

    # if here, we've found a lane faster than our lane by a margin and it's been faster for long enough
    self.fastest_lane = fastest_lane.name

  def send_status(self):
    new_fastest = self.fastest_lane in ['left', 'right'] and self.last_fastest_lane not in ['left', 'right']
    ls_send = messaging.new_message('laneSpeed')

    ls_send.laneSpeed.leftLaneSpeeds = [trk.vRel + self.v_ego for trk in self.lanes['left'].tracks]
    ls_send.laneSpeed.middleLaneSpeeds = [trk.vRel + self.v_ego for trk in self.lanes['middle'].tracks]
    ls_send.laneSpeed.rightLaneSpeeds = [trk.vRel + self.v_ego for trk in self.lanes['right'].tracks]
    ls_send.laneSpeed.leftLaneDistances = [trk.dRel for trk in self.lanes['left'].tracks]
    ls_send.laneSpeed.middleLaneDistances = [trk.dRel for trk in self.lanes['middle'].tracks]
    ls_send.laneSpeed.rightLaneDistances = [trk.dRel for trk in self.lanes['right'].tracks]

    ls_send.laneSpeed.status = self.fastest_lane
    ls_send.laneSpeed.new = new_fastest  # only send audible alert once when a lane becomes fastest, then continue to show silent alert
    self.pm.send('laneSpeed', ls_send)

    if self.fastest_lane != self.last_fastest_lane and self.fastest_lane == 'none':  # if lane stops being fastest
      self.last_alert_end_time = sec_since_boot()
    elif self.last_fastest_lane in ['left', 'right'] and self.fastest_lane == self.opposite_lane(self.last_fastest_lane):  # or fastest switches
      self.last_alert_end_time = sec_since_boot()

    self.last_fastest_lane = self.fastest_lane

  def opposite_lane(self, name):
    """Returns name of opposite lane name"""
    return {'left': 'right', 'right': 'left'}[name]

  def reset(self, reset_tracks=False, reset_fastest=False, reset_avg_speed=False):
    for lane in self.lanes:
      if reset_tracks:
        self.lanes[lane].tracks = []
        self.lanes[lane].oncoming_tracks = []

      if reset_avg_speed:
        self.lanes[lane].avg_speed = None

      if reset_fastest:
        self.lanes[lane].fastest_count = 0

  # def log_data(self):
  #   live_tracks = [{'vRel': trk.vRel, 'yRel': trk.yRel, 'dRel': trk.dRel} for trk in self.live_tracks]
  #   with open('/data/lane_speed', 'a') as f:
  #     f.write('{}\n'.format({'v_ego': self.v_ego, 'd_poly': self.d_poly, 'steer_angle': self.steer_angle, 'live_tracks': live_tracks}))


class Track:
  def __init__(self, vRel, yRel, dRel):
    self.vRel = vRel
    self.yRel = yRel
    self.dRel = dRel

TEMP_LIVE_TRACKS = [Track(10, 0, 20), Track(10, 0, 20), Track(20, 4, 20), Track(25, 4, 20)]

def main():
  lane_speed = LaneSpeed()
  lane_speed.start()


if __name__ == '__main__':
  main()


# DEBUG = False
#
# if DEBUG:
#   def circle_y(_x, _angle):  # fixme: not sure if this is correct
#     return -(_x * _angle) ** 2 / (1000 * (_angle * 2))
#
#   ls = LaneSpeed()
#
#   class Track:
#     def __init__(self, vRel, yRel, dRel):
#       self.vRel = vRel
#       self.yRel = yRel
#       self.dRel = dRel
#
#   d_poly = [3.2945357553160193e-10, -0.0009911218658089638, -0.009723401628434658, 0.14891201257705688]
#
#   keys = ['v_ego', 'a_ego', 'v_lead', 'lead_status', 'x_lead', 'y_lead', 'a_lead', 'a_rel', 'v_lat', 'steer_angle', 'steer_rate', 'track_data', 'time', 'gas', 'brake', 'car_gas', 'left_blinker', 'right_blinker', 'set_speed', 'new_accel', 'gyro']
#
#   sample = [8.013258934020996, 0.14726917445659637, 8.45051383972168, True, 12.680000305175781, 0.19999998807907104, 0.7618321180343628, 0.0, 0.0, -0.30000001192092896, 0.0, {'tracks': [{'trackID': 13482, 'yRel': 0.1599999964237213, 'dRel': 12.680000305175781, 'vRel': 0.4000000059604645, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13652, 'yRel': -0.03999999910593033, 'dRel': 19.360000610351562, 'vRel': 0.5249999761581421, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13690, 'yRel': -0.20000000298023224, 'dRel': 22.639999389648438, 'vRel': 0.25, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13691, 'yRel': 4.440000057220459, 'dRel': 27.520000457763672, 'vRel': 8.824999809265137, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13692, 'yRel': 2.8399999141693115, 'dRel': 36.68000030517578, 'vRel': -5.099999904632568, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13694, 'yRel': 2.9600000381469727, 'dRel': 36.68000030517578, 'vRel': -5.074999809265137, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13698, 'yRel': -1.1200000047683716, 'dRel': 17.040000915527344, 'vRel': 0.32499998807907104, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13700, 'yRel': -0.20000000298023224, 'dRel': 25.31999969482422, 'vRel': 0.699999988079071, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13703, 'yRel': -0.11999999731779099, 'dRel': 19.84000015258789, 'vRel': 0.20000000298023224, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13704, 'yRel': 0.23999999463558197, 'dRel': 12.680000305175781, 'vRel': 0.4749999940395355, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13705, 'yRel': 0.03999999910593033, 'dRel': 25.360000610351562, 'vRel': 0.3499999940395355, 'stationary': False, 'oncoming': False, 'status': 0.0}, {'trackID': 13706, 'yRel': -5.599999904632568, 'dRel': 116.4800033569336, 'vRel': 8.675000190734863, 'stationary': False, 'oncoming': False, 'status': 0.0}], 'live': True}, 1571441322.0375044, 26.91699981689453, 0.0, 0.07500000298023224, False, False, 21.94444465637207, 0.06090415226828825, [0.0047149658203125, -0.039764404296875, 0.029388427734375]]
#   sample = dict(zip(keys, sample))
#   trks = sample['track_data']['tracks']
#   trks = [Track(trk['vRel'], trk['yRel'], trk['dRel']) for trk in trks]
#   trks.append(Track(4, -12.8, 103))
#   trks.append(Track(12, -11, 115))
#   trks.append(Track(32, -11, 115))
#
#   dRel = [t.dRel for t in trks]
#   yRel = [t.yRel for t in trks]
#   steerangle = sample['steer_angle']
#   plt.scatter(dRel, yRel, label='tracks')
#   x_path = np.linspace(0, 130, 100)
#   # y_path = circle_y(x_path, steerangle)
#   y_path = np.polyval(d_poly, x_path)
#   plt.plot([0, 130], [0, 0])
#   # plt.plot(x_path, y_path, label='dPoly')
#   plt.plot(x_path, y_path + 3.7 / 2, 'r--', label='left line')
#   plt.plot(x_path, y_path - 3.7 / 2, 'r--', label='right line')
#
#   plt.plot(x_path, y_path + 3.7 / 2 + 3.7, 'g--')
#   plt.plot(x_path, y_path - 3.7 / 2 - 3.7, 'g--')
#
#
#   plt.legend()
#   plt.show()
#
#   for _ in range(1):
#     out = ls.update(10, None, steerangle, d_poly, trks)  # v_ego, lead, steer_angle, d_poly, live_tracks
#   print([(lane.name, lane.fastest_count) for lane in ls.lanes.values()])
#   print('out: {}'.format(out))
#   print([len(ls.lanes[l].tracks) for l in ls.lanes])
