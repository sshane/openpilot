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


class LaneSpeedState:
  off = 0
  audible = 1
  silent = 2
  to_state = {off: 'off', silent: 'silent', audible: 'audible'}
  to_idx = {v: k for k, v in to_state.items()}

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


LANE_SPEED_RATE = 1 / 5.

class LaneSpeed:
  def __init__(self):
    self.op_params = opParams()

    self._track_speed_margin = 0.05  # track has to be above X% of v_ego (excludes oncoming and stopped)
    self._faster_than_margin = 0.075  # avg of secondary lane has to be faster by X% to show alert
    self._min_enable_speed = 35 * CV.MPH_TO_MS
    self._min_fastest_time = 3 / LANE_SPEED_RATE  # how long should we wait for a specific lane to be faster than middle before alerting
    self._max_steer_angle = 100  # max supported steering angle
    self._extra_wait_time = 5  # in seconds, how long to wait after last alert finished before allowed to show next alert

    self.fastest_lane = 'none'  # always will be either left, right, or none as a string, never middle or NoneType
    self.last_fastest_lane = 'none'
    self._setup()

  def _setup(self):
    t_start = sec_since_boot()
    self.ls_state = self.op_params.get('lane_speed_alerts', 'audible').strip().lower()
    if not isinstance(self.ls_state, str) or self.ls_state not in LaneSpeedState.to_idx:
      self.ls_state = LaneSpeedState.audible
      self.op_params.put('lane_speed_alerts', LaneSpeedState.to_state[self.ls_state])
    else:
      self.ls_state = LaneSpeedState.to_idx[self.ls_state]
    self.last_ls_state = self.ls_state
    self.offset = self.ls_state

    self.lane_width = 3.7  # in meters, just a starting point
    self.sm = messaging.SubMaster(['carState', 'liveTracks', 'pathPlan', 'laneSpeedButton', 'controlsState'])
    self.pm = messaging.PubMaster(['laneSpeed'])

    lane_positions = {'left': self.lane_width, 'middle': 0, 'right': -self.lane_width}  # lateral position in meters from center of car to center of lane
    lane_names = ['left', 'middle', 'right']
    self.lanes = {name: Lane(name, lane_positions[name]) for name in lane_names}

    self.oncoming_lanes = {'left': False, 'right': False}

    self.last_alert_end_time = 0
    print('Setup time: {}'.format(sec_since_boot() - t_start))

  def start(self, temp_v_ego=None, temp_steer_angle=None, temp_d_poly=None, temp_tracks=None):
    while True:  # this loop can take up 0.049_ seconds without lagging
      t_start = sec_since_boot()
      self.sm.update(0)

      if temp_v_ego is not None:
        self.v_ego = temp_v_ego
      else:
        self.v_ego = self.sm['carState'].vEgo
      if temp_steer_angle is not None:
        self.steer_angle = temp_steer_angle
      else:
        self.steer_angle = self.sm['carState'].steeringAngle
      if temp_d_poly is not None:
        self.d_poly = temp_d_poly
      else:
        self.d_poly = np.array(list(self.sm['pathPlan'].dPoly))
      if temp_tracks is not None:
        self.live_tracks = temp_tracks
      else:
        self.live_tracks = self.sm['liveTracks']

      self.update_lane_bounds()
      self.update()
      self.send_status()

      t_sleep = LANE_SPEED_RATE - (sec_since_boot() - t_start)
      if t_sleep > 0:
        time.sleep(t_sleep)
      else:  # don't sleep if lagging
        print('lane_speed lagging by: {} ms'.format(round(-t_sleep * 1000, 3)))

  def update(self):
    self.reset(reset_tracks=True, reset_avg_speed=True)
    self.ls_state = (self.sm['laneSpeedButton'].status + self.offset) % len(LaneSpeedState.to_state)

    # checks that we have dPoly, dPoly is not NaNs, and steer angle is less than max allowed
    if len(self.d_poly) and not np.isnan(self.d_poly[0]):
      # self.filter_tracks()  # todo: will remove tracks very close to other tracks to make averaging more robust
      self.group_tracks()
      self.find_oncoming_lanes()
      self.get_fastest_lane()
    else:
      self.reset(reset_fastest=True)

  def update_lane_bounds(self):
    t_start = sec_since_boot()
    lane_width = self.sm['pathPlan'].laneWidth
    if isinstance(lane_width, float) and lane_width > 1:
      self.lane_width = min(lane_width, 4)  # LanePlanner uses 4 as max width for dPoly calculation

    self.lanes['left'].pos = self.lane_width  # update with new lane center positions
    self.lanes['right'].pos = -self.lane_width

    # and now update bounds
    self.lanes['left'].bounds = np.array([self.lanes['left'].pos * 1.5, self.lanes['left'].pos / 2])
    self.lanes['middle'].bounds = np.array([self.lanes['left'].pos / 2, self.lanes['right'].pos / 2])
    self.lanes['right'].bounds = np.array([self.lanes['right'].pos / 2, self.lanes['right'].pos * 1.5])
    print('update_lane_bounds: {}'.format(sec_since_boot() - t_start))

  # def filter_tracks(self):  # todo: make cluster() return indexes of live_tracks instead
  #   print(type(self.live_tracks))
  #   clustered = cluster(self.live_tracks, 0.048)  # clusters tracks based on dRel
  #   clustered = [clstr for clstr in clustered if len(clstr) > 1]
  #   print([[trk.dRel for trk in clstr] for clstr in clustered])
  #   for clstr in clustered:
  #     pass
  #   # print(c)

  def group_tracks(self):
    t_start = sec_since_boot()
    """Groups tracks based on lateral position, dPoly offset, and lane width"""
    y_offsets = np.polyval(self.d_poly, [trk.dRel for trk in self.live_tracks])  # it's faster to calculate all at once
    for track, y_offset in zip(self.live_tracks, y_offsets):
      for lane_name in self.lanes:
        lane_bounds = self.lanes[lane_name].bounds + y_offset  # offset lane bounds based on our future lateral position (dPoly) and track's distance
        if lane_bounds[0] >= track.yRel >= lane_bounds[1]:  # track is in a lane
          if track.vRel + self.v_ego >= 2.24:
            self.lanes[lane_name].tracks.append(track)
          elif track.vRel + self.v_ego <= -2.24:  # make sure we don't add stopped tracks at high speeds
            self.lanes[lane_name].oncoming_tracks.append(track)
          break  # skip to next track
    print('group_tracks: {}'.format(sec_since_boot() - t_start))

  def find_oncoming_lanes(self):
    t_start = sec_since_boot()
    """If number of oncoming tracks is greater than tracks going our direction, set lane to oncoming"""
    for lane in self.oncoming_lanes:
      self.oncoming_lanes[lane] = False
      if len(self.lanes[lane].oncoming_tracks) > len(self.lanes[lane].tracks):  # 0 can't be > 0 so 0 oncoming tracks will be handled correctly
        self.oncoming_lanes[lane] = True
    print('find_oncoming_lanes: {}'.format(sec_since_boot() - t_start))

  def lanes_with_avg_speeds(self):
    """Returns a dict of lane objects where avg_speed not None"""
    return {lane: self.lanes[lane] for lane in self.lanes if self.lanes[lane].avg_speed is not None}

  def get_fastest_lane(self):
    t_start = sec_since_boot()
    self.fastest_lane = 'none'
    if self.ls_state == LaneSpeedState.off:
      return

    v_cruise_setpoint = self.sm['controlsState'].vCruise * CV.KPH_TO_MS
    for lane_name in self.lanes:
      lane = self.lanes[lane_name]
      track_speeds = [track.vRel + self.v_ego for track in lane.tracks]
      track_speeds = [speed for speed in track_speeds if self.v_ego * self._track_speed_margin < speed <= v_cruise_setpoint]
      if len(track_speeds):  # filters out very slow tracks
        lane.avg_speed = np.mean(track_speeds)  # todo: something with std?
    print('get_fastest_lane avg_speeds: {}'.format(sec_since_boot() - t_start))

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

  # def log_data(self):  # DON'T USE AGAIN until I fix live tracks formatting
  #   log_file = '/data/lane_speed_log'
  #   lanes_tracks = {}
  #   lanes_oncoming_tracks = {}
  #   bounds = {}
  #   for lane in self.lanes:
  #     bounds[lane] = self.lanes[lane].bounds
  #     lanes_tracks[lane] = [{'vRel': trk.vRel, 'dRel': trk.dRel, 'yRel': trk.yRel} for trk in self.lanes[lane].tracks]
  #     lanes_oncoming_tracks[lane] = [{'vRel': trk.vRel, 'dRel': trk.dRel, 'yRel': trk.yRel} for trk in self.lanes[lane].oncoming_tracks]
  #
  #   log_data = {'v_ego': self.v_ego, 'd_poly': self.d_poly, 'lane_tracks': lanes_tracks, 'lane_oncoming_tracks': lanes_oncoming_tracks,
  #               'live_tracks': self.live_tracks, 'oncoming_lanes': self.oncoming_lanes, 'bounds': bounds}
  #   with open(log_file, 'a') as f:
  #     f.write('{}\n'.format(log_data))

  def send_status(self):
    new_fastest = self.fastest_lane in ['left', 'right'] and self.last_fastest_lane not in ['left', 'right']
    fastest_lane = self.fastest_lane
    if self.ls_state == LaneSpeedState.silent:
      new_fastest = False  # be silent
    if self.v_ego < self._min_enable_speed or abs(self.steer_angle) > self._max_steer_angle:  # keep sending updates, but not fastestLane
      fastest_lane = 'none'

    ls_send = messaging.new_message('laneSpeed')
    ls_send.laneSpeed.fastestLane = fastest_lane
    ls_send.laneSpeed.new = new_fastest  # only send audible alert once when a lane becomes fastest, then continue to show silent alert

    ls_send.laneSpeed.leftLaneSpeeds = [trk.vRel + self.v_ego for trk in self.lanes['left'].tracks]
    ls_send.laneSpeed.middleLaneSpeeds = [trk.vRel + self.v_ego for trk in self.lanes['middle'].tracks]
    ls_send.laneSpeed.rightLaneSpeeds = [trk.vRel + self.v_ego for trk in self.lanes['right'].tracks]

    ls_send.laneSpeed.leftLaneDistances = [trk.dRel for trk in self.lanes['left'].tracks]
    ls_send.laneSpeed.middleLaneDistances = [trk.dRel for trk in self.lanes['middle'].tracks]
    ls_send.laneSpeed.rightLaneDistances = [trk.dRel for trk in self.lanes['right'].tracks]

    ls_send.laneSpeed.leftLaneOncoming = self.oncoming_lanes['left']
    ls_send.laneSpeed.rightLaneOncoming = self.oncoming_lanes['right']

    if self.last_ls_state != self.ls_state:  # show alert if button tapped and write to opParams
      self.op_params.put('lane_speed_alerts', LaneSpeedState.to_state[self.ls_state])
      ls_send.laneSpeed.state = LaneSpeedState.to_state[self.ls_state]

    self.pm.send('laneSpeed', ls_send)

    if self.fastest_lane != self.last_fastest_lane and self.fastest_lane == 'none':  # if lane stops being fastest
      self.last_alert_end_time = sec_since_boot()
    elif self.last_fastest_lane in ['left', 'right'] and self.fastest_lane == self.opposite_lane(self.last_fastest_lane):  # or fastest switches
      self.last_alert_end_time = sec_since_boot()

    self.last_fastest_lane = self.fastest_lane
    self.last_ls_state = self.ls_state

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


class Track:
  def __init__(self, vRel, yRel, dRel):
    self.vRel = vRel
    self.yRel = yRel
    self.dRel = dRel

TEMP_LIVE_TRACKS = [Track(np.random.uniform(-40, 40), np.random.uniform(-12, 12), np.random.uniform(0, 180)) for _ in range(16)]
TEMP_D_POLY = np.array([1.3839008e-06, 0.00017540116, -0.0059070284, 0.25009361])

def main():
  lane_speed = LaneSpeed()
  lane_speed.start(temp_v_ego=np.random.uniform(0, 40), temp_steer_angle=0, temp_d_poly=TEMP_D_POLY, temp_tracks=TEMP_LIVE_TRACKS)


if __name__ == '__main__':
  main()
