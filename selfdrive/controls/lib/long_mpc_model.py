import numpy as np
import math

from selfdrive.config import Conversions as CV
from cereal.messaging import SubMaster
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.longitudinal_mpc_model import libmpc_py


class LongitudinalMpcModel():
  def __init__(self):

    self.setup_mpc()
    self.v_mpc = 0.0
    self.v_mpc_future = 0.0
    self.a_mpc = 0.0
    self.last_cloudlog_t = 0.0
    self.ts = list(range(10))
    self.sm = SubMaster(['liveTracks'])

    self.valid = False

  def setup_mpc(self, v_ego=0.0):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(1.0, 1.0, 1.0, 1.0, 1.0)
    self.libmpc.init_with_simulation(v_ego)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")

    self.cur_state[0].x_ego = 0
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0

  def set_cur_state(self, v, a):
    self.cur_state[0].x_ego = 0.0
    self.cur_state[0].v_ego = v
    self.cur_state[0].a_ego = a

  @property
  def use_model(self):
    # Decides if we should use model. Planner still uses the slowest
    # track attrs: dRel, vRel, aRel, yRel
    # v_ego = self.v_ego * CV.MS_TO_MPH
    if self.v_ego <= 10 * CV.MPH_TO_MS:  # always use model under 10 mph (if slowest)
      return True

    tracks = self.sm['liveTracks']
    track_len = len(tracks)
    if track_len == 0:
      return False
    speeds = [trk.vRel + self.v_ego for trk in tracks]
    moving = [spd for spd in speeds if spd > 1.5 * CV.MPH_TO_MS]
    if len(moving) >= 3:  # if enough moving tracks
      return True
    return False

    # Not sure what this is
    # v_rels = [trk.vRel for trk in tracks]
    #
    # above_len = len([vr for vr in v_rels if vr >= v_ego + 3])
    # if above_len / track_len > 1 / 4:
    #   return False
    #
    # max_speed_diff = 3.8e-05 * v_ego ** 2 + -0.00627 * v_ego + 1.0114
    # max_speed_diff *= v_ego
    #
    # # Now filter
    # v_rels = [vr for vr in v_rels if abs(vr) <= max_speed_diff and vr < 0]
    # avg_v_rel = np.mean(v_rels)
    #
    # if abs((-avg_v_rel * (len(v_rels) / 16)) / 5.625) >= 0.115:
    #   return True
    # return False

  def update(self, v_ego, a_ego, poss, speeds, accels):
    self.sm.update(0)
    self.v_ego = v_ego
    self.a_ego = a_ego
    if len(poss) == 0:
      self.valid = False
      return

    x_poly = list(map(float, np.polyfit(self.ts, poss, 3)))
    v_poly = list(map(float, np.polyfit(self.ts, speeds, 3)))
    a_poly = list(map(float, np.polyfit(self.ts, accels, 3)))

    # Calculate mpc
    self.libmpc.run_mpc(self.cur_state, self.mpc_solution, x_poly, v_poly, a_poly)

    # Get solution. MPC timestep is 0.2 s, so interpolation to 0.05 s is needed
    self.v_mpc = self.mpc_solution[0].v_ego[1]
    self.a_mpc = self.mpc_solution[0].a_ego[1]
    # if not self.use_model:
    #   self.valid = False
    #   return
    self.v_mpc_future = self.mpc_solution[0].v_ego[10]
    self.valid = True

    # Reset if NaN or goes through lead car
    nans = any(math.isnan(x) for x in self.mpc_solution[0].v_ego)

    t = sec_since_boot()
    if nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal model mpc reset - backwards")

      self.libmpc.init(1.0, 1.0, 1.0, 1.0, 1.0)
      self.libmpc.init_with_simulation(v_ego)

      self.cur_state[0].v_ego = v_ego
      self.cur_state[0].a_ego = 0.0

      self.v_mpc = v_ego
      self.a_mpc = a_ego
      self.valid = False
