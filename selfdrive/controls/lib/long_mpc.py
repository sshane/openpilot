import os
import math
import time

import cereal.messaging as messaging
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU
from selfdrive.controls.lib.longitudinal_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from common.op_params import opParams
from common.numpy_fast import interp, clip
from common.travis_checker import travis
from selfdrive.config import Conversions as CV

LOG_MPC = os.environ.get('LOG_MPC', False)


class LongitudinalMpc():
  def __init__(self, mpc_id):
    self.mpc_id = mpc_id
    self.op_params = opParams()

    self.setup_mpc()
    self.v_mpc = 0.0
    self.v_mpc_future = 0.0
    self.a_mpc = 0.0
    self.v_cruise = 0.0
    self.prev_lead_status = False
    self.prev_lead_x = 0.0
    self.new_lead = False
    self.last_cloudlog_t = 0.0

    if not travis and mpc_id == 1:
      self.pm = messaging.PubMaster(['smiskolData'])
    else:
      self.pm = None
    self.car_data = {'v_ego': 0.0, 'a_ego': 0.0}
    self.lead_data = {'v_lead': None, 'x_lead': None, 'a_lead': None, 'status': False}
    self.df_data = {"v_leads": [], "v_egos": []}  # dynamic follow data
    self.last_cost = 0.0
    self.df_profile = self.op_params.get('dynamic_follow', 'relaxed').strip().lower()
    self.sng = False

  def send_mpc_solution(self, pm, qp_iterations, calculation_time):
    qp_iterations = max(0, qp_iterations)
    dat = messaging.new_message()
    dat.init('liveLongitudinalMpc')
    dat.liveLongitudinalMpc.xEgo = list(self.mpc_solution[0].x_ego)
    dat.liveLongitudinalMpc.vEgo = list(self.mpc_solution[0].v_ego)
    dat.liveLongitudinalMpc.aEgo = list(self.mpc_solution[0].a_ego)
    dat.liveLongitudinalMpc.xLead = list(self.mpc_solution[0].x_l)
    dat.liveLongitudinalMpc.vLead = list(self.mpc_solution[0].v_l)
    dat.liveLongitudinalMpc.cost = self.mpc_solution[0].cost
    dat.liveLongitudinalMpc.aLeadTau = self.a_lead_tau
    dat.liveLongitudinalMpc.qpIterations = qp_iterations
    dat.liveLongitudinalMpc.mpcId = self.mpc_id
    dat.liveLongitudinalMpc.calculationTime = calculation_time
    pm.send('liveLongitudinalMpc', dat)

  def setup_mpc(self):
    ffi, self.libmpc = libmpc_py.get_libmpc(self.mpc_id)
    self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                     MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)

    self.mpc_solution = ffi.new("log_t *")
    self.cur_state = ffi.new("state_t *")
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0
    self.a_lead_tau = _LEAD_ACCEL_TAU

  def set_cur_state(self, v, a):
    self.cur_state[0].v_ego = v
    self.cur_state[0].a_ego = a

  def get_TR(self, CS):
    if not self.lead_data['status'] or travis:
      TR = 1.8
    else:
      self.store_df_data()
      TR = self.dynamic_follow(CS)

    if not travis:
      self.change_cost(TR)
      self.send_cur_TR(TR)
    return TR

  def send_cur_TR(self, TR):
    if self.mpc_id == 1 and self.pm is not None:
      dat = messaging.new_message()
      dat.init('smiskolData')
      dat.smiskolData.mpcTR = TR
      self.pm.send('smiskolData', dat)

  def change_cost(self, TR):
    TRs = [0.9, 1.8, 2.7]
    costs = [1.0, 0.1, 0.05]
    cost = interp(TR, TRs, costs)
    if self.last_cost != cost:
      self.libmpc.change_tr(MPC_COST_LONG.TTC, cost, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
      self.last_cost = cost

  def store_df_data(self):
    v_lead_retention = 1.9  # keep only last x seconds
    v_ego_retention = 2.5

    cur_time = time.time()
    if self.lead_data['status']:
      self.df_data['v_leads'] = [sample for sample in self.df_data['v_leads'] if
                                 cur_time - sample['time'] <= v_lead_retention
                                 and not self.new_lead]  # reset when new lead
      self.df_data['v_leads'].append({'v_lead': self.lead_data['v_lead'], 'time': cur_time})

    self.df_data['v_egos'] = [sample for sample in self.df_data['v_egos'] if cur_time - sample['time'] <= v_ego_retention]
    self.df_data['v_egos'].append({'v_ego': self.car_data['v_ego'], 'time': cur_time})

  def calculate_lead_accel(self):
    min_consider_time = 1.0  # minimum amount of time required to consider calculation
    a_lead = self.lead_data['a_lead']
    if len(self.df_data['v_leads']):  # if not empty
      elapsed = self.df_data['v_leads'][-1]['time'] - self.df_data['v_leads'][0]['time']
      if elapsed > min_consider_time:  # if greater than min time (not 0)
        a_calculated = (self.df_data['v_leads'][-1]['v_lead'] - self.df_data['v_leads'][0]['v_lead']) / elapsed  # delta speed / delta time
        # old version: # if abs(a_calculated) > abs(a_lead) and a_lead < 0.33528:  # if a_lead is greater than calculated accel (over last 1.5s, use that) and if lead accel is not above 0.75 mph/s
        #   a_lead = a_calculated

        # long version of below: if (a_calculated < 0 and a_lead >= 0 and a_lead < -a_calculated * 0.5) or (a_calculated > 0 and a_lead <= 0 and -a_lead > a_calculated * 0.5) or (a_lead * a_calculated > 0 and abs(a_calculated) > abs(a_lead)):
        if (a_calculated < 0 <= a_lead < -a_calculated * 0.55) or (a_calculated > 0 >= a_lead and -a_lead < a_calculated * 0.45) or (a_lead * a_calculated > 0 and abs(a_calculated) > abs(a_lead)):  # this is a mess, fix
          a_lead = a_calculated
    return a_lead  # if above doesn't execute, we'll return a_lead from radar

  def dynamic_follow(self, CS):
    self.df_profile = self.op_params.get('dynamic_follow', 'relaxed').strip().lower()
    x_vel = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]  # velocities
    profile_mod_x = [2.2352, 13.4112, 24.5872, 35.7632]  # profile mod speeds, mph: [5., 30., 55., 80.]
    if self.df_profile == 'roadtrip':
      y_dist = [1.3847, 1.3946, 1.4078, 1.4243, 1.4507, 1.4837, 1.5327, 1.553, 1.581, 1.617, 1.653, 1.687, 1.74]  # TRs
      profile_mod_pos = [0.99, 0.9025, 0.815, 0.55]
      profile_mod_neg = [1.0, 1.18, 1.382, 1.787]
    elif self.df_profile == 'traffic':  # for in congested traffic
      x_vel = [0.0, 1.892, 3.7432, 5.8632, 8.0727, 10.7301, 14.343, 17.6275, 22.4049, 28.6752, 34.8858, 40.35]
      y_dist = [1.3781, 1.3791, 1.3802, 1.3825, 1.3984, 1.4249, 1.4194, 1.3162, 1.1916, 1.0145, 0.9855, 0.9562]
      profile_mod_pos = [1.05, 1.375, 2.99, 3.8]
      profile_mod_neg = [0.79, .1, 0.0, 0.0]
    else:  # default to relaxed/stock
      y_dist = [1.385, 1.394, 1.406, 1.421, 1.444, 1.474, 1.516, 1.534, 1.546, 1.568, 1.579, 1.593, 1.614]
      profile_mod_pos = [1.0] * 4
      profile_mod_neg = [1.0] * 4

    sng_TR = 1.7  # reacceleration stop and go TR
    sng_speed = 15.0 * CV.MPH_TO_MS

    if self.car_data['v_ego'] > sng_speed:  # keep sng distance until we're above sng speed again
      self.sng = False

    if (self.car_data['v_ego'] >= sng_speed or self.df_data['v_egos'][0]['v_ego'] >= self.car_data['v_ego']) and not self.sng:  # if above 15 mph OR we're decelerating to a stop, keep shorter TR. when we reaccelerate, use sng_TR and slowly decrease
      TR = interp(self.car_data['v_ego'], x_vel, y_dist)
    else:  # this allows us to get closer to the lead car when stopping, while being able to have smooth stop and go when reaccelerating
      self.sng = True
      x = [sng_speed / 3.0, sng_speed]  # decrease TR between 5 and 15 mph from 1.8s to defined TR above at 15mph while accelerating
      y = [sng_TR, interp(sng_speed, x_vel, y_dist)]
      TR = interp(self.car_data['v_ego'], x, y)

    TR_mod = []
    # Dynamic follow modifications (the secret sauce)
    x = [-20.0383, -15.6978, -11.2053, -7.8781, -5.0407, -3.2167, -1.6122, 0.0, 0.6847, 1.3772, 1.9007, 2.7452]  # relative velocity values
    y = [0.641, 0.506, 0.418, 0.334, 0.24, 0.115, 0.065, 0.0, -0.049, -0.068, -0.142, -0.221]  # modification values
    TR_mod.append(interp(self.lead_data['v_lead'] - self.car_data['v_ego'], x, y))

    x = [-4.4795, -2.8122, -1.5727, -1.1129, -0.6611, -0.2692, 0.0, 0.1466, 0.5144, 0.6903, 0.9302]  # lead acceleration values
    y = [0.265, 0.187, 0.096, 0.057, 0.033, 0.024, 0.0, -0.009, -0.042, -0.053, -0.059]  # modification values
    TR_mod.append(interp(self.calculate_lead_accel(), x, y))

    profile_mod_pos = interp(self.car_data['v_ego'], profile_mod_x, profile_mod_pos)
    profile_mod_neg = interp(self.car_data['v_ego'], profile_mod_x, profile_mod_neg)

    TR_mod = sum([mod * profile_mod_neg if mod < 0 else mod * profile_mod_pos for mod in TR_mod])  # alter TR modification according to profile
    TR += TR_mod

    if CS.leftBlinker or CS.rightBlinker and self.df_profile != 'traffic':
      x = [8.9408, 22.352, 31.2928]  # 20, 50, 70 mph
      y = [1.0, .75, .65]  # reduce TR when changing lanes
      TR *= interp(self.car_data['v_ego'], x, y)

    return clip(TR, 0.9, 2.7)

  def process_lead(self, v_lead, a_lead, x_lead, status):
    self.lead_data['v_lead'] = v_lead
    self.lead_data['a_lead'] = a_lead
    self.lead_data['x_lead'] = x_lead
    self.lead_data['status'] = status

  def update(self, pm, CS, lead, v_cruise_setpoint):
    v_ego = CS.vEgo
    self.car_data = {'v_ego': CS.vEgo, 'a_ego': CS.aEgo}

    # Setup current mpc state
    self.cur_state[0].x_ego = 0.0

    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = max(0.0, lead.vLead)
      a_lead = lead.aLeadK

      if (v_lead < 0.1 or -a_lead / 2.0 > v_lead):
        v_lead = 0.0
        a_lead = 0.0
      self.process_lead(v_lead, a_lead, x_lead, lead.status)

      self.a_lead_tau = lead.aLeadTau
      self.new_lead = False
      if not self.prev_lead_status or abs(x_lead - self.prev_lead_x) > 2.5:
        self.libmpc.init_with_simulation(self.v_mpc, x_lead, v_lead, a_lead, self.a_lead_tau)
        self.new_lead = True

      self.prev_lead_status = True
      self.prev_lead_x = x_lead
      self.cur_state[0].x_l = x_lead
      self.cur_state[0].v_l = v_lead
    else:
      self.process_lead(None, None, None, False)
      self.prev_lead_status = False
      # Fake a fast lead car, so mpc keeps running
      self.cur_state[0].x_l = 50.0
      self.cur_state[0].v_l = v_ego + 10.0
      a_lead = 0.0
      self.a_lead_tau = _LEAD_ACCEL_TAU

    # Calculate mpc
    t = sec_since_boot()
    n_its = self.libmpc.run_mpc(self.cur_state, self.mpc_solution, self.a_lead_tau, a_lead, self.get_TR(CS))
    duration = int((sec_since_boot() - t) * 1e9)

    if LOG_MPC:
      self.send_mpc_solution(pm, n_its, duration)

    # Get solution. MPC timestep is 0.2 s, so interpolation to 0.05 s is needed
    self.v_mpc = self.mpc_solution[0].v_ego[1]
    self.a_mpc = self.mpc_solution[0].a_ego[1]
    self.v_mpc_future = self.mpc_solution[0].v_ego[10]

    # Reset if NaN or goes through lead car
    crashing = any(lead - ego < -50 for (lead, ego) in zip(self.mpc_solution[0].x_l, self.mpc_solution[0].x_ego))
    nans = any(math.isnan(x) for x in self.mpc_solution[0].v_ego)
    backwards = min(self.mpc_solution[0].v_ego) < -0.01

    if ((backwards or crashing) and self.prev_lead_status) or nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal mpc %d reset - backwards: %s crashing: %s nan: %s" % (
                          self.mpc_id, backwards, crashing, nans))

      self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                       MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
      self.cur_state[0].v_ego = v_ego
      self.cur_state[0].a_ego = 0.0
      self.v_mpc = v_ego
      self.a_mpc = CS.aEgo
      self.prev_lead_status = False
