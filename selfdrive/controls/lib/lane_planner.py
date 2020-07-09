import numpy as np
from common.op_params import opParams
from common.numpy_fast import interp
from cereal import log
from cereal import messaging
from cereal.messaging import SubMaster, PubMaster
from selfdrive.config import Conversions as CV


def compute_path_pinv(l=50):
  deg = 3
  x = np.arange(l*1.0)
  X = np.vstack(tuple(x**n for n in range(deg, -1, -1))).T
  pinv = np.linalg.pinv(X)
  return pinv


def model_polyfit(points, path_pinv):
  return np.dot(path_pinv, [float(x) for x in points])


def eval_poly(poly, x):
  return poly[3] + poly[2]*x + poly[1]*x**2 + poly[0]*x**3


def calc_d_poly(l_poly, r_poly, p_poly, l_prob, r_prob, lane_width, v_ego):
  # This will improve behaviour when lanes suddenly widen
  # these numbers were tested on 2000segments and found to work well
  lane_width = min(4.0, lane_width)
  width_poly = l_poly - r_poly
  prob_mods = []
  for t_check in [0.0, 1.5, 3.0]:
    width_at_t = eval_poly(width_poly, t_check * (v_ego + 7))
    prob_mods.append(interp(width_at_t, [4.0, 5.0], [1.0, 0.0]))
  mod = min(prob_mods)
  l_prob = mod * l_prob
  r_prob = mod * r_prob

  path_from_left_lane = l_poly.copy()
  path_from_left_lane[3] -= lane_width / 2.0
  path_from_right_lane = r_poly.copy()
  path_from_right_lane[3] += lane_width / 2.0

  lr_prob = l_prob + r_prob - l_prob * r_prob

  d_poly_lane = (l_prob * path_from_left_lane + r_prob * path_from_right_lane) / (l_prob + r_prob + 0.0001)
  return lr_prob * d_poly_lane + (1.0 - lr_prob) * p_poly


class DynamicCameraOffset:
  def __init__(self):
    self.sm = SubMaster(['laneSpeed'])
    self.pm = PubMaster(['dynamicCameraOffset'])
    self.op_params = opParams()
    self.camera_offset = self.op_params.get('camera_offset', 0.06)
    self.leftLaneOncoming = False
    self.rightLaneOncoming = False
    self.min_enable_speed = 15 * CV.MPH_TO_MS
    self.min_lane_width_certainty = 0.3
    self.hug_left_ratio = 0.25
    self.hug_right_ratio = 0.75  # todo: verify these are correct, think they are now

    self.i = 0  # current integral, always changing
    self.i_rate = 1 / 20

    self.poly_prob_speeds = [0, 25 * CV.MPH_TO_MS, 35 * CV.MPH_TO_MS, 60 * CV.MPH_TO_MS]
    self.poly_probs = [0.1, 0.2, 0.55, 0.65]  # lane line must exist in direction we're offsetting towards

  def update(self, v_ego, lane_width_estimate, lane_width_certainty, l_poly, r_poly, l_prob, r_prob):
    self.sm.update(0)
    self.camera_offset = self.op_params.get('camera_offset', 0.06)
    self.leftLaneOncoming = self.sm['laneSpeed'].leftLaneOncoming
    self.rightLaneOncoming = self.sm['laneSpeed'].rightLaneOncoming
    self.l_prob = l_prob
    self.r_prob = r_prob

    dynamic_offset = self._get_camera_offset(v_ego, lane_width_estimate, lane_width_certainty, l_poly, r_poly)
    self._send_state()  # for alerts, before speed check so alerts don't get stuck on
    if dynamic_offset is not None:
      self.integral = 0
      return dynamic_offset
    return self.camera_offset  # don't offset if no lane line in direction we're going to hug

  def _send_state(self):
    dco_send = messaging.new_message('dynamicCameraOffset')
    dco_send.dynamicCameraOffset.keepingLeft = self.keeping_left
    dco_send.dynamicCameraOffset.keepingRight = self.keeping_right
    self.pm.send('dynamicCameraOffset', dco_send)

  def _get_camera_offset(self, v_ego, lane_width_estimate, lane_width_certainty, l_poly, r_poly):
    self.keeping_left, self.keeping_right = False, False  # reset keeping
    if self.leftLaneOncoming == self.rightLaneOncoming:  # if both false or both true do nothing
      return
    if v_ego < self.min_enable_speed:
      return
    if np.isnan(l_poly[0]) or np.isnan(r_poly[0]):
      return
    min_poly_prob = 0.0  # np.interp(v_ego, self.poly_prob_speeds, self.poly_probs)
    if self.l_prob < min_poly_prob or self.r_prob < min_poly_prob:  # require both lane lines
      return
    if lane_width_certainty < self.min_lane_width_certainty:  # we need to know the current lane width
      return
    left_line_pos = l_poly[3] + self.camera_offset  # just for accuracy when calculating where car is in lane
    right_line_pos = r_poly[3] + self.camera_offset  # todo: possibly factor in an average car width so we don't go over line?
    estimated_lane_pos_left = left_line_pos / lane_width_estimate  # estimated position of car in lane based on left line
    estimated_lane_pos_right = 1 - abs(right_line_pos) / lane_width_estimate  # estimated position of car in lane based on right line

    # find car's lane position using weighted average of lane poly certainty
    # if certainty of both lines are high, then just average equally
    l_prob = self.l_prob / (self.l_prob + self.r_prob)  # this and next line sum to 1
    r_prob = self.r_prob / (self.l_prob + self.r_prob)
    # be biases towards position found from most probable lane line. left is 1, right is 0
    estimated_lane_position = estimated_lane_pos_left * l_prob + estimated_lane_pos_right * r_prob  # should be a float from 0 to 1, 0.5 is center

    k_i = 0.5  # integral gain, needs to be tuned
    # k_p = 1.0  # proportional gain, needs to be tuned
    k_p = self.op_params.get('dyn_camera_offset_p', 1.0)  # proportional gain, needs to be tuned

    if self.leftLaneOncoming:
      self.keeping_right = True
      error = estimated_lane_position - self.hug_right_ratio
      # self.i = self.i + (error * self.k_i * self.i_rate)  # todo: integral
      p = error * k_p
      offset = p  # + self.i
    else:  # right lane oncoming
      self.keeping_left = True
      error = estimated_lane_position - self.hug_left_ratio
      # self.i = self.i + (error * self.k_i * self.i_rate)  # todo: integral
      p = error * k_p
      offset = p  # + self.i
    return self.camera_offset + offset


class LanePlanner():
  def __init__(self):
    self.l_poly = [0., 0., 0., 0.]
    self.r_poly = [0., 0., 0., 0.]
    self.p_poly = [0., 0., 0., 0.]
    self.d_poly = [0., 0., 0., 0.]

    self.lane_width_estimate = 3.7
    self.lane_width_certainty = 1.0
    self.lane_width = 3.7

    self.l_prob = 0.
    self.r_prob = 0.

    self.l_lane_change_prob = 0.
    self.r_lane_change_prob = 0.

    self._path_pinv = compute_path_pinv()
    self.x_points = np.arange(50)
    self.dynamic_camera_offset = DynamicCameraOffset()

  def parse_model(self, md):
    if len(md.leftLane.poly):
      self.l_poly = np.array(md.leftLane.poly)
      self.r_poly = np.array(md.rightLane.poly)
      self.p_poly = np.array(md.path.poly)
    else:
      self.l_poly = model_polyfit(md.leftLane.points, self._path_pinv)  # left line
      self.r_poly = model_polyfit(md.rightLane.points, self._path_pinv)  # right line
      self.p_poly = model_polyfit(md.path.points, self._path_pinv)  # predicted path
    self.l_prob = md.leftLane.prob  # left line prob
    self.r_prob = md.rightLane.prob  # right line prob

    if len(md.meta.desireState):
      self.l_lane_change_prob = md.meta.desireState[log.PathPlan.Desire.laneChangeLeft - 1]
      self.r_lane_change_prob = md.meta.desireState[log.PathPlan.Desire.laneChangeRight - 1]

  def update_d_poly(self, v_ego):
    # only offset left and right lane lines; offsetting p_poly does not make sense (or does it?)
    CAMERA_OFFSET = self.dynamic_camera_offset.update(v_ego, self.lane_width, self.lane_width_certainty, self.l_poly, self.r_poly, self.l_prob, self.r_prob)
    self.l_poly[3] += CAMERA_OFFSET
    self.r_poly[3] += CAMERA_OFFSET
    # self.p_poly[3] += CAMERA_OFFSET

    # Find current lanewidth
    self.lane_width_certainty += 0.05 * (self.l_prob * self.r_prob - self.lane_width_certainty)
    current_lane_width = abs(self.l_poly[3] - self.r_poly[3])
    self.lane_width_estimate += 0.005 * (current_lane_width - self.lane_width_estimate)
    speed_lane_width = interp(v_ego, [0., 31.], [2.8, 3.5])
    self.lane_width = self.lane_width_certainty * self.lane_width_estimate + \
                      (1 - self.lane_width_certainty) * speed_lane_width

    self.d_poly = calc_d_poly(self.l_poly, self.r_poly, self.p_poly, self.l_prob, self.r_prob, self.lane_width, v_ego)

  def update(self, v_ego, md):
    self.parse_model(md)
    self.update_d_poly(v_ego)
