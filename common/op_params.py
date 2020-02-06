import os
import json
import time
import string
import random
from common.travis_checker import travis


def write_params(params, params_file):
  if not travis:
    with open(params_file, "w") as f:
      json.dump(params, f, indent=2, sort_keys=True)
    os.chmod(params_file, 0o764)


def read_params(params_file, default_params):
  try:
    with open(params_file, "r") as f:
      params = json.load(f)
    return params, True
  except Exception as e:
    print(e)
    params = default_params
    return params, False


class opParams:
  def __init__(self):
    """
      To add your own parameter to opParams in your fork, simply add a new dictionary entry with the name of your parameter and its default value to save to new users' op_params.json file.
      The description, allowed_types, and live keys are no longer required but recommended to help users edit their parameters with opEdit and opTune correctly.
        - The description value will be shown to users when they use opEdit or opTune to change the value of the parameter.
        - The allowed_types key is used to restrict what kinds of values can be entered with opEdit so that users can't reasonably break the fork with unintended behavior.
          Limiting the range of floats or integers is still recommended when `.get`ting the parameter.
          When a None value is allowed, use `type(None)` instead of None, as opEdit checks the type against the values in the key with `isinstance()`.
        - Finally, the live key tells both opParams and opTune that it's a live parameter that will change. Therefore, you must place the `op_params.get()` call in the update function so that it can update.
      Here's an example of the minimum required dictionary:

      self.default_params = {'camera_offset': {'default': 0.06}}
    """

    self.default_params = {'camera_offset': {'default': 0.06, 'allowed_types': [float, int], 'description': 'Your camera offset to use in lane_planner.py', 'live': True},
                           'awareness_factor': {'default': 3.0, 'allowed_types': [float, int], 'description': 'Multiplier for the awareness times', 'live': False},
                           'lane_hug_direction': {'default': None, 'allowed_types': [type(None), str], 'description': "(None, 'left', 'right'): Direction of your lane hugging, if present. None will disable this modification", 'live': False},
                           'lane_hug_angle_offset': {'default': 0.0, 'allowed_types': [float, int], 'description': ('This is the angle your wheel reads when driving straight at highway speeds.\n'
                                                                                                                    'Replaces both offsets from the calibration learner to help fix lane hugging.\n'
                                                                                                                    'Enter absolute value here, direction is determined by parameter \'lane_hug_direction\''), 'live': True},
                           'dynamic_follow': {'default': 'relaxed', 'allowed_types': [str], 'description': "Can be: ('traffic', 'relaxed', 'roadtrip'): Left to right increases in following distance.\n"
                                                                                                           "All profiles support dynamic follow so you'll get your preferred distance while\n"
                                                                                                           "retaining the smoothness and safety of dynamic follow!", 'live': True},
                           'alca_nudge_required': {'default': True, 'allowed_types': [bool], 'description': ('Whether to wait for applied torque to the wheel (nudge) before making lane changes. '
                                                                                                             'If False, lane change will occur IMMEDIATELY after signaling'), 'live': False},
                           'alca_min_speed': {'default': 25.0, 'allowed_types': [float, int], 'description': 'The minimum speed allowed for an automatic lane change (in MPH)', 'live': False},
                           'steer_ratio': {'default': None, 'allowed_types': [type(None), float, int], 'description': 'Can be: (None, 17.8, 14.6) If you enter None, openpilot will use the learned sR.\n'
                                                                                                                      'If you use a number, openpilot will use that steer ratio instead', 'live': True},
                           'use_dynamic_lane_speed': {'default': True, 'allowed_types': [bool], 'description': 'Whether you want openpilot to adjust your speed based on surrounding vehicles', 'live': False},
                           'min_dynamic_lane_speed': {'default': 20.0, 'allowed_types': [float, int], 'description': 'The minimum speed to allow dynamic lane speed to operate (in MPH)', 'live': False},
                           'upload_on_hotspot': {'default': False, 'allowed_types': [bool], 'description': 'If False, openpilot will not upload driving data while connected to your phone\'s hotspot', 'live': False},
                           'reset_integral': {'default': False, 'allowed_types': [bool], 'description': 'This resets integral whenever the longitudinal PID error crosses or is zero.\nShould help it recover from overshoot quicker', 'live': False}}

    self.params = {}
    self.params_file = "/data/op_params.json"
    self.kegman_file = "/data/kegman.json"
    self.last_read_time = time.time()
    self.read_frequency = 5.0  # max frequency to read with self.get(...) (sec)
    self.force_update = False  # replaces values with default params if True, not just add add missing key/value pairs
    self.to_delete = ['dynamic_lane_speed', 'longkiV', 'following_distance', 'static_steer_ratio']
    self.run_init()  # restores, reads, and updates params

  def create_id(self):  # creates unique identifier to send with sentry errors. please update uniqueID with op_edit.py to your username!
    need_id = False
    if "uniqueID" not in self.params:
      need_id = True
    if "uniqueID" in self.params and self.params["uniqueID"] is None:
      need_id = True
    if need_id:
      random_id = ''.join([random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for i in range(15)])
      self.params["uniqueID"] = random_id

  def add_default_params(self):
    prev_params = dict(self.params)
    if not travis:
      self.create_id()
      for key in self.default_params:
        if self.force_update:
          self.params[key] = self.default_params[key]['default']
        elif key not in self.params:
          self.params[key] = self.default_params[key]['default']
    return prev_params == self.params

  def format_default_params(self):
    return {key: self.default_params[key]['default'] for key in self.default_params}

  def run_init(self):  # does first time initializing of default params, and/or restoring from kegman.json
    if travis:
      self.params = self.format_default_params()
      return
    self.params = self.format_default_params()  # in case any file is corrupted
    to_write = False
    no_params = False
    if os.path.isfile(self.params_file):
      self.params, read_status = read_params(self.params_file, self.format_default_params())
      if read_status:
        to_write = not self.add_default_params()  # if new default data has been added
        if self.delete_old():  # or if old params have been deleted
          to_write = True
      else:  # don't overwrite corrupted params, just print to screen
        print("ERROR: Can't read op_params.json file")
    elif os.path.isfile(self.kegman_file):
      to_write = True  # write no matter what
      try:
        with open(self.kegman_file, "r") as f:  # restore params from kegman
          self.params = json.load(f)
          self.add_default_params()
      except:
        print("ERROR: Can't read kegman.json file")
    else:
      no_params = True  # user's first time running a fork with kegman_conf or op_params
    if to_write or no_params:
      write_params(self.params, self.params_file)

  def delete_old(self):
    prev_params = dict(self.params)
    for i in self.to_delete:
      if i in self.params:
        del self.params[i]
    return prev_params == self.params

  def put(self, key, value):
    self.params.update({key: value})
    write_params(self.params, self.params_file)

  def get(self, key=None, default=None, force_update=False):  # can specify a default value if key doesn't exist
    self.update_params(key, force_update)
    if key is None:
      return self.params
    return self.params[key] if key in self.params else default

  def update_params(self, key, force_update):
    if force_update or (key in self.default_params and 'live' in self.default_params[key] and self.default_params[key]['live']):  # if is a live param, we want to get updates while openpilot is running
      if not travis and time.time() - self.last_read_time >= self.read_frequency:  # make sure we aren't reading file too often
        self.params, read_status = read_params(self.params_file, self.format_default_params())
        if not read_status:
          time.sleep(1/100.)
          self.params, _ = read_params(self.params_file, self.format_default_params())  # if the file was being written to, retry once
        self.last_read_time = time.time()

  def delete(self, key):
    if key in self.params:
      del self.params[key]
      write_params(self.params, self.params_file)
