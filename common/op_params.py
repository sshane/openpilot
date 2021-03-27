#!/usr/bin/env python3
import os
import json
from common.colors import COLORS
from common.travis_checker import travis
try:
  from common.realtime import sec_since_boot
except ImportError:
  import time
  sec_since_boot = time.time

warning = lambda msg: print('{}opParams WARNING: {}{}'.format(COLORS.WARNING, msg, COLORS.ENDC))
error = lambda msg: print('{}opParams ERROR: {}{}'.format(COLORS.FAIL, msg, COLORS.ENDC))

NUMBER = [float, int]  # value types
NONE_OR_NUMBER = [type(None), float, int]

BASE_DIR = '/data' if not travis else '/tmp'
PARAMS_PATH = os.path.join(BASE_DIR, 'community', 'params')
IMPORTED_PATH = os.path.join(PARAMS_PATH, '.imported')
OLD_PARAMS_FILE = os.path.join(BASE_DIR, 'op_params.json')


class Param:
  def __init__(self, default=None, allowed_types=[], description=None, *, static=False, live=True, hidden=False):
    self.default_value = default  # value first saved and returned if actual value isn't a valid type
    if not isinstance(allowed_types, list):
      allowed_types = [allowed_types]
    self.allowed_types = allowed_types  # allowed python value types for opEdit
    self.description = description  # description to be shown in opEdit
    self.hidden = hidden  # hide this param to user in opEdit
    self.live = live  # show under the live menu in opEdit
    self.static = static  # use cached value, never reads to update
    self._create_attrs()

  def is_valid(self, value):
    if not self.has_allowed_types:  # always valid if no allowed types, otherwise checks to make sure
      return True
    return type(value) in self.allowed_types

  def _create_attrs(self):  # Create attributes and check Param is valid
    self.has_allowed_types = isinstance(self.allowed_types, list) and len(self.allowed_types) > 0
    self.has_description = self.description is not None
    self.is_list = list in self.allowed_types
    self.read_frequency = None if self.static else (1 if self.live else 10)  # how often to read param file (sec)
    if self.has_allowed_types:
      assert type(self.default_value) in self.allowed_types, 'Default value type must be in specified allowed_types!'
    if self.is_list:
      self.allowed_types.remove(list)


def _read_param(key):  # Returns None, False if a json error occurs
  try:
    with open(os.path.join(PARAMS_PATH, key), 'r') as f:
      value = json.loads(f.read())
    return value, True
  except json.decoder.JSONDecodeError:
    return None, False


def _write_param(key, value):
  tmp = os.path.join(PARAMS_PATH, '.' + key)
  with open(tmp, 'w') as f:
    f.write(json.dumps(value))
    f.flush()
    os.fsync(f.fileno())
  os.rename(tmp, os.path.join(PARAMS_PATH, key))
  os.chmod(os.path.join(PARAMS_PATH, key), 0o777)


def _import_params(can_import):
  needs_import = False  # if opParams needs to import from old params file
  if not os.path.exists(PARAMS_PATH):
    os.makedirs(PARAMS_PATH)
    needs_import = True
  needs_import &= os.path.exists(OLD_PARAMS_FILE)
  needs_import &= not os.path.exists(IMPORTED_PATH)
  needs_import &= can_import

  if needs_import:
    try:
      with open(OLD_PARAMS_FILE, 'r') as f:
        old_params = json.loads(f.read())
      for key in old_params:
        if not os.path.exists(os.path.join(PARAMS_PATH, key)):
          _write_param(key, old_params[key])
      return True, old_params
    except:
      pass
  return False, None


class opParams:
  def __init__(self):
    """
      To add your own parameter to opParams in your fork, simply add a new entry in self.fork_params, instancing a new Param class with at minimum a default value.
      The allowed_types and description args are not required but highly recommended to help users edit their parameters with opEdit safely.
        - The description value will be shown to users when they use opEdit to change the value of the parameter.
        - The allowed_types arg is used to restrict what kinds of values can be entered with opEdit so that users can't crash openpilot with unintended behavior.
              (setting a param intended to be a number with a boolean, or viceversa for example)
          Limiting the range of floats or integers is still recommended when `.get`ting the parameter.
          When a None value is allowed, use `type(None)` instead of None, as opEdit checks the type against the values in the arg with `isinstance()`.
        - Finally, the live arg tells both opParams and opEdit that it's a live parameter that will change. Therefore, you must place the `op_params.get()` call in the update function so that it can update.

      Here's an example of a good fork_param entry:
      self.fork_params = {'camera_offset': Param(default=0.06, allowed_types=NUMBER), live=True}  # NUMBER allows both floats and ints
    """

    self.fork_params = {'camera_offset': Param(0.06, NUMBER, 'Your camera offset to use in lane_planner.py'),
                        'dynamic_follow': Param('auto', str, 'Can be: (\'traffic\', \'relaxed\', \'stock\'): Left to right increases in following distance.\n'
                                                             'All profiles support dynamic follow except stock so you\'ll get your preferred distance while\n'
                                                             'retaining the smoothness and safety of dynamic follow!', static=True),
                        'global_df_mod': Param(1.0, NUMBER, 'The multiplier for the current distance used by dynamic follow. The range is limited from 0.85 to 2.5\n'
                                                            'Smaller values will get you closer, larger will get you farther\n'
                                                            'This is multiplied by any profile that\'s active. Set to 1. to disable'),
                        'min_TR': Param(0.9, NUMBER, 'The minimum allowed following distance in seconds. Default is 0.9 seconds.\n'
                                                     'The range is limited from 0.85 to 1.6.'),
                        'alca_nudge_required': Param(True, bool, 'Whether to wait for applied torque to the wheel (nudge) before making lane changes. '
                                                                 'If False, lane change will occur IMMEDIATELY after signaling'),
                        'alca_min_speed': Param(25.0, NUMBER, 'The minimum speed allowed for an automatic lane change (in MPH)'),
                        'steer_ratio': Param(None, NONE_OR_NUMBER, '(Can be: None, or a float) If you enter None, openpilot will use the learned sR.\n'
                                                                   'If you use a float/int, openpilot will use that steer ratio instead'),
                        # 'lane_speed_alerts': Param('silent', str, 'Can be: (\'off\', \'silent\', \'audible\')\n'
                        #                                           'Whether you want openpilot to alert you of faster-traveling adjacent lanes'),
                        'upload_on_hotspot': Param(False, bool, 'If False, openpilot will not upload driving data while connected to your phone\'s hotspot'),
                        'enable_long_derivative': Param(False, bool, 'If you have longitudinal overshooting, enable this! This enables derivative-based\n'
                                                                     'integral wind-down to help reduce overshooting within the long PID loop'),
                        'disengage_on_gas': Param(False, bool, 'Whether you want openpilot to disengage on gas input or not'),
                        'update_behavior': Param('auto', str, 'Can be: (\'off\', \'alert\', \'auto\') without quotes\n'
                                                              'off will never update, alert shows an alert on-screen\n'
                                                              'auto will reboot the device when an update is seen', static=True),
                        'dynamic_gas': Param(False, bool, 'Whether to use dynamic gas if your car is supported'),
                        'hide_auto_df_alerts': Param(False, bool, 'Hides the alert that shows what profile the model has chosen'),
                        'log_auto_df': Param(False, bool, 'Logs dynamic follow data for auto-df'),
                        # 'dynamic_camera_offset': Param(False, bool, 'Whether to automatically keep away from oncoming traffic.\n'
                        #                                             'Works from 35 to ~60 mph (requires radar)'),
                        # 'dynamic_camera_offset_time': Param(3.5, NUMBER, 'How long to keep away from oncoming traffic in seconds after losing lead'),
                        'support_white_panda': Param(False, bool, 'Enable this to allow engagement with the deprecated white panda.\n'
                                                                  'localizer might not work correctly', static=True),
                        'disable_charging': Param(30, NUMBER, 'How many hours until charging is disabled while idle', static=True),

                        'prius_use_pid': Param(False, bool, 'This enables the PID lateral controller with new a experimental derivative tune\n'
                                                            'False: stock INDI, True: TSS2-tuned PID', static=True),
                        'use_lqr': Param(False, bool, 'Enable this to use LQR as your lateral controller over default with any car', static=True),
                        'corollaTSS2_use_indi': Param(False, bool, 'Enable this to use INDI for lat with your TSS2 Corolla', static=True),
                        'rav4TSS2_use_indi': Param(False, bool, 'Enable this to use INDI for lat with your TSS2 RAV4', static=True),
                        'standstill_hack': Param(False, bool, 'Some cars support stop and go, you just need to enable this', static=True)}

    self._to_delete = ['steer_rate_fix', 'uniqueID']  # a list of unused params you want to delete from users' params file
    self._to_reset = []  # a list of params you want reset to their default values
    self._run_init()  # restores, reads, and updates params

  def _run_init(self):  # does first time initializing of default params
    # Two required parameters for opEdit
    self.fork_params['username'] = Param(None, [type(None), str, bool], 'Your identifier provided with any crash logs sent to Sentry.\nHelps the developer reach out to you if anything goes wrong')
    self.fork_params['op_edit_live_mode'] = Param(False, bool, 'This parameter controls which mode opEdit starts in', hidden=True)

    self.params = self._load_params(can_import=not travis)
    self._add_default_params()  # adds missing params and resets values with invalid types to self.params
    self._delete_and_reset()  # removes old params
    self._last_read_times = {p: sec_since_boot() for p in self.params}

  def get(self, key=None, *, force_update=False):  # key=None returns dict of all params
    if key is None:
      return self._get_all_params(to_update=force_update)
    self._check_key_exists(key, 'get')
    param_info = self.fork_params[key]
    rate = param_info.read_frequency  # will be None if param is static, so check below

    if (not param_info.static and sec_since_boot() - self._last_read_times[key] >= rate) or force_update:
      value, success = _read_param(key)
      self._last_read_times[key] = sec_since_boot()
      if not success:  # in case of read error, use default and overwrite param
        value = param_info.default_value
        _write_param(key, value)
      self.params[key] = value

    if param_info.is_valid(value := self.params[key]):
      return value  # all good, returning user's value
    print(warning('User\'s value type is not valid! Returning default'))  # somehow... it should always be valid
    return param_info.default_value  # return default value because user's value of key is not in allowed_types to avoid crashing openpilot

  def put(self, key, value):
    self._check_key_exists(key, 'put')
    if not self.fork_params[key].is_valid(value):
      raise Exception('opParams: Tried to put a value of invalid type!')
    self.params.update({key: value})
    _write_param(key, value)

  def _load_params(self, can_import=False):
    ret = _import_params(can_import)  # returns success (bool), params (dict)
    if ret[0]:
      open(IMPORTED_PATH, 'w').close()
      return ret[1]

    params = {}
    for key in os.listdir(PARAMS_PATH):  # PARAMS_PATH is guaranteed to exist
      if key.startswith('.') or key not in self.fork_params:
        continue
      value, success = _read_param(key)
      if not success:
        value = self.fork_params[key].default_value
        _write_param(key, value)
      params[key] = value
    return params

  def _get_all_params(self, to_update=False):
    if to_update:
      self.params = self._load_params()
    return {k: self.params[k] for k, p in self.fork_params.items() if k in self.params and not p.hidden}

  def _check_key_exists(self, key, met):
    if key not in self.fork_params:
      raise Exception('opParams: Tried to {} an unknown parameter! Key not in fork_params: {}'.format(met, key))

  def _add_default_params(self):
    for key, param in self.fork_params.items():
      if key not in self.params:
        self.params[key] = param.default_value
        _write_param(key, self.params[key])
      elif not param.is_valid(self.params[key]):
        print(warning('Value type of user\'s {} param not in allowed types, replacing with default!'.format(key)))
        self.params[key] = param.default_value
        _write_param(key, self.params[key])

  def _delete_and_reset(self):
    for key in list(self.params):
      if key in self._to_delete:
        del self.params[key]
        os.remove(os.path.join(PARAMS_PATH, key))
      elif key in self._to_reset and key in self.fork_params:
        self.params[key] = self.fork_params[key].default_value
        _write_param(key, self.params[key])
