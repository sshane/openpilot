#!/usr/bin/env python3
import os
import json
try:
  from common.realtime import sec_since_boot
except ImportError:
  import time
  sec_since_boot = time.time
  print("opParams WARNING: using python time.time() instead of faster sec_since_boot")

travis = False


# class KeyInfo:
#   default = None
#   allowed_types = []
#   is_list = False
#   has_allowed_types = False
#   live = False
#   has_default = False
#   has_description = False
#   hidden = False

class ValueTypes:
  number = [float, int]
  none_or_number = [type(None), float, int]


class Param:
  def __init__(self, default, allowed_types=None, description=None, hidden=False):
    self.default = default
    if not isinstance(allowed_types, list):
      allowed_types = [allowed_types]
    self.allowed_types = allowed_types
    self.description = description
    self.live = False
    self.hidden = hidden
    self._create_attrs()

  def set_live(self):
    self.live = True

  def _create_attrs(self):
    self.has_allowed_types = isinstance(self.allowed_types, list) and len(self.allowed_types) > 0
    self.has_description = self.description is not None
    if self.has_allowed_types:
      assert type(self.default) in self.allowed_types, 'Default value type must be in specified allowed_types!'


class opParams:
  def __init__(self):
    # self.default_params = {'camera_offset': Param(0.06, [int, float], 'Your camera offset to use in lane_planner.py')}

    VT = ValueTypes()
    self.fork_params = {'camera_offset': Param(0.06, VT.number, 'Your camera offset to use in lane_planner.py'),
                        'dynamic_follow': Param('auto', str, 'Can be: (\'traffic\', \'relaxed\', \'roadtrip\'): Left to right increases in following distance.\n'
                                                             'All profiles support dynamic follow so you\'ll get your preferred distance while\n'
                                                             'retaining the smoothness and safety of dynamic follow!'),
                        'global_df_mod': Param(None, VT.none_or_number, 'The multiplier for the current distance used by dynamic follow. The range is limited from 0.85 to 1.2\n'
                                                                        'Smaller values will get you closer, larger will get you farther\n'
                                                                        'This is multiplied by any profile that\'s active. Set to None to disable'),
                        'min_TR': Param(None, VT.none_or_number, 'The minimum allowed following distance in seconds. Default is 0.9 seconds.\n'
                                                                 'The range is limited from 0.85 to 1.3. Set to None to disable'),
                        'alca_nudge_required': Param(True, bool, 'Whether to wait for applied torque to the wheel (nudge) before making lane changes. '
                                                                 'If False, lane change will occur IMMEDIATELY after signaling'),
                        'alca_min_speed': Param(25.0, VT.number, 'The minimum speed allowed for an automatic lane change (in MPH)'),
                        'steer_ratio': Param(None, VT.none_or_number, '(Can be: None, or a float) If you enter None, openpilot will use the learned sR.\n'
                                                                      'If you use a float/int, openpilot will use that steer ratio instead'),
                        'lane_speed_alerts': Param('silent', str, 'Can be: (\'off\', \'silent\', \'audible\')\n'
                                                                  'Whether you want openpilot to alert you of faster-traveling adjacent lanes'),
                        'upload_on_hotspot': Param(False, bool, 'If False, openpilot will not upload driving data while connected to your phone\'s hotspot'),
                        'enable_long_derivative': Param(False, bool, 'If you have longitudinal overshooting, enable this! This enables derivative-based\n'
                                                                     'integral wind-down to help reduce overshooting within the long PID loop'),
                        'disengage_on_gas': Param(True, bool, 'Whether you want openpilot to disengage on gas input or not'),
                        'no_ota_updates': Param(False, bool, 'Set this to True to disable all automatic updates. Reboot to take effect'),
                        'dynamic_gas': Param(True, bool, 'Whether to use dynamic gas if your car is supported'),
                        'hide_auto_df_alerts': Param(False, bool, 'Hides the alert that shows what profile the model has chosen'),
                        'log_auto_df': Param(False, bool, 'Logs dynamic follow data for auto-df'),
                        'dynamic_camera_offset': Param(True, bool, 'Whether to automatically keep away from oncoming traffic.\n'
                                                                   'Works from 35 to ~60 mph (requires radar)'),
                        'dynamic_camera_offset_time': Param(2.5, VT.number, 'How long to keep away from oncoming traffic in seconds after losing lead'),
                        'support_white_panda': Param(False, bool, 'Enable this to allow engagement with the deprecated white panda.\n'
                                                                  'localizer might not work correctly'),
                        'prius_use_lqr': Param(False, bool, 'If you have a newer Prius with a good angle sensor, you can try enabling this to use LQR'),


                        'op_edit_live_mode': Param(False, bool, 'This parameter controls which mode opEdit starts in. It should be hidden from the user with the hide key', hidden=True)}

    live_params = ['camera_offset', 'global_df_mod', 'min_TR', 'steer_ratio']
    [self.fork_params[p].set_live() for p in live_params]

    self.params = {}
    self.params_file = "/data/op_params.json"
    self.last_read_time = sec_since_boot()
    self.read_frequency = 2.5  # max frequency to read with self.get(...) (sec)
    self.force_update = False  # replaces values with default params if True, not just add add missing key/value pairs
    self.to_delete = ['reset_integral', 'dyn_camera_offset_i', 'dyn_camera_offset_p', 'lane_hug_direction', 'lane_hug_angle_offset']  # a list of params you want to delete (unused)
    # self.run_init()  # restores, reads, and updates params

  def run_init(self):  # does first time initializing of default params
    if travis:
      self.params = self._format_default_params()
      return

    self.params = self._format_default_params()  # in case any file is corrupted

    to_write = False
    if os.path.isfile(self.params_file):
      if self._read():
        to_write = not self._add_default_params()  # if new default data has been added
        to_write = self._delete_old or to_write  # or if old params have been deleted
      else:  # don't overwrite corrupted params, just print
        print("opParams ERROR: Can't read op_params.json file")
    else:
      to_write = True  # user's first time running a fork with op_params, write default params

    if to_write:
      self._write()
      os.chmod(self.params_file, 0o764)

  def get(self, key=None, default=None, force_update=False):  # can specify a default value if key doesn't exist
    if key is None:
      return self._get_all()

    key_info = self.key_info(key)
    self._update_params(key_info, force_update)
    if key in self.params:
      if key_info.has_allowed_types:
        value = self.params[key]
        if type(value) in key_info.allowed_types:
          return value  # all good, returning user's value

        print('opParams WARNING: User\'s value is not valid!')
        if key_info.has_default:  # invalid value type, try to use default value
          if type(key_info.default) in key_info.allowed_types:  # actually check if the default is valid
            # return default value because user's value of key is not in the allowed_types to avoid crashing openpilot
            return key_info.default

        return self._value_from_types(key_info.allowed_types)  # else use a standard value based on type (last resort to keep openpilot running if user's value is of invalid type)
      else:
        return self.params[key]  # no defined allowed types, returning user's value

    return default  # not in params

  def put(self, key, value):
    self.params.update({key: value})
    self._write()

  def delete(self, key):
    if key in self.params:
      del self.params[key]
      self._write()

  def key_info(self, key):
    key_info = KeyInfo()
    if key is None or key not in self.default_params:
      return key_info
    if key in self.default_params:
      if 'allowed_types' in self.default_params[key]:
        allowed_types = self.default_params[key]['allowed_types']
        if isinstance(allowed_types, list) and len(allowed_types) > 0:
          key_info.has_allowed_types = True
          key_info.allowed_types = list(allowed_types)
          if list in [type(typ) for typ in allowed_types]:
            key_info.is_list = True
            key_info.allowed_types.remove(list)
            key_info.allowed_types = key_info.allowed_types[0]

      if 'live' in self.default_params[key]:
        key_info.live = self.default_params[key]['live']

      if 'default' in self.default_params[key]:
        key_info.has_default = True
        key_info.default = self.default_params[key]['default']

      key_info.has_description = 'description' in self.default_params[key]

      if 'hide' in self.default_params[key]:
        key_info.hidden = self.default_params[key]['hide']

    return key_info

  def _add_default_params(self):
    prev_params = dict(self.params)
    for key in self.default_params:
      if self.force_update:
        self.params[key] = self.default_params[key]['default']
      elif key not in self.params:
        self.params[key] = self.default_params[key]['default']
    return prev_params == self.params

  def _format_default_params(self):
    return {key: self.default_params[key]['default'] for key in self.default_params}

  @property
  def _delete_old(self):
    deleted = False
    for param in self.to_delete:
      if param in self.params:
        del self.params[param]
        deleted = True
    return deleted

  def _get_all(self):  # returns all non-hidden params
    return {k: v for k, v in self.params.items() if not self.key_info(k).hidden}

  def _value_from_types(self, allowed_types):
    if list in allowed_types:
      return []
    elif float in allowed_types or int in allowed_types:
      return 0
    elif type(None) in allowed_types:
      return None
    elif str in allowed_types:
      return ''
    return None  # unknown type

  def _update_params(self, key_info, force_update):
    if force_update or key_info.live:  # if is a live param, we want to get updates while openpilot is running
      if not travis and (sec_since_boot() - self.last_read_time >= self.read_frequency or force_update):  # make sure we aren't reading file too often
        if self._read():
          self.last_read_time = sec_since_boot()

  def _read(self):
    try:
      with open(self.params_file, "r") as f:
        self.params = json.loads(f.read())
      return True
    except Exception as e:
      print('opParams ERROR: {}'.format(e))
      self.params = self._format_default_params()
      return False

  def _write(self):
    if not travis:
      with open(self.params_file, "w") as f:
        f.write(json.dumps(self.params, indent=2))  # can further speed it up by remove indentation but makes file hard to read
