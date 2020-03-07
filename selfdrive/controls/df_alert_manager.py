class DfAlertManager:
  def __init__(self, op_params):
    self.op_params = op_params
    self.current_profile = self.op_params.get('dynamic_follow', default='relaxed').lower()
    self.profiles = ['traffic', 'relaxed', 'roadtrip']

    self.idx_to_profile = {0: 'traffic', 1: 'relaxed', 2: 'roadtrip'}
    self.profile_to_idx = {v: k for k, v in self.idx_to_profile.items()}

    self.last_button_status = None

  def run_init(self):
    if self.profile_to_idx[self.current_profile] != 0:  # the following line and loop ensure we start at the user's current profile
      self.idx_to_profile[0] = self.current_profile
      self.profiles.remove(self.current_profile)
      for idx, profile in enumerate(self.profiles):
        self.idx_to_profile[idx + 1] = profile
    self.last_button_status = 0

  def update(self, sm_smiskol):
    if self.last_button_status is None:
      self.run_init()
    else:
      df_profile = sm_smiskol['dynamicFollowButton'].status
      if self.last_button_status != df_profile:
        self.last_button_status = df_profile
        df_profile_string = self.idx_to_profile[df_profile]
        self.op_params.put('dynamic_follow', df_profile_string)  # this sets our param so long_mpc will change profiles
        return df_profile_string

    return None
