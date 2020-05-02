from selfdrive.controls.lib.dynamic_follow import dfProfiles


class dfAlertManager:
  def __init__(self, op_params):
    self.op_params = op_params
    self.df_profiles = dfProfiles()
    self.current_profile = self.df_profiles.to_idx[self.op_params.get('dynamic_follow', default='relaxed').strip().lower()]

    self.offset = None
    self.profile_pred = None
    self.last_button_status = 0

  @property
  def is_auto(self):
    return self.current_profile == self.df_profiles.auto

  def update(self, sm):
    changed = False
    if self.offset is None:
      changed = True
      self.offset = self.current_profile  # ensure we start at the user's current profile
    else:
      status = sm['dynamicFollowButton'].status
      new_profile = (status + self.offset) % len(self.df_profiles.to_profile)
      if self.last_button_status != status:
        changed = True
        self.current_profile = new_profile
        self.last_button_status = status
      elif self.is_auto:
        profile_pred = sm['dynamicFollowData'].profilePred
        if profile_pred != self.profile_pred:
          changed = True
          self.current_profile = profile_pred
    return self.current_profile, changed
