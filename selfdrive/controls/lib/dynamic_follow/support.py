class LeadData:
  v_lead = None
  x_lead = None
  a_lead = None
  status = False
  new_lead = False


class CarData:
  v_ego = 0.0
  a_ego = 0.0


class dfData:
  v_leads = []
  v_egos = []


class dfProfiles:
  traffic = 0
  relaxed = 1
  roadtrip = 2
  auto = 3
  to_profile = {0: 'traffic', 1: 'relaxed', 2: 'roadtrip', 3: 'auto'}
  to_idx = {v: k for k, v in to_profile.items()}