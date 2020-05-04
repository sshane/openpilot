from numpy import interp


def to_mph(x):
  return [i * 2.23694 for i in x]

def to_ms(x):
  return [i * 0.44704 for i in x]

p_mod_x = [5., 30., 55., 80.]
for v_ego in p_mod_x:
  if v_ego != 5.:
    continue
  # roadtrip
  x_vel = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]
  x_vel = to_mph(x_vel)
  y_dist = [1.4109, 1.4196, 1.431, 1.4454, 1.4684, 1.4971, 1.5397, 1.56, 1.588, 1.624, 1.66, 1.694, 1.747]  # TRs
  TR_traffic = interp(v_ego, x_vel, y_dist)

  traffic_mod_pos = [0.98, 0.915, 0.83, 0.55]
  traffic_mod_neg = [1.02, 1.18, 1.39, 1.825]

  traffic_mod_pos = interp(v_ego, p_mod_x, traffic_mod_pos)
  traffic_mod_neg = interp(v_ego, p_mod_x, traffic_mod_neg)

  # relaxed
  x_vel = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]
  x_vel = to_mph(x_vel)
  y_dist = [1.385, 1.394, 1.406, 1.421, 1.444, 1.474, 1.516, 1.534, 1.546, 1.568, 1.579, 1.593, 1.614]
  TR_relaxed = interp(v_ego, x_vel, y_dist)
  relaxed_mod_pos = [1.0, 1.0, 1.0, 1.0]
  relaxed_mod_neg = [1.0, 1.0, 1.0, 1.0]
  relaxed_mod_pos = interp(v_ego, p_mod_x, relaxed_mod_pos)
  relaxed_mod_neg = interp(v_ego, p_mod_x, relaxed_mod_neg)


  x_rel = [-44.824474802000005, -35.11503673200001, -25.065583782, -17.622837014, -11.275743458, -7.195564898000001, -3.6063946680000005, 0.0, 1.531632818, 3.0807137680000003, 4.251751858, 6.140847688000001]
  y_rel = [0.641, 0.506, 0.418, 0.334, 0.24, 0.115, 0.065, 0.0, -0.049, -0.068, -0.142, -0.221]  # modification values
  TR_mod_pos = interp(-10, x_rel, y_rel)
  TR_mod_neg = interp(3.6, x_rel, y_rel)
  print('v_ego: {}'.format(v_ego))
  print('traffic: {}'.format(TR_traffic))
  print('pos: {}, neg: {}'.format(TR_traffic + TR_mod_pos * traffic_mod_pos, TR_traffic + TR_mod_neg * traffic_mod_neg))
  print()
  print('relaxed: {}'.format(TR_relaxed))
  print('pos: {}, neg: {}'.format(TR_relaxed + TR_mod_pos * relaxed_mod_pos, TR_relaxed + TR_mod_neg * relaxed_mod_neg))
  print('------')