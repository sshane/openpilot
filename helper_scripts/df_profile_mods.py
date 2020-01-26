from numpy import interp

p_mod_x = [6.71081, 44.7387, 78.2928]
for v_ego in p_mod_x:
  if v_ego != 78.2928:
    continue
  # roadtrip
  x_vel = [0.0, 4.166748138, 8.333272582, 12.50002072, 16.666768858, 20.833293302, 25.858579012, 30.5230463, 50.00008288, 70.00011603200001, 75.00012432, 80.000132608, 90.00014918400001]
  y_dist = [1.41, 1.419, 1.431, 1.446, 1.47, 1.5, 1.542, 1.563, 1.581, 1.617, 1.653, 1.687, 1.74]  # TRs
  TR_roadtrip = interp(v_ego, x_vel, y_dist)
  roadtrip_mod_pos = [0.85, 0.815, 0.57]
  roadtrip_mod_neg = [1.18, 1.27, 1.675]
  roadtrip_mod_pos = interp(v_ego, p_mod_x, roadtrip_mod_pos)
  roadtrip_mod_neg = interp(v_ego, p_mod_x, roadtrip_mod_neg)

  # relaxed
  x_vel = [0.0, 4.166748138, 8.333272582, 12.50002072, 16.666768858, 20.833293302, 25.858579012, 30.5230463, 50.00008288, 70.00011603200001, 75.00012432, 80.000132608, 90.00014918400001]
  y_dist = [1.385, 1.394, 1.406, 1.421, 1.444, 1.474, 1.516, 1.534, 1.546, 1.568, 1.579, 1.593, 1.614]
  TR_relaxed = interp(v_ego, x_vel, y_dist)
  relaxed_mod_pos = [1.0, 1.0, 1.0]
  relaxed_mod_neg = [1.0, 1.0, 1.0]
  relaxed_mod_pos = interp(v_ego, p_mod_x, relaxed_mod_pos)
  relaxed_mod_neg = interp(v_ego, p_mod_x, relaxed_mod_neg)

  x_rel = [-44.824474802000005, -35.11503673200001, -25.065583782, -17.622837014, -11.275743458, -7.195564898000001, -3.6063946680000005, 0.0, 1.531632818, 3.0807137680000003, 4.251751858, 6.140847688000001]
  y_rel = [0.641, 0.506, 0.418, 0.334, 0.24, 0.115, 0.065, 0.0, -0.049, -0.068, -0.142, -0.221]  # modification values
  TR_mod_pos = interp(-10, x_rel, y_rel)
  TR_mod_neg = interp(3.6, x_rel, y_rel)
  print('v_ego: {}'.format(v_ego))
  print('roadtrip: {}'.format(TR_roadtrip))
  print('pos: {}, neg: {}'.format(TR_roadtrip + TR_mod_pos * roadtrip_mod_pos, TR_roadtrip + TR_mod_neg * roadtrip_mod_neg))
  print()
  print('relaxed: {}'.format(TR_relaxed))
  print('pos: {}, neg: {}'.format(TR_relaxed + TR_mod_pos * relaxed_mod_pos, TR_relaxed + TR_mod_neg * relaxed_mod_neg))
  print('------')