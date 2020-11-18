from cereal.messaging import SubMaster

sm = SubMaster(['modelV2'], poll=['modelV2'])
model_t = [0, 0.009765625, 0.0390625, 0.087890625, 0.15625, 0.24414062, 0.3515625, 0.47851562, 0.625, 0.79101562, 0.9765625, 1.1816406, 1.40625, 1.6503906, 1.9140625, 2.1972656, 2.5, 2.8222656, 3.1640625, 3.5253906, 3.90625, 4.3066406, 4.7265625, 5.1660156, 5.625, 6.1035156, 6.6015625, 7.1191406, 7.65625, 8.2128906, 8.7890625, 9.3847656, 10]
mpc_idxs = list(range(10))

model_t_idx = [sorted(range(len(model_t)), key=[abs(idx - t) for t in model_t].__getitem__)[0] for idx in mpc_idxs]  # matches 0 to 9 interval to idx from t
# speed_curr_idx = sorted(range(len(model_t)), key=[abs(t - .1) for t in model_t].__getitem__)[0]  # idx used for current speed, position still uses model_t_idx


while 1:
  sm.update()

  modelV2 = sm['modelV2']
  if not sm.updated['modelV2'] or len(modelV2.position.x) == 0:
    continue

  distances = []  # everything is derived from x position since velocity is outputting weird values
  speeds = []
  accelerations = [0]
  for t in model_t_idx:
    distances.append(modelV2.position.x[t])

    # if t == 0:
    #   speeds.append(modelV2.position.x[speed_curr_idx] / model_t[speed_curr_idx])
    # else:
    #   speeds.append(modelV2.position.x[t] / model_t[t])
    speeds.append(modelV2.velocity.x[t])

    if model_t_idx.index(t) > 0:  # skip first since we can't calculate (and don't want to use v_ego)
      accelerations.append((speeds[-1] - speeds[-2]) / model_t[t])

  accelerations[0] = accelerations[1] - (accelerations[2] - accelerations[1])  # extrapolate back first accel from second and third, less weight
  print(speeds[:5])
