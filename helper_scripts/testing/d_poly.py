import numpy as np
l_prob = 0.25
r_prob = 0.25
path_from_left_lane = np.array([0.25, 0.25, 0.25, 0.25])
path_from_right_lane = np.array([0.75, 0.75, 0.75, 0.75])
lr_prob = l_prob + r_prob - l_prob * r_prob
d_poly_lane = (l_prob * path_from_left_lane + r_prob * path_from_right_lane) / (l_prob + r_prob + 0.0001)
print(lr_prob)
print(d_poly_lane)  # should be [0.5, 0.5, 0.5, 0.5]

p_poly = np.array([1, 1, 1, 1])
print(lr_prob * d_poly_lane + (1.0 - lr_prob) * p_poly)
