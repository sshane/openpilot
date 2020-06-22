import numpy as np
import matplotlib.pyplot as plt
import common.transformations.orientation as orient

d_poly = [1.17423917e-06, -0.000185162149, -0.00574009353, -0.0913085938]
x = np.linspace(0, 182, 182)
y = np.polyval(d_poly, x)


device_frame_from_view_frame = np.array([
  [ 0.,  0.,  1.],
  [ 1.,  0.,  0.],
  [ 0.,  1.,  0.]
])
eon_intrinsics = np.array([[910.0, 0.0, 582.0],
                           [0.0, 910.0, 437.0],
                           [0.0, 0.0, 1.0]])

ground_from_medmodel_frame = np.array([[0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                       [-1.09890110e-03, 0.00000000e+00, 2.81318681e-01],
                                       [-1.84808520e-20, 9.00738606e-04, -4.28751576e-02]])

FULL_FRAME_SIZE = (1164, 874)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0

intrinsic_matrix = np.array([
  [FOCAL,   0.,   W/2.],
  [  0.,  FOCAL,  H/2.],
  [  0.,    0.,     1.]])

MODEL_PATH_MAX_VERTICES_CNT = 98

view_frame_from_device_frame = device_frame_from_view_frame.T


roll = 0
pitch = 0
yaw = 0
height = 1
device_from_road = orient.rot_from_euler([roll, pitch, yaw]).dot(np.diag([1, -1, -1]))
view_from_road = view_frame_from_device_frame.dot(device_from_road)
extrinsic_matrix = np.hstack((view_from_road, [[0], [height], [0]])).flatten()

extrinsic_matrix_eigen = np.zeros((3, 4))
i = 0
while i < 4*3:
  print(int(i / 4), int(i % 4))
  extrinsic_matrix_eigen[int(i / 4), int(i % 4)] = extrinsic_matrix[i]

  i += 1

# camera_frame_from_road_frame = eon_intrinsics * extrinsic_matrix_eigen
camera_frame_from_road_frame = np.dot(eon_intrinsics, extrinsic_matrix_eigen)

camera_frame_from_ground = np.zeros((3, 3))
camera_frame_from_ground[:,0] = camera_frame_from_road_frame[:,0]
camera_frame_from_ground[:,1] = camera_frame_from_road_frame[:,1]
camera_frame_from_ground[:,2] = camera_frame_from_road_frame[:,3]
# camera_frame_from_ground.col(0) = camera_frame_from_road_frame.col(0);
# camera_frame_from_ground.col(1) = camera_frame_from_road_frame.col(1);
# camera_frame_from_ground.col(2) = camera_frame_from_road_frame.col(3);
# warp_matrix = camera_frame_from_ground * ground_from_medmodel_frame
warp_matrix = np.dot(camera_frame_from_ground, ground_from_medmodel_frame)

cur_transform_v = np.zeros((3*3))
i = 0
while i < 3*3:
  cur_transform_v[i] = warp_matrix[int(i / 3), int(i % 3)]
  i += 1

i = 0
new_x = []
new_y = []
OFF = 0.7
while i < MODEL_PATH_MAX_VERTICES_CNT / 2:
  _x = x[i]
  _y = y[i]
  p_car_space = np.array([_x, _y, 0., 1.])
  Ep4 = np.matmul(extrinsic_matrix_eigen, p_car_space)
  Ep = np.array([Ep4[0], Ep4[1], Ep4[2]])
  KEp = np.dot(intrinsic_matrix, Ep)
  p_image = p_full_frame = np.array([KEp[0] / KEp[2], KEp[1] / KEp[2], 1.])
  print(p_image)
  new_x.append(p_full_frame[0])
  new_y.append(p_full_frame[1])

  i += 1

plt.plot(new_x, new_y, label='transformed')


# ep4 = np.matmul(extrinsic_matrix, p_car_space)


# plt.plot(x, y, label='original')
plt.legend()
plt.show()
