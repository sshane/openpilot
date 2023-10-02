#!/usr/bin/env python3
from tools.lib.framereader import FrameReader
import os
from tools.lib.logreader import LogReader
import json
import matplotlib.pyplot as plt
import numpy as np
from common.basedir import BASEDIR
import copy

import cv2

# FrameReader()
FRAME_SIZE = (1928, 1208)

BODY_DATA_DIR = os.path.join(BASEDIR, 'bad/body-data')
OUT_DIR = os.path.join(BASEDIR, 'bad/data')

# CLASSES = {'nn': [0, 0], 'fn': [0, 0], 'bn': [0, 0], 'nl': [0, 0], 'nr': [0, 0], 'fr': [0, 0], 'fn': [0, 0]}
# CLASSES = {'fl': [-1, 0], 'fn': [-1, 0], 'fr': [-1, 0]}
CLASSES = {'fl': 1, 'fn': 0, 'fr': -1}

for route_name in os.listdir(BODY_DATA_DIR):
  # if 'c452f8ee3bec47d6|2023-10-01--12-49-07' not in route_name:  # route gathered avoiding obstacles
  #   continue
  # print(route_name)
  # if '9bb74d06b7261e2c|2023-09-27--17-02-23' not in route_name:
  #   continue
  for segment_fn in os.listdir(os.path.join(BODY_DATA_DIR, route_name)):
    # if segment_fn != '3':
    #   continue
    # rlog_path = os.path.join(BODY_DATA_DIR, route_name, segment_fn, 'rlog.bz2')
    # dcamera_path = os.path.join(BODY_DATA_DIR, route_name, segment_fn, 'dcamera.hevc')
    rlog_name = next(i for i in os.listdir(os.path.join(BODY_DATA_DIR, route_name, segment_fn)) if i.endswith('.bz2'))
    rlog_path = os.path.join(BODY_DATA_DIR, route_name, segment_fn, rlog_name)
    dcamera_name = next(i for i in os.listdir(os.path.join(BODY_DATA_DIR, route_name, segment_fn)) if i.endswith('.hevc'))
    dcamera_path = os.path.join(BODY_DATA_DIR, route_name, segment_fn, dcamera_name)

    lr = LogReader(rlog_path)
    # fr = FrameReader(dcamera_path)

    all_msgs = sorted(lr, key=lambda m: m.logMonoTime)

    joystick_packets = []
    actuators_packets = []
    cnt = 0
    for msg in all_msgs:
      # TODO: think this is a minimum of ~10Hz?
      if msg.which() == 'testJoystick':
        if len(msg.testJoystick.axes):
          # print(msg.testJoystick.axes)
          joystick_packets.append((msg.logMonoTime, list(msg.testJoystick.axes)))
      elif msg.which() == 'carControl':
        # actuators_packets.append((msg.carControl.actuators.steer, msg.carControl.actuators.accel))
        actuators_packets.append((msg.carControl.actuators.accel / 4, msg.carControl.actuators.steer))
        # print(msg.carControl.actuators.steer, msg.carControl.actuators.accel)

    # fr.get(0, fr.frame_count, pix_fmt='rgb24')
    # print('don')
    # continue
    print(len(actuators_packets), 'len')
    # raise Exception

    vidcap = cv2.VideoCapture(dcamera_path)
    # count = 0
    frames = []
    while 1:
      success, image = vidcap.read()
      if not success:
        break
      image = cv2.resize(image, (round(FRAME_SIZE[0] / 5), round(FRAME_SIZE[1] / 5)))
      frames.append(image)

    prev_frame = None
    for idx, frame in enumerate(frames):
      if idx % 10 != 0:
        continue

      if prev_frame is not None:
        # frames are 20hz, predict half a second in future (10 frames)
        actuators_idx = min(round(np.interp(idx + 10, [0, len(frames)], [0, len(actuators_packets)])),
                            len(actuators_packets) - 1)
        # print(idx, joystick_idx)
        print(idx, actuators_idx)

        if TO_PLOT := False:
          plt.clf()
          plt.imshow(frame)
          # plt.title(str(joystick_packets[joystick_idx]))
          plt.title(str(actuators_packets[actuators_idx]))
          plt.pause(0.5)

        if TO_SAVE := True:
          actuators_pred = actuators_packets[actuators_idx]
          # img_fn = "/mnt/c/Users/Shane/bad/new/dongle{}-seg{}_image{:>4}_y_{}_{}_.png".format(route_name.split('_')[0],
          sample_folder = "{}/bad/data/all/dongle{}-seg{}_sample{:04}_y_{}_{}_".format(BASEDIR,
                                                                                       route_name.split('_')[0],
                                                                                       segment_fn, idx,
                                                                                       round(actuators_pred[0]),
                                                                                       round(actuators_pred[1]))

          os.makedirs(sample_folder, exist_ok=True)
          cv2.imwrite(os.path.join(sample_folder, 'image_0_prev.png'), prev_frame)
          cv2.imwrite(os.path.join(sample_folder, 'image_1_cur.png'), frame)

      prev_frame = copy.copy(frame)

    # frames = fr.get(0, fr.frame_count)
    # print(len(frames), np.array(frames).shape)
    # print(fr.frame_count)
    # for frame_idx in range(fr.frame_count):
    #   print(fr.get(frame_idx, frame_idx))
    # # # fr.get(pix_fmt=)


    # plt.figure()
    # plt.plot([i[0] for i in joystick_packets], 'bo')
    # plt.title(f'{len(joystick_packets)}, {(joystick_packets[-1][0] - joystick_packets[0][0]) * 1e-9}')
    # plt.show()
    # print((joystick_packets[-1][0] - joystick_packets[0][0]) * 1e-9)
    # print((all_msgs[-1].logMonoTime - all_msgs[0].logMonoTime) * 1e-9)
    # print(len(joystick_packets), cnt)

    print(rlog_path, dcamera_path)
    print('Done with seg', route_name, segment_fn)
    # raise Exception('Done with first seg!')
  print()


raise Exception
# TEST_SEG = '/mnt/c/Users/Shane/Downloads/body-data/c81675c456f9f72d_2023-09-27--17-23-04/5'
TEST_SEG = '/mnt/c/Users/Shane/Downloads/body-data/c81675c456f9f72d_2023-09-27--17-23-04/17'
vidcap = cv2.VideoCapture(f'{TEST_SEG}/dcamera.hevc')
lr = LogReader(f'{TEST_SEG}/rlog.bz2')

tjs = 0
train_y = []
for msg in sorted(lr, key=lambda x: x.logMonoTime):
  if msg.which() == 'testJoystick':
    if len(msg.testJoystick.axes):
      print(msg.testJoystick.axes)
      train_y.append(msg.testJoystick.axes)  # fwd/bck is idx 0, left/right is idx 1
      tjs += 1
# raise Exception
print(tjs)

# forward is -1, backward is 1
# left is 1, right is -1
count = 0
success = True
every = True
while success:
  success, image = vidcap.read()
  image = cv2.resize(image, (round(FRAME_SIZE[0] / 3), round(FRAME_SIZE[1] / 3)))
  if count % 20 == 0:
    train = 'train' if every else 'test'
    every = not every
    img_fn = "/mnt/c/Users/Shane/bad/new/{}/image{:>3}_y_{}_{}_seg17.png".format(train, count,
                                                                                 round(train_y[count + 10][0]),
                                                                                 round(train_y[count + 10][1]))
    cv2.imwrite(img_fn, image)
    print(img_fn)
  print(image.shape)
  # break
  print('Read a new frame: ', success, count)
  count += 1
