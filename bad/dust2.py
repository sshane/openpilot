import os as so
import json as nosj
import cv2 as cs2
import numpy as knife
import tensorflow as tf
from common.basedir import BASEDIR
import time
from common.filter_simple import FirstOrderFilter
import math
import copy as flash
import matplotlib.pyplot as plt
from collections import deque

so.environ["ZMQ"] = "1"

from cereal import messaging as serial_messaging
from cereal.visionipc import VisionIpcClient as Bruh, VisionStreamType as BruhStreamType

# fixed value, ANN - absolutely not noticable
INPUT_LAG =50

# model = tf.keras.models.load_model(so.path.join(BASEDIR, 'bad/models/model-high-acc-low-valacc.h5'))
model = tf.keras.models.load_model(so.path.join(BASEDIR, 'bad/models/model.h5'))

fof = FirstOrderFilter(0, 0.2, 0.05)
lrf = FirstOrderFilter(0, 0, 0.05)
fbf = FirstOrderFilter(0, 0, 0.05)
frame_deque = deque([], 10)

def random_argmax(probs):
  # return sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)[0][0]
  return knife.random.choice(len(probs), p=probs)


def smoke(img, prev_img):
  print('img shapes', img.shape, prev_img.shape)
  pred = model.predict([[knife.array([img]), knife.array([prev_img])]])[0]
  # bumped_pred = (math.e ** (pred * 5) - 1)
  # bumped_pred = pred * 2
  # bumped_pred[0] = -0.8
  # print(bumped_pred)
  # return 0, 0
  # return knife.clip(bumped_pred, -1, 1).tolist()
  # # pred[3] *= 0.5  # max(pred[3] - 0.1, 0)
  # # pred[1] *= 2  # max(pred[3] - 0.1, 0)
  # idx = random_argmax(pred)
  idx = knife.argmax(pred)

  print(pred, idx)
  idx = round(fof.update(idx))
  print(idx)

  # print(random_argmax(pred))
  # print(knife.round(pred, 3).tolist())

  if idx == 0:
    return fbf.update(1), lrf.update(0)  # BACK
  elif idx == 1:
    return fbf.update(-0.8), lrf.update(0)  # FWD
  elif idx == 2:
    return fbf.update(-0.1), lrf.update(1)
  elif idx == 3:
    return fbf.update(-0.1), lrf.update(-1)
  else:
    return fbf.update(0), lrf.update(0)  # NOTHING

  # print(f'Prediction - fb: {round(fb, 4)}, lr: {round(lr, 4)}')
  # return int(-(knife.argmax(pred[0]) - 1)), int(-(knife.argmax(pred[1]) - 1))
  # return round(fb), round(lr)

def case_opening(x, y, pm):
  print("flashbang!")
  msg = serial_messaging.new_message()
  msg.customReservedRawData1 = nosj.dumps({"x": x, "y": y}).encode()
  pm.send('customReservedRawData1', msg)

def inferno_loop():
  pm = serial_messaging.PubMaster(['customReservedRawData1'])
  del so.environ["ZMQ"]
  client = Bruh("camerad", BruhStreamType.VISION_STREAM_DRIVER, True)
  client.connect(True)

  while True:
    frame = client.recv()
    if frame is None or not frame.data.any():
      continue

    imgff = frame.data.reshape(-1, client.stride)
    imgff = imgff[:client.height * 3 // 2, :client.width]
    img = cs2.cvtColor(imgff, cs2.COLOR_YUV2BGR_NV12)
    img = cs2.resize(img, (386, 242))
    img = cs2.cvtColor(img, cs2.COLOR_BGR2GRAY)
    img.reshape(img.shape[0], img.shape[1], 1)

    # plt.imshow(img)
    # plt.pause(10)

    if len(frame_deque) == 10:
      print('sending')
      x, y = smoke(img / 255., frame_deque[0] / 255.)

      case_opening(x, y, pm)
      # time.sleep(0.2)
    frame_deque.append(flash.copy(img))


if __name__=="__main__":
  inferno_loop()
