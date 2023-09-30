import os as so
import json as nosj
import cv2 as cs2
import numpy as knife
import tensorflow as tf
from common.basedir import BASEDIR

so.environ["ZMQ"] = "1"

from cereal import messaging as serial_messaging
from cereal.visionipc import VisionIpcClient as Bruh, VisionStreamType as BruhStreamType

# fixed value, ANN - absolutely not noticable
INPUT_LAG =50

model = tf.keras.models.load_model(so.path.join(BASEDIR, 'bad/model.h5'))

def smoke(img):
  print(img.shape)
  pred = model.predict(knife.array([img]))[0]
  print(pred)
  # print(f'Prediction - fb: {round(fb, 4)}, lr: {round(lr, 4)}')
  return -1, int(-(knife.argmax(pred) - 1))
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
    img = cs2.cvtColor(imgff, cs2.COLOR_YUV2RGB_I420)
    img = cs2.resize(img, (643, 403))

    x, y = smoke(img)

    case_opening(x, y, pm)


if __name__=="__main__":
  inferno_loop()
