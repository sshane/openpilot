# from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader

import cv2

# FrameReader()
FRAME_SIZE = (1928, 1208)

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
  if count % 5 == 0:
    train = 'train' if every else 'test'
    every = not every
    img_fn = "/mnt/c/Users/Shane/bad/{}/image{:>3}_y_{}_{}_seg17.png".format(train, count,
                                                                        round(train_y[count + 10][0]),
                                                                        round(train_y[count + 10][1]))
    cv2.imwrite(img_fn, image)
    print(img_fn)
  print(image.shape)
  # break
  print('Read a new frame: ', success, count)
  count += 1
