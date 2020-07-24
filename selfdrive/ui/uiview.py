#!/usr/bin/env python3
import os
import subprocess
import multiprocessing
import signal
import time

import cereal.messaging as messaging
from common.params import Params

from common.basedir import BASEDIR

KILL_TIMEOUT = 15


def send_controls_packet(pm):
  while True:
    dat = messaging.new_message('controlsState')
    dat.controlsState = {
      "rearViewCam": False,
    }
    pm.send('controlsState', dat)
    time.sleep(0.01)

def send_thermal_packet(pm):
  while True:
    dat = messaging.new_message('thermal')
    dat.thermal = {
      'started': True,
    }
    pm.send('thermal', dat)
    time.sleep(0.01)

def main():
  pm = messaging.PubMaster(['controlsState', 'thermal'])
  controls_sender = multiprocessing.Process(target=send_controls_packet, args=[pm])
  thermal_sender = multiprocessing.Process(target=send_thermal_packet, args=[pm])
  controls_sender.start()
  thermal_sender.start()

  # TODO: refactor with manager start/kill
  proc_cam = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/camerad/camerad"), cwd=os.path.join(BASEDIR, "selfdrive/camerad"))
  proc_ui = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/ui/ui"), cwd=os.path.join(BASEDIR, "selfdrive/ui"))

  params = Params()
  is_rhd_checked = False
  should_exit = False

  def terminate(signalNumber, frame):
    print('got SIGTERM, exiting..')
    should_exit = True
    proc_cam.send_signal(signal.SIGINT)
    proc_ui.send_signal(signal.SIGINT)
    kill_start = time.time()
    while proc_cam.poll() is None:
      if time.time() - kill_start > KILL_TIMEOUT:
        from selfdrive.swaglog import cloudlog
        cloudlog.critical("FORCE REBOOTING PHONE!")
        os.system("date >> /sdcard/unkillable_reboot")
        os.system("reboot")
        raise RuntimeError
      continue
    controls_sender.terminate()
    exit()

  signal.signal(signal.SIGTERM, terminate)

  # while True:
  #   time.sleep(0.01)


if __name__ == '__main__':
  main()
