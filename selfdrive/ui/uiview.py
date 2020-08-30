#!/usr/bin/env python3
import os
import time
import signal
import subprocess
import multiprocessing
import cereal.messaging as messaging

from common.basedir import BASEDIR

KILL_TIMEOUT = 15


def send_controls_packet(pm):
  while True:
    dat = messaging.new_message('controlsState')
    dat.controlsState.rearViewCam = False
    pm.send('controlsState', dat)
    time.sleep(1 / 100.)


def send_thermal_packet(pm):
  while True:
    dat = messaging.new_message('thermal')
    dat.thermal.started = True
    pm.send('thermal', dat)
    time.sleep(1 / 2.)  # 2 hz


def main():
  pm = messaging.PubMaster(['controlsState', 'thermal'])
  controls_sender = multiprocessing.Process(target=send_controls_packet, args=[pm])
  controls_sender.start()
  thermal_sender = multiprocessing.Process(target=send_thermal_packet, args=[pm])
  thermal_sender.start()

  # TODO: refactor with manager start/kill
  proc_cam = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/camerad/camerad"), cwd=os.path.join(BASEDIR, "selfdrive/camerad"))
  proc_ui = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/ui/ui"), cwd=os.path.join(BASEDIR, "selfdrive/ui"))
  proc_mod = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/modeld/modeld"), cwd=os.path.join(BASEDIR, "selfdrive/modeld"))
  proc_cal = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/locationd/calibrationd.py"), cwd=os.path.join(BASEDIR, "selfdrive/locationd"))

  def terminate(signalNumber, frame):
    print('got SIGTERM, exiting..')
    proc_cam.send_signal(signal.SIGINT)
    proc_ui.send_signal(signal.SIGINT)
    proc_cal.send_signal(signal.SIGINT)
    thermal_sender.terminate()
    controls_sender.terminate()
    proc_mod.terminate()
    proc_cal.terminate()
  exit()

  signal.signal(signal.SIGTERM, terminate)
  signal.signal(signal.SIGINT, terminate)  # catch ctrl-c as well


if __name__ == '__main__':
  main()
