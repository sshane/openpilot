#!/usr/bin/env python3
import os
import time
import signal
import subprocess
import cereal.messaging as messaging
from common.basedir import BASEDIR

services = ['controlsState', 'deviceState', 'radarState']  # the services needed to be spoofed to start ui offroad
procs = {'camerad': 'selfdrive/camerad/camerad', 'ui': 'selfdrive/ui/ui',
         'modeld': 'selfdrive/modeld/modeld', 'calibrationd': 'selfdrive/locationd/calibrationd.py'}
started_procs = [subprocess.Popen(os.path.join(BASEDIR, procs[p]), cwd=os.path.join(BASEDIR, os.path.dirname(procs[p]))) for p in procs]  # start needed processes
pm = messaging.PubMaster(services)

dat_cs, dat_ds, dat_radar = [messaging.new_message(s) for s in services]
dat_cs.controlsState.rearViewCam = False  # ui checks for these two messages
dat_ds.deviceState.started = True

try:
  while True:
    pm.send('controlsState', dat_cs)
    pm.send('deviceState', dat_ds)
    pm.send('radarState', dat_radar)
    time.sleep(1 / 20)  # continually send, rate doesn't matter
except KeyboardInterrupt:
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLING')
  [p.send_signal(signal.SIGINT) for p in started_procs]
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLING')
  print('KILLED')
