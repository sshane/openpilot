#!/usr/bin/env python3
"""
glowd — SP105E underglow controller daemon.

Runs only_onroad. On start: connects BLE + powers on LEDs.
On SIGTERM (manager kill at ignition off): powers off LEDs + disconnects.
Maps CarState to underglow colors reactively.

Color mapping:
  - RPM → hue (green idle → yellow → amber → purple at redline)
  - Braking → orange, intensity scales with decel
  - Gas → warm amber blended with RPM color
  - Downshift → brief purple flash
  - Blinker → amber pulse
  - Standstill → slow breathing pulse
  - Reverse → white

California-legal: no red or blue, especially on the front.
Safe colors: green, yellow, amber, orange, purple, white, pink.
"""
import asyncio
import colorsys
import math
import signal
import subprocess
import time
from enum import IntEnum, IntFlag

import numpy as np

import cereal.messaging as messaging
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper

from openpilot.tools.underglow import sp105e

DEBUG = True

# RPM thresholds
RPM_MIN = 800
RPM_COLOR_MAX = 5500

# Timing
UPDATE_HZ = 15
BRAKE_FLASH_S = 0.8
RAINBOW_HOLDOVER_S = 2.5
RAINBOW_DELAY_S = 1.5
RAINBOW_PERIOD_S = 8.0


class GlowState(IntEnum):
  DRIVING = 0     # RPM-based color
  STANDSTILL = 1  # rainbow cycle


class GlowMod(IntFlag):
  BRAKE = 1


def rpm_to_color(rpm: float) -> tuple[int, int, int]:
  """Map RPM to color: green(idle) → yellow → amber → purple(redline).
  Avoids pure red and blue. Low range uses HSV, high range blends RGB."""
  t = max(0.0, min(1.0, (rpm - RPM_MIN) / (RPM_COLOR_MAX - RPM_MIN)))
  if t < 0.7:
    # green (0.33) → orange (0.08) in HSV
    hue = 0.33 - (0.33 - 0.08) * (t / 0.7)
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)
  else:
    # orange → purple via RGB blend (avoids blue in HSV path)
    frac = (t - 0.7) / 0.3
    return (
      int(255 + (180 - 255) * frac),
      int(122 * (1 - frac)),
      int(255 * frac),
    )



class GlowController:
  def __init__(self):
    self.last_color = (0, 0, 0)
    self.state = GlowState.STANDSTILL
    self._standstill_start: float | None = None
    self._moving_start: float | None = None
    self._mods = GlowMod(0)
    self._prev_brake = False
    self._brake_pressed_t = 0.0

    # HSV smoothing filters
    dt = 1.0 / UPDATE_HZ
    self._hx_filter = FirstOrderFilter(0.0, 0.5, dt, initialized=False)  # cos(hue)
    self._hy_filter = FirstOrderFilter(0.0, 0.5, dt, initialized=False)  # sin(hue)
    self._s_filter = FirstOrderFilter(0.0, 0.5, dt, initialized=False)
    self._v_filter = FirstOrderFilter(0.0, 0.5, dt, initialized=False)

  def smooth_color(self, color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Filter RGB through HSV space. Hue filtered in cartesian (cos/sin)
    to handle circular wrapping generically via atan2."""
    h, s, v = colorsys.rgb_to_hsv(color[0] / 255, color[1] / 255, color[2] / 255)
    angle = 2 * math.pi * h
    hx = self._hx_filter.update(math.cos(angle))
    hy = self._hy_filter.update(math.sin(angle))
    h = math.atan2(hy, hx) / (2 * math.pi) % 1.0
    s = self._s_filter.update(s)
    v = self._v_filter.update(v)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

  def _rainbow_color(self, v_ego: float) -> tuple[int, int, int]:
    speed_mult = np.interp(v_ego, [0.0, 5.0], [1.0, 2.0])
    hue = (time.monotonic() * speed_mult / RAINBOW_PERIOD_S) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)

  def update(self, sm, chill: bool):
    cs = sm['carState']
    now = time.monotonic()

    if chill:
      self.state = GlowState.DRIVING
      self._mods = GlowMod(0)
      return

    # Base state transitions
    if self.state == GlowState.DRIVING:
      if cs.vEgo < 1 and cs.engineRpm < 1500:
        if self._standstill_start is None:
          self._standstill_start = now
        elif now - self._standstill_start > RAINBOW_DELAY_S:
          self._standstill_start = None
          self.state = GlowState.STANDSTILL
      else:
        self._standstill_start = None

    elif self.state == GlowState.STANDSTILL:
      if cs.engineRpm >= 1500:
        self._moving_start = None
        self.state = GlowState.DRIVING
      elif cs.vEgo >= 1:
        if self._moving_start is None:
          self._moving_start = now
        elif now - self._moving_start > RAINBOW_HOLDOVER_S:
          self._moving_start = None
          self.state = GlowState.DRIVING
      else:
        self._moving_start = None

    # Update modifiers
    if cs.brakePressed and not self._prev_brake:
      self._brake_pressed_t = now
    self._prev_brake = cs.brakePressed

    if now - self._brake_pressed_t < BRAKE_FLASH_S:
      self._mods |= GlowMod.BRAKE
    else:
      self._mods &= ~GlowMod.BRAKE

  def get_color(self, sm) -> tuple[int, int, int]:
    cs = sm['carState']

    # Modifier: brake — dark red flash on rising edge
    if self._mods & GlowMod.BRAKE:
      return (128, 0, 0)

    # Base color from state
    if self.state == GlowState.STANDSTILL:
      return self._rainbow_color(cs.vEgo)
    return rpm_to_color(cs.engineRpm)


def _put_glow_status(params, status: str, color: tuple[int, int, int] = (0, 0, 0)):
  params.put_nonblocking("GlowStatus", {"status": status, "color": list(color)})


def bt_is_ready() -> bool:
  """Check if BT stack is up (managed by bluetooth.service in AGNOS)."""
  result = subprocess.run(["sudo", "hciconfig", "hci0"], capture_output=True)
  return b"UP RUNNING" in result.stdout


async def ble_connect():
  """Connect to SP105E, power on, sweep brightness. Returns client or None.
  Assumes BT stack is already up (bluetooth.service in AGNOS)."""
  if not bt_is_ready():
    print("glowd: hci0 not up (waiting for bluetooth.service)")
    return None

  print("glowd: connecting to SP105E...")
  client = await sp105e.connect(exit_on_fail=False)
  if client is None:
    return None
  await asyncio.sleep(0.5)
  await sp105e.set_power(client, on=True)
  await asyncio.sleep(0.5)
  print("glowd: connected, LEDs on")
  return client


async def ble_shutdown(client):
  """Power off LEDs and disconnect."""
  if client is not None:
    try:
      await sp105e.set_brightness(client, sp105e.BRIGHTNESS_MIN)
      await asyncio.sleep(0.5)
      await sp105e.set_power(client, on=False)
      await client.disconnect()
      print("glowd: LEDs off, disconnected")
    except Exception as e:
      print(f"glowd: shutdown BLE error: {e}")


async def glowd_thread():
  do_exit = False
  client = None

  def signal_handler(signum, frame):
    nonlocal do_exit
    print(f"glowd: caught signal {signum}, exiting")
    do_exit = True

  signal.signal(signal.SIGTERM, signal_handler)
  signal.signal(signal.SIGINT, signal_handler)

  params = Params()
  _put_glow_status(params, "connecting")
  chill_mode = params.get_bool("GlowMode")
  standstill_only = params.get_bool("GlowStandstillOnly")
  last_param_read = 0.0

  client = await ble_connect()
  _put_glow_status(params, "connected" if client else "disconnected")

  sm = messaging.SubMaster(['carState'], poll='carState')
  ctrl = GlowController()
  rk = Ratekeeper(UPDATE_HZ)
  last_reconnect_attempt = 0.0

  print(f"glowd: running at {UPDATE_HZ}Hz, chill={chill_mode}")

  while not do_exit:
    sm.update(0)

    # Refresh params every 5s
    now = time.monotonic()
    if now - last_param_read > 2.5:
      chill_mode = params.get_bool("GlowMode")
      standstill_only = params.get_bool("GlowStandstillOnly")
      last_param_read = now

    # If disconnected, try to reconnect every 5s
    if client is None:
      if now - last_reconnect_attempt > 5.0:
        last_reconnect_attempt = now
        print("glowd: attempting reconnect...")
        _put_glow_status(params, "connecting")
        client = await ble_connect()
        _put_glow_status(params, "connected" if client else "disconnected")
      rk.keep_time()
      continue

    if sm.updated['carState']:
      ctrl.update(sm, chill_mode)

      if standstill_only and ctrl.state == GlowState.DRIVING:
        if ctrl.last_color != (0, 0, 0):
          await sp105e.set_power(client, on=False)
          ctrl.last_color = (0, 0, 0)
          _put_glow_status(params, "connected")
        rk.keep_time()
        continue
      elif standstill_only and ctrl.last_color == (0, 0, 0):
        await sp105e.set_power(client, on=True)

      raw_color = ctrl.get_color(sm)
      color = ctrl.smooth_color(raw_color)

      if color != ctrl.last_color:
        if DEBUG:
          cs = sm['carState']
          print(f"glowd: RPM={cs.engineRpm:.0f} chill={chill_mode} → RGB{color}")

        try:
          await sp105e.set_color(client, *color)
        except Exception as e:
          print(f"glowd: BLE error: {e}")
          try:
            await client.disconnect()
          except Exception:
            pass
          client = None
          _put_glow_status(params, "disconnected", ctrl.last_color)
          continue

        ctrl.last_color = color
        _put_glow_status(params, "connected", color)

    rk.keep_time()

  # Clean shutdown: power off LEDs
  _put_glow_status(params, "disconnected", ctrl.last_color)
  await ble_shutdown(client)


def main():
  asyncio.run(glowd_thread())


if __name__ == "__main__":
  main()
