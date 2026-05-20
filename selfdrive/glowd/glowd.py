#!/usr/bin/env python3
"""
glowd — SP105E underglow controller daemon.

Runs only_onroad. On start: connects BLE, powers on LEDs, sets brightness.
On SIGTERM (manager kill at ignition off): dims to min, powers off, disconnects.

Color mapping:
  - RPM → hue (green idle → yellow → amber → purple at redline)
  - RPM rate-of-change → bounce filter overshoots color on downshifts/rev matches
  - Standstill → safe rainbow (green ↔ yellow, filter-friendly)
  - Standstill 60s+ → full hue rainbow

All colors smoothed through HSV filter (cos/sin for hue wrapping).
BLE writes skip frames when lagging to prevent queue snowball.
California-legal: no red or blue, especially on the front.
"""
import asyncio
import colorsys
import math
import signal
import subprocess
import time
from enum import IntEnum, IntFlag

import cereal.messaging as messaging
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper

from openpilot.tools.underglow import sp105e

DEBUG = True

# RPM thresholds
RPM_MIN = 1500
RPM_COLOR_MAX = 5500

DEFAULT_BRIGHTNESS = 4  # ~57%, level 0-6
DIFF_SCALE = 2.5        # multiplier on rpm-baseline diff for color mapping

# Timing
UPDATE_HZ = 15
BRAKE_FLASH_S = 0.4
RAINBOW_HOLDOVER_S = 2.5
RAINBOW_DELAY_S = 1.5
RAINBOW_PERIOD_S = 8.0
FULL_RAINBOW_DELAY_S = 60.0


class GlowState(IntEnum):
  DRIVING = 0          # RPM-based color
  STANDSTILL = 1       # safe rainbow (green ↔ amber/purple, filter-friendly)
  STANDSTILL_FULL = 2  # full rainbow (after 1min standstill)


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
    dt = 1.0 / UPDATE_HZ

    # RPM baseline filter — tracks steady-state RPM, color driven by rpm - baseline
    self._rpm_baseline = FirstOrderFilter(0.0, 5.0, dt, initialized=False)

    # HSV smoothing filters
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

  def _rainbow_safe_color(self) -> tuple[int, int, int]:
    """Cycle green(0.33) ↔ yellow(0.14). Smooth enough for the HSV filter."""
    t = (time.monotonic() / RAINBOW_PERIOD_S) % 1.0
    # Ping-pong between green and yellow
    hue = 0.14 + (0.33 - 0.14) * (0.5 + 0.5 * math.sin(2 * math.pi * t))
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)

  def _rainbow_full_color(self) -> tuple[int, int, int]:
    """Full hue cycle. Only used after extended standstill."""
    hue = (time.monotonic() / RAINBOW_PERIOD_S) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)

  def update(self, sm):
    cs = sm['carState']
    now = time.monotonic()

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

    elif self.state in (GlowState.STANDSTILL, GlowState.STANDSTILL_FULL):
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

      # Upgrade to full rainbow after extended standstill
      if self.state == GlowState.STANDSTILL:
        if self._standstill_start is None:
          self._standstill_start = now
        elif now - self._standstill_start > FULL_RAINBOW_DELAY_S:
          self._standstill_start = None
          self.state = GlowState.STANDSTILL_FULL

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
    # if self._mods & GlowMod.BRAKE:
    #   return (128, 0, 0)

    # Base color from state
    if self.state == GlowState.STANDSTILL:
      return self._rainbow_safe_color()
    if self.state == GlowState.STANDSTILL_FULL:
      return self._rainbow_full_color()
    # Color driven by rpm - baseline: cruise = green, rev changes = color
    baseline = self._rpm_baseline.update(cs.engineRpm)
    diff = max(0, cs.engineRpm - baseline) * DIFF_SCALE
    return rpm_to_color(RPM_MIN + diff)


def _put_glow_status(params, status: str, color: tuple[int, int, int] = (0, 0, 0)):
  params.put("GlowStatus", {"status": status, "color": list(color)})


def bt_is_ready() -> bool:
  """Check if BT stack is up (managed by bluetooth.service in AGNOS)."""
  result = subprocess.run(["sudo", "hciconfig", "hci0"], capture_output=True)
  return b"UP RUNNING" in result.stdout


async def ble_connect(brightness: int = DEFAULT_BRIGHTNESS):
  """Connect to SP105E, power on, set brightness. Returns client or None."""
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
  await sp105e.set_brightness(client, brightness)
  print(f"glowd: connected, LEDs on, brightness={brightness}")
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
  if not params.get_bool("GlowEnabled"):
    print("glowd: disabled via GlowEnabled param")
    _put_glow_status(params, "disabled")
    return

  _put_glow_status(params, "connecting")
  brightness = params.get("GlowBrightness") or DEFAULT_BRIGHTNESS

  client = await ble_connect(brightness)
  _put_glow_status(params, "connected" if client else "disconnected")

  sm = messaging.SubMaster(['carState'], poll='carState')
  ctrl = GlowController()
  rk = Ratekeeper(UPDATE_HZ)
  last_reconnect_attempt = 0.0

  print(f"glowd: running at {UPDATE_HZ}Hz")

  while not do_exit:
    sm.update(0)

    now = time.monotonic()

    # If disconnected, try to reconnect every 5s
    if client is None:
      if now - last_reconnect_attempt > 5.0:
        last_reconnect_attempt = now
        print("glowd: attempting reconnect...")
        _put_glow_status(params, "connecting")
        client = await ble_connect(brightness)
        _put_glow_status(params, "connected" if client else "disconnected")
      rk.keep_time()
      continue

    if sm.updated['carState']:
      ctrl.update(sm)

      raw_color = ctrl.get_color(sm)
      color = ctrl.smooth_color(raw_color)

      if color != ctrl.last_color:
        if DEBUG:
          cs = sm['carState']
          print(f"glowd: state={ctrl.state.name} RPM={cs.engineRpm:.0f} v={cs.vEgo:.1f} brake={cs.brakePressed} → RGB{color}")

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
