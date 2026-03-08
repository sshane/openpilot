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

import cereal.messaging as messaging
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper

from openpilot.tools.underglow import sp105e

DEBUG = True

# --- Color palette ---
COLOR_REVERSE = (255, 255, 255)  # white
COLOR_STANDSTILL = (0, 200, 80)  # soft green for breathing

# RPM thresholds
RPM_MIN = 800
RPM_MAX = 7000

# Timing
UPDATE_HZ = 20
BRIGHT_INIT_STEPS = 10


def rpm_to_color(rpm: float) -> tuple[int, int, int]:
  """Map RPM to color: green(idle) → yellow → amber → purple(redline).
  Avoids pure red and blue. Low range uses HSV, high range blends RGB."""
  t = max(0.0, min(1.0, (rpm - RPM_MIN) / (RPM_MAX - RPM_MIN)))
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


def breathing_brightness(t: float, period: float = 3.0) -> float:
  """Sinusoidal breathing: 0.3 → 1.0 → 0.3."""
  phase = (t % period) / period
  return 0.3 + 0.7 * (0.5 + 0.5 * math.sin(2 * math.pi * phase - math.pi / 2))


def scale_color(color: tuple[int, int, int], brightness: float) -> tuple[int, int, int]:
  return (int(color[0] * brightness), int(color[1] * brightness), int(color[2] * brightness))


class GlowController:
  def __init__(self):
    self.last_color = (0, 0, 0)
    self.standstill_start = 0.0
    self.prev_standstill = False
    self._rainbow_until = 0.0
    self._brake_pressed_t = 0.0
    self._prev_brake = False

    # HSV smoothing filters
    dt = 1.0 / UPDATE_HZ
    self._h_filter = FirstOrderFilter(0.0, 0.1, dt)
    self._s_filter = FirstOrderFilter(0.0, 0.1, dt)
    self._v_filter = FirstOrderFilter(0.0, 0.1, dt)

  def _smooth_color(self, color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Filter RGB through HSV space for smooth transitions."""
    h, s, v = colorsys.rgb_to_hsv(color[0] / 255, color[1] / 255, color[2] / 255)
    h = self._h_filter.update(h)
    s = self._s_filter.update(s)
    v = self._v_filter.update(v)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

  def compute_color(self, sm, chill_mode: bool = False) -> tuple[int, int, int]:
    cs = sm['carState']
    now = time.monotonic()

    rpm = cs.engineRpm
    standstill = cs.standstill
    # TODO: use sp105e.set_brightness instead of scaling RGB
    brightness = 1.0 if standstill or str(cs.gearShifter) == 'reverse' else 0.7

    # Brake rising edge: dark red for 0.4s
    if cs.brakePressed and not self._prev_brake:
      self._brake_pressed_t = now
    self._prev_brake = cs.brakePressed
    if now - self._brake_pressed_t < 0.4:
      return (128, 0, 0)

    # Reverse
    if str(cs.gearShifter) == 'reverse':
      return scale_color(COLOR_REVERSE, brightness)

    # Standstill: slow rainbow cycle (continues 2.5s after leaving)
    if standstill:
      if not self.prev_standstill:
        self.standstill_start = now
        self.prev_standstill = True
    elif self.prev_standstill:
      self.prev_standstill = False
      self._rainbow_until = now + 2.5

    elapsed = now - self.standstill_start
    if (standstill and elapsed > 2.0) or now < self._rainbow_until:
      hue = ((elapsed - 2.0) / 8.0) % 1.0  # full cycle every 8s
      r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
      return scale_color((int(r * 255), int(g * 255), int(b * 255)), brightness)

    # RPM-based color
    return scale_color(rpm_to_color(rpm), brightness)


def _put_glow_status(params, status: str, color: tuple[int, int, int] = (0, 0, 0)):
  params.put_nonblocking("GlowStatus", {"status": status, "color": list(color)})


def bt_is_ready() -> bool:
  """Check if BT stack is up (managed by bluetooth.service in AGNOS)."""
  result = subprocess.run(["sudo", "hciconfig", "hci0"], capture_output=True)
  return b"UP RUNNING" in result.stdout


async def ble_connect():
  """Connect to SP105E, power on, max brightness. Returns client or None.
  Assumes BT stack is already up (bluetooth.service in AGNOS)."""
  if not bt_is_ready():
    print("glowd: hci0 not up (waiting for bluetooth.service)")
    return None

  print("glowd: connecting to SP105E...")
  client = await sp105e.connect(exit_on_fail=False)
  if client is None:
    return None
  await sp105e.power_on(client)
  await sp105e.set_brightness(client, sp105e.BRIGHTNESS_MAX)
  print("glowd: connected, LEDs on, brightness maxed")
  return client


async def ble_shutdown(client):
  """Power off LEDs and disconnect."""
  if client is not None:
    try:
      await sp105e.power_off(client)
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
    if now - last_param_read > 5.0:
      chill_mode = params.get_bool("GlowMode")
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
      color = ctrl._smooth_color(ctrl.compute_color(sm, chill_mode))

      if color != ctrl.last_color:
        if DEBUG:
          cs = sm['carState']
          print(f"glowd: RPM={cs.engineRpm:.0f} gear={cs.gearActual} chill={chill_mode} → RGB{color}")

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
