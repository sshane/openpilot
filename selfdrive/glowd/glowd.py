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
import time

import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper

from openpilot.tools.underglow.sp105e import (
  connect, set_color, set_brightness, dim, power_toggle, send, packet, Command,
)

DEBUG = True

# --- Color palette (California-legal: no red, no blue) ---
COLOR_IDLE = (0, 180, 60)        # green at idle
COLOR_BRAKE = (255, 40, 0)       # deep orange-red (more orange than red)
COLOR_BRAKE_HARD = (255, 80, 0)  # bright orange on hard brake
COLOR_GAS = (255, 140, 0)        # warm amber
COLOR_REVERSE = (255, 255, 255)  # white
COLOR_BLINKER = (255, 160, 0)    # amber
COLOR_DOWNSHIFT = (180, 0, 255)  # purple flash
COLOR_STANDSTILL = (0, 200, 80)  # soft green for breathing

# RPM thresholds
RPM_MIN = 800
RPM_MAX = 7000

# Timing
UPDATE_HZ = 20
BLINKER_HZ = 1.5
DOWNSHIFT_FLASH_DURATION = 0.4
BRIGHT_INIT_STEPS = 10

# Gear debounce: ignore gearActual == 0 briefly (neutral between shifts)
GEAR_ZERO_DEBOUNCE_S = 0.5


def rpm_to_color(rpm: float) -> tuple[int, int, int]:
  """Map RPM to hue: green(idle) → yellow → amber → purple(redline).
  Avoids pure red (0.0) and blue (0.66)."""
  t = max(0.0, min(1.0, (rpm - RPM_MIN) / (RPM_MAX - RPM_MIN)))
  if t < 0.7:
    hue = 0.33 - (0.33 - 0.08) * (t / 0.7)
  else:
    hue = 0.08 + (0.78 - 0.08) * ((t - 0.7) / 0.3)
  r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
  return int(r * 255), int(g * 255), int(b * 255)


def breathing_brightness(t: float, period: float = 3.0) -> float:
  """Sinusoidal breathing: 0.3 → 1.0 → 0.3."""
  phase = (t % period) / period
  return 0.3 + 0.7 * (0.5 + 0.5 * math.sin(2 * math.pi * phase - math.pi / 2))


def scale_color(color: tuple[int, int, int], brightness: float) -> tuple[int, int, int]:
  return (int(color[0] * brightness), int(color[1] * brightness), int(color[2] * brightness))


class GlowController:
  def __init__(self):
    self.last_valid_gear = 0
    self.gear_zero_since = 0.0
    self.effective_gear = 0
    self._prev_effective_gear = 0

    self.downshift_until = 0.0
    self.last_blinker_toggle = 0.0
    self.blinker_on = True
    self.last_color = (0, 0, 0)
    self.standstill_start = 0.0
    self.was_standstill = False

  def _update_gear(self, raw_gear: int, now: float) -> int:
    """Debounce gearActual: hold last valid gear when it drops to 0 briefly."""
    if raw_gear > 0:
      self.last_valid_gear = raw_gear
      self.gear_zero_since = 0.0
      self.effective_gear = raw_gear
    else:
      if self.gear_zero_since == 0.0:
        self.gear_zero_since = now
      if now - self.gear_zero_since < GEAR_ZERO_DEBOUNCE_S:
        self.effective_gear = self.last_valid_gear
      else:
        self.effective_gear = 0
    return self.effective_gear

  def compute_color(self, sm) -> tuple[int, int, int]:
    cs = sm['carState']
    now = time.monotonic()

    rpm = cs.engineRpm
    brake = cs.brakePressed
    gas = cs.gasPressed
    standstill = cs.standstill
    left_blinker = cs.leftBlinker
    right_blinker = cs.rightBlinker
    raw_gear = cs.gearActual

    gear = self._update_gear(raw_gear, now)
    prev_gear = self._prev_effective_gear

    # --- Priority 1: Downshift flash ---
    if gear > 0 and prev_gear > 0 and gear < prev_gear:
      self.downshift_until = now + DOWNSHIFT_FLASH_DURATION
      if DEBUG:
        print(f"glowd: DOWNSHIFT {prev_gear} → {gear}")
    self._prev_effective_gear = gear

    if now < self.downshift_until:
      return COLOR_DOWNSHIFT

    # --- Priority 2: Braking ---
    if brake:
      a_ego = cs.aEgo
      if a_ego < -3.0:
        return COLOR_BRAKE_HARD
      else:
        intensity = min(1.0, max(0.5, abs(a_ego) / 4.0))
        return scale_color(COLOR_BRAKE, intensity)

    # --- Priority 3: Blinker amber pulse ---
    if left_blinker or right_blinker:
      period = 1.0 / BLINKER_HZ
      if now - self.last_blinker_toggle >= period / 2:
        self.blinker_on = not self.blinker_on
        self.last_blinker_toggle = now
      if self.blinker_on:
        return COLOR_BLINKER

    # --- Priority 4: Reverse ---
    if str(cs.gearShifter) == 'reverse':
      return COLOR_REVERSE

    # --- Priority 5: Standstill breathing ---
    if standstill:
      if not self.was_standstill:
        self.standstill_start = now
        self.was_standstill = True
      elapsed = now - self.standstill_start
      if elapsed > 2.0:
        bright = breathing_brightness(elapsed - 2.0, period=3.0)
        return scale_color(COLOR_STANDSTILL, bright)
    else:
      self.was_standstill = False

    # --- Priority 6: Gas pressed (warm amber overlay) ---
    if gas and rpm > RPM_MIN:
      rpm_color = rpm_to_color(rpm)
      return (
        (rpm_color[0] + COLOR_GAS[0]) // 2,
        (rpm_color[1] + COLOR_GAS[1]) // 2,
        (rpm_color[2] + COLOR_GAS[2]) // 2,
      )

    # --- Priority 7: RPM-based color ---
    return rpm_to_color(rpm)


async def ble_connect():
  """Connect to SP105E, power on, max brightness. Returns client or None."""
  print("glowd: connecting to SP105E...")
  client = await connect(exit_on_fail=False)
  if client is None:
    return None
  # Toggle on + max brightness
  await power_toggle(client)
  await asyncio.sleep(0.3)
  for _ in range(BRIGHT_INIT_STEPS):
    await set_brightness(client, 16)
    await asyncio.sleep(0.03)
  print("glowd: connected, LEDs on, brightness maxed")
  return client


async def ble_shutdown(client):
  """Power off LEDs and disconnect."""
  if client is not None:
    try:
      await power_toggle(client)
      await asyncio.sleep(0.1)
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

  client = await ble_connect()

  sm = messaging.SubMaster(['carState'], poll='carState')
  ctrl = GlowController()
  rk = Ratekeeper(UPDATE_HZ)
  last_reconnect_attempt = 0.0

  print(f"glowd: running at {UPDATE_HZ}Hz")

  while not do_exit:
    sm.update(timeout=100)

    # If disconnected, try to reconnect every 5s
    if client is None:
      now = time.monotonic()
      if now - last_reconnect_attempt > 5.0:
        last_reconnect_attempt = now
        print("glowd: attempting reconnect...")
        client = await ble_connect()
      rk.keep_time()
      continue

    if sm.updated['carState']:
      color = ctrl.compute_color(sm)

      if color != ctrl.last_color:
        if DEBUG:
          cs = sm['carState']
          print(f"glowd: RPM={cs.engineRpm:.0f} gear={cs.gearActual} → RGB{color}")

        try:
          await set_color(client, *color)
        except Exception as e:
          print(f"glowd: BLE error: {e}")
          try:
            await client.disconnect()
          except Exception:
            pass
          client = None
          continue

        ctrl.last_color = color

    rk.keep_time()

  # Clean shutdown: power off LEDs
  await ble_shutdown(client)


def main():
  asyncio.run(glowd_thread())


if __name__ == "__main__":
  main()
