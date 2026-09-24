#!/usr/bin/env python3
"""
SP105E BLE LED controller for LowGlow underglow kit.

Protocol (reverse-engineered March 2026):
  Packet format: 38 [D1] [D2] [D3] [CMD] 83
  Sets GRB color order (factory default) on connect. API accepts RGB.

Confirmed commands:
  SET_COLOR:      38 GG RR BB 1E 83  (GRB wire order, API takes RGB)
  POWER_TOGGLE:   38 00 00 00 AA 83  (toggle only, 0xAB does nothing on SP105E)
  SET_MODE:       38 MM 00 00 2C 83  (mode 0-202, sets pattern/animation)
  BRIGHT_UP:      38 00 00 00 2A 83  (relative +1, all data bytes ignored)
  BRIGHT_DOWN:    38 00 00 00 28 83  (relative -1, all data bytes ignored)
  Brightness: 7 levels (0-6), clamps at both ends (no wrap)
  COLOR_ORDER:    38 NN 00 00 3C 83  (0=GRB, 1=GBR, 2=RGB, 3=BGR, 4=RBG, 5=BRG)

Modes (via SET_MODE 0x2C with mode number in D1):
  - Modes 0-202 accepted (state byte 1 reflects mode number)
  - Mode 0xC9 (201) = static color (also set implicitly by SET_COLOR)
  - Modes 0xFE-0xFF wrap to mode 1
  - Mode 1 = rainbow flow (confirmed visually)
  - Speed command not found yet (0x03 doesn't work, unlike SP110E)

GET_STATE response (8 bytes via notify on 0xFFE1):
  Byte 0: power (1=on, 0=off)
  Byte 1: mode number (0xC9=static, 0x01-0xCA=patterns)
  Byte 2: unknown (typically 5-6)
  Byte 3: brightness (0-6)
  Byte 4: unknown (0x03)
  Byte 5: color order (0=GRB, etc.)
  Byte 6: unknown (0x02)
  Byte 7: 0x58 (88) — likely pixel count

SP110E vs SP105E differences:
  - SP110E has no packet framing (no 0x38/0x83), SP105E requires it
  - SP110E: 0xAA=on, 0xAB=off. SP105E: 0xAA=toggle, 0xAB=no-op
  - SP110E: 0x2A=absolute brightness (D1=level). SP105E: 0x2A=relative +1 (D1 ignored)
  - SP110E: 0x03=speed. SP105E: 0x03 does nothing useful
  - SP110E: 12-byte state (includes color, white, pixel count). SP105E: 8-byte state
  - SP110E: 122 modes. SP105E: 202 modes

Notes:
  - Sending SET_COLOR stops any active pattern and sets mode to 0xC9 (static)
  - Device must be ON for commands to work
  - 0xAA is a toggle (on->off, off->on), not absolute
  - Color order (0x3C) persists to flash — script sets GRB on connect

BLE timing (measured on comma four):
  - SET_COLOR at ~33Hz sustained with response=True (no sleep needed)
  - Brightness steps: 50ms sleep before each required (0ms drops ~50%)
  - GET_STATE round-trip: 150-400ms (start_notify → send → wait → stop_notify)
  - GET_STATE unreliable if interleaved between brightness steps — verify only after all steps done
  - After connect: 0.5s sleep before first command
  - After rapid color writes: 0.5s sleep before GET_STATE works
  - response=True on all commands prevents BLE write buffer buildup
  - Stale BlueZ connections after kill -9: `bluetoothctl disconnect` clears them
"""
import argparse
import asyncio
import subprocess
import sys
from enum import IntEnum
from bleak import BleakScanner, BleakClient

CHAR = "0000ffe1-0000-1000-8000-00805f9b34fb"

CONNECT_RETRIES = 3
SCAN_TIMEOUT = 8

PACKET_START = 0x38
PACKET_END = 0x83


class Command(IntEnum):
  SET_COLOR = 0x1E
  POWER_TOGGLE = 0xAA
  BRIGHT_UP = 0x2A    # relative +1, all data bytes ignored, clamps at 6
  BRIGHT_DOWN = 0x28  # relative -1, all data bytes ignored, clamps at 0
  SET_MODE = 0x2C
  GET_STATE = 0x10    # triggers notify with 8-byte state response
  COLOR_ORDER = 0x3C
  # DANGEROUS on SP105E — do not send (requires power cycle to recover):
  # 0x1C — SP110E: set IC model. SP105E: bright white, ignores all commands after
  # 0x2D — SP110E: set pixel count. SP105E: brief off/on, may wedge state


class ColorOrder(IntEnum):
  GRB = 0  # default
  GBR = 1
  RGB = 2
  BGR = 3
  RBG = 4
  BRG = 5


MODE_STATIC = 0xC9  # set implicitly by SET_COLOR
MODE_MAX = 202      # modes 0-202 accepted, 0xFE+ wraps


def packet(d1: int, d2: int, d3: int, cmd: int) -> bytes:
  return bytes([PACKET_START, d1, d2, d3, cmd, PACKET_END])


def color_packet(r: int, g: int, b: int) -> bytes:
  """Color packet. Swaps to GRB wire order so callers use standard RGB."""
  return packet(g, r, b, Command.SET_COLOR)


async def find_sp105e(timeout=SCAN_TIMEOUT):
  """Scan for SP105E by name. Returns BLEDevice or None."""
  scanner = BleakScanner()
  await scanner.start()
  for _ in range(timeout * 10):
    await asyncio.sleep(0.1)
    for d in scanner.discovered_devices:
      if d.name and "SP" in d.name:
        await scanner.stop()
        return d
  await scanner.stop()
  return None


async def connect(retries=CONNECT_RETRIES, exit_on_fail=True):
  """Connect to SP105E with retries. Returns BleakClient."""
  for attempt in range(1, retries + 1):
    try:
      dev = await find_sp105e()
      if not dev:
        print(f"SP105E not found (attempt {attempt}/{retries})")
        if attempt < retries:
          await asyncio.sleep(2)
        continue
      print(f"Found {dev.address}")
      # Clear any stale BlueZ connection from a crashed process
      subprocess.run(["bluetoothctl", "disconnect", dev.address], capture_output=True, timeout=5)
      client = BleakClient(dev.address, timeout=20)
      await client.connect()
      # Always set GRB (factory default) on connect to ensure known state
      await send(client, packet(ColorOrder.GRB, 0, 0, Command.COLOR_ORDER))
      print(f"Connected to {dev.address} (GRB order set)")
      return client
    except Exception as e:
      print(f"Connect failed (attempt {attempt}/{retries}): {e}")
      if attempt < retries:
        await asyncio.sleep(2)

  if exit_on_fail:
    print("SP105E: all connection attempts failed")
    sys.exit(1)
  return None


async def send(client, data: bytes, response=False):
  await client.write_gatt_char(CHAR, data, response=response)


# --- State reading ---

async def get_state(client, retries: int = 2) -> bytes | None:
  """Send GET_STATE (0x10) and return 8-byte notify response.
  Returns None on timeout. Byte 0: 1=on, 0=off."""
  await asyncio.sleep(0.1)
  for attempt in range(retries):
    result = None
    event = asyncio.Event()

    def on_notify(sender, data):
      nonlocal result
      result = data
      event.set()

    await client.start_notify(CHAR, on_notify)
    await send(client, packet(0, 0, 0, Command.GET_STATE))
    try:
      await asyncio.wait_for(event.wait(), timeout=2.0)
    except asyncio.TimeoutError:
      print(f"sp105e: get_state timeout (attempt {attempt + 1}/{retries})")
    await client.stop_notify(CHAR)
    if result is not None:
      return result
    await asyncio.sleep(0.5)
  return None


async def is_on(client) -> bool:
  """Returns True if LEDs are on."""
  state = await get_state(client)
  return state is not None and state[0] == 1


# --- High-level commands ---

async def set_color(client, r, g, b):
  await send(client, color_packet(r, g, b))


async def power_toggle(client):
  await send(client, packet(0, 0, 0, Command.POWER_TOGGLE), response=True)


async def set_power(client, on: bool, retries: int = 2):
  """Set power state. No-op if already in desired state."""
  target = 1 if on else 0
  label = "on" if on else "off"
  for attempt in range(retries):
    state = await get_state(client)
    if state is None:
      print(f"WARNING: power_{label} state read failed (attempt {attempt + 1}/{retries})")
      await asyncio.sleep(0.5)
      continue
    if state[0] == target:
      return
    await power_toggle(client)
    return
  print(f"WARNING: power_{label} failed after retries")


BRIGHTNESS_MAX = 6
BRIGHTNESS_MIN = 0


async def get_brightness(client) -> int | None:
  """Read current brightness level (0-6). Returns None on error."""
  state = await get_state(client)
  if state is not None and len(state) >= 4:
    return state[3]
  return None


async def set_brightness(client, level: int, retries: int = 2):
  """Set absolute brightness (0-6). Reads current level, steps to target, verifies."""
  level = max(BRIGHTNESS_MIN, min(BRIGHTNESS_MAX, level))
  for attempt in range(retries):
    current = await get_brightness(client)
    if current is None:
      print("WARNING: set_brightness can't read current level, skipping")
      return
    diff = level - current
    if diff == 0:
      return
    if diff > 0:
      for _ in range(diff):
        await brightness_step_up(client)
    else:
      for _ in range(-diff):
        await brightness_step_down(client)
    # verify
    actual = await get_brightness(client)
    if actual == level:
      return
    print(f"set_brightness: attempt {attempt + 1} wanted {level}, got {actual}, retrying")


async def brightness_step_up(client):
  """Step brightness up by 1. All data bytes ignored by controller."""
  await asyncio.sleep(0.1)
  await send(client, packet(0, 0, 0, Command.BRIGHT_UP), response=True)


async def brightness_step_down(client):
  """Step brightness down by 1. All data bytes ignored by controller."""
  await asyncio.sleep(0.1)
  await send(client, packet(0, 0, 0, Command.BRIGHT_DOWN), response=True)


async def set_mode(client, mode):
  """Set animation mode (0-202). Mode 0xC9=static (also set by SET_COLOR)."""
  await send(client, packet(mode, 0, 0, Command.SET_MODE), response=True)


async def set_color_order(client, order=ColorOrder.GRB):
  """Set color byte order. WARNING: persists to flash! Leave as GRB (default)."""
  await send(client, packet(order, 0, 0, Command.COLOR_ORDER))


# --- CLI ---

async def cmd_color(args):
  client = await connect()
  await set_color(client, args.r, args.g, args.b)
  print(f"Color set to ({args.r}, {args.g}, {args.b})")
  await client.disconnect()


async def cmd_toggle(args):
  client = await connect()
  await power_toggle(client)
  print("Power toggled")
  await client.disconnect()


async def cmd_on(args):
  client = await connect()
  await set_power(client, on=True)
  print("ON")
  await client.disconnect()


async def cmd_off(args):
  client = await connect()
  await set_power(client, on=False)
  print("OFF")
  await client.disconnect()


async def cmd_state(args):
  client = await connect()
  state = await get_state(client)
  if state is None:
    print("No response")
  else:
    print(f"Power: {'ON' if state[0] == 1 else 'OFF'}")
    print(f"Brightness: {state[3]}/{BRIGHTNESS_MAX}")
    print(f"Mode: {state[1]} ({'static' if state[1] == 0xC9 else 'pattern'})")
    print(f"Color order: {state[5]} ({ColorOrder(state[5]).name})")
    print(f"Raw: {' '.join(f'{b:02x}' for b in state)}")
  await client.disconnect()


async def cmd_bright(args):
  client = await connect()
  current = await get_brightness(client)
  await set_brightness(client, args.value)
  after = await get_brightness(client)
  print(f"Brightness: {current} → {after} (range 0-{BRIGHTNESS_MAX})")
  await client.disconnect()


async def cmd_mode(args):
  client = await connect()
  await set_mode(client, args.mode)
  print(f"Mode set to {args.mode}")
  await client.disconnect()


async def run_demo(client):
  """HSV color cycle → brightness ramp → repeat. Ctrl+C to stop."""
  import colorsys
  await set_power(client, on=True)
  await set_brightness(client, BRIGHTNESS_MAX)

  print("Demo: colors → brightness → colors. Ctrl+C to stop.")
  try:
    while True:
      print("  color cycle...")
      for step in range(200):
        hue = step / 200.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        await set_color(client, int(r * 255), int(g * 255), int(b * 255))

      print("  brightness ramp...")
      await set_color(client, 255, 0, 0)
      await asyncio.sleep(0.1)
      for _ in range(BRIGHTNESS_MAX):
        await brightness_step_down(client)
      for _ in range(BRIGHTNESS_MAX):
        await brightness_step_up(client)
  except KeyboardInterrupt:
    pass
  await set_brightness(client, BRIGHTNESS_MAX)
  print("\n  demo done.")


async def cmd_demo(args):
  client = await connect()
  await run_demo(client)
  await client.disconnect()


async def cmd_interactive(args):
  client = await connect()
  print("\nCommands:")
  print("  color R G B       set static color (RGB 0-255)")
  print("  bright N          set brightness (0-6)")
  print("  on / off / toggle power control")
  print("  state             read device state")
  print("  mode N            set mode (0-202, decimal or 0xNN hex)")
  print("  cmd XX [D1 D2 D3]  send command byte (hex), wraps in packet")
  print("  raw HH HH ...     send raw hex bytes")
  print("  demo              color cycle")
  print("  quit\n")

  while True:
    try:
      line = input("sp105e> ").strip()
    except (EOFError, KeyboardInterrupt):
      break
    if not line:
      continue
    parts = line.split()
    c = parts[0].lower()
    try:
      if c == "color" and len(parts) == 4:
        await set_color(client, int(parts[1]), int(parts[2]), int(parts[3]))
      elif c in ("bright", "brightness") and len(parts) == 2:
        level = int(parts[1])
        current = await get_brightness(client)
        await set_brightness(client, level)
        after = await get_brightness(client)
        print(f"  Brightness: {current} → {after}")
      elif c == "on":
        await set_power(client, on=True)
        print("  ON")
      elif c == "off":
        await set_power(client, on=False)
        print("  OFF")
      elif c == "toggle":
        await power_toggle(client)
      elif c == "state":
        state = await get_state(client)
        if state is None:
          print("  No response")
        else:
          print(f"  Power: {'ON' if state[0] == 1 else 'OFF'}")
          print(f"  Brightness: {state[3]}/{BRIGHTNESS_MAX}")
          print(f"  Mode: {state[1]} ({'static' if state[1] == 0xC9 else 'pattern'})")
          print(f"  Color order: {state[5]} ({ColorOrder(state[5]).name})")
          print(f"  Raw: {' '.join(f'{b:02x}' for b in state)}")
      elif c == "mode" and len(parts) == 2:
        val = parts[1]
        mode = int(val, 16) if val.startswith("0x") else int(val)
        await set_mode(client, mode)
        print(f"  mode {mode}")
      elif c == "cmd" and len(parts) >= 2:
        cmd = int(parts[1], 16)
        d1 = int(parts[2], 16) if len(parts) > 2 else 0
        d2 = int(parts[3], 16) if len(parts) > 3 else 0
        d3 = int(parts[4], 16) if len(parts) > 4 else 0
        await send(client, packet(d1, d2, d3, cmd), response=True)
        print(f"  sent: {packet(d1, d2, d3, cmd).hex()}")
      elif c == "raw":
        data = bytes([int(x, 16) for x in parts[1:]])
        await send(client, data)
        print(f"  sent: {data.hex()}")
      elif c == "demo":
        await run_demo(client)
      elif c in ("quit", "exit", "q"):
        break
      else:
        print("Unknown command.")
    except Exception as e:
      print(f"Error: {e}")

  await client.disconnect()
  print("Bye.")


async def cmd_scan(args):
  print("Scanning 10s...")
  devices = await BleakScanner.discover(timeout=10, return_adv=True)
  for addr, (dev, adv) in sorted(devices.items(), key=lambda x: x[1][1].rssi, reverse=True):
    if dev.name:
      print(f"  {dev.address}  {dev.name}  rssi={adv.rssi}")


def main():
  parser = argparse.ArgumentParser(description="SP105E BLE LED controller")
  sub = parser.add_subparsers(dest="command")

  p_color = sub.add_parser("color", help="Set color (RGB 0-255)")
  p_color.add_argument("r", type=int)
  p_color.add_argument("g", type=int)
  p_color.add_argument("b", type=int)

  sub.add_parser("toggle", help="Toggle power on/off")
  sub.add_parser("on", help="Turn on (no-op if already on)")
  sub.add_parser("off", help="Turn off (no-op if already off)")
  sub.add_parser("state", help="Read device state")
  sub.add_parser("demo", help="Color cycle demo")
  sub.add_parser("interactive", help="Interactive REPL")
  sub.add_parser("scan", help="Scan for BLE devices")

  p_bright = sub.add_parser("bright", help="Set brightness (0-6)")
  p_bright.add_argument("value", type=int, help="brightness level 0-6")

  p_mode = sub.add_parser("mode", help="Set mode (decimal)")
  p_mode.add_argument("mode", type=int)

  args = parser.parse_args()

  commands = {
    "color": cmd_color,
    "toggle": cmd_toggle,
    "on": cmd_on,
    "off": cmd_off,
    "state": cmd_state,
    "bright": cmd_bright,
    "mode": cmd_mode,
    "demo": cmd_demo,
    "interactive": cmd_interactive,
    "scan": cmd_scan,
  }

  if not args.command:
    args.command = "interactive"

  asyncio.run(commands[args.command](args))


if __name__ == "__main__":
  main()
