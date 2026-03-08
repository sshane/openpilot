#!/usr/bin/env python3
"""
SP105E BLE LED controller for LowGlow underglow kit.

Protocol (reverse-engineered March 2026):
  Packet format: 38 [D1] [D2] [D3] [CMD] 83
  Sets GRB color order (factory default) on connect. API accepts RGB.

Confirmed commands:
  SET_COLOR:      38 GG RR BB 1E 83  (GRB wire order, API takes RGB)
  POWER_TOGGLE:   38 00 00 00 AA 83  (toggle only, 0xAB does nothing)
  SET_MODE:       38 MM 00 00 2C 83  (mode number in D1)
  BRIGHT_UP:      38 SS 00 00 2A 83  (relative step up, S=step size 1-16)
  BRIGHT_DOWN:    38 SS 00 00 28 83  (relative step down, S=step size 1-8)
  COLOR_ORDER:    38 NN 00 00 3C 83  (0=GRB, 1=GBR, 2=RGB, 3=BGR, 4=RBG, 5=BRG)

Pattern modes (as CMD byte directly, D1-D3 ignored):
  See Pattern enum below.

Notes:
  - Sending SET_COLOR stops any active pattern and goes to static
  - Device must be ON for commands to work
  - 0xAA is a toggle (on->off, off->on), not absolute
  - Speed command not found yet
  - Color order (0x3C) persists to flash — script sets GRB on connect
"""
import argparse
import asyncio
import sys
from enum import IntEnum
from bleak import BleakScanner, BleakClient

CHAR = "0000ffe1-0000-1000-8000-00805f9b34fb"

PACKET_START = 0x38
PACKET_END = 0x83


class Command(IntEnum):
  SET_COLOR = 0x1E
  POWER_TOGGLE = 0xAA
  BRIGHT_UP = 0x2A    # relative: step brighter, D1=step size (1-16)
  BRIGHT_DOWN = 0x28  # relative: step dimmer, D1=step size (1-8)
  SET_MODE = 0x2C
  COLOR_ORDER = 0x3C
  # DANGEROUS — do not send, will soft-brick (requires power cycle):
  # 0x1C — bright white, ignores all commands after
  # 0x2D — brief off/on, may wedge state


class ColorOrder(IntEnum):
  GRB = 0  # default
  GBR = 1
  RGB = 2
  BGR = 3
  RBG = 4
  BRG = 5


class Pattern(IntEnum):
  RAINBOW_FLOW = 0x03
  RAINBOW_1 = 0x05
  RAINBOW_2 = 0x06
  BREATHING = 0x07       # fade through colors, slow
  BREATHING_2 = 0x08
  BREATHING_3 = 0x09
  BREATHING_4 = 0x0A
  BREATHING_5 = 0x0B
  COLOR_CYCLE = 0x0D     # yellow->orange->red, no fade
  COLOR_CYCLE_SLOW = 0x0E
  RAINBOW_FAST = 0x0F
  RAINBOW_FAST_2 = 0x10


def packet(d1: int, d2: int, d3: int, cmd: int) -> bytes:
  return bytes([PACKET_START, d1, d2, d3, cmd, PACKET_END])


def color_packet(r: int, g: int, b: int) -> bytes:
  """Color packet. Swaps to GRB wire order so callers use standard RGB."""
  return packet(g, r, b, Command.SET_COLOR)


async def find_sp105e(timeout=10):
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


async def connect():
  dev = await find_sp105e()
  if not dev:
    print("SP105E not found")
    sys.exit(1)
  print(f"Found {dev.address}")
  client = BleakClient(dev.address, timeout=20)
  await client.connect()
  # Always set GRB (factory default) on connect to ensure known state
  await send(client, packet(ColorOrder.GRB, 0, 0, Command.COLOR_ORDER))
  print(f"Connected to {dev.address} (GRB order set)")
  return client


async def send(client, data: bytes):
  await client.write_gatt_char(CHAR, data, response=False)


# --- High-level commands ---

async def set_color(client, r, g, b):
  await send(client, color_packet(r, g, b))


async def power_toggle(client):
  await send(client, packet(0, 0, 0, Command.POWER_TOGGLE))


async def set_brightness(client, val):
  """Step brightness up. Relative, not absolute. Step size 1-16."""
  await send(client, packet(val, 0, 0, Command.BRIGHT_UP))


async def dim(client, val):
  """Step brightness down. Relative, not absolute. Step size 1-8."""
  await send(client, packet(val, 0, 0, Command.BRIGHT_DOWN))


async def set_mode(client, mode):
  """Set animation mode via SET_MODE with mode number in D1."""
  await send(client, packet(mode, 0, 0, Command.SET_MODE))


async def set_pattern(client, pattern):
  """Set pattern directly via CMD byte."""
  await send(client, packet(0, 0, 0, pattern))


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


async def cmd_bright(args):
  client = await connect()
  steps = abs(args.value)
  if args.value >= 0:
    for _ in range(steps):
      await set_brightness(client, 8)
      await asyncio.sleep(0.1)
    print(f"Brightness up {steps} steps")
  else:
    for _ in range(steps):
      await dim(client, 8)
      await asyncio.sleep(0.1)
    print(f"Brightness down {steps} steps")
  await client.disconnect()


async def cmd_mode(args):
  client = await connect()
  await set_mode(client, args.mode)
  print(f"Mode set to {args.mode}")
  await client.disconnect()


async def cmd_pattern(args):
  client = await connect()
  await set_pattern(client, args.pattern)
  print(f"Pattern set to {args.pattern.name}")
  await client.disconnect()


async def run_demo(client):
  """HSV color cycle → brightness ramp → repeat. Ctrl+C to stop."""
  import colorsys
  # Max brightness
  for _ in range(10):
    await set_brightness(client, 16)
    await asyncio.sleep(0.05)

  print("Demo: colors → brightness → colors. Ctrl+C to stop.")
  try:
    while True:
      print("  color cycle...")
      for step in range(300):
        hue = step / 300.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        await set_color(client, int(r * 255), int(g * 255), int(b * 255))
        await asyncio.sleep(0.01)

      print("  brightness ramp...")
      await set_color(client, 255, 0, 0)
      await asyncio.sleep(0.1)
      for _ in range(6):
        await dim(client, 1)
        await asyncio.sleep(0.05)
      for _ in range(6):
        await set_brightness(client, 1)
        await asyncio.sleep(0.05)
  except KeyboardInterrupt:
    pass
  # Restore full bright
  for _ in range(10):
    await set_brightness(client, 16)
    await asyncio.sleep(0.05)
  print("\n  demo done.")


async def cmd_demo(args):
  client = await connect()
  await run_demo(client)
  await client.disconnect()


async def cmd_interactive(args):
  client = await connect()
  print("\nCommands:")
  print("  color R G B       set static color (RGB 0-255)")
  print("  bright N          step brighter (N steps, use -N to dim)")
  print("  toggle            power on/off")
  print("  mode N            set mode (decimal, via SET_MODE)")
  print("  pattern NAME      set pattern (e.g. BREATHING, RAINBOW_FLOW)")
  print("  raw HH HH ...     send raw hex bytes")
  print("  demo              color cycle")
  print("  quit\n")
  print(f"  Available patterns: {', '.join(p.name for p in Pattern)}\n")

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
        val = int(parts[1])
        if val >= 0:
          for _ in range(val):
            await set_brightness(client, 8)
            await asyncio.sleep(0.1)
        else:
          for _ in range(-val):
            await dim(client, 8)
            await asyncio.sleep(0.1)
      elif c == "toggle":
        await power_toggle(client)
      elif c == "mode" and len(parts) == 2:
        await set_mode(client, int(parts[1]))
      elif c == "pattern" and len(parts) == 2:
        name = parts[1].upper()
        try:
          p = Pattern[name]
        except KeyError:
          print(f"  Unknown pattern. Options: {', '.join(p.name for p in Pattern)}")
          continue
        await set_pattern(client, p)
        print(f"  {p.name}")
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
  sub.add_parser("demo", help="Color cycle demo")
  sub.add_parser("interactive", help="Interactive REPL")
  sub.add_parser("scan", help="Scan for BLE devices")

  p_bright = sub.add_parser("bright", help="Step brightness (positive=up, negative=down)")
  p_bright.add_argument("value", type=int, help="number of steps (negative to dim)")

  p_mode = sub.add_parser("mode", help="Set mode (decimal)")
  p_mode.add_argument("mode", type=int)

  p_pattern = sub.add_parser("pattern", help="Set pattern by name")
  p_pattern.add_argument("pattern", type=lambda x: Pattern[x.upper()],
                          choices=list(Pattern), metavar="PATTERN")

  args = parser.parse_args()

  commands = {
    "color": cmd_color,
    "toggle": cmd_toggle,
    "bright": cmd_bright,
    "mode": cmd_mode,
    "pattern": cmd_pattern,
    "demo": cmd_demo,
    "interactive": cmd_interactive,
    "scan": cmd_scan,
  }

  if not args.command:
    args.command = "interactive"

  asyncio.run(commands[args.command](args))


if __name__ == "__main__":
  main()
