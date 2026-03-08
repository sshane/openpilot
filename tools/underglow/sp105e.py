#!/usr/bin/env python3
"""
SP105E BLE LED controller for LowGlow underglow kit.

Protocol (reverse-engineered March 2026):
  Packet format: 38 [D1] [D2] [D3] [CMD] 83
  Color byte order: GRB (not RGB!)

Confirmed commands:
  0x1E = Set color:      38 RR GG BB 1E 83  (after setting order=2 for RGB)
  0xAA = Power toggle:   38 00 00 00 AA 83  (toggle only, 0xAB does nothing)
  0x2C = Set mode:       38 MM 00 00 2C 83  (mode number in D1)
  0x2A = Brightness:     38 BB 00 00 2A 83  (higher = brighter)
  0x3C = Color order:    38 NN 00 00 3C 83  (0=GRB, 1=GBR, 2=RGB, 3=BGR, 4=RBG, 5=BRG)
  0x2D = Pixel count?:   38 NN 00 00 2D 83  (causes brief off/on, might set LED count)

Pattern modes (as CMD byte directly, D1-D3 ignored):
  0x03 = rainbow animation (blue/red/green flowing)
  0x05 = rainbow pattern 1
  0x06 = rainbow pattern 2
  0x07 = breathing: fade through colors (red->blue->yellow etc), slow
  0x08-0x0B = breathing variations (similar to 0x07)
  0x0D = color cycle: yellow->orange->red, no fade between colors
  0x0E = same as 0x0D but very slow
  0x0F = fast flowing rainbow
  0x10 = same as 0x0F

Notes:
  - Send 38 02 00 00 3C 83 on connect to set RGB order
  - Sending color (0x1E) stops any active pattern and goes to static
  - Device must be ON for commands to work
  - 0xAA is a toggle (on->off, off->on), not absolute
  - 0xAB does nothing
  - Speed command not found yet
  - bleak needs: source /etc/profile && python3
"""
import argparse
import asyncio
import sys
from bleak import BleakScanner, BleakClient

CHAR = "0000ffe1-0000-1000-8000-00805f9b34fb"


def packet(d1: int, d2: int, d3: int, cmd: int) -> bytes:
  return bytes([0x38, d1, d2, d3, cmd, 0x83])


def color_packet(r: int, g: int, b: int) -> bytes:
  """Color packet. Assumes RGB order has been set via set_color_order(2)."""
  return packet(r, g, b, 0x1E)


async def find_sp105e(timeout=10):
  scanner = BleakScanner()
  await scanner.start()
  await asyncio.sleep(timeout)
  await scanner.stop()
  for d in scanner.discovered_devices:
    if d.name and "SP" in d.name:
      return d
  return None


async def connect():
  dev = await find_sp105e()
  if not dev:
    print("SP105E not found")
    sys.exit(1)
  print(f"Found {dev.address}")
  client = BleakClient(dev.address, timeout=20)
  await client.connect()
  print("Connected.")
  return client


async def send(client, data: bytes):
  await client.write_gatt_char(CHAR, data, response=False)


# --- High-level commands ---

async def set_color(client, r, g, b):
  await send(client, color_packet(r, g, b))


async def power_toggle(client):
  await send(client, packet(0, 0, 0, 0xAA))


async def set_brightness(client, val):
  """Set brightness. 0-255, higher = brighter."""
  await send(client, packet(val, 0, 0, 0x2A))


async def set_mode(client, mode):
  """Set animation mode via 0x2C with mode number in D1."""
  await send(client, packet(mode, 0, 0, 0x2C))


async def set_pattern(client, pattern):
  """Set pattern directly via CMD byte (0x03, 0x05-0x10, etc.)."""
  await send(client, packet(0, 0, 0, pattern))


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
  await set_brightness(client, args.value)
  print(f"Brightness set to {args.value}")
  await client.disconnect()


async def cmd_mode(args):
  client = await connect()
  await set_mode(client, args.mode)
  print(f"Mode set to {args.mode}")
  await client.disconnect()


async def cmd_pattern(args):
  client = await connect()
  await set_pattern(client, args.pattern)
  print(f"Pattern set to 0x{args.pattern:02X}")
  await client.disconnect()


async def cmd_demo(args):
  client = await connect()
  colors = [
    (255, 0, 0, "red"),
    (0, 255, 0, "green"),
    (0, 0, 255, "blue"),
    (255, 255, 0, "yellow"),
    (0, 255, 100, "comma green"),
    (255, 0, 255, "magenta"),
    (255, 128, 0, "orange"),
    (255, 255, 255, "white"),
  ]
  for r, g, b, name in colors:
    print(f"  {name} ({r},{g},{b})")
    await set_color(client, r, g, b)
    await asyncio.sleep(1.5)
  print("Demo done.")
  await client.disconnect()


async def cmd_interactive(args):
  client = await connect()
  print("\nCommands:")
  print("  color R G B       set static color")
  print("  bright N          brightness 0-255")
  print("  toggle            power on/off")
  print("  mode N            set mode (decimal, via 0x2C)")
  print("  pattern HH        set pattern (hex CMD byte)")
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
        await set_brightness(client, int(parts[1]))
      elif c == "toggle":
        await power_toggle(client)
      elif c == "mode" and len(parts) == 2:
        await set_mode(client, int(parts[1]))
      elif c == "pattern" and len(parts) == 2:
        await set_pattern(client, int(parts[1], 16))
      elif c == "raw":
        data = bytes([int(x, 16) for x in parts[1:]])
        await send(client, data)
        print(f"  sent: {data.hex()}")
      elif c == "demo":
        colors = [
          (255, 0, 0, "red"), (0, 255, 0, "green"), (0, 0, 255, "blue"),
          (255, 255, 0, "yellow"), (0, 255, 100, "comma green"),
        ]
        for r, g, b, name in colors:
          print(f"  {name}")
          await set_color(client, r, g, b)
          await asyncio.sleep(1.5)
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

  p_bright = sub.add_parser("bright", help="Set brightness (0-255)")
  p_bright.add_argument("value", type=int)

  p_mode = sub.add_parser("mode", help="Set mode (decimal, via 0x2C)")
  p_mode.add_argument("mode", type=int)

  p_pattern = sub.add_parser("pattern", help="Set pattern (hex CMD byte)")
  p_pattern.add_argument("pattern", type=lambda x: int(x, 16))

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
