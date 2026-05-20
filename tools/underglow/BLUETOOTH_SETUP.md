# Bluetooth on comma four (SDM845 / WCN3990)

## Status: Working (March 2026)

## Kernel Changes (in agnos-builder)

### Branch: `bluetooth-support` in agnos-builder

### 1. Defconfig (`agnos-kernel-sdm845/arch/arm64/configs/tici_defconfig`)
```
CONFIG_BT=y
CONFIG_BT_BREDR=y
CONFIG_BT_RFCOMM=y
CONFIG_BT_LE=y
CONFIG_BT_HCIUART=y
CONFIG_BT_HCIUART_H4=y
CONFIG_BT_HCIUART_QCA=y
CONFIG_MSM_BT_POWER=y
CONFIG_BTFM_SLIM=y
CONFIG_BTFM_SLIM_WCN3990=y
```

### 2. Device Tree (`agnos-kernel-sdm845/arch/arm64/boot/dts/qcom/comma_common.dtsi`)
Added BT UART (SE6 at 0x898000, GPIOs 45-48):
```dts
&qupv3_se6_4uart {
  status = "ok";
};
```

## Userspace Init Sequence

### Prerequisites (apt)
- `bluez` (provides hciattach, hciconfig, bluetoothctl)
- `rfkill`

### Init Steps (in order)
1. Power on BT chip via btpower ioctl:
   ```python
   import fcntl, os
   fd = os.open('/dev/btpower', os.O_RDWR)
   fcntl.ioctl(fd, 0xbfad, 1)  # BT_CMD_PWR_CTRL = 0xbfad
   os.close(fd)
   ```
2. `rfkill unblock bluetooth`
3. `hciattach -s 115200 /dev/ttyHS0 qualcomm 115200 flow`
4. `hciconfig hci0 up`

### Key Gotchas
- ttyHS0 is SE6 (0x898000) = BT UART. Before DTS change, ttyHS0 was GPS UART (0x88c000) — returned all zeros
- `hciattach any` creates hci0 but `hci0 up` fails with EBUSY. Must use `qualcomm` type with `-s 115200`
- bluez `qualcomm` init generates garbled firmware filename ("201 PF_ BUI.bin") from WCN3990 version string — but chip works without firmware file
- BT firmware partition (sde5, vfat) has files at `/image/crbtfw21.tlv` etc — not needed for basic BLE
- Root fs is read-only in most places; `/data/` is writable

## Hardware Details
- Chip: WCN3990 (Qualcomm, integrated WiFi+BT)
- BT version: 4.2 (LMP 0x08, sub 0x02be)
- Firmware string: "Release 10.0201 PF=WCN3990"
- BD Address: partially populated (00:00:00:00:5A:AD)
- BT UART: QUPv3 SE6 4-wire UART at 0x898000

## LowGlow Underglow Controller
- Controller: **SP105E** (not SP110E as initially assumed)
- BLE device name: `SP105E`
- MAC seen: `BA:AB:05:04:02:BD`
- Protocol: SP110E-compatible (same BLE service 0xFFE0, characteristic 0xFFE1)
- Can set: static color, brightness (relative), mode (1-120 presets), on/off. Speed not found yet
- Controller addresses LEDs individually for built-in patterns (rainbow flow etc.), but no per-LED BLE command found yet. May require longer payloads — only 6-byte packets tested so far
- Python library: `sp110e` (pip) or raw `bleak`

## Remaining TODO
1. **Speed command** - Not found yet (0x24/0x26 don't work), need more testing
2. **Map remaining state bytes** - Bytes 2, 4, 6, 7 still unknown
3. **Bake into AGNOS** - Add bluez+rfkill to agnos-builder system image so they persist across reboots
4. **App "apply configuration" recovery sequence** - Reverse-engineer what the app sends to recover from soft-brick
5. **Test glowd on device while driving**

## SP105E BLE Protocol (Reverse-Engineered March 2026)
- Service: 0xFFE0, Write characteristic: 0xFFE1 (read/write-without-response/write/notify)
- SP105E does NOT have 0xFFE2 init char (SP110E does) — no init needed
- Also has battery service 0x180F char 0x2A19 (read/notify)
- **Packet format: `38 [D1] [D2] [D3] [CMD] 83`** (6 bytes, 38/83 framing)
- **Color byte order: GRB (factory default) — persists to flash. Script sets GRB on connect to ensure known state**

### Confirmed Commands
| Command | Format | Notes |
|---------|--------|-------|
| SET_COLOR | `38 GG RR BB 1E 83` | GRB wire order (set on connect), API takes RGB |
| POWER_TOGGLE | `38 00 00 00 AA 83` | Toggle only, 0xAB does nothing |
| BRIGHT_UP | `38 SS 00 00 2A 83` | Relative step brighter, S=step size (1-16) |
| BRIGHT_DOWN | `38 SS 00 00 28 83` | Relative step dimmer, S=step size (1-8) |
| SET_MODE | `38 MM 00 00 2C 83` | Mode number in D1 (01, 05, 0A, etc.) |
| GET_STATE | `38 00 00 00 10 83` | Triggers notify with 8-byte state (see below) |
| COLOR_ORDER | `38 NN 00 00 3C 83` | 0=GRB 1=GBR 2=RGB 3=BGR 4=RBG 5=BRG. Persists to flash! |

### State Response (GET_STATE 0x10, via notify on FFE1)
8 bytes returned via BLE notify after sending GET_STATE. Also fires automatically on POWER_TOGGLE.
```
Byte 0: Power (1=ON, 0=OFF) — confirmed with visual correlation
Byte 1: Mode (SET_MODE value: 0xC9=201=static color, 1-120+=patterns)
Byte 2: 0x06 (unknown — never changed)
Byte 3: Brightness (0=min, 6=max, 7 levels) — confirmed stepping 0→1→2→3→4→5→6
Byte 4: 0x03 (unknown)
Byte 5: Color order (0=GRB, 1=GBR, 2=RGB, etc.) — confirmed
Byte 6: 0x02 (unknown)
Byte 7: 0x58 (88 — LED count?)
```
Power + brightness + mode + color order are live-readable.
Enables: deterministic on/off (`power_on`/`power_off`), absolute brightness (`set_brightness(0-6)`).

### Color Order Map (0x3C)
| Value | D1 | D2 | D3 | Name |
|-------|----|----|-----|------|
| 0 | G | R | B | GRB (default) |
| 1 | G | B | R | GBR |
| 2 | R | G | B | RGB |
| 3 | B | G | R | BGR |
| 4 | R | B | G | RBG |
| 5 | B | R | G | BRG |

### Pattern Modes (CMD byte directly, D1-D3 ignored)
| CMD | Description |
|-----|-------------|
| 0x03 | Rainbow animation (blue/red/green flowing) |
| 0x05 | Rainbow pattern 1 |
| 0x06 | Rainbow pattern 2 |
| 0x07 | Breathing: fade through colors (slow) |
| 0x08-0x0B | Breathing variations |
| 0x0D | Color cycle: yellow→orange→red, no fade |
| 0x0E | Same as 0x0D but very slow |
| 0x0F | Fast flowing rainbow |
| 0x10 | Same as 0x0F |

### Other findings
- Sending color (0x1E) stops any active pattern → returns to static
- SET_COLOR does NOT change power state — byte 0 stays the same, but LEDs visually respond even when "off"
- `0xAB` (OFF) does nothing
- Speed command not found yet (0x24/0x26 don't work, patterns auto-cycle between effects)
- Brightness: 7 levels (0-6), commands are relative (0x2A up, 0x28 down) but absolute control via state read + stepping
- Pattern commands (0x03, 0x07, etc as CMD byte) don't update state byte 1 — only SET_MODE (0x2C) does
- BLE direct read: FFE1 returns 128 bytes of zeros. State only readable via notify after GET_STATE (0x10) or POWER_TOGGLE (0xAA)
- Battery service (0x180F/0x2A19) returns 0 (not useful)
- Color order (0x3C) persists to flash across power cycles — script sets GRB (0) on connect
- Don't rapid-fire toggles — 0.3s gap between toggles can drop one
- **DANGEROUS commands** (soft-brick, ignores all commands after):
  - `0x1C` — sets bright white, unresponsive
  - `0x2D` — brief off/on, may wedge device state
  - **Recovery without power cycle**: App "apply configuration" (change color order to RGB then back to GRB + apply) recovers the device. Sending 0x3C alone does NOT work — app sends a config bundle that reinitializes the controller. Exact recovery sequence unknown.
  - LowGlow LED app config options: controller type (LowGlow V1), IC model (OG Kit vs Standard Kit), color order
- SP110E gist (partial overlap): https://gist.github.com/mbullington/37957501a07ad065b67d4e8d39bfe012

## glowd — Underglow Daemon
- Location: `selfdrive/glowd/glowd.py`
- Registered in `system/manager/process_config.py` as `only_onroad`, `enabled=TICI`
- Initializes BT adapter on start (btpower, rfkill, hciattach, hciconfig)
- Uses `power_on()`/`power_off()` for deterministic on/off with ignition
- SIGTERM handler (from manager) cleanly powers off LEDs on ignition off
- Auto-reconnects every 5s on BLE failure
- 20Hz update loop reading CarState

### Color Mapping (California-legal: no red or blue)
Priority order:
1. **Downshift** (gearActual decreases): purple flash 0.4s — gear debounced 0.5s to ignore neutral
2. **Braking** (brakePressed): deep orange, bright orange on hard brake (aEgo < -3)
3. **Blinker** (leftBlinker/rightBlinker): amber pulse at 1.5Hz
4. **Reverse** (gearShifter==reverse): white
5. **Standstill** (>2s): slow green breathing
6. **Gas** (gasPressed): RPM color blended toward warm amber
7. **RPM** (default): green(800rpm) → yellow → amber → purple(7000rpm)

### CarState fields used (brzpilot fork)
engineRpm, gearActual, shiftGrade, clutchPressed, brakePressed, gasPressed,
aEgo, vEgo, standstill, leftBlinker, rightBlinker, gearShifter

## Quick Reference Commands
- Scan: `adb shell "timeout 10 hcitool -i hci0 lescan 2>&1 | grep -v '(unknown)' | sort -u -k2"`
- Full init (after kernel+DTS already flashed):
  ```
  apt-get update && apt-get install -y bluez rfkill
  python3 -c "import fcntl,os; fd=os.open('/dev/btpower',os.O_RDWR); fcntl.ioctl(fd,0xbfad,1); os.close(fd)"
  rfkill unblock bluetooth
  hciattach -s 115200 /dev/ttyHS0 qualcomm 115200 flow
  hciconfig hci0 up
  ```
