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
- Can set: static color, brightness, mode (1-120 presets), speed, on/off
- Cannot address individual LEDs over BT (controller limitation)
- Python library: `sp110e` (pip) or raw `bleak`

## Remaining TODO
1. **UI setup button** - install bluez+rfkill, run init sequence, confirm slider pattern
2. **SP105E control script** - Python script using bleak to connect to SP105E and send commands
3. **CarState reactive colors** - Map vEgo/steeringAngleDeg/brakePressed/etc to SP105E color commands
4. **Bake into AGNOS** - Add bluez+rfkill to agnos-builder system image so they persist across reboots

## SP105E BLE Protocol (Reverse-Engineered March 2026)
- Service: 0xFFE0, Write characteristic: 0xFFE1 (read/write-without-response/write/notify)
- SP105E does NOT have 0xFFE2 init char (SP110E does) — no init needed
- Also has battery service 0x180F char 0x2A19 (read/notify)
- **Packet format: `38 [D1] [D2] [D3] [CMD] 83`** (6 bytes, 38/83 framing)
- **Color byte order: GRB (not RGB!)**

### Confirmed Commands
| Command | Format | Notes |
|---------|--------|-------|
| SET_COLOR | `38 GG RR BB 1E 83` | GRB byte order! |
| POWER_TOGGLE | `38 00 00 00 AA 83` | Toggle only, 0xAB does nothing |
| BRIGHTNESS | `38 BB 00 00 2A 83` | 0-255, higher=brighter |
| SET_MODE | `38 MM 00 00 2C 83` | Mode number in D1 (01, 05, 0A, etc.) |

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
- `0x28` also affects brightness (inverse: higher=dimmer)
- `0xAB` (OFF) does nothing
- `0x03` as speed command didn't work — triggers pattern instead
- Device must be ON for commands to work; color cmd alone doesn't turn it on
- `0x1C`-`0x29` (except 0x28) had no visible effect
- SP110E gist (partial overlap): https://gist.github.com/mbullington/37957501a07ad065b67d4e8d39bfe012

## Color Ideas for CarState Mapping
- **Startup** (park→drive): rainbow chase → settle to base color
- **Speed** (vEgo): blue(0)→purple(30mph)→pink/red(60+mph), brightness scales with speed
- **Braking** (brakePressed/brake): deep red, brightness = brake pressure, flash on hard brake (aEgo < -3)
- **Acceleration** (gasPressed + aEgo): orange→red fire gradient
- **Steering** (steeringAngleDeg): color shifts left=blue/purple, right=orange/amber
- **Blinker** (leftBlinker/rightBlinker): amber pulse at ~1.5Hz
- **openpilot engaged** (cruiseState.enabled): comma green (0,255,100)
- **Reverse** (gearShifter==reverse): white glow
- **Parked/idle** (standstill): slow breathing pulse
- Note: engineRpm is DEPRECATED in car.capnp, use vEgo/aEgo instead

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
