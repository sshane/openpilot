"""
Live Manual Stats Widget

Small onroad overlay showing current drive statistics, RPM meter with rev-match helper,
shift grade feedback, and launch progress.
"""

import json
import pyray as rl

from openpilot.common.params import Params
from opendbc.car.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget


# Colors
GREEN = rl.Color(46, 204, 113, 220)
YELLOW = rl.Color(241, 196, 15, 220)
RED = rl.Color(231, 76, 60, 220)
ORANGE = rl.Color(230, 126, 34, 220)
CYAN = rl.Color(52, 152, 219, 220)
WHITE = rl.Color(255, 255, 255, 220)
GRAY = rl.Color(150, 150, 150, 200)
BG_COLOR = rl.Color(0, 0, 0, 160)

# RPM zones for BRZ (7500 redline)
RPM_REDLINE = 7500
RPM_ECONOMY_MAX = 2500
RPM_POWER_MIN = 4000
RPM_DANGER_MIN = 6500
RPM_TARGET_MIN_DISPLAY = 750  # Don't show upshift indicator below this RPM

# 2024 BRZ gear ratios for rev-match calculation
BRZ_GEAR_RATIOS = {1: 3.626, 2: 2.188, 3: 1.541, 4: 1.213, 5: 1.000, 6: 0.767}
BRZ_FINAL_DRIVE = 4.10
BRZ_TIRE_CIRCUMFERENCE = 1.977


def rpm_for_speed_and_gear(speed_ms: float, gear: int) -> float:
  """Calculate expected RPM for a given speed and gear"""
  if gear not in BRZ_GEAR_RATIOS or speed_ms <= 0:
    return 0.0
  return (speed_ms * BRZ_FINAL_DRIVE * BRZ_GEAR_RATIOS[gear] * 60) / BRZ_TIRE_CIRCUMFERENCE


class ManualStatsWidget(Widget):
  """Widget showing live manual driving stats, RPM meter, and feedback"""

  def __init__(self):
    super().__init__()
    self._params = Params()
    self._visible = False
    self._stats: dict = {}
    self._update_counter = 0
    # Shift grade flash state
    self._last_shift_grade = 0
    self._shift_flash_frames = 0
    self._flash_grade = 0  # The grade to display during flash
    # Track gear before clutch for rev-match display
    self._gear_before_clutch = 0
    self._last_clutch_state = False
    # Filtered RPM for smooth label display (0.1s time constant, ~60fps)
    self._rpm_filter = FirstOrderFilter(0, 0.1, 1/60)

  def _render(self, rect: rl.Rectangle):
    # Update stats every ~15 frames (0.25s at 60fps)
    self._update_counter += 1
    if self._update_counter >= 15:
      self._update_counter = 0
      self._load_stats()

    # Get live data from CarState
    cs = ui_state.sm['carState']# if ui_state.sm.valid['carState'] else None
    if not cs:
      return

    # Widget dimensions - extend to bottom with same margin as top
    margin = 10
    w = 250
    h = int(rect.height - 2 * margin)  # Full height minus top and bottom margin
    x = int(rect.x + rect.width - w - margin)
    y = int(rect.y + margin)

    # Background
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, h), 0.08, 10, BG_COLOR)

    font = gui_app.font(FontWeight.MEDIUM)
    font_bold = gui_app.font(FontWeight.BOLD)
    px = x + 14
    py = y + 12

    # === RPM METER ===
    rpm = cs.engineRpm
    self._draw_rpm_meter(px, py, w - 28, 50, rpm, cs)
    py += 62

    # === GEAR + SHIFT GRADE FLASH ===
    gear = cs.gearActual
    gear_text = str(gear) if gear > 0 else "N"

    # Check for new shift - only trigger when shiftGrade goes from 0 to non-zero
    if cs.shiftGrade > 0 and self._last_shift_grade == 0:
      # New shift detected - start flash with this grade
      self._shift_flash_frames = 150  # Flash for 2.5s at 60fps
      self._flash_grade = cs.shiftGrade  # Store the grade to display
    # Track the raw shiftGrade value
    self._last_shift_grade = cs.shiftGrade

    # Draw gear with flash color if recently shifted
    if self._shift_flash_frames > 0:
      self._shift_flash_frames -= 1
      if self._flash_grade == 1:
        gear_color = GREEN
        grade_text = "✓"
      elif self._flash_grade == 2:
        gear_color = YELLOW
        grade_text = "~"
      else:
        gear_color = RED
        grade_text = "✗"
      rl.draw_text_ex(font_bold, gear_text, rl.Vector2(px, py), 55, 0, gear_color)
      rl.draw_text_ex(font_bold, grade_text, rl.Vector2(px + 42, py + 8), 40, 0, gear_color)
    else:
      rl.draw_text_ex(font_bold, gear_text, rl.Vector2(px, py), 55, 0, WHITE)

    # Shift suggestion arrow
    suggestion = self._stats.get('shift_suggestion', 'ok')
    if suggestion == 'upshift':
      rl.draw_text_ex(font_bold, "↑", rl.Vector2(px + 95, py + 8), 43, 0, GREEN)
    elif suggestion == 'downshift':
      rl.draw_text_ex(font_bold, "↓", rl.Vector2(px + 95, py + 8), 43, 0, YELLOW)

    py += 62

    # === LAUNCH FEEDBACK ===
    launches = self._stats.get('launches', 0)
    good_launches = self._stats.get('good_launches', 0)
    # Detect if currently launching (low speed, was stopped)
    if cs.vEgo < 5.0 and cs.vEgo > 0.5 and not cs.clutchPressed:
      rl.draw_text_ex(font, "LAUNCHING...", rl.Vector2(px, py), 26, 0, CYAN)
    elif launches > 0:
      pct = int(good_launches / launches * 100) if launches > 0 else 0
      color = GREEN if pct >= 75 else (YELLOW if pct >= 50 else GRAY)
      rl.draw_text_ex(font, f"Launch: {good_launches}/{launches}", rl.Vector2(px, py), 26, 0, color)
    else:
      rl.draw_text_ex(font, "Launch: -", rl.Vector2(px, py), 26, 0, GRAY)
    py += 34

    # === STATS ROW ===
    font_size = 24

    # Stalls & Lugs on same line
    stalls = self._stats.get('stalls', 0)
    lugs = self._stats.get('lugs', 0)
    is_lugging = cs.isLugging

    if is_lugging:
      rl.draw_text_ex(font, "LUGGING!", rl.Vector2(px, py), font_size, 0, RED)
    else:
      stall_color = GREEN if stalls == 0 else RED
      lug_color = GREEN if lugs == 0 else YELLOW
      rl.draw_text_ex(font, f"S:{stalls}", rl.Vector2(px, py), font_size, 0, stall_color)
      rl.draw_text_ex(font, f"L:{lugs}", rl.Vector2(px + 65, py), font_size, 0, lug_color)

    # Shift quality
    shifts = self._stats.get('shifts', 0)
    good_shifts = self._stats.get('good_shifts', 0)
    if shifts > 0:
      pct = int(good_shifts / shifts * 100)
      color = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
      rl.draw_text_ex(font, f"Sh:{pct}%", rl.Vector2(px + 135, py), font_size, 0, color)
    else:
      rl.draw_text_ex(font, "Sh:-", rl.Vector2(px + 135, py), font_size, 0, GRAY)

  def _draw_rpm_meter(self, x: int, y: int, w: int, h: int, rpm: float, cs):
    """Draw RPM bar with color zones and rev-match target"""
    font = gui_app.font(FontWeight.MEDIUM)

    # Bar background (pushed down for bigger RPM text)
    bar_h = 20
    bar_y = y + 32
    rl.draw_rectangle_rounded(rl.Rectangle(x, bar_y, w, bar_h), 0.3, 5, rl.Color(40, 40, 40, 200))

    # Calculate fill width
    rpm_pct = min(rpm / RPM_REDLINE, 1.0)
    fill_w = int(w * rpm_pct)

    # Color based on RPM zone
    if rpm < RPM_ECONOMY_MAX:
      bar_color = GREEN
    elif rpm < RPM_POWER_MIN:
      bar_color = YELLOW
    elif rpm < RPM_DANGER_MIN:
      bar_color = ORANGE
    else:
      bar_color = RED

    # Draw filled portion
    if fill_w > 0:
      rl.draw_rectangle_rounded(rl.Rectangle(x, bar_y, fill_w, bar_h), 0.3, 5, bar_color)

    # Track gear before clutch press for rev-match display
    if not cs.clutchPressed and cs.gearActual > 0:
      self._gear_before_clutch = cs.gearActual

    # Rev-match target lines when clutch pressed OR shift suggestion showing
    suggestion = self._stats.get('shift_suggestion', 'ok')
    show_rev_targets = (cs.clutchPressed or suggestion != 'ok') and self._gear_before_clutch > 0
    if show_rev_targets:
      # 65% opacity when showing due to suggestion only (not clutch)
      alpha = 220 if cs.clutchPressed else 143
      cyan = rl.Color(CYAN.r, CYAN.g, CYAN.b, alpha)
      red = rl.Color(RED.r, RED.g, RED.b, alpha)
      white = rl.Color(WHITE.r, WHITE.g, WHITE.b, alpha)

      # Calculate both targets first
      down_rpm = 0
      up_rpm = 0
      if self._gear_before_clutch > 1:
        down_rpm = rpm_for_speed_and_gear(cs.vEgo, self._gear_before_clutch - 1)
      if self._gear_before_clutch < 6:
        up_rpm = rpm_for_speed_and_gear(cs.vEgo, self._gear_before_clutch + 1)

      # Downshift target - cyan if safe, red if over redline
      if down_rpm >= RPM_REDLINE:
        # Over redline - show red warning clipped to right side
        down_x = x + w
        rl.draw_rectangle(down_x - 4, bar_y - 5, 4, bar_h + 10, red)
        rl.draw_text_ex(font, f"{int(round(down_rpm / 10) * 10)}!", rl.Vector2(down_x - 45, bar_y + bar_h + 3), 20, 0, red)
      elif down_rpm > RPM_TARGET_MIN_DISPLAY:
        # Safe downshift target (cyan)
        down_x = x + int(w * (down_rpm / RPM_REDLINE))
        rl.draw_rectangle(down_x - 2, bar_y - 5, 4, bar_h + 10, cyan)
        rl.draw_text_ex(font, f"{int(round(down_rpm / 10) * 10)}", rl.Vector2(down_x - 20, bar_y + bar_h + 3), 20, 0, cyan)

      # Upshift target (white) - only show if above minimum display threshold
      if up_rpm > RPM_TARGET_MIN_DISPLAY and up_rpm < RPM_REDLINE:
        up_x = x + int(w * (up_rpm / RPM_REDLINE))
        rl.draw_rectangle(up_x - 2, bar_y - 5, 4, bar_h + 10, white)
        rl.draw_text_ex(font, f"{int(round(up_rpm / 10) * 10)}", rl.Vector2(up_x - 20, bar_y + bar_h + 3), 20, 0, white)

    # RPM text (filtered for smooth display, rounded to nearest 10)
    self._rpm_filter.update(rpm)
    rpm_text = f"{int(round(self._rpm_filter.x / 10) * 10)}"
    rl.draw_text_ex(font, rpm_text, rl.Vector2(x, y), 28, 0, WHITE)
    rl.draw_text_ex(font, "rpm", rl.Vector2(x + 70, y + 5), 20, 0, GRAY)

  def _load_stats(self):
    """Load current session stats"""
    data = self._params.get("ManualDriveLiveStats")
    self._stats = data if data else {}
