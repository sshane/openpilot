"""
Manual Driving Stats Settings Page

Shows historical stats and trends for manual transmission driving.
"""

import datetime
import random
import pyray as rl

from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.scroll_panel2 import GuiScrollPanel2
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget, NavWidget
from openpilot.selfdrive.ui.mici.layouts.manual_drive_summary import ManualDriveSummaryDialog


# Colors
GREEN = rl.Color(46, 204, 113, 255)
YELLOW = rl.Color(241, 196, 15, 255)
RED = rl.Color(231, 76, 60, 255)
GRAY = rl.Color(100, 100, 100, 255)
LIGHT_GRAY = rl.Color(180, 180, 180, 255)
WHITE = rl.Color(255, 255, 255, 255)
BG_CARD = rl.Color(45, 45, 45, 255)


class ManualStatsLayout(NavWidget):
  """Settings page showing historical manual driving stats"""

  def __init__(self, back_callback):
    super().__init__()
    self._params = Params()
    self._scroll_panel = GuiScrollPanel2(horizontal=False)
    self._stats: dict = {}
    self._hand_text: str = ""
    self._hand_color: rl.Color = GRAY
    self._encouragement_text: str = ""
    self._section_comments: dict[str, tuple[str, rl.Color]] = {}
    self.set_back_callback(back_callback)

  def show_event(self):
    super().show_event()
    self._scroll_panel.set_offset(0)
    self._load_stats()

  def _load_stats(self):
    """Load historical stats from Params and cache random text picks"""
    data = self._params.get("ManualDriveStats")
    if data:
      self._stats = data
    else:
      self._stats = {}
    # Pick random texts once per page visit (not every frame)
    self._hand_text, self._hand_color = self._get_overall_hand()
    self._encouragement_text = self._get_encouragement()
    self._section_comments = self._pick_section_comments()

  def _draw_comment(self, x: int, y: int, w: int, key: str) -> int:
    """Draw a section comment if one exists. Returns updated y."""
    if key not in self._section_comments:
      return y
    text, color = self._section_comments[key]
    font_roman = gui_app.font(FontWeight.ROMAN)
    wrapped = wrap_text(font_roman, text, 18, w)
    for line in wrapped:
      rl.draw_text_ex(font_roman, line, rl.Vector2(x, y), 18, 0, color)
      y += 22
    return y + 5

  def _render(self, rect: rl.Rectangle):
    content_height = self._measure_content_height(rect)
    scroll_offset = round(self._scroll_panel.update(rect, content_height))

    x = int(rect.x + 20)
    y = int(rect.y + 20 + scroll_offset)
    w = int(rect.width - 40)

    # Title
    font_bold = gui_app.font(FontWeight.BOLD)
    font_medium = gui_app.font(FontWeight.MEDIUM)
    font_roman = gui_app.font(FontWeight.ROMAN)

    rl.draw_text_ex(font_bold, "Manual Driving Stats", rl.Vector2(x, y), 48, 0, WHITE)
    y += 60

    # View Last Drive button
    btn_w, btn_h = 340, 65
    btn_rect = rl.Rectangle(x, y, btn_w, btn_h)
    btn_color = rl.Color(60, 60, 60, 255) if not rl.check_collision_point_rec(rl.get_mouse_position(), btn_rect) else rl.Color(80, 80, 80, 255)
    rl.draw_rectangle_rounded(btn_rect, 0.3, 10, btn_color)
    rl.draw_text_ex(font_medium, "View Last Drive Summary", rl.Vector2(x + 20, y + 18), 26, 0, WHITE)
    if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT) and rl.check_collision_point_rec(rl.get_mouse_position(), btn_rect):
      gui_app.set_modal_overlay(ManualDriveSummaryDialog())
    y += btn_h + 25

    if not self._stats or self._stats.get('total_drives', 0) == 0:
      rl.draw_text_ex(font_roman, "No driving data yet. Get out there and practice!",
                      rl.Vector2(x, y), 28, 0, GRAY)
      return

    # Overall hand rating
    y = self._draw_card(x, y, w, "Your Hand", [
      ("Overall Rating", self._hand_text, self._hand_color),
      ("Total Drives", str(self._stats.get('total_drives', 0)), WHITE),
      ("Total Drive Time", self._format_time(self._stats.get('total_drive_time', 0)), WHITE),
      ("Total Stalls", str(self._stats.get('total_stalls', 0)), self._stall_color(self._stats.get('total_stalls', 0))),
      ("Total Lugs", str(self._stats.get('total_lugs', 0)), LIGHT_GRAY),
    ])
    y += 15

    # Shift quality card
    total_up = self._stats.get('total_upshifts', 0)
    total_down = self._stats.get('total_downshifts', 0)
    up_good = self._stats.get('upshifts_good', 0)
    down_good = self._stats.get('downshifts_good', 0)

    up_pct = f"{int(up_good / total_up * 100)}%" if total_up > 0 else "N/A"
    down_pct = f"{int(down_good / total_down * 100)}%" if total_down > 0 else "N/A"

    y = self._draw_card(x, y, w, "Shift Quality", [
      ("Total Upshifts", str(total_up), WHITE),
      ("Good Upshifts", f"{up_good} ({up_pct})", self._pct_color(up_good, total_up)),
      ("Total Downshifts", str(total_down), WHITE),
      ("Good Downshifts", f"{down_good} ({down_pct})", self._pct_color(down_good, total_down)),
    ])
    y = self._draw_comment(x, y, w, 'shifts')
    y += 15

    # Launch quality card
    total_launches = self._stats.get('total_launches', 0)
    good_launches = self._stats.get('launches_good', 0)
    stalled_launches = self._stats.get('launches_stalled', 0)

    launch_pct = f"{int(good_launches / total_launches * 100)}%" if total_launches > 0 else "N/A"

    y = self._draw_card(x, y, w, "Launch Quality", [
      ("Total Launches", str(total_launches), WHITE),
      ("Good Launches", f"{good_launches} ({launch_pct})", self._pct_color(good_launches, total_launches)),
      ("Stalled Launches", str(stalled_launches), RED if stalled_launches > 0 else GREEN),
    ])
    y = self._draw_comment(x, y, w, 'launches')
    y += 15

    # Trend card - aggregate by day for consistency with charts
    session_history = self._stats.get('session_history', [])
    recent_days = self._aggregate_by_day(session_history)[-10:]
    num_days = len(recent_days)

    recent_stalls = [d.get('stalls', 0) for d in recent_days]
    recent_shifts = []
    for d in recent_days:
      total = d.get('upshifts', 0) + d.get('downshifts', 0)
      good = d.get('upshifts_good', 0) + d.get('downshifts_good', 0)
      recent_shifts.append(int(good / total * 100) if total > 0 else 100)

    trend_items = []
    if len(recent_stalls) >= 2:
      trend = self._calculate_trend(recent_stalls)
      trend_text, trend_color = self._trend_text(trend, lower_better=True)
      trend_items.append((f"Stall Trend (last {num_days}d)", trend_text, trend_color))

    if len(recent_shifts) >= 2:
      trend = self._calculate_trend(recent_shifts)
      trend_text, trend_color = self._trend_text(trend, lower_better=False)
      trend_items.append((f"Shift Score Trend (last {num_days}d)", trend_text, trend_color))

    if recent_shifts:
      avg_score = sum(recent_shifts) / len(recent_shifts)
      trend_items.append((f"Avg Shift Score (last {num_days}d)", f"{int(avg_score)}/100", self._score_color(avg_score)))

    if trend_items:
      y = self._draw_card(x, y, w, "Recent Trends", trend_items)
      y += 15

    # Per-gear smoothness chart
    gear_counts = self._stats.get('gear_shift_counts', {})
    gear_jerks = self._stats.get('gear_shift_jerk_totals', {})
    if gear_counts and any(gear_counts.values()):
      y = self._draw_gear_chart(x, y, w, gear_counts, gear_jerks)
      y = self._draw_comment(x, y, w, 'gears')
      y += 15

    # Session history charts
    session_history = self._stats.get('session_history', [])
    if session_history:
      y = self._draw_shift_chart(x, y, w, session_history)
      y = self._draw_comment(x, y, w, 'shift_chart')
      y += 15
      y = self._draw_stalls_chart(x, y, w, session_history)
      y = self._draw_comment(x, y, w, 'stalls_chart')
      y += 15
      y = self._draw_launch_chart(x, y, w, session_history)
      y = self._draw_comment(x, y, w, 'launch_chart')
      y += 15

    # Encouragement based on progress (with text wrapping)
    y += 10
    wrapped_lines = wrap_text(font_roman, self._encouragement_text, 24, w - 10)
    for line in wrapped_lines:
      rl.draw_text_ex(font_roman, line, rl.Vector2(x, y), 24, 0, LIGHT_GRAY)
      y += 30

  def _draw_card(self, x: int, y: int, w: int, title: str, items: list) -> int:
    """Draw a card with title and stat items, with wrapping for long values"""
    font_bold = gui_app.font(FontWeight.BOLD)
    font_medium = gui_app.font(FontWeight.MEDIUM)

    # Calculate height - check for items that need wrapping
    extra_lines = 0
    max_value_width = w - 220  # Leave space for label, trigger wrap earlier
    for _, value, _ in items:
      value_width = rl.measure_text_ex(font_medium, value, 24, 0).x
      if value_width > max_value_width:
        extra_lines += 1

    card_h = 50 + len(items) * 38 + extra_lines * 32
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, card_h), 0.02, 10, BG_CARD)

    # Title
    rl.draw_text_ex(font_bold, title, rl.Vector2(x + 15, y + 12), 32, 0, WHITE)
    y += 50

    # Items
    for label, value, color in items:
      value_width = rl.measure_text_ex(font_medium, value, 24, 0).x

      # Check if value needs to wrap to next line (below label)
      if value_width > max_value_width:
        # Draw label
        rl.draw_text_ex(font_medium, label, rl.Vector2(x + 15, y), 26, 0, LIGHT_GRAY)
        y += 32
        # Draw value on next line, wrapped if needed
        wrapped = wrap_text(font_medium, value, 22, w - 40)
        for line in wrapped:
          rl.draw_text_ex(font_medium, line, rl.Vector2(x + 25, y), 22, 0, color)
          y += 26
        y += 6
      else:
        # Draw label and value on same line
        rl.draw_text_ex(font_medium, label, rl.Vector2(x + 15, y), 26, 0, LIGHT_GRAY)
        rl.draw_text_ex(font_medium, value, rl.Vector2(x + w - 15 - value_width, y), 24, 0, color)
        y += 38

    return y

  def _aggregate_by_day(self, sessions: list) -> list:
    """Aggregate sessions into per-day summaries, summing counts"""
    import math
    days: dict[str, dict] = {}  # date_str -> aggregated dict
    for s in sessions:
      ts = s.get('timestamp', 0)
      if not ts or not isinstance(ts, (int, float)) or math.isnan(ts) or ts <= 0:
        continue
      date_key = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
      if date_key not in days:
        days[date_key] = {
          'timestamp': ts,  # Keep last timestamp for the day label
          'duration': 0, 'stalls': 0, 'lugs': 0,
          'upshifts': 0, 'upshifts_good': 0,
          'downshifts': 0, 'downshifts_good': 0,
          'launches': 0, 'launches_good': 0,
        }
      d = days[date_key]
      d['timestamp'] = max(d['timestamp'], ts)
      for k in ('duration', 'stalls', 'lugs', 'upshifts', 'upshifts_good',
                'downshifts', 'downshifts_good', 'launches', 'launches_good'):
        d[k] = d.get(k, 0) + s.get(k, 0)
    return list(days.values())

  def _draw_shift_chart(self, x: int, y: int, w: int, sessions: list) -> int:
    """Draw a bar chart showing shift score history (aggregated by day)"""
    font_bold = gui_app.font(FontWeight.BOLD)
    font_small = gui_app.font(FontWeight.ROMAN)

    chart_h = 200
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, chart_h), 0.02, 10, BG_CARD)

    # Title
    rl.draw_text_ex(font_bold, "Shift Score History", rl.Vector2(x + 15, y + 12), 28, 0, WHITE)

    # Chart area
    chart_x = x + 40
    chart_y = y + 50
    chart_w = w - 60
    chart_inner_h = 90

    # Draw axis
    rl.draw_line(chart_x, chart_y + chart_inner_h, chart_x + chart_w, chart_y + chart_inner_h, GRAY)
    rl.draw_line(chart_x, chart_y, chart_x, chart_y + chart_inner_h, GRAY)

    # Y-axis labels
    rl.draw_text_ex(font_small, "100", rl.Vector2(x + 10, chart_y - 5), 14, 0, GRAY)
    rl.draw_text_ex(font_small, "50", rl.Vector2(x + 15, chart_y + chart_inner_h // 2 - 5), 14, 0, GRAY)
    rl.draw_text_ex(font_small, "0", rl.Vector2(x + 22, chart_y + chart_inner_h - 5), 14, 0, GRAY)

    days = self._aggregate_by_day(sessions)
    display_days = days[-12:] if len(days) > 12 else days
    if not display_days:
      return y + chart_h

    bar_spacing = 4
    bar_w = max(8, (chart_w - bar_spacing * len(display_days)) // len(display_days))

    for i, day in enumerate(display_days):
      ups = day.get('upshifts', 0)
      ups_good = day.get('upshifts_good', 0)
      downs = day.get('downshifts', 0)
      downs_good = day.get('downshifts_good', 0)
      total = ups + downs
      score = ((ups_good + downs_good) / total * 100) if total > 0 else 100

      bar_h = int((score / 100) * chart_inner_h)
      bar_x = chart_x + i * (bar_w + bar_spacing)
      bar_y_top = chart_y + chart_inner_h - bar_h

      color = GREEN if score >= 80 else (YELLOW if score >= 50 else RED)
      rl.draw_rectangle(int(bar_x), int(bar_y_top), int(bar_w), int(bar_h), color)

      # Day label
      timestamp = day.get('timestamp', 0)
      if timestamp > 0:
        dt = datetime.datetime.fromtimestamp(timestamp)
        day_x = bar_x + bar_w // 2 - 4
        rl.draw_text_ex(font_small, str(dt.day), rl.Vector2(day_x, chart_y + chart_inner_h + 4), 13, 0, GRAY)

    # Legend
    legend_y = chart_y + chart_inner_h + 22
    rl.draw_text_ex(font_small, "Higher = better shifts. Green 80%+, Yellow 50%+, Red <50%", rl.Vector2(chart_x, legend_y), 14, 0, GRAY)

    return y + chart_h

  def _draw_stalls_chart(self, x: int, y: int, w: int, sessions: list) -> int:
    """Draw a bar chart showing stalls and lugs per day"""
    font_bold = gui_app.font(FontWeight.BOLD)
    font_small = gui_app.font(FontWeight.ROMAN)

    chart_h = 180
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, chart_h), 0.02, 10, BG_CARD)

    # Title
    rl.draw_text_ex(font_bold, "Stalls & Lugs (Jackets)", rl.Vector2(x + 15, y + 12), 28, 0, WHITE)

    # Chart area
    chart_x = x + 40
    chart_y = y + 50
    chart_w = w - 60
    chart_inner_h = 70

    days = self._aggregate_by_day(sessions)
    display_days = days[-12:] if len(days) > 12 else days

    # Find max for scaling
    max_issues = max((d.get('stalls', 0) + d.get('lugs', 0) for d in display_days), default=1) if display_days else 1
    max_issues = max(max_issues, 5)  # Min scale of 5

    # Draw axis
    rl.draw_line(chart_x, chart_y + chart_inner_h, chart_x + chart_w, chart_y + chart_inner_h, GRAY)
    rl.draw_line(chart_x, chart_y, chart_x, chart_y + chart_inner_h, GRAY)

    # Y-axis labels
    rl.draw_text_ex(font_small, str(max_issues), rl.Vector2(x + 15, chart_y - 5), 14, 0, GRAY)
    rl.draw_text_ex(font_small, "0", rl.Vector2(x + 22, chart_y + chart_inner_h - 5), 14, 0, GRAY)

    if not display_days:
      return y + chart_h

    bar_spacing = 4
    bar_w = max(8, (chart_w - bar_spacing * len(display_days)) // len(display_days))

    for i, day in enumerate(display_days):
      stalls = day.get('stalls', 0)
      lugs = day.get('lugs', 0)
      bar_x = chart_x + i * (bar_w + bar_spacing)

      # Stacked bar: stalls (red) on bottom, lugs (orange) on top
      stall_h = int((stalls / max_issues) * chart_inner_h)
      lug_h = int((lugs / max_issues) * chart_inner_h)

      # Lugs (yellow/orange) - bottom
      if lug_h > 0:
        rl.draw_rectangle(int(bar_x), int(chart_y + chart_inner_h - lug_h), int(bar_w), int(lug_h), YELLOW)

      # Stalls (red) - stacked on top of lugs
      if stall_h > 0:
        rl.draw_rectangle(int(bar_x), int(chart_y + chart_inner_h - lug_h - stall_h), int(bar_w), int(stall_h), RED)

      # Day label
      timestamp = day.get('timestamp', 0)
      if timestamp > 0:
        dt = datetime.datetime.fromtimestamp(timestamp)
        day_x = bar_x + bar_w // 2 - 4
        rl.draw_text_ex(font_small, str(dt.day), rl.Vector2(day_x, chart_y + chart_inner_h + 4), 13, 0, GRAY)

    # Legend
    legend_y = chart_y + chart_inner_h + 22
    rl.draw_rectangle(int(chart_x), int(legend_y + 2), 12, 12, RED)
    rl.draw_text_ex(font_small, "Stalls", rl.Vector2(chart_x + 16, legend_y), 14, 0, GRAY)
    rl.draw_rectangle(int(chart_x + 70), int(legend_y + 2), 12, 12, YELLOW)
    rl.draw_text_ex(font_small, "Lugs", rl.Vector2(chart_x + 86, legend_y), 14, 0, GRAY)
    rl.draw_text_ex(font_small, "Lower = fewer jackets!", rl.Vector2(chart_x + 140, legend_y), 14, 0, GRAY)

    return y + chart_h

  def _draw_launch_chart(self, x: int, y: int, w: int, sessions: list) -> int:
    """Draw a bar chart showing launch success rate per day"""
    font_bold = gui_app.font(FontWeight.BOLD)
    font_small = gui_app.font(FontWeight.ROMAN)

    chart_h = 180
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, chart_h), 0.02, 10, BG_CARD)

    # Title
    rl.draw_text_ex(font_bold, "Launch Success (Waddle Rate)", rl.Vector2(x + 15, y + 12), 28, 0, WHITE)

    # Chart area
    chart_x = x + 40
    chart_y = y + 50
    chart_w = w - 60
    chart_inner_h = 70

    # Draw axis
    rl.draw_line(chart_x, chart_y + chart_inner_h, chart_x + chart_w, chart_y + chart_inner_h, GRAY)
    rl.draw_line(chart_x, chart_y, chart_x, chart_y + chart_inner_h, GRAY)

    # Y-axis labels
    rl.draw_text_ex(font_small, "100%", rl.Vector2(x + 5, chart_y - 5), 14, 0, GRAY)
    rl.draw_text_ex(font_small, "0%", rl.Vector2(x + 15, chart_y + chart_inner_h - 5), 14, 0, GRAY)

    days = self._aggregate_by_day(sessions)
    display_days = days[-12:] if len(days) > 12 else days
    if not display_days:
      return y + chart_h

    bar_spacing = 4
    bar_w = max(8, (chart_w - bar_spacing * len(display_days)) // len(display_days))

    for i, day in enumerate(display_days):
      launches = day.get('launches', 0)
      launches_good = day.get('launches_good', 0)
      bar_x = chart_x + i * (bar_w + bar_spacing)

      if launches > 0:
        pct = (launches_good / launches) * 100
        bar_h = int((pct / 100) * chart_inner_h)
        bar_y_top = chart_y + chart_inner_h - bar_h
        color = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
        rl.draw_rectangle(int(bar_x), int(bar_y_top), int(bar_w), int(bar_h), color)
      else:
        # No launches - draw thin gray bar
        rl.draw_rectangle(int(bar_x), int(chart_y + chart_inner_h - 2), int(bar_w), 2, GRAY)

      # Day label
      timestamp = day.get('timestamp', 0)
      if timestamp > 0:
        dt = datetime.datetime.fromtimestamp(timestamp)
        day_x = bar_x + bar_w // 2 - 4
        rl.draw_text_ex(font_small, str(dt.day), rl.Vector2(day_x, chart_y + chart_inner_h + 4), 13, 0, GRAY)

    # Legend
    legend_y = chart_y + chart_inner_h + 22
    rl.draw_text_ex(font_small, "Higher = smoother launches = more waddle, less jacket!", rl.Vector2(chart_x, legend_y), 14, 0, GRAY)

    return y + chart_h

  def _draw_gear_chart(self, x: int, y: int, w: int, gear_counts: dict, gear_jerks: dict) -> int:
    """Draw a bar chart showing shift smoothness into each gear (1-6)"""
    font_bold = gui_app.font(FontWeight.BOLD)
    font_small = gui_app.font(FontWeight.ROMAN)

    chart_h = 180
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, chart_h), 0.02, 10, BG_CARD)

    # Title
    rl.draw_text_ex(font_bold, "Waddle Smoothness by Gear", rl.Vector2(x + 15, y + 12), 28, 0, WHITE)

    # Chart area
    chart_x = x + 50
    chart_y = y + 50
    chart_w = w - 70
    chart_inner_h = 70

    # Draw axis
    rl.draw_line(chart_x, chart_y + chart_inner_h, chart_x + chart_w, chart_y + chart_inner_h, GRAY)
    rl.draw_line(chart_x, chart_y, chart_x, chart_y + chart_inner_h, GRAY)

    # Y-axis labels (smoothness score, higher = better)
    rl.draw_text_ex(font_small, "Smooth", rl.Vector2(x + 5, chart_y - 2), 12, 0, GREEN)
    rl.draw_text_ex(font_small, "Jerky", rl.Vector2(x + 10, chart_y + chart_inner_h - 10), 12, 0, RED)

    # Calculate smoothness scores for each gear (invert jerk - lower jerk = higher score)
    bar_spacing = 12
    bar_w = (chart_w - bar_spacing * 5) // 6

    for gear in range(1, 7):
      count = gear_counts.get(gear, gear_counts.get(str(gear), 0))
      jerk_total = gear_jerks.get(gear, gear_jerks.get(str(gear), 0.0))

      bar_x = chart_x + (gear - 1) * (bar_w + bar_spacing)

      if count > 0:
        avg_jerk = jerk_total / count
        # Convert jerk to smoothness score (0-100), lower jerk = higher score
        # Jerk of 0 = 100, jerk of 5+ = 0
        smoothness = max(0, min(100, 100 - (avg_jerk * 20)))

        bar_h = int((smoothness / 100) * chart_inner_h)
        bar_y = chart_y + chart_inner_h - bar_h

        # Color based on smoothness
        if smoothness >= 80:
          color = GREEN
        elif smoothness >= 50:
          color = YELLOW
        else:
          color = RED

        rl.draw_rectangle(int(bar_x), int(bar_y), int(bar_w), int(bar_h), color)
      else:
        # No data - draw thin gray bar
        rl.draw_rectangle(int(bar_x), int(chart_y + chart_inner_h - 2), int(bar_w), 2, GRAY)

      # Gear label
      gear_label = str(gear)
      label_x = bar_x + bar_w // 2 - 5
      rl.draw_text_ex(font_small, gear_label, rl.Vector2(label_x, chart_y + chart_inner_h + 6), 16, 0, WHITE)

    # Legend
    legend_y = chart_y + chart_inner_h + 28
    rl.draw_text_ex(font_small, "Green = waddle smooth, Red = jerky jackets. Practice weak gears!", rl.Vector2(x + 15, legend_y), 14, 0, GRAY)

    return y + chart_h

  def _measure_content_height(self, rect: rl.Rectangle) -> int:
    """Measure total content height for scrolling"""
    y = 20 + 60  # Title
    y += 90  # View Last Drive button (65 + 25)

    if not self._stats or self._stats.get('total_drives', 0) == 0:
      return y + 40

    comment_h = 27  # height per section comment line

    # Overview card (now has 5 items with hand rating, +60 for potential wrapped lines)
    y += 50 + 5 * 38 + 60 + 15
    # Shift card + comment
    y += 50 + 4 * 38 + 15
    if 'shifts' in self._section_comments:
      y += comment_h
    # Launch card + comment
    y += 50 + 3 * 38 + 15
    if 'launches' in self._section_comments:
      y += comment_h
    # Trend card (estimate)
    y += 50 + 3 * 38 + 15
    # Gear chart + comment
    if self._stats.get('gear_shift_counts'):
      y += 180 + 15
      if 'gears' in self._section_comments:
        y += comment_h

    # Charts (3 charts) + comments
    if self._stats.get('session_history'):
      y += 200 + 15  # Shift score chart
      if 'shift_chart' in self._section_comments:
        y += comment_h
      y += 180 + 15  # Stalls/lugs chart
      if 'stalls_chart' in self._section_comments:
        y += comment_h
      y += 180 + 15  # Launch chart
      if 'launch_chart' in self._section_comments:
        y += comment_h
    # Encouragement (estimate 2-3 lines wrapped)
    y += 100

    return y + 40  # padding

  def _format_time(self, seconds: float) -> str:
    """Format seconds as hours:minutes"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
      return f"{hours}h {minutes}m"
    return f"{minutes}m"

  def _stall_color(self, stalls: int) -> rl.Color:
    if stalls == 0:
      return GREEN
    elif stalls < 5:
      return YELLOW
    return RED

  def _pct_color(self, good: int, total: int) -> rl.Color:
    if total == 0:
      return GRAY
    pct = good / total
    if pct >= 0.8:
      return GREEN
    elif pct >= 0.5:
      return YELLOW
    return RED

  def _score_color(self, score: float) -> rl.Color:
    if score >= 80:
      return GREEN
    elif score >= 50:
      return YELLOW
    return RED

  def _calculate_trend(self, values: list) -> float:
    """Calculate trend as average change over recent values"""
    if len(values) < 2:
      return 0.0
    # Compare first half avg to second half avg
    mid = len(values) // 2
    first_half = sum(values[:mid]) / mid if mid > 0 else 0
    second_half = sum(values[mid:]) / (len(values) - mid) if len(values) - mid > 0 else 0
    return second_half - first_half

  def _trend_text(self, trend: float, lower_better: bool) -> tuple[str, rl.Color]:
    """Get trend text and color"""
    if abs(trend) < 0.5:
      return "Stable", LIGHT_GRAY

    if lower_better:
      if trend < 0:
        return "Improving!", GREEN
      return random.choice(["Getting worse", "Getting worse - Weixing is shaking his head. Kirby turned around."]), RED
    else:
      if trend > 0:
        return "Improving!", GREEN
      return random.choice(["Getting worse", "Getting worse - Weixing is shaking his head. Kirby turned around."]), RED

  def _get_overall_hand(self) -> tuple[str, rl.Color]:
    """Calculate overall poker hand rating based on all stats"""
    total_drives = self._stats.get('total_drives', 0)
    if total_drives == 0:
      return "No Cards Yet", GRAY

    total_stalls = self._stats.get('total_stalls', 0)
    total_shifts = self._stats.get('total_upshifts', 0) + self._stats.get('total_downshifts', 0)
    good_shifts = self._stats.get('upshifts_good', 0) + self._stats.get('downshifts_good', 0)

    stall_rate = total_stalls / total_drives
    shift_pct = (good_shifts / total_shifts * 100) if total_shifts > 0 else 100

    # Calculate overall score
    score = shift_pct - (stall_rate * 10)

    # Recent improvement bonus - aggregate by day
    session_history = self._stats.get('session_history', [])
    recent_days = self._aggregate_by_day(session_history)[-10:]
    recent_scores = []
    for d in recent_days:
      total = d.get('upshifts', 0) + d.get('downshifts', 0)
      good = d.get('upshifts_good', 0) + d.get('downshifts_good', 0)
      recent_scores.append(int(good / total * 100) if total > 0 else 100)
    if len(recent_scores) >= 3:
      if recent_scores[-1] > recent_scores[0]:
        score += 5  # Bonus for improving

    if score >= 98 and stall_rate == 0:
      return random.choice([
        "Royal Flush - Waddle is driving! Kacper threw his glasses!",
        "Royal Flush - Perfection! Pure waddle energy!",
        "Royal Flush - Legendary! KP maxed out!",
        "Royal Flush - CCR material! Waddle certified!",
        "Royal Flush - Elite! Priest-approved waddle!",
        "Royal Flush - Weixing just shed a tear of joy. Kirby is star-spinning.",
      ]), GREEN
    elif score >= 95 and stall_rate == 0:
      return random.choice([
        "Royal Flush - Porch-worthy waddle! KP earned!",
        "Royal Flush - Top-tier driving, almost flawless!",
        "Royal Flush - So close to perfection! Waddle approved!",
        "Royal Flush - KP is proud, keep this up!",
        "Royal Flush - Premium waddle, just shy of legendary!",
        "Royal Flush - Weixing put your photo on his fridge. Kirby gave you a star.",
      ]), GREEN
    elif score >= 90:
      return random.choice([
        "Straight Flush - Elite waddle, CCM vibes!",
        "Straight Flush - Near-perfect, porch is calling!",
        "Straight Flush - Waddle royalty!",
        "Straight Flush - Weixing raised an eyebrow, in a good way. Kirby did a little twirl.",
      ]), GREEN
    elif score >= 85:
      return random.choice([
        "Four of a Kind - Priest-approved waddle!",
        "Four of a Kind - Strong waddle game!",
        "Four of a Kind - CCR energy building!",
        "Four of a Kind - Weixing almost smiled. Kirby perked up.",
      ]), GREEN
    elif score >= 80:
      return random.choice([
        "Full House - Solid waddle, not SS!",
        "Full House - Consistent waddle vibes!",
        "Full House - QG territory!",
        "Full House - Weixing didn't complain. For Weixing, that's a compliment. Kirby is chilling.",
      ]), GREEN
    elif score >= 70:
      return random.choice([
        "Flush - Good waddle, almost KP",
        "Flush - Getting there, waddle incoming!",
        "Flush - Shedding jackets nicely!",
        "Flush - Weixing looked up from his phone briefly. Kirby yawned.",
      ]), YELLOW
    elif score >= 60:
      return random.choice([
        "Straight - Improving, not SS yet",
        "Straight - Progress! Keep pushing!",
        "Straight - Jacket count dropping!",
        "Straight - Weixing checked his watch. Kirby fell asleep.",
      ]), YELLOW
    elif score >= 50:
      return random.choice([
        "Three of a Kind - Getting there, shake off jackets",
        "Three of a Kind - Waddle is within reach!",
        "Three of a Kind - Keep at it, less jackets soon!",
        "Three of a Kind - Weixing pinched the bridge of his nose. Kirby deflated.",
      ]), YELLOW
    elif score >= 40:
      return random.choice([
        "Two Pair - Jackets territory",
        "Two Pair - Room to grow, QG!",
        "Two Pair - Still shedding jackets",
        "Two Pair - Weixing closed his laptop and stared out the window. Kirby popped.",
      ]), YELLOW
    elif score >= 30:
      return random.choice([
        "One Pair - Jacketed, huge oof",
        "One Pair - Jacket city, but improving?",
        "One Pair - SS vibes, keep practicing!",
        "One Pair - Weixing blocked your number. Kirby ate your clutch.",
      ]), RED
    else:
      return random.choice([
        "High Card - SS! Full jackets!",
        "High Card - Jacketed hard! QG needed!",
        "High Card - Waddle disapproves. Keep going!",
        "High Card - Weixing pretends he doesn't know you. Kirby swallowed the car whole.",
      ]), RED

  def _pick_section_comments(self) -> dict[str, tuple[str, 'rl.Color']]:
    """Pick a contextual comment for each section based on the data"""
    comments: dict[str, tuple[str, rl.Color]] = {}

    # Shift quality
    total_up = self._stats.get('total_upshifts', 0)
    total_down = self._stats.get('total_downshifts', 0)
    up_good = self._stats.get('upshifts_good', 0)
    down_good = self._stats.get('downshifts_good', 0)
    total_shifts = total_up + total_down
    if total_shifts > 0:
      shift_pct = (up_good + down_good) / total_shifts * 100
      if shift_pct >= 90:
        comments['shifts'] = random.choice([
          "Butter smooth! Weixing almost smiled.",
          "Shifts are dialed. Kirby did a little twirl.",
          "Priest-approved shifting right here.",
        ]), GREEN
      elif shift_pct >= 70:
        comments['shifts'] = random.choice([
          "Solid shifts, room to polish.",
          "Getting cleaner. Kirby is watching.",
          "Not bad! Weixing looked up briefly.",
        ]), YELLOW
      else:
        comments['shifts'] = random.choice([
          "Those shifts need some love.",
          "Weixing felt that from across the room.",
          "Kirby is concerned about your synchros.",
        ]), RED

    # Launch quality
    total_launches = self._stats.get('total_launches', 0)
    good_launches = self._stats.get('launches_good', 0)
    stalled_launches = self._stats.get('launches_stalled', 0)
    if total_launches > 0:
      launch_pct = good_launches / total_launches * 100
      if launch_pct >= 90:
        comments['launches'] = random.choice([
          "Smooth off the line! Clutch control on point.",
          "Launch game is strong. Kirby approves.",
          "Clean launches. The bite point is your friend.",
        ]), GREEN
      elif launch_pct >= 70:
        comments['launches'] = random.choice([
          "Launches are OK. Find that bite point more consistently.",
          "Getting there! A little more clutch feel needed.",
          "Decent launches, some room to grow.",
        ]), YELLOW
      else:
        comments['launches'] = random.choice([
          "Launches need work. Easy on the clutch!",
          "Kirby is bracing for impact every launch.",
          "More revs, slower clutch release. You'll get it.",
        ]), RED
      if stalled_launches > 2:
        comments['launches'] = random.choice([
          f"{stalled_launches} stalled launches - find that bite point!",
          f"{stalled_launches} stalled launches - Kirby is hiding the keys.",
          f"{stalled_launches} stalled launches - more gas before you release!",
        ]), RED

    # Gear chart
    gear_counts = self._stats.get('gear_shift_counts', {})
    gear_jerks = self._stats.get('gear_shift_jerk_totals', {})
    if gear_counts and any(gear_counts.values()):
      worst_gear, worst_score = None, 101
      best_gear, best_score = None, -1
      for gear in range(1, 7):
        count = gear_counts.get(gear, gear_counts.get(str(gear), 0))
        jerk = gear_jerks.get(gear, gear_jerks.get(str(gear), 0.0))
        if count > 0:
          smoothness = max(0, min(100, 100 - (jerk / count * 20)))
          if smoothness < worst_score:
            worst_gear, worst_score = gear, smoothness
          if smoothness > best_score:
            best_gear, best_score = gear, smoothness
      if worst_gear and best_gear and worst_gear != best_gear:
        comments['gears'] = random.choice([
          f"Gear {best_gear} is your smoothest. Gear {worst_gear} needs practice.",
          f"Cleanest into gear {best_gear}. Gear {worst_gear} is your weak spot.",
          f"Gear {worst_gear} is where the jackets live. Gear {best_gear} is waddle territory.",
        ]), YELLOW

    # Shift score chart
    session_history = self._stats.get('session_history', [])
    days = self._aggregate_by_day(session_history)
    if len(days) >= 3:
      recent = days[-3:]
      scores = []
      for d in recent:
        t = d.get('upshifts', 0) + d.get('downshifts', 0)
        g = d.get('upshifts_good', 0) + d.get('downshifts_good', 0)
        scores.append(int(g / t * 100) if t > 0 else 100)
      avg = sum(scores) / len(scores)
      if avg >= 85:
        comments['shift_chart'] = random.choice([
          "Recent shifts looking clean!",
          "Shift scores are up. Waddle energy.",
          "Consistency is showing. Kirby is pleased.",
        ]), GREEN
      elif scores[-1] > scores[0]:
        comments['shift_chart'] = random.choice([
          "Trending up! Keep this momentum.",
          "Scores climbing. Weixing might notice soon.",
          "Getting better day by day.",
        ]), YELLOW
      else:
        comments['shift_chart'] = random.choice([
          "Shift scores dipping. Focus up!",
          "Weixing is watching these numbers drop.",
          "Time to tighten up those shifts.",
        ]), RED

    # Stalls chart
    if len(days) >= 3:
      recent_stalls = [d.get('stalls', 0) for d in days[-3:]]
      if all(s == 0 for s in recent_stalls):
        comments['stalls_chart'] = random.choice([
          "Stall-free streak! Don't break it.",
          "Zero stalls lately. Weixing is watching... approvingly.",
          "Clean streak. Kirby is relaxed.",
        ]), GREEN
      elif recent_stalls[-1] < recent_stalls[0]:
        comments['stalls_chart'] = random.choice([
          "Stalls trending down. Shedding jackets!",
          "Fewer stalls recently. Progress!",
          "The jacket count is dropping. Keep going.",
        ]), YELLOW
      elif recent_stalls[-1] > recent_stalls[0]:
        comments['stalls_chart'] = random.choice([
          "Stalls creeping up. Deep breath, find the bite point.",
          "More stalls lately. Weixing noticed.",
          "Jacket count rising. Kirby is concerned.",
        ]), RED

    # Launch chart
    if len(days) >= 3:
      recent_launch_pcts = []
      for d in days[-3:]:
        l_total = d.get('launches', 0)
        l_good = d.get('launches_good', 0)
        if l_total > 0:
          recent_launch_pcts.append(l_good / l_total * 100)
      if len(recent_launch_pcts) >= 2:
        if all(p >= 80 for p in recent_launch_pcts):
          comments['launch_chart'] = random.choice([
            "Launches looking consistent!",
            "Smooth off the line, day after day.",
            "Kirby trusts your launches now.",
          ]), GREEN
        elif recent_launch_pcts[-1] > recent_launch_pcts[0]:
          comments['launch_chart'] = random.choice([
            "Launch success trending up!",
            "Getting smoother off the line.",
            "Clutch control improving. Waddle incoming.",
          ]), YELLOW
        elif recent_launch_pcts[-1] < recent_launch_pcts[0]:
          comments['launch_chart'] = random.choice([
            "Launches getting rougher. Easy on the clutch!",
            "Launch success dipping. Find that bite point.",
            "Kirby is bracing again.",
          ]), RED

    return comments

  def _get_encouragement(self) -> str:
    """Get encouragement based on overall progress"""
    total_drives = self._stats.get('total_drives', 0)
    total_stalls = self._stats.get('total_stalls', 0)
    # Aggregate by day for consistent messaging
    session_history = self._stats.get('session_history', [])
    recent_days = self._aggregate_by_day(session_history)[-10:]
    num_days = len(recent_days)
    recent_stalls = [d.get('stalls', 0) for d in recent_days]
    recent_scores = []
    for d in recent_days:
      total = d.get('upshifts', 0) + d.get('downshifts', 0)
      good = d.get('upshifts_good', 0) + d.get('downshifts_good', 0)
      recent_scores.append(int(good / total * 100) if total > 0 else 100)

    if total_drives == 0:
      return random.choice([
        "Start driving to see your stats! Time to earn your first waddle KP.",
        "No drives yet! Get out there and start your waddle journey!",
        "Empty stats - the porch awaits your first drive!",
      ])

    if total_drives <= 2:
      if total_stalls == 0:
        return random.choice([
          "No stalls yet! Waddle energy from day 1. Keep it up!",
          "Zero stalls early on! Natural waddle talent?!",
          "Clean start! Priest-approved from the jump!",
        ])
      return random.choice([
        f"{total_stalls} stall{'s' if total_stalls > 1 else ''} so far - every waddle driver starts somewhere. QG!",
        f"{total_stalls} stall{'s' if total_stalls > 1 else ''} early on - totally normal, waddle is coming!",
        f"{total_stalls} stall{'s' if total_stalls > 1 else ''} - shedding jackets already. Keep going!",
      ])

    stall_rate = total_stalls / total_drives

    # Check for improvement
    improving = False
    if len(recent_scores) >= 3:
      if recent_scores[-1] > recent_scores[0] + 5:
        improving = True

    if len(recent_stalls) >= 3:
      recent_avg = sum(recent_stalls[-3:]) / 3
      if recent_avg == 0:
        # Check for crazy good performance
        if len(recent_scores) >= 3 and all(s >= 95 for s in recent_scores[-3:]):
          return random.choice([
            f"Last {num_days}d: 95%+ shifts, NO stalls?! Waddle is driving! Kacper threw his glasses!",
            f"Last {num_days}d: near-perfect shifts, zero stalls! Legendary waddle!",
            f"Last {num_days}d: flawless! Porch-worthy, priest-approved, KP maxed!",
          ])
        if improving:
          return random.choice([
            f"Last {num_days}d: no stalls AND improving? Waddle energy! QG to KP!",
            f"Last {num_days}d: stall-free and trending up! CCR energy!",
            f"Last {num_days}d: zero stalls, scores climbing! Porch incoming!",
          ])
        return random.choice([
          f"Last {num_days}d: no stalls - waddle game strong! Not SS, priest-approved!",
          f"Last {num_days}d: stall-free! Solid waddle vibes!",
          f"Last {num_days}d: clean driving, no jackets! Keep it up!",
        ])
      elif recent_avg < stall_rate:
        return random.choice([
          f"Last {num_days}d: better than avg - shedding jackets, channeling waddle!",
          f"Last {num_days}d: fewer stalls than usual! De-jacketing in progress!",
          f"Last {num_days}d: improving! Waddle is within reach!",
        ])

    if total_stalls == 0:
      return random.choice([
        "Zero stalls overall! Waddle game is immaculate!",
        "Not a single stall! Priest-approved driving!",
        "Stall-free career! Pure waddle energy!",
      ])

    drives_per_stall = round(total_drives / total_stalls)

    if stall_rate < 1:
      if improving:
        return random.choice([
          f"1 stall every {drives_per_stall} drives AND improving (last {num_days}d)! Porch-worthy waddle progress!",
          f"1 stall every {drives_per_stall} drives AND getting better (last {num_days}d)! CCR material!",
          f"1 stall every {drives_per_stall} drives AND trending up (last {num_days}d)! KP earned!",
        ])
      return random.choice([
        f"1 stall every {drives_per_stall} drives - solid waddle vibes, not SS!",
        f"1 stall every {drives_per_stall} drives - consistent waddle energy!",
        f"1 stall every {drives_per_stall} drives - jacket count staying low!",
      ])
    else:
      stalls_per_drive = round(total_stalls / total_drives)
      return random.choice([
        f"About {stalls_per_drive} stall{'s' if stalls_per_drive > 1 else ''} every drive - keep at it! QG to KP!",
        f"About {stalls_per_drive} stall{'s' if stalls_per_drive > 1 else ''} every drive - still jacketed, but every drive is practice!",
        f"About {stalls_per_drive} stall{'s' if stalls_per_drive > 1 else ''} every drive - jackets happen! The porch is waiting!",
        f"About {stalls_per_drive} stall{'s' if stalls_per_drive > 1 else ''} every drive - Weixing left the room. Kirby followed him out.",
      ])
