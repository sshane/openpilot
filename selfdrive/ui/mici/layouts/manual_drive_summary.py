"""
Manual Drive Summary Dialog

Shows end-of-drive statistics for manual transmission driving with
encouraging or critical feedback based on performance.
Poker hand themed with waddle/jacket references.
"""

import random
import pyray as rl
from typing import Optional

from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.scroll_panel2 import GuiScrollPanel2
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets.nav_widget import NavWidget


# Colors
GREEN = rl.Color(46, 204, 113, 255)
YELLOW = rl.Color(241, 196, 15, 255)
RED = rl.Color(231, 76, 60, 255)
GRAY = rl.Color(150, 150, 150, 255)
LIGHT_GRAY = rl.Color(200, 200, 200, 255)
WHITE = rl.Color(255, 255, 255, 255)
BG_CARD = rl.Color(45, 45, 45, 255)

# Poker hand names
HAND_NAMES = {
  "A": "Aces",
  "K": "Kings",
  "Q": "Queens",
  "J": "Jacks",
  "10": "10s"
}

HAND_SUBTITLES = {
  "A": "Porch-worthy! KP!",
  "K": "CCM vibes! QG!",
  "Q": "Priest-approved",
  "J": "Not SS... yet",
  "10": "Jacketed! Huge oof"
}


class ManualDriveSummaryDialog(NavWidget):
  """Modal dialog showing end-of-drive manual transmission stats"""

  def __init__(self):
    super().__init__()
    self._scroll_panel = GuiScrollPanel2(horizontal=False)
    self._session_data: Optional[dict] = None
    self._overall_grade: str = "good"  # good, ok, poor
    self._card_rank: str = "10"  # Poker card rank: 10, J, Q, K, A
    self._shift_score: float = 0.0
    self._avg_shift_score: float = 0.0

    # Load all data from one param read
    self._load_data()

    # Pick random texts once for this instance
    self._header_text, self._header_color = self._pick_header()
    self._encouragement_text = self._pick_encouragement()

    self.set_back_callback(lambda: gui_app.pop_widget())

  def _load_data(self):
    """Load session and historical data from ManualDriveStats (single read)"""
    data = Params().get("ManualDriveStats")
    if not data:
      return

    stats = data
    history = stats.get('session_history', [])

    # Last session
    if history:
      self._session_data = history[-1]
      self._calculate_grade()

    # Average shift score from recent history
    scores = []
    for s in history[-10:]:
      ups = s.get('upshifts', 0)
      ups_good = s.get('upshifts_good', 0)
      downs = s.get('downshifts', 0)
      downs_good = s.get('downshifts_good', 0)
      total = ups + downs
      if total > 0:
        scores.append((ups_good + downs_good) / total * 100)
    if scores:
      self._avg_shift_score = sum(scores) / len(scores)

  def _calculate_grade(self):
    """Calculate overall grade based on session performance"""
    if not self._session_data:
      self._overall_grade = "ok"
      self._card_rank = "10"
      self._shift_score = 0
      return

    # Calculate grade based on stalls, shifts, and launches
    stalls = self._session_data.get('stalls', 0)
    lugs = self._session_data.get('lugs', 0)

    # Shift quality
    upshift_total = self._session_data.get('upshifts', 0)
    upshift_good = self._session_data.get('upshifts_good', 0)
    downshift_total = self._session_data.get('downshifts', 0)
    downshift_good = self._session_data.get('downshifts_good', 0)

    # Launch quality
    launch_total = self._session_data.get('launches', 0)
    launch_good = self._session_data.get('launches_good', 0)
    launch_stalled = self._session_data.get('launches_stalled', 0)

    # Calculate scores
    total_shifts = upshift_total + downshift_total
    self._shift_score = ((upshift_good + downshift_good) / total_shifts * 100) if total_shifts > 0 else 100
    launch_score = (launch_good / launch_total * 100) if launch_total > 0 else 100

    # Penalties
    stall_penalty = stalls * 20
    lug_penalty = lugs * 5
    launch_stall_penalty = launch_stalled * 15

    overall_score = max(0, min(100, (self._shift_score + launch_score) / 2 - stall_penalty - lug_penalty - launch_stall_penalty))

    # Poker card ranking: 10, J, Q, K, A
    if overall_score >= 90 and stalls == 0:
      self._card_rank = "A"
      self._overall_grade = "good"
    elif overall_score >= 75 and stalls == 0:
      self._card_rank = "K"
      self._overall_grade = "good"
    elif overall_score >= 60 and stalls <= 1:
      self._card_rank = "Q"
      self._overall_grade = "ok"
    elif overall_score >= 40:
      self._card_rank = "J"
      self._overall_grade = "ok"
    else:
      self._card_rank = "10"
      self._overall_grade = "poor"

  def _pick_header(self) -> tuple[str, rl.Color]:
    if self._overall_grade == "good":
      return random.choice([
        "Waddle Driver!",
        "KP Earned!",
        "Porch-worthy!",
        "CCR Energy!",
        "Priest-approved!",
        "Pure Waddle!",
      ]), GREEN
    elif self._overall_grade == "ok":
      return random.choice([
        "Decent Drive",
        "Getting There!",
        "Not SS... Yet",
        "Shedding Jackets",
        "Almost Waddle",
      ]), YELLOW
    else:
      return random.choice([
        "Jackets...",
        "Huge Oof",
        "SS Vibes",
        "Full Jackets!",
        "Jacketed!",
      ]), RED

  def _pick_encouragement(self) -> str:
    if not self._session_data:
      return "No data available for this drive."

    stalls = self._session_data.get('stalls', 0)
    lugs = self._session_data.get('lugs', 0)
    launch_stalled = self._session_data.get('launches_stalled', 0)

    upshift_good = self._session_data.get('upshifts_good', 0)
    upshift_total = self._session_data.get('upshifts', 0)
    downshift_good = self._session_data.get('downshifts_good', 0)
    downshift_total = self._session_data.get('downshifts', 0)
    launch_good = self._session_data.get('launches_good', 0)
    launch_total = self._session_data.get('launches', 0)

    messages = []

    if self._overall_grade == "good":
      # Check for perfect drive - Kacper glasses moment
      total_shifts = upshift_total + downshift_total
      total_good = upshift_good + downshift_good
      perfect_shifts = total_shifts > 0 and total_good == total_shifts
      perfect_launches = launch_total > 0 and launch_good == launch_total

      if self._card_rank == "A" and stalls == 0 and lugs == 0 and perfect_shifts and perfect_launches:
        messages.append(random.choice([
          "PERFECT! Waddle is driving! Kacper threw his glasses!",
          "FLAWLESS! Even Kacper couldn't believe it!",
          "LEGENDARY! Full waddle, zero jackets, KP maxed!",
          "PERFECT! Weixing just shed a tear of joy. Kirby is star-spinning.",
        ]))
      elif self._card_rank == "A":
        messages.append(random.choice([
          "Aces! Porch-worthy waddle, KP earned!",
          "Aces! CCR material right here!",
          "Aces! Waddle would be proud!",
          "Aces! Weixing raised an eyebrow, in a good way. Kirby did a little twirl.",
        ]))
      elif self._card_rank == "K":
        messages.append(random.choice([
          "Kings! Waddle energy, CCM vibes!",
          "Kings! Solid drive, almost porch-worthy!",
          "Kings! Not SS, definitely QG!",
          "Kings! Weixing didn't complain. For Weixing, that's a compliment. Kirby is chilling.",
        ]))
      if stalls == 0 and launch_stalled == 0:
        messages.append(random.choice(["No stalls!", "Zero stalls, clean!", "Stall-free!"]))
      if perfect_shifts:
        messages.append(random.choice([
          "Perfect shifts - priest-approved!",
          "Every shift was butter!",
          "Flawless shifting, pure waddle!",
        ]))
      elif upshift_total > 0 and upshift_good == upshift_total:
        messages.append(random.choice(["Perfect upshifts!", "Upshifts on point!", "Clean upshifts!"]))
      if downshift_total > 0 and downshift_good >= downshift_total * 0.8:
        messages.append(random.choice(["Great rev matching!", "Rev matching on point!", "Heel-toe vibes!"]))
      if perfect_launches:
        messages.append(random.choice(["Flawless launches!", "Every launch was smooth!", "Launch game maxed!"]))
      elif launch_total > 0 and launch_good >= launch_total * 0.8:
        messages.append(random.choice(["Smooth launches!", "Launches looking clean!", "Good clutch control!"]))
      if not messages:
        messages.append(random.choice(["Keep channeling waddle!", "Waddle energy maintained!", "Stay on this path!"]))

    elif self._overall_grade == "ok":
      if self._card_rank == "Q":
        messages.append(random.choice([
          "Queens - almost there!",
          "Queens - one step from waddle!",
          "Queens - so close to KP!",
          "Queens - Weixing checked his watch. Kirby yawned.",
        ]))
      else:
        messages.append(random.choice([
          "Jacks - improving, not SS!",
          "Jacks - shedding jackets slowly!",
          "Jacks - waddle is within reach!",
          "Jacks - Weixing pinched the bridge of his nose. Kirby deflated.",
        ]))
      if stalls > 0:
        messages.append(f"Only {stalls} stall{'s' if stalls > 1 else ''} - {random.choice(['shedding jackets!', 'getting better!', 'less than before?'])}")
      if lugs > 0:
        messages.append(f"{random.choice(['Watch RPMs', 'Easy on the low RPMs'])} - {lugs} lug{'s' if lugs > 1 else ''}.")
      if upshift_total > 0 and upshift_good < upshift_total:
        messages.append(random.choice(["Smoother upshifts needed.", "Upshifts could be cleaner.", "Work on those upshifts!"]))

    else:  # poor - jackets
      messages.append(random.choice([
        "Jacketed! Huge oof. SS vibes!",
        "Full jackets! CCR this is not.",
        "Oof. Jacket city. QG needed!",
        "Jacketed hard. Waddle disapproves.",
        "Weixing pretends he doesn't know you. Kirby swallowed the car whole.",
      ]))
      if stalls > 2:
        messages.append(f"{stalls} stalls - {random.choice(['more gas, slower clutch!', 'find that bite point!', 'easy on the clutch!'])}")
      if launch_stalled > 0:
        messages.append(f"{launch_stalled} stalled launch{'es' if launch_stalled > 1 else ''} - {random.choice(['find bite point!', 'more revs before release!', 'hold clutch longer!'])}")
      if lugs > 3:
        messages.append(f"Lugging {lugs}x - {random.choice(['downshift sooner!', 'drop a gear!', 'RPMs too low!'])}")
      if not messages[1:]:
        messages.append(random.choice(["Even the best got jacketed at first. QG!", "Keep practicing, waddle awaits!", "Every driver starts here. KP is coming!"]))

    return " ".join(messages)

  def _measure_content_height(self) -> int:
    """Calculate total content height for scrolling"""
    font_roman = gui_app.font(FontWeight.ROMAN)
    h = 0
    h += 50   # Header
    h += 38   # Card rank
    h += 35   # Duration
    h += 75   # Shift score bar
    h += 195  # Stats card
    # Encouragement text (estimate)
    wrapped = wrap_text(font_roman, self._encouragement_text, 22, 500)
    h += len(wrapped) * 28 + 20
    return h

  def _render(self, rect: rl.Rectangle):
    # Content area with scrolling
    content_rect = rl.Rectangle(rect.x + 10, rect.y + 10, rect.width - 20, rect.height - 20)
    content_height = self._measure_content_height()
    scroll_offset = round(self._scroll_panel.update(content_rect, content_height))

    x = int(content_rect.x) + 20  # Padding on left
    y = int(content_rect.y) + scroll_offset
    w = int(content_rect.width) - 40  # Padding on both sides

    font_bold = gui_app.font(FontWeight.BOLD)
    font_medium = gui_app.font(FontWeight.MEDIUM)
    font_roman = gui_app.font(FontWeight.ROMAN)

    # Enable scissor mode to clip content
    rl.begin_scissor_mode(int(content_rect.x), int(content_rect.y), int(content_rect.width), int(content_rect.height))

    # Top section card background (header, hand, duration, score bar)
    top_card_h = 200
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, top_card_h), 0.02, 10, BG_CARD)

    # Header
    rl.draw_text_ex(font_bold, self._header_text, rl.Vector2(x + 15, y + 12), 44, 0, self._header_color)
    y += 58

    # Card rank display - poker hand style with subtitle
    card_color = GREEN if self._card_rank in ("A", "K") else (YELLOW if self._card_rank in ("Q", "J") else RED)
    card_text = f"Your hand: {HAND_NAMES[self._card_rank]}"
    rl.draw_text_ex(font_medium, card_text, rl.Vector2(x + 15, y), 28, 0, card_color)
    # Subtitle
    subtitle = HAND_SUBTITLES[self._card_rank]
    subtitle_width = rl.measure_text_ex(font_roman, subtitle, 20, 0).x
    rl.draw_text_ex(font_roman, subtitle, rl.Vector2(x + w - subtitle_width - 35, y + 4), 20, 0, card_color)
    y += 38

    # Duration
    duration = self._session_data.get('duration', 0) if self._session_data else 0
    duration_min = int(duration // 60)
    duration_sec = int(duration % 60)
    rl.draw_text_ex(font_roman, f"Drive: {duration_min}:{duration_sec:02d}",
                    rl.Vector2(x + 15, y), 22, 0, GRAY)
    y += 35

    # Shift Score Progress Bar with comparison
    y = self._draw_score_bar(x + 15, y, w - 30, "Shift Score", self._shift_score, self._avg_shift_score)
    y += 15

    # Stats in a card
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, 190), 0.02, 10, BG_CARD)
    card_x = x + 15
    card_y = y + 12

    # Jackets section (stalls + lugs)
    stalls = self._session_data.get('stalls', 0) if self._session_data else 0
    lugs = self._session_data.get('lugs', 0) if self._session_data else 0
    jackets_text = "Jackets:" if (stalls > 0 or lugs > 0) else "No Jackets!"
    jackets_color = RED if stalls > 0 else (YELLOW if lugs > 0 else GREEN)
    rl.draw_text_ex(font_medium, jackets_text, rl.Vector2(card_x, card_y), 24, 0, jackets_color)
    card_y += 30

    card_y = self._draw_mini_stat(card_x, card_y, w - 30, "Stalls", stalls, 0, True)
    card_y = self._draw_mini_stat(card_x, card_y, w - 30, "Lugs", lugs, 0, True)

    # Waddle section (launches + shifts)
    card_y += 8
    rl.draw_text_ex(font_medium, "Waddle Stats:", rl.Vector2(card_x, card_y), 24, 0, WHITE)
    card_y += 30

    upshift_total = self._session_data.get('upshifts', 0) if self._session_data else 0
    upshift_good = self._session_data.get('upshifts_good', 0) if self._session_data else 0
    downshift_total = self._session_data.get('downshifts', 0) if self._session_data else 0
    downshift_good = self._session_data.get('downshifts_good', 0) if self._session_data else 0
    launch_total = self._session_data.get('launches', 0) if self._session_data else 0
    launch_good = self._session_data.get('launches_good', 0) if self._session_data else 0

    if launch_total > 0:
      card_y = self._draw_mini_stat(card_x, card_y, w - 30, "Launches", f"{launch_good}/{launch_total}", launch_total, False, launch_good)

    total_shifts = upshift_total + downshift_total
    total_good = upshift_good + downshift_good
    if total_shifts > 0:
      card_y = self._draw_mini_stat(card_x, card_y, w - 30, "Shifts", f"{total_good}/{total_shifts}", total_shifts, False, total_good)

    y += 200

    # Encouragement/criticism text
    wrapped = wrap_text(font_roman, self._encouragement_text, 22, w)
    for line in wrapped:
      rl.draw_text_ex(font_roman, line, rl.Vector2(x, y), 22, 0, LIGHT_GRAY)
      y += 28

    rl.end_scissor_mode()

    return -1  # Keep showing dialog

  def _draw_score_bar(self, x: int, y: int, w: int, label: str, score: float, avg_score: float) -> int:
    """Draw a progress bar showing score vs average"""
    font_medium = gui_app.font(FontWeight.MEDIUM)
    font_roman = gui_app.font(FontWeight.ROMAN)

    # Label and score
    rl.draw_text_ex(font_medium, label, rl.Vector2(x, y), 22, 0, WHITE)
    score_text = f"{int(score)}%"
    score_color = GREEN if score >= 80 else (YELLOW if score >= 50 else RED)
    score_width = rl.measure_text_ex(font_medium, score_text, 22, 0).x
    rl.draw_text_ex(font_medium, score_text, rl.Vector2(x + w - score_width, y), 22, 0, score_color)
    y += 28

    # Progress bar background
    bar_h = 16
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, bar_h), 0.3, 10, rl.Color(60, 60, 60, 255))

    # Progress bar fill
    fill_w = int((score / 100) * w)
    if fill_w > 0:
      rl.draw_rectangle_rounded(rl.Rectangle(x, y, fill_w, bar_h), 0.3, 10, score_color)

    # Average marker line
    if avg_score > 0:
      avg_x = x + int((avg_score / 100) * w)
      rl.draw_rectangle(avg_x - 1, y - 2, 3, bar_h + 4, WHITE)

    y += bar_h + 6

    # Comparison text
    if avg_score > 0:
      diff = score - avg_score
      if diff > 5:
        comp_text = f"Above avg (+{int(diff)})"
        comp_color = GREEN
      elif diff < -5:
        comp_text = f"Below avg ({int(diff)})"
        comp_color = RED
      else:
        comp_text = "Near average"
        comp_color = GRAY
      rl.draw_text_ex(font_roman, comp_text, rl.Vector2(x, y), 16, 0, comp_color)
      rl.draw_text_ex(font_roman, "| = your avg", rl.Vector2(x + w - 80, y), 16, 0, GRAY)
    y += 22

    return y

  def _draw_mini_stat(self, x: int, y: int, w: int, label: str, value, target, lower_better: bool, current=None) -> int:
    """Draw a compact stat row"""
    font_roman = gui_app.font(FontWeight.ROMAN)
    font_size = 20

    # Determine color
    if lower_better:
      if isinstance(value, int):
        color = GREEN if value == 0 else (YELLOW if value <= 2 else RED)
      else:
        color = LIGHT_GRAY
    else:
      if current is not None and target > 0:
        ratio = current / target
        color = GREEN if ratio >= 0.8 else (YELLOW if ratio >= 0.5 else RED)
      else:
        color = LIGHT_GRAY

    rl.draw_text_ex(font_roman, label, rl.Vector2(x, y), font_size, 0, LIGHT_GRAY)
    value_str = str(value)
    value_width = rl.measure_text_ex(font_roman, value_str, font_size, 0).x
    rl.draw_text_ex(font_roman, value_str, rl.Vector2(x + w - value_width, y), font_size, 0, color)

    return y + 26
