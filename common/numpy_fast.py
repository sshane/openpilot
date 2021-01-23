def int_rnd(x):
  return int(round(x))

def clip(x, lo, hi):
  return max(lo, min(hi, x))

def interp(x, xp, fp):
  N = len(xp)

  def get_interp(xv):
    hi = 0
    while hi < N and xv > xp[hi]:
      hi += 1
    low = hi - 1
    return fp[-1] if hi == N and xv > xp[low] else (
      fp[0] if hi == 0 else
      (xv - xp[low]) * (fp[hi] - fp[low]) / (xp[hi] - xp[low]) + fp[low])

  return [get_interp(v) for v in x] if hasattr(x, '__iter__') else get_interp(x)

def interp_2d(x, y, bp, v):
  N_x = len(bp[0])
  N_y = len(bp[1])

  def get_interp(xv, yv):
    hi_x = 0
    hi_y = 0
    while hi_x < N_x and xv > bp[0][hi_x]:  # get hi for x bp
      hi_x += 1
    while hi_y < N_y and yv > bp[1][hi_y]:  # then get hi for y bp
      hi_y += 1
    low_x = hi_x - 1
    low_y = hi_y - 1

    if ((hi_x == N_x or hi_y == N_y) and xv > bp[0][low_x] and yv > bp[1][low_y]) or hi_x == 0 or hi_y == 0:
      # This branch is taken if either x or y is at top or bottom of their respective BPs (just need to interpolate one input)
      if hi_x == N_x and hi_y == N_y:  # both at top
        return v[-1][-1]
      elif hi_x == 0 and hi_y == 0:  # both at bottom
        return v[0][0]

      bp_idx = 1 if hi_x in [N_x, 0] else 0  # Use y BPs if x is at top or bottom, else use x BPs (y is at top or bottom)
      if hi_x in [N_x, 0]:  # if x is at top or bottom, use last or first y values
        new_v = v[-1 if hi_x == N_x else 0]
      else:  # if y is at top or bottom, get last or first x value of each y values list
        new_v = [_v[-1 if hi_y == N_y else 0] for _v in v]

      return interp(yv if bp_idx == 1 else xv, bp[bp_idx], new_v)
    else:  # This branch is taken if both x and y are not top or bottom (interpolate both)
      # Gets us a new values list by interpolating x input first
      new_v = [interp(xv, [bp[0][low_x], bp[0][hi_x]], [low_xv, hi_xv]) for
               low_xv, hi_xv in zip(v[low_x], v[hi_x])]
      return interp(yv, [bp[1][low_y], bp[1][hi_y]], new_v)  # Finally we can interpolate y input

  return [get_interp(v1, v2) for v1, v2 in zip(x, y)] if hasattr(x, '__iter__') else get_interp(x, y)

def mean(x):
  return sum(x) / len(x)
