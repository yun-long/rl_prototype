"""Modified FrozenLake environment that does not have absorbing states."""

import numpy as np
import sys
from six import StringIO

from gym import utils
from gym.envs.toy_text.discrete import DiscreteEnv

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
  "4x4": [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
  ],
  "8x8": [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG"
  ],
}


class FrozenLakeEnv(DiscreteEnv):
  metadata = {'render.modes': ['human', 'ansi']}

  def __init__(self, desc=None, map_name="4x4", is_slippery=True):
    if desc is None and map_name is None:
      raise ValueError('Must provide either desc or map_name')
    elif desc is None:
      desc = MAPS[map_name]
    self.desc = desc = np.asarray(desc, dtype='c')
    self.nrow, self.ncol = nrow, ncol = desc.shape

    nA = 4
    nS = nrow * ncol

    # NOTE: Start at s = 0
    isd = np.array(desc == b'S').astype('float64').ravel()
    # isd /= isd.sum()
    # NOTE: Start in safe states
    # isd = np.logical_or(desc == b'S', desc == b'F').astype('float64').ravel()
    # isd /= np.sum(isd)
    # NOTE: Start anywhere (including finish and holes)
    # isd = np.ones(nS) / nS

    P = {s: {a: [] for a in range(nA)} for s in range(nS)}

    def to_s(row, col):
      return row * ncol + col

    def inc(row, col, a):
      if a == 0:  # left
        col = max(col - 1, 0)
      elif a == 1:  # down
        row = min(row + 1, nrow - 1)
      elif a == 2:  # right
        col = min(col + 1, ncol - 1)
      elif a == 3:  # up
        row = max(row - 1, 0)
      return (row, col)

    # I added this big reward at the goal state
    big_rew = 5.0
    norm_rew = -0.5
    hole_rew = -5.0
    for row in range(nrow):
      for col in range(ncol):
        s = to_s(row, col)
        for a in range(4):
          li = P[s][a]
          letter = desc[row, col]
          # if letter in b'GH':
          #     li.append((1.0, s, hole_rew, True))
          # else:
          if is_slippery:
            for b in [(a - 1) % 4, a, (a + 1) % 4]:
              newrow, newcol = inc(row, col, b)
              newstate = to_s(newrow, newcol)
              newletter = desc[newrow, newcol]
              done = bytes(newletter) in b'GH'
              if newletter == b'G':
                rew = big_rew
              elif newletter == b'H':
                rew = hole_rew
              else:
                rew = norm_rew
              # rew = float(newletter == b'G')
              li.append((0.8 if b == a else 0.1, newstate, rew, done))
          else:
            newrow, newcol = inc(row, col, a)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b'GH'
            rew = float(newletter == b'G')
            li.append((1.0, newstate, rew, done))

    super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

  def _render(self, mode='human', close=False):
    if close:
      return
    outfile = StringIO() if mode == 'ansi' else sys.stdout

    row, col = self.s // self.ncol, self.s % self.ncol
    desc = self.desc.tolist()
    desc = [[c.decode('utf-8') for c in line] for line in desc]
    desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
    if self.lastaction is not None:
      outfile.write(
        "  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
    else:
      outfile.write("\n")
    outfile.write("\n".join(''.join(line) for line in desc) + "\n")

    return outfile
