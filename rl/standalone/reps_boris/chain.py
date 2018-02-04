"""Chain environment with stochastic transitions."""

import numpy as np
import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text.discrete import DiscreteEnv


class ChainEnv(DiscreteEnv):
  metadata = {'render.modes': ['human', 'ansi']}

  def __init__(self, nS=5, ps=0.9, small=2, large=10):
    pf = 1 - ps
    nA = 2
    isd = np.zeros(nS)
    isd[0] = 1  # initial state is 0

    P = {s: {a: [] for a in range(nA)} for s in range(nS)}
    # Add backwards action
    for s in range(nS-1):
      P[s][0].append((ps, 0, small, False))
      P[s][0].append((pf, s+1, 0, False))
    P[nS-1][0].append((ps, 0, small, False))
    P[nS-1][0].append((pf, nS-1, large, False))

    # Add forward action
    for s in range(nS-1):
      P[s][1].append((ps, s+1, 0, False))
      P[s][1].append((pf, 0, small, False))
    P[nS-1][1].append((ps, nS-1, large, False))
    P[nS-1][1].append((pf, 0, small, False))

    super().__init__(nS, nA, P, isd)
    self._seed()

  def _render(self, mode='human', close=False):
    if close:
      return
    outfile = StringIO() if mode == 'ansi' else sys.stdout

    states = [str(i) for i in range(self.nS)]
    current = self.s
    states[current] = utils.colorize(states[current], "red", highlight=True)

    outfile.write('\n' + ' '.join(states) + '\n')

    if mode != 'human':
      return outfile
