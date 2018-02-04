"""Scalable CliffWalking environment."""

import numpy as np
import sys
from gym.envs.toy_text.discrete import DiscreteEnv

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class CliffWalkingEnv(DiscreteEnv):
  metadata = {'render.modes': ['human', 'ansi']}

  def __init__(self, shape=(2, 3)):
    self.shape = shape
    self.start_state_index =\
      np.ravel_multi_index((self.shape[0]-1, 0), self.shape)
    self.terminal_state = (self.shape[0] - 1, self.shape[1] - 1)

    nS = int(np.prod(self.shape))
    nA = 4

    # Cliff Location
    self._cliff = np.zeros(self.shape, dtype=np.bool)
    self._cliff[self.shape[0]-1, 1:-1] = True

    # Calculate transition probabilities and rewards
    P = {}
    for s in range(nS):
      position = np.unravel_index(s, self.shape)
      P[s] = {a: [] for a in range(nA)}
      P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
      P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
      P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
      P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

    # Calculate initial state distribution
    # We always start in state (3, 0)
    isd = np.zeros(nS)
    isd[self.start_state_index] = 1.0

    super(CliffWalkingEnv, self).__init__(nS, nA, P, isd)

  def _limit_coordinates(self, coord):
    """
    Prevent the agent from falling out of the grid world
    :param coord:
    :return:
    """
    coord[0] = min(coord[0], self.shape[0] - 1)
    coord[0] = max(coord[0], 0)
    coord[1] = min(coord[1], self.shape[1] - 1)
    coord[1] = max(coord[1], 0)
    return coord

  def _calculate_transition_prob(self, current, delta):
    """
    Determine the outcome for an action. Transition Prob is always 1.0.
    :param current: Current position on the grid as (row, col)
    :param delta: Change in position for transition
    :return: (1.0, new_state, reward, done)
    """
    if current == self.terminal_state or self._cliff[current]:
      return [(1.0, self.start_state_index, 0, False)]

    new_position = np.array(current) + np.array(delta)
    new_position = self._limit_coordinates(new_position).astype(int)
    if self._cliff[tuple(new_position)]:
      return [(1.0, self.start_state_index, -10, False)]

    is_done = tuple(new_position) == self.terminal_state
    rew = -1 if not is_done else 100
    new_state = np.ravel_multi_index(tuple(new_position), self.shape)
    new_state = new_state if not is_done else self.start_state_index
    return [(1.0, new_state, rew, False)]

  def _render(self, mode='human', close=False):
    if close:
      return

    outfile = sys.stdout

    for s in range(self.nS):
      position = np.unravel_index(s, self.shape)
      if self.s == s:
        output = " x "
      # Print terminal state
      elif position == self.terminal_state:
        output = " T "
      elif self._cliff[position]:
        output = " C "
      else:
        output = " o "

      if position[1] == 0:
        output = output.lstrip()
      if position[1] == self.shape[1] - 1:
        output = output.rstrip()
        output += '\n'

      outfile.write(output)
    outfile.write('\n')
