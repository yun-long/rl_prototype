from collections import namedtuple, defaultdict


IDs = namedtuple("IDs", 'Pendulum, MCarCon')
#
env_IDs = IDs
#
env_IDs.Pendulum = 'Pendulum-v0'
env_IDs.MounCarCon = 'MountainCarContinuous-v0'


