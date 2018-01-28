from rl.policy.numpy.gp_linear_mean import GPLinearMean
from rl.featurizer.rbf_featurizer import RBFFeaturizer
#
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
#
env = Continuous_MountainCarEnv()
print("Action space : ", env.action_space)
print("Action space low : ", env.action_space.low)
print("Action space high: ", env.action_space.high)
print("Observation space : ", env.observation_space)
print("Observation space low : ", env.observation_space.low)
print("Observation space high: ", env.observation_space.high)
#
dim_features = 10
rbf_featurizer = RBFFeaturizer(env, dim_features=dim_features)
# rbf_featurizer.plot_examples()


