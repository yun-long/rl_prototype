import numpy as np
from rl.featurizer.one_hot_featurizer import OneHotFeaturizer
from rl.value.value_estimator import ValueEstimator
from rl.policy.discrete_policy import DistributionPolicy
from rl.sampler.standard_sampler import StandardSampler
from rl.misc.dual_function import optimize_fdual_fn_v1, optimize_dual_fn_paramv
from rl.misc.plot_rewards import plot_coeff_tr_ep_rs
from rl.misc.plot_value import plot_2D_value
from gym.envs.toy_text.nchain import NChainEnv

def f_reps(i_trial, val_fn, pol_fn, sampler, num_ep, num_sp, alpha, rnd):
    # env.seed(seed=147691)
    # rnd = np.random.RandomState(seed=43225801)
    ep_rewards = []
    etap = (15.0, 0.9, 0.1)
    etaf = lambda i: max(etap[0] * etap[1]**i, etap[2])
    for i_ep in range(num_ep):
        eta0 = etaf(i_ep)
        # sample data set from sampler
        data_set, mean_r = sampler.sample_data(policy=pol_fn, N=num_sp)
        #
        rewards, feat_diff, sa_pair_n, sa_pairs = sampler.count_data(data=data_set,
                                                                     featurizer=val_fn.featurizer)
        #
        ep_rewards.append(mean_r)
        if alpha == 1.0:
            v, A, g = optimize_dual_fn_paramv(rewards=rewards,
                                              features_diff=feat_diff,
                                              init_eta=eta0,
                                              init_v=val_fn.param_v,
                                              epsilon=1.,
                                              sa_n=sa_pair_n)
            pol_fn.update_reps(A=A, param_eta=eta0, param_v=v, g=g, keys=sa_pairs)
            val_fn.update(new_param_v=v)
        else:
            v, lamda, kappa, A, fcp = optimize_fdual_fn_v1(rewards=rewards,
                                                           features_diff=feat_diff,
                                                           sa_n=sa_pair_n,
                                                           eta=eta0,
                                                           alpha=alpha,
                                                           rnd=rnd)
            pol_fn.update_freps(A, eta0, lamda, v, sa_pairs, fcp, param_kappa=kappa)
            val_fn.update(new_param_v=v)
        print("\ralpha {}, Trails {}, Episode {}, Expected Return {}.".format(alpha, i_trial, i_ep, mean_r))
    return ep_rewards
