
class GaussianPolicy(object):

    def __init__(self):
        pass

    def predict_action(self, state):
        """
        For step based usually
        :param state:
        :return:
        """
        raise NotImplementedError

    def sample_theta(self, num_samples):
        """
        For episode based usually
        :param num_samples:
        :return:
        """
        raise NotImplementedError

    def update_pg(self, *args, **kwargs):
        raise NotImplementedError

    def update_wml(self, *args, **kwargs):
        raise NotImplementedError

    def optimal_policy_demo(self, num_demos):
        raise NotImplementedError