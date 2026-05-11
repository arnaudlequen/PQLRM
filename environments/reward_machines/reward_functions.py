import math

class RewardFunction:
    def __init__(self):
        pass

    # To implement...
    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")


class ConstantRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def get_reward(self, s_info):
        return self.c


class SumRewardFunction(RewardFunction):
    """
    Weighted sum of two reward functions.
    """
    def __init__(self, left: RewardFunction, right: RewardFunction, left_weight: float = 1.0, right_weight: float = 1.0):
        super().__init__()
        self.left = left
        self.right = right
        self.left_weight = left_weight
        self.right_weight = right_weight

    def get_type(self):
        return "sum"

    def get_reward(self, s_info):
        return self.left_weight * self.left.get_reward(s_info) + self.right_weight * self.right.get_reward(s_info)

class RewardControl(RewardFunction):
    """
    Gives a reward for moving forward
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "ctrl"

    def get_reward(self, s_info):
        return s_info['reward_ctrl']

class RewardForward(RewardFunction):
    """
    Gives a reward for moving forward
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "forward"

    def get_reward(self, s_info):
        return s_info['reward_run'] + s_info['reward_ctrl']  #Cheetah


class RewardBackwards(RewardFunction):
    """
    Gives a reward for moving backwards
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "backwards"

    def get_reward(self, s_info):
        return -s_info['reward_run'] + s_info['reward_ctrl']  #Cheetah
