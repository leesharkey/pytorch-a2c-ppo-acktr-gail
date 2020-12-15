import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class BanditEnv(gym.Env):

    def __init__(self, p_dist_dist_name=None):

        self.r_dist = [1, 1]
        self.p_dist_dist_name = 'Dirichlet' # From 'Dirichlet', 'Dirichlet_support_subset', 'FixedEasy'

        self.n_bandits = len(self.r_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.n_bandits + 1,),
                                            dtype=np.float32)
        self.prev_act, self.prev_rew = None, 0.

        # Define the restricted subset of the support of the Dirichlet
        # distribution
        num_ele_subset = 51
        increment = 1/num_ele_subset
        self.dirichlet_support_subset = [[p,1-p] for p in
                                         np.arange(0,1+increment,increment)]

        self.seed()
        self.sample_p_dist_dist()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if type(action) == np.ndarray:
            action = action[0]
        elif type(action) == np.int64 or type(action) == np.int32:
            pass
        else:
            raise ValueError("Action is the wrong type. Is " + \
                             "{}. Can be np.ndarray ".format(type(action)) + \
                             "or np.int32 or np.int64")
        assert self.action_space.contains(action)

        # By default, the agent is punished unless it chooses the correct
        # action.
        reward = -1 / (self.n_bandits - 1)

        if np.random.uniform() < self.p_dist[action]:
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]
            else:
                reward = np.random.normal(self.r_dist[action][0], self.r_dist[action][1])

        self.prev_act, self.prev_rew = action, reward

        info = {'p_dist': self.p_dist}

        return self._get_obs(), reward, False, info

    def _get_obs(self):
        return self.prev_act_and_reward_to_vec(self.prev_act, self.prev_rew)

    def reset(self):
        self.sample_p_dist_dist()
        self.prev_act, self.prev_rew = None, 0.
        return self._get_obs()

    def to_one_hot(self, idx):
        one_hot_vec = np.zeros(self.n_bandits)
        one_hot_vec[idx] = 1
        return one_hot_vec

    def prev_act_and_reward_to_vec(self, prev_act, reward):
        if prev_act is None:
            one_hot_act = np.zeros(2)
        else:
            one_hot_act = self.to_one_hot(prev_act)
        one_hot_act_and_rew = np.append(one_hot_act, reward)
        # print(one_hot_act_and_rew)
        return one_hot_act_and_rew

    def sample_p_dist_dist(self):
        # self.seed(self.orig_seed)

        if self.p_dist_dist_name == 'Dirichlet':
            self.p_dist = self.np_random.dirichlet([1]*self.n_bandits)# 0.1 is easy. 1 is average. 1000 is very hard.
        elif self.p_dist_dist_name == 'Dirichlet_support_subset':
            self.p_dist = self.np_random.choice(self.dirichlet_support_subset)
        elif self.p_dist_dist_name == 'FixedEasy':
            self.p_dist = np.array([0.9, 0.1])
        else:
            raise ValueError("Invalid distribution name for bandits")