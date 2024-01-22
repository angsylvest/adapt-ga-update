import numpy as np 

"""
# bayesian updating approach to updating actions
# adapted from simple-n-armed bandit code 

# source code: 
https://github.com/iosband/ts_tutorial/blob/master/ts_tutorial_intro.ipynb

"""

class NArmedBandit():

    def __init__(self, probs):
        self.probs = np.array(probs)
        assert np.all(self.probs >= 0)
        assert np.all(self.probs <= 1)


        self.optimal_reward = np.max(self.probs)
        self.n_arm = len(self.probs)


    def get_observation(self):
        return self.n_arm
    
    def get_optimal_reward(self):
        return np.max(self.probs)
    
    def get_expected_reward(self, action):
        return self.probs[action]
    
    def get_stochastic_reward(self, action):
        return np.random.binomial(1, self.probs[action])
    

class NArmedBanditDrift(NArmedBandit):
    def __init__(self, n_arm, a0=1., b0=1., gamma=0.01): # self.gamma = 0 means no drift
        self.n_arm = n_arm 
        self.a0 = a0
        self.b0 = b0 

        self.prior_success = np.array([a0 for a in range(n_arm)])
        self.prior_failure = np.array([b0 for b in range(n_arm)])

        self.gamma = gamma 
        self.probs = np.array([np.random.beta(a0, b0) for a in range(n_arm)])

    
    def set_prior(self, prior_success, prior_failure):
        self.prior_success = np.array(prior_success)
        self.prior_failure = np.array(prior_failure)

    def get_optimal_reward(self):
        return np.max(self.probs)
    
    def advance(self, action, reward):
        self.prior_success = self.prior_success * (1 - self.gamma) + self.a0 * self.gamma
        self.prior_success = self.prior_failure * (1 - self.gamma) + self.b0 * self.gamma

        self.prior_success[action] += reward
        self.prior_failure[action] += 1 - reward

        # resample posterior probabilities 
        self.probs = np.array([np.random.beta(self.prior_success[a], self.prior_failure[a]) for a in range(self.n_arm)])
    

    
