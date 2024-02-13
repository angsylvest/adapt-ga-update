import torch 
import numpy as np 
import torch.nn as nn 
import gym 
import time 

from torch.optim import Adam
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt


import torch.nn.functional as F

import sys 
sys.path.append('../../')
import utils.global_var as globals
online = globals.online
use_batch = globals.use_batch

"""Majority taken from PPO-for-Beginners github repo"""


class PPO():
    def __init__(self, policy_class, env, agent_info, **hyperparameters):
        # validate that env is compatible (most likely gym)
        
        print(f'PPO env type {type(env.observation_space[0])}')
        print(f'PPO action space {type(env.action_space[0])}')
        assert(type(env.observation_space[0]) == gym.spaces.Dict)
        assert(type(env.action_space[0]) == gym.spaces.Discrete)

        # initialize hyperparameters 
        self._init_hyperparameters(hyperparameters)

        # extract env info 
        self.env = env 
        # self.obs_dim = env.observation_space[0].shape[0]
        self.obs_dim = env.observation_space[0]["agent"].shape[0] # + env.observation_space[0]["target"].shape[0]
        
        # self.act_dim = env.action_space[0].shape[0]
        self.act_dim = env.action_space[0].n # is 4 now
        self.agent_info = agent_info
        # update graph for agent 
        self.graph_path = self.agent_info["path"]
        self.avg_rewards_over_time = []


        # initialize actor and critic 
        self.actor = policy_class(self.obs_dim, self.act_dim)
        self.critic = policy_class(self.obs_dim, 1) # was 1

        # initialize optimizers 
        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

		# This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }

    # TODO: eventually get rid of, this is not used...
    def learn(self, total_timesteps):
        # training actor and critic networks 
        t_so_far = 0  # ts simulated so far 
        i_so_far = 0  # iterations ran so far 
        
        while t_so_far < total_timesteps: 
            # collect batch sims here 
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.roll_out()

            # calc ts collected this batch 
            t_so_far += np.sum(batch_lens)

            # incre # of iterations 
            i_so_far += 1

            # log timesteps + iterations 
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # calc advantage at k-th iteration 
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach # diff between observed and estiamted return 


            # TODO: might need to comment out, not officially in psuedo-code
            # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # update network and n epochs 
            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # calc surrogate losses 
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1- self.clip, 1 + self.clip) * A_k

                # calculate actor + critic losses 
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs) # fit value function by regression on mean-squared error

                self.actor_optim.zero_grad()
                critic_loss.backwards()
                self.critic_optim.step()

                # log actor loss 
                self.logger['actor_losses'].append(actor_loss.detach())

            # print training performance so far 
            self._log_summary()

            # save model if it's time 
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def learn_adjusted(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rewards):
        
        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rewards
        self.logger['batch_lens'] = batch_lens
        self.logger['i_so_far'] += 1
        
        # training actor and critic networks 
        t_so_far = 0  # ts simulated so far 
        # i_so_far = 0  # iterations ran so far 
        
        # while t_so_far < total_timesteps: 
            # collect batch sims here 
            # batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.roll_out()

            # calc ts collected this batch 
        t_so_far += np.sum(batch_lens)

        # incre # of iterations 
        # i_so_far += 1

        # log timesteps + iterations 
        self.logger['t_so_far'] = t_so_far
        # self.logger['i_so_far'] = i_so_far

        # calc advantage at k-th iteration 
        # print(f'in evaluate() batch_obs: {batch_obs} and batch_acts: {batch_acts}')
        V, _ = self.evaluate(batch_obs, batch_acts)
        # print(f'batch_rtgs: {batch_rtgs} and V.detach(): {V.detach()}')
        A_k = batch_rtgs - V.detach() # diff between observed and estimated return 


        # TODO: might need to comment out, not officially in psuedo-code
        # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        # update network and n epochs 
        for _ in range(self.n_updates_per_iteration):
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # calc surrogate losses 
            # print(f'surrogate loss params: \n ratios: {ratios} and A_k: {A_k}')
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1- self.clip, 1 + self.clip) * A_k

            # calculate actor + critic losses 
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs) # fit value function by regression on mean-squared error

            self.actor_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # log actor loss 
            self.logger['actor_losses'].append(actor_loss.detach())


        # print training performance so far 
        self._log_summary()

        # save model if it's time 
        if self.logger['i_so_far'] % self.save_freq == 0:
            torch.save(self.actor.state_dict(), './ppo_actor.pth')
            torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def roll_out(self):
        # collect info during iteration 
        
        # batch data 
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        ep_rews = 0 
        t = 0 

        while t < self.timesteps_per_batch:
            ep_rews = []

            obs = self.env.reset() # will begin simulation iteration (make webots compatible)
            done = False 

            for ep_t in range(self.max_timesteps_per_episode): 

                t += 1 
                # run an episode (probably could be collected in real time and requested)

                batch_obs.append(obs)

                # calc action + make step in env 
                action, log_prob = self.get_action()

                # TODO: need way to acquire this info without needing step 
                obs, rew, done, _ = self.env.step(action)

                # track reward, action, action log probabilities 
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done: # env will tell when to terminate 
                    break 

            # track ep length + rewards 
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # reshape data as tesnros 
        batch_obs = torch.tensor(batch_obs, dtype = torch.float)
        batch_acts = torch.tensor(batch_acts, dtype = torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype = torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        # log 

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    

    def compute_rtgs(self, batch_rews):
        # compute rewards to go 

        batch_rtgs = []

        # iterate through each episode
        for ep_rews in reversed(batch_rews): 
            discounted_reward = 0 

            # iterate through all rewards in the episode 
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma # self.gamma less emphasis on future rewards 
                batch_rtgs.insert(0, discounted_reward)

        # convert rtg to tensor 
        batch_rtgs = torch.tensor(batch_rtgs, dtype = torch.float)

        return batch_rtgs
        
        

    def get_action(self, obs): 
        # query from actor network  
        mean = self.actor(obs)

        # dist = MultivariateNormal(mean, self.cov_mat)
        
        # create a categorical distribution
        dist = torch.distributions.Categorical(logits=mean)
        
        # sample action from distribution 
        action = dist.sample()

        # calc log probability for action 
        log_prob = dist.log_prob(action)

        # returned sample action + log probability 
        return action.detach().numpy(), log_prob.detach()
    

    def evaluate(self, batch_obs, batch_acts):
        # get value from critic and log prob from actor 
        
        """ 
        batch_obs shape:  (num ts in batch, dim of obs)
        batch_acts shape: (num ts in batch, dim of action)
        """
        V = self.critic(batch_obs).squeeze() # should be same size as batch_rtgs
        # print(f'critic output from evaluate(): {V.shape}')

        # calc log probabilities 
        mean = self.actor(batch_obs)
        # print(f'mean from self.actor: {mean.shape}')
        # dist = MultivariateNormal(mean, self.cov_mat)
        
        dist = torch.distributions.Categorical(logits=mean)
        
        # Flatten batch_acts to ensure it's a 1D tensor
        # batch_acts_flat = batch_acts.view(-1)

        
        # Assuming mean has shape (batch_size, num_actions)
        # print(f' batch acts shape -- {batch_acts.shape}' )
        # print(batch_acts)
        # target_values = batch_acts[:, 0].long()
        # print(f'target values: {target_values}')
        
        # print(f'dist output: {dist}')
        log_probs = dist.log_prob(batch_acts)
        
        # log_probs = F.cross_entropy(mean, batch_acts[:, 0].long(), reduction='none') # with batches, instead of just 1 obs
        # print(f'log probs from evaluate(): {log_probs}')

        # return value vector of each obs in batch + log probs 
        return V, log_probs

        


    def _init_hyperparameters(self, hyperparameters): 
        # initialize default + custom values 

        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 600
        self.n_updates_per_iteration = 5
        self.lr = 0.005 
        self.gamma = 0.95
        self.clip = 0.2

        # miscel parameters 
        self.render = True 
        self.render_every_i = 10
        self.save_freq = 10 
        self.seed = 11

        # change default values to custom values 
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
        if self.seed != None:
			# Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")


    def _log_summary(self): # will update to store info as csv if needed 
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        print(f'batch rews from logger: {self.logger["batch_rews"]}')

        if not online: # not updating with only one obs at a time 
            avg_ep_lens = np.mean(self.logger["batch_lens"])
            avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
            avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

            # Round decimal places for more aesthetic logging messages
            avg_ep_lens = str(round(avg_ep_lens, 2))
            avg_ep_rews = str(round(avg_ep_rews, 2))
            avg_actor_loss = str(round(avg_actor_loss, 5))

            # print(f"batch rews: {self.logger['batch_rews']}")
            self.avg_rewards_over_time.append(avg_ep_rews)
            plt.plot(self.avg_rewards_over_time)
            plt.xlabel('Iterations')
            plt.ylabel('Average Reward')
            plt.title('Average Reward Over Time')
            plt.grid(True)
            plt.savefig(self.graph_path + 'avg_reward_over_time.png')  # Save the plot
            # plt.show()


            # Print logging statements
            print(flush=True)
            print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
            print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
            print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
            print(f"Average Loss: {avg_actor_loss}", flush=True)
            print(f"Timesteps So Far: {t_so_far}", flush=True)
            print(f"Iteration took: {delta_t} secs", flush=True)
            print(f"------------------------------------------------------", flush=True)
            print(flush=True)

        else: # diff output for online 
            print(flush=True)
            print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
            # print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
            # print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
            # print(f"Average Loss: {avg_actor_loss}", flush=True)
            # print(f"Timesteps So Far: {t_so_far}", flush=True)
            # print(f"Iteration took: {delta_t} secs", flush=True)
            # print(f"------------------------------------------------------", flush=True)
            # print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

