# attempt at online version of online sequential markov monte carlo 
import numpy as np
import torch 
from torch.distributions import MultivariateNormal

import torch.nn.functional as F

class MarkovMonteCarlo():
    def __init__(self, state_dim, act_dim):
        self.state_dim = state_dim
        self.act_dim = act_dim

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)


    def initialize_particles(self, num_particles, mean): 
        self.num_particles = num_particles

        # mean is query from actor network
        covariance_matrix = self.cov_mat
        particles = np.random.multivariate_normal(mean, covariance_matrix, size=num_particles)
        # each particle represents agent's belief in the state at a given time 
        return particles
    
    # adapted from PPO code for sampling from multivariate distribution
    def get_transition_probabilities(self, particles, action):
        # particles: Tensor of shape (num_particles, state_dim)
        
        # Assuming you have some method to compute the mean for the next state based on particles and action
        next_state_means = self.compute_mean_transition(particles, action)

        # Create a multivariate normal distribution for each particle
        dists = [MultivariateNormal(mean, self.cov_mat) for mean in next_state_means]

        # Sample from each distribution to get the next states for all particles
        next_states = [dist.sample() for dist in dists]

        return torch.stack(next_states)
    

    def compute_mean_transition(self, particles, action):
                # particles: Tensor of shape (num_particles, state_dim)

        # Expand action to match the shape of particles
        expanded_action = action.unsqueeze(0).expand(self.num_particles, -1)

        # Use the actor network to get the mean of the action distribution for each particle
        next_state_means = self.actor(particles)

        # Add the action to the mean to get the mean for the next state
        next_state_means += expanded_action

        return next_state_means 
    
    def sample_from_transition_model(self, current_state, action):
        # Define your transition dynamics (this is just a simple example)
        # Replace this with the actual transition model of your environment
        transition_probabilities = self.get_transition_probabilities(current_state, action)

        # Sample the next state based on the transition probabilities
        next_state = np.random.choice(len(transition_probabilities), p=transition_probabilities)

        return next_state

    # Prediction step
    def predict(self, particles, action):
        for i in range(self.num_particles):
            particles[i] = self.sample_from_transition_model(particles[i], action)
        return particles
    
    def resample(self, particles, weights):
        # particles: Tensor of shape (num_particles, state_dim)
        # weights: Tensor of shape (num_particles,)

        # Normalize weights
        normalized_weights = F.normalize(weights, p=1, dim=0)

        # Resample particles based on the weights
        resampled_indices = torch.multinomial(normalized_weights, self.num_particles, replacement=True)
        resampled_particles = particles[resampled_indices]

        return resampled_particles
    

    def compute_mean_observation(particles):
        return np.mean(particles, axis=0) 
    

    def compute_likelihood(self, observation, particles): 
        # observation: Tensor of shape (observation_dim,)
        # particles: Tensor of shape (num_particles, state_dim)

        # Assuming you have some method to compute the mean for the observation based on particles
        observation_means = self.compute_mean_observation(particles)

        # Create a multivariate normal distribution for each particle's observation mean
        dists = [MultivariateNormal(mean, self.observation_cov_mat) for mean in observation_means]

        # Compute the log-likelihood for each particle using the multivariate normal distributions
        log_likelihoods = torch.stack([dist.log_prob(observation) for dist in dists])

        # Convert log-likelihoods to probabilities
        likelihoods = torch.exp(log_likelihoods)

        return likelihoods

    # Update step
    def update(self, particles, observation):
        weights = self.compute_likelihood(observation, particles)
        particles = self.resample(particles, weights)
        return particles

    # Online Reinforcement Learning using SMC
    # def online_rl_smc(self, num_particles, timestep):
    #     particles = initialize_particles(num_particles)

    #     for each timestep:
    #         # Agent selects an action based on the belief state (e.g., using a policy)
    #         action = select_action(particles)

    #         # Agent takes the selected action, receives observation, and updates belief state
    #         particles = predict(particles, action)
    #         particles = update(particles, observation)

    #         # Optional: Agent can learn from the updated belief state (e.g., update policy or value function)

    #     return particles