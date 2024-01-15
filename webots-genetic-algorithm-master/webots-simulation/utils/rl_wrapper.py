import gym
from gym import spaces
from matplotlib import pyplot as plt
import pygame
import numpy as np

from utils.rl_agent import *

from utils.ppo import * 
from utils.nn import * 

class ForagingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, num_agents = 2):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.num_agents = num_agents

        # self._action_to_direction = {
            # 0: np.array([1, 0]),
            # 1: np.array([0, 1]),
            # 2: np.array([-1, 0]),
            # 3: np.array([0, -1]),
        # }
        
        self._action_to_direction = {
            0: [1, 0],
            1: [0, 1],
            2: [-1, 0],
            3: [0, -1],
        }

        self.agent_list = []

        # ---- multi-agent version of action space ---- #
        self.action_space = []
        self.observation_space = []

        for i in range(num_agents):

            agent = RLAgent()
            self.agent_list.append(agent)

            self.action_space.append(spaces.Discrete(4))

            obs_space = spaces.Dict(
                    {
                        "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                        "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    }
                )
            
            self.observation_space.append(obs_space)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = "rgb_array"

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        # update obs for all agents here 
        for a in range(self.num_agents): 
            agent_loc = self.agent_list[a].location
            agent_tar = self.agent_list[a].goal

            dic_update = {"agent": agent_loc, "target": agent_tar}
            self.observation_space[a] = dic_update

        agent_locations = self.observation_space  # Assuming _agent_locations is a list of agent locations
        return [{"agent": agent_loc["agent"], "target": agent_loc["target"]} for agent_loc in agent_locations]


    def _get_info(self):
        return [{"distance": np.linalg.norm(
                np.array([agent_loc["agent"].x, agent_loc["agent"].y]) - np.array([agent_loc["target"].x, agent_loc["target"].y]), ord=1
            )} for agent_loc in self.observation_space]


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # TODO: ensure that env elements are update or correct method is called to this action

        return observation, info
    
    def step(self, actions):
        rewards = []
        observations = []
        terminated = []
        infos = []

        for agent_id, action in enumerate(actions):
            direction = self._action_to_direction[action]
            loc = self.observation_space[agent_id]["agent"]

            self.observation_space[agent_id]["agent"] = np.clip(
                loc + direction, 0, self.size - 1
            )

            self.agent_list[agent_id].set_location(self.observation_space[agent_id]["agent"])

            terminated_agent = np.array_equal(
                loc, self._target_location
            )

            terminated.append(terminated_agent)
            reward = 1 if terminated_agent else 0 # simple reward, doesn't use ttc or collision 
            rewards.append(reward)
            observation = self._get_obs()
            observations.append(observation)
            info = self._get_info()
            infos.append(info)

        if self.render_mode == "human":
            self._render_frame()

        # Assuming your environment considers an episode terminated if any agent reaches the target
        episode_terminated = any(terminated)

        return observations, rewards, episode_terminated, {}, infos
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels


        for i in range(self.num_agents): # hard-coded, would need to update to make improved
            self.display_w = 500
            screen_left = -300
            screen_right = 300
            goal_radius1 = 10
            agent_radius = 10
            screen_w = self.display_w
            screen_h = int(screen_w/1)
            screen_up = -300
            screen_down = 300

            goalx, goaly = self.agent_list[i].goal
            posx, posy = self.agent_list[i].location

            pygame.draw.circle(canvas, (0, 0, 255), (int((goalx-screen_left)/(screen_right-screen_left)*screen_w), int((goaly-screen_up)/(screen_down-screen_up)*screen_h)), int(goal_radius1/(screen_right-screen_left)*screen_w), 1)
            pygame.draw.circle(canvas, (0, 0, 255), (int((posx-screen_left)/(screen_right-screen_left)*screen_w), int((posy-screen_up)/(screen_down-screen_up)*screen_h)), int(agent_radius/(screen_right-screen_left)*screen_w), 0)

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def main(args): 

    hyperparameters = {
            'timesteps_per_batch': 2048, 
            'max_timesteps_per_episode': 200, 
            'gamma': 0.99, 
            'n_updates_per_iteration': 10,
            'lr': 3e-4, 
            'clip': 0.2,
            'render': True,
            'render_every_i': 10
            }
    

    # create env here 
    env = ForagingEnv()

    # begin training 
    train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)


def train(env, hyperparameters, actor_model, critic_model): # other params used if loading from previous training
	# Create a model for PPO.
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

    # will train from scratch 
    model.learn(total_timesteps=200_000_000)

