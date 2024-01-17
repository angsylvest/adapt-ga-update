'''
Discretized local environment for each agent
'''
import numpy as np

class MapInfo():
    # map info gets updated every sec 

    def __init__(self, upper_left=(-1, 1), lower_right=(1, -1), num_spaces=5):
        # create discretized form of env
        self.complete_env = np.zeros((num_spaces * (lower_right[0] - upper_left[0]), num_spaces * (upper_left[1] - lower_right[1])))
        self.space_dimensions = np.array([lower_right[0] - upper_left[0], upper_left[1] - lower_right[1]])
        self.num_spaces = num_spaces

        self.upper_left = upper_left

    def cont_to_discrete(self, continuous_position):
        # compute relative position within the continuous space
        relative_position = continuous_position - self.upper_left  # distance 
        # print(f'relative_position: {relative_position}')

        # compute size of each grid cell
        grid_cell_size = self.space_dimensions / np.array(self.complete_env.shape)
        # print(f'grid_cell_size: {grid_cell_size}')

        # calculate indices for each dimension
        indices = (relative_position / grid_cell_size).astype(int)

        # clip indices to stay within array bounds
        clipped_indices = np.clip(indices, 0, np.array(self.complete_env.shape) - 1)

        return tuple(clipped_indices)

    def format_local_env(self, curr_pos):  # assuming curr_pos is np array 
        # returns indices of curr_pos and neighboring grid spaces
        discrete_pos = self.cont_to_discrete(curr_pos)

        # Extract the subset of the environment around the current position
        local_env_indices = np.indices((2 * self.num_spaces + 1, 2 * self.num_spaces + 1)).reshape(2, -1).T
        neighbor_indices = local_env_indices + np.array(discrete_pos) - self.num_spaces

        # Identify neighboring cells that are one grid space away
        one_space_away_indices = [
            tuple(idx)
            for idx in neighbor_indices
            if np.abs(np.array(idx) - np.array(discrete_pos)).sum() == 1
        ]

        # Get values from self.complete_env for the subset
        subset_values = self.get_subset_values(discrete_pos, one_space_away_indices)

        return discrete_pos, one_space_away_indices, subset_values
    
    def get_subset_values(self, curr_index, neighbor_indices):
        # Get values from self.complete_env for the subset
        subset_values = [self.complete_env[idx] for idx in neighbor_indices]

        return subset_values


    def update_env(self, agent_pos, obj_pos): 
        # after certain amount of time, will update env info globally 
        for pos in agent_pos:
            index = self.cont_to_discrete(pos)
            self.complete_env[index] = 1 

        for pos in obj_pos: 
            index = self.cont_to_discrete(pos)
            self.complete_env[index] = 1  

def main():        
    test = MapInfo()
    print(test.format_local_env(np.array([0,0]) ))


