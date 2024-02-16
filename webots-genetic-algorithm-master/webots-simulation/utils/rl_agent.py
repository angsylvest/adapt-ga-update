import numpy as np 

class RLAgent():
    def __init__(self):
        self.observation = np.concatenate((np.array([0,0]), np.array([1000]), np.array([1000]), np.array([1000])))
        self.action = [] 
        self.goal = ()
        self.isAvoiding = False 
        self.isHoming = False 
        self.numCollected = 0
        self.agent_id = 0 
        self.location = (0,0)
        self.reward = 0
        self.done = False
        self.log_prob = 0

    def set_location(self, pos):
        self.location = pos

    def set_id(self, id):
        self.agent_id = id

    def set_state(self, state, isTrue):
        if state == "homing" and isTrue: 
            self.isHoming = True 
        elif state == "avoiding" and isTrue:
            self.isAvoiding = True
        elif state == "homing" and not isTrue: 
            self.isHoming = False 
        elif state == "avoiding" and not isTrue:
            self.isAvoiding = False

    def update_collection(self):
        self.numCollected = self.numCollected + 1 

    def set_reward(self, reward): 
        self.reward = reward

    def reset(self): 
        self.observation = []
        self.action = [] 
        self.goal = ()
        self.isAvoiding = False 
        self.isHoming = False 
        self.numCollected = 0
        self.reward = 0

    def step_agent(self, action): # use if not synchronous
        self.set_location(np.clip(self.observation + action))
        self.set_observation = self.observation # maybe be subject to change
        
        terminated_agent = np.array_equal(
            self.location, self.goal
        )
        reward = 1 if terminated_agent else 0 # simple reward, doesn't use ttc or collision 

        infos = self.get_info()

        return self.observation, reward, {}, infos
        

    def set_observation(self, obs):
        self.observation = obs

    def get_info(self):
        posx, posy = self.location
        goalx, goaly = self.goal
        return {"distance": np.linalg.norm(
        np.array([posx, posy]) - np.array([goalx, goaly]), ord=1
        )}