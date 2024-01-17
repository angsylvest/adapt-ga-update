from controller import Supervisor, Node, Keyboard, Emitter, Receiver, Field
import math 
from robot_pop import * 
import random
import time 

import sys 
sys.path.append('../../')
from utils.rl_agent import *
from utils.ppo import * 
from utils.nn import * 
from utils.rl_wrapper import * 

# ideally would have global env variable available here 
from controllers.hybrid_rl_supervisor.hybrid_rl_supervisor import grid_env 

"""
Main supervisor base 
Optimization algorithm - Collaboration-oriented 
Angel Sylvester 2022
"""
columns = 'agent id' + ',time step' + ',fitness' + ',xpos'+ ',ypos' + ',num col' + ',genotype' 

# global collected_count 
collected_count = []

# statistics collected 
total_found = 0
pairs = []
population = []

# set-up robot 
TIME_STEP = 32
robot = Supervisor()  # create Supervisor instance
timestep = int(robot.getBasicTimeStep())

# generalize id acquisition (will be same as robot assigned to), might change to someting different  
given_id = int(robot.getName()[11:-1])
assigned_r_name = "k0" if given_id == 0 else "k0(" + str(given_id) + ")" 
Agent = RLAgent()
# Agent.set_id(assigned_r_name)
# Agent.set_location((population[given_id].getPosition()[0], population[given_id].getPosition()[1]))

# individual PPO model for each agent 
hyperparamters = {}
env = ForagingEnv()
model = PPO(FeedForwardNN, env, **hyperparamters)
    
#### allow personal supervisor to send important encounter info back to supervisor ## 
emitter = robot.getDevice("emitter")
emitter.setChannel(2)

#### receive info from supervisor regarding robot #####
receiver = robot.getDevice("receiver") 
receiver.enable(TIME_STEP)
receiver.setChannel(5) 

### allow personal supervisor to recieve encounter info from robot 
receiver_individual = robot.getDevice("receiver_processor") 
receiver_individual.enable(TIME_STEP)
receiver_individual.setChannel(int(given_id * 10)) # will be updated to be robot_id * 10 
emitter_individual = robot.getDevice("emitter")
emitter_individual.setChannel((int(given_id) * 10) - 1)

updated = False
fit_update = False 
start = 0 

prev_msg = ""
random.seed(10)
curr_fitness = 0
child = ""

comparing_genes = False 
complete = True 

ep_rews = []
ep_actions = []
# log_probs = []
batch_observations = []
batch_rewards = []
batch_actions = []
batch_rtgs = []
batch_lens = []
batch_log_probs = []

# time keeping 
prev_time = robot.getTime()
update_sec = 1 


# based off given id of robot assigned 
def find_nearest_robot_genotype(r_index):
    global population 
    global collected_count 
    global pairs 
    global prev_msg 

    closest_neigh = " "
    curr_robot = population[r_index]
    curr_dist = 1000 # arbitrary value 

    
    curr_pos = [curr_robot.getPosition()[0], curr_robot.getPosition()[1]]
    other_index = r_index
    
    for i in range(len(population)):
        if (i != r_index): 
            other_pos = [population[i].getPosition()[0], population[i].getPosition()[1]]
            dis = math.dist(curr_pos, other_pos)
            if closest_neigh == " ":
                closest_neigh = str(population[i].getId())
                curr_dist = dis
                other_index = i
            elif dis < curr_dist: 
                closest_neigh = str(population[i].getId())
                curr_dist = dis 
                other_index = i
                
    return other_index

def message_listener(time_step):
    global total_found 
    global collected_count 
    global population
    global pairs 
    global child 
    global comparing_genes
    global curr_action
    global complete 

    global batch_rewards
    global ep_rews
    global batch_lens
    global Agent


    if receiver.getQueueLength()>0:
        message = receiver.getString()
        ## access to robot id for node acquisition    
        if 'ids' in message: 
            id_msg = message.split(" ")[1:]
            population = []
            
            for id in id_msg: # will convert to nodes to eventual calculation 
                node = robot.getFromId(int(id))
                population.append(node)
                
            Agent.set_id(assigned_r_name)
            Agent.set_location((population[given_id].getPosition()[0], population[given_id].getPosition()[1]))
            
            # set action for corresponding robot 
            pos = np.array([population[given_id].getPosition()[0], population[given_id].getPosition()[1]])
            curr_action, log_probs = model.get_action(pos)
            discretized_action = np.argmax(curr_action).item() # TODO: might not be correct, index of the maximum value in your continuous vector as a discrete action
            curr_action = env._action_to_direction[discretized_action]
            Agent.action = curr_action 
            # print(f'initial action for agent {assigned_r_name} with action : {curr_action}')

            msg = 'agent_action:'+ str(curr_action[0]) + "," + str(curr_action[1])
            emitter_individual.send(msg.encode('utf-8'))
                
            receiver.nextPacket() 
            
        elif 'size' in message: 
            population = []
            size = int(message[4:]) 
            receiver.nextPacket()

        elif 'action-request' in message: 
            curr_action = Agent.action
            action, log_prob = env._action_to_direction(model.get_action())
            Agent.set_location((population[given_id].getPosition()[0], population[given_id].getPosition()[1]))

            Agent.action = action 
            Agent.log_prob = log_prob

            # if complete: batch_log_probs.append(log_prob) 
            curr_action = Agent.action if not complete else action # TODO: must do next action once done with previous

            # ep_actions.append(curr_action) 

            receiver.nextPacket()

        elif 'episode-complete' in message: 
            # TODO: reset agent + reset actions for agent
            Agent.reset() 
            msg = 'episode-agent-complete'
            emitter_individual.send(msg.encode('utf-8'))

            batch_lens.append(600) # TODO: make more dynamic 
            batch_rewards.append(ep_rews)
            
            receiver.nextPacket()

        elif 'updating-network' in message: 
            # TODO: make network update pause sim or stop further collection of statistics 
            batch_obs = torch.tensor(batch_obs, dtype = torch.float)
            batch_acts = torch.tensor(batch_acts, dtype = torch.float)
            batch_log_probs = torch.tensor(batch_log_probs, dtype = torch.float)
            batch_rtgs = model.compute_rtgs(batch_rewards)
            
            
            model.learn_adjusted(batch_observations, batch_acts, batch_log_probs, batch_rtgs, batch_lens) # TODO: update 
            msg = 'update-complete'
            emitter.send(msg.encode('utf-8'))
            
        else: 
            receiver.nextPacket()
            
    if receiver_individual.getQueueLength()>0:  
        # message_individual = receiver_individual.getData().decode('utf-8')
        message_individual = receiver_individual.getString()

        if 'action-complete' in message_individual: 
            Agent.reward = message_individual.split(":")[-1]
            obs, rew, done = Agent.observation, Agent.reward, Agent.done
            ep_rews.append(rew)

            # update action and tell agent to execute 
            # curr_action = Agent.action  
            pos = np.array([population[given_id].getPosition()[0], population[given_id].getPosition()[1]])
            Agent.set_location((population[given_id].getPosition()[0], population[given_id].getPosition()[1])) 
            action, log_prob = model.get_action(pos) 
            
            discretized_action = np.argmax(action).item() # TODO: might not be correct, index of the maximum value in your continuous vector as a discrete action
            curr_action = env._action_to_direction[discretized_action]
            Agent.action = curr_action 

            # Agent.action = env._action_to_direction(action).tolist() 
            Agent.log_prob = log_prob 

            # if complete: batch_log_probs.append(log_prob) 
            # curr_action = Agent.action if not complete else action # TODO: must do next action once done with previous
            msg = 'agent_action:'+ str(curr_action[0]) + "," + str(curr_action[1])
            emitter_individual.send(msg.encode('utf-8'))
            receiver_individual.nextPacket()

        else: 
            # print('indiviudal msgs --', message_individual)
            receiver_individual.nextPacket()


# TODO: make adjusted function for batch_obs, back_acts, batch_log_probs, batch_rtgs, batch_lens
     
def update_batch_info():
    global prev_time
    global update_sec
    global given_id

    global batch_observations
    global batch_actions
    global ep_rews
    global batch_log_probs 

    if robot.getTime() - prev_time > update_sec: # update every second 
        prev_time = robot.getTime()
        # update info 
        batch_observations.append(population[given_id].getPosition())
        batch_actions.append(Agent.action)
        ep_rews.append(Agent.reward)
        batch_log_probs.append(Agent.log_prob)


def run_optimization():
    global updated
    global total_found 
    global collected_count
    global population 
    global prev_msg 
    global timestep 
    
    while robot.step(timestep) != -1: 
        message_listener(timestep)    
        update_batch_info()    
     
  
def main(): 
    run_optimization()
         
main()
                    
            
            
