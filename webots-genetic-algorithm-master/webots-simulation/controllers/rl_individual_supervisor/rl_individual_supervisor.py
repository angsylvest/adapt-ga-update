from controller import Supervisor, Node, Keyboard, Emitter, Receiver, Field
import math 
from robot_pop import * 
import random
import time 
import re

import sys 
sys.path.append('../../')
from utils.rl_agent import *
from utils.ppo import * 
from utils.nn import * 
from utils.rl_wrapper import * 
import utils.global_var as globals

# from controllers.pure_rl_supervisor.pure_rl_supervisor import ex

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
batch = globals.use_batch
online = globals.online
sim_type = globals.sim_type
communication = globals.communication 
using_high_dens = globals.using_high_dense 
hyperparamters = {}
agent_info = {"path": f"../../graph-generation/collection-data/env_{sim_type}_batch_{batch}_online_{online}_agent_{assigned_r_name}_", "name": assigned_r_name}
env = ForagingEnv()
model = PPO(FeedForwardNN, env, agent_info, **hyperparamters)

# # collected counts csv generation 
agent_df = open(f'../../graph-generation/collection-data/overall-df-{sim_type}-comm_{communication}-agent-{assigned_r_name}.csv', 'w')
overall_columns = 'episode,' + 'avg_reward' + ',objects retrieved' 
agent_df.write(str(overall_columns) + '\n')
agent_df.close()
    
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
num_updates_per_episode = 600 # 1 per second 
curr_index = 0

prev_pos = ()
agent_pos = ()
prev_population_pos = []

cum_reward = 0 
episode_length = 0 
curr_episode = 0 



def generate_vector(agent_pos, prev_pos, prev_population_pos):
    pass 


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
    global batch_observations
    global batch_log_probs
    global batch_lens
    global batch_rtgs
    global batch_actions
    global ep_rews
    global batch_lens
    global Agent
    global curr_index
    
    global prev_pos
    global agent_pos
    global prev_population_pos
    global curr_episode
    global cum_reward 
    global episode_length 
    global curr_episode 


    if receiver.getQueueLength()>0:
        message = receiver.getString()
        # print('individual messages: ', message) 
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
            # pos = np.array([population[given_id].getPosition()[0], population[given_id].getPosition()[1]])
            # curr_action, log_probs = model.get_action(pos)
            # Agent.action = curr_action[0] 
            # # print(f'initial action for agent {assigned_r_name} with action : {curr_action} for {type(curr_action)}')
            # # discretized_action = np.argmax(curr_action).item() # TODO: might not be correct, index of the maximum value in your continuous vector as a discrete action
            # curr_action = env._action_to_direction[int(curr_action[0])]
            # # Agent.action = curr_action 
            # # print(f'initial action for agent {assigned_r_name} with action : {curr_action} and agent reward {Agent.reward}')

            # msg = 'agent_action:'+ str(curr_action[0]) + "," + str(curr_action[1])
            # emitter_individual.send(msg.encode('utf-8'))
                
            receiver.nextPacket() 
            
        elif 'size' in message: 
            population = []
            size = int(message[4:]) 
            receiver.nextPacket()

        elif 'action-request' in message: 
            Agent.reward = float(message_individual.split(":")[-2])
            Agent.observation = float(message_individual.split(":")[-1]) # TODO: not correct yet
            
            pos = np.array([population[given_id].getPosition()[0], population[given_id].getPosition()[1]])
            Agent.action, log_prob = model.get_action(pos)
            curr_action = env._action_to_direction(Agent.action)
            Agent.set_location((population[given_id].getPosition()[0], population[given_id].getPosition()[1]))

            # Agent.action = action 
            Agent.log_prob = log_prob
            
            curr_action = Agent.action
            
            msg = 'agent_action:'+ str(curr_action[0]) + "," + str(curr_action[1])
            emitter_individual.send(msg.encode('utf-8'))
            
            # print(f'after action-request action for agent {assigned_r_name} with action : {curr_action}')

            # if complete: batch_log_probs.append(log_prob) 
            curr_action = Agent.action if not complete else action # TODO: must do next action once done with previous

            # ep_actions.append(curr_action) 

            receiver.nextPacket()

        elif 'episode-complete' in message: 
            # TODO: reset agent + reset actions for agent
            Agent.reset() 
            msg = 'episode-agent-complete'
            emitter_individual.send(msg.encode('utf-8'))
            curr_episode += 1 

            agent_df = open(f'../../graph-generation/collection-data/overall-df-{sim_type}-comm_{communication}-agent-{assigned_r_name}.csv', 'a')
            col_update = f"{curr_episode}, {cum_reward/episode_length}, NA"
            agent_df.write(str(col_update) + '\n')
            agent_df.close()

            cum_reward = 0 
            episode_length = 0 

            print('episode complete')

            batch_lens.append(600) # TODO: make more dynamic 
            batch_rewards.append(ep_rews)
            # print(batch_rewards)
            ep_rews = []
            ep_rews.append(Agent.reward)
            
            # set action for corresponding robot 
            pos = np.array([population[given_id].getPosition()[0], population[given_id].getPosition()[1]])
            curr_action, log_probs = model.get_action(Agent.observation)
            Agent.action = curr_action[0] 
            # print(f'initial action for agent {assigned_r_name} with action : {curr_action} for {type(curr_action)}')
            # discretized_action = np.argmax(curr_action).item() # TODO: might not be correct, index of the maximum value in your continuous vector as a discrete action
            curr_action = env._action_to_direction[int(curr_action[0])]

            msg = 'agent_action:'+ str(curr_action[0]) + "," + str(curr_action[1])
            emitter_individual.send(msg.encode('utf-8'))
            # Agent.action = curr_action 
            # batch
            
            receiver.nextPacket()

        elif 'updating-network' in message: 
            # this instance of update occurs at the end of the episode 

            # TODO: make network update pause sim or stop further collection of statistics 
            print('updating network')
            
            
            # TODO: need to make work across episodes updated code with corrections 
            # print(f'init batch sizes \n batch_observations: {len(batch_observations)} \n batch_actions: {len(batch_actions)}')
            # print('batch actions --', batch_actions)

            if not online: 
                batch_observations = torch.tensor(batch_observations, dtype = torch.float)
                batch_acts = torch.tensor(batch_actions, dtype = torch.float)
                batch_log_probs = torch.tensor(batch_log_probs, dtype = torch.float)
                # print('intial batch_rewards - ', batch_rewards)

                # if not globals.use_batch: 
                #     batch_rewards.append(ep_rews) # TODO: do i need this? 
                # print('final batch_rewards - ', len(batch_rewards))
                batch_rtgs = model.compute_rtgs(batch_rewards) # TODO: should be batch_rewards instead of ep_rews
                batch_lens = [600] # TODO: update so correct 
                # print('converted to torch')
                # print(f'info updates : \n batch_observations: {batch_observations} \n batch_actions: {batch_acts} \n ep_rews: {ep_rews} \n batch_rtgs: {batch_rtgs}') 
                # print(f'batch info dims: \n batch_observations: {batch_observations.shape} \n batch_actions: {batch_acts.shape} \n batch_rtgs: {batch_rtgs.shape}')
                
                model.learn_adjusted(batch_observations, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rewards) # TODO: update 
                print('updated model for episode based update')
            
            # reset each batch 
            batch_observations = []
            batch_actions = []
            ep_rews = []
            batch_log_probs = [] 
            batch_lens = [] 
            curr_index = 0
            batch_rewards = []
            
            msg = "update-complete"

            emitter.send(msg.encode('utf-8'))
            
            receiver.nextPacket()
            
        else: 
            print(f'skipping: {message}')
            receiver.nextPacket()
            
    if receiver_individual.getQueueLength()>0:  
        # message_individual = receiver_individual.getData().decode('utf-8')
        message_individual = receiver_individual.getString()
        # print('indiviudal msgs --', message_individual)

        if 'action-complete' in message_individual: 
            Agent.reward = float(message_individual.split(":")[1])

            # Use regular expression to find the dictionary portion
            pattern = r'{.*}'
            match = re.search(pattern, message_individual)
            dictionary = "" 

            if match:
                dictionary_string = match.group(0)
                dictionary = eval(dictionary_string)

            else:
                print("Dictionary not found in input string.")

            Agent.observation = np.concatenate((np.array(dictionary["agent"]),np.array(dictionary["ultrasonic"]), np.array(dictionary["ultrasonic_left"]), np.array(dictionary["ultrasonic_right"])))                 
            obs, rew, done = Agent.observation, Agent.reward, Agent.done

            if len(batch_observations) > 0 and online: 
                batch_observations = torch.tensor(batch_observations, dtype = torch.float)
                batch_acts = torch.tensor(batch_actions, dtype = torch.float)
                batch_log_probs = torch.tensor(batch_log_probs, dtype = torch.float)
                # print('intial batch_rewards - ', batch_rewards)
                batch_rewards.append(ep_rews)
                # print('final batch_rewards - ', batch_rewards)
                batch_rtgs = model.compute_rtgs(batch_rewards) # TODO: should be batch_rewards instead of ep_rews
                batch_lens = [600] # TODO: update so correct 
                print('converted to torch')
                print(f'info updates : \n batch_observations: {batch_observations.shape} \n batch_actions: {batch_acts.shape} \n ep_rews: {ep_rews} \n batch_rtgs: {batch_rtgs}') 
                
                
                model.learn_adjusted(batch_observations, batch_acts, batch_log_probs, batch_rtgs, batch_lens, 0) # TODO: update 
                print('updated model')
                
                # reset each batch 
                batch_observations = []
                batch_actions = []
                ep_rews = []
                batch_log_probs = [] 
                batch_lens = [] 
                curr_index = 0
                batch_rewards = []


            # ep_rews.append(rew)

            # update action and tell agent to execute 
            # curr_action = Agent.action  
            pos = np.array([population[given_id].getPosition()[0], population[given_id].getPosition()[1]])
            Agent.set_location(pos) 
            action, log_prob = model.get_action(Agent.observation) 
            Agent.action = action[0] 
            
            discretized_action = np.argmax(action).item() # TODO: might not be correct, index of the maximum value in your continuous vector as a discrete action
            curr_action = env._action_to_direction[discretized_action]
            # Agent.action = curr_action 

            # Agent.action = env._action_to_direction(action).tolist() 
            Agent.log_prob = log_prob 

            # if complete: batch_log_probs.append(log_prob) 
            # curr_action = Agent.action if not complete else action # TODO: must do next action once done with previous
            msg = 'agent_action:'+ str(curr_action[0]) + "," + str(curr_action[1])
            emitter_individual.send(msg.encode('utf-8'))
            receiver_individual.nextPacket()

        else: 
            #
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
    global batch_rewards

    global curr_index 
    global episode_length
    global cum_reward
   

    if robot.getTime() - prev_time > update_sec and curr_index <= num_updates_per_episode: # update every second 
        prev_time = robot.getTime()
        # print(f'batch_observations: {batch_observations}')
        # update info 
        batch_observations.append(Agent.observation)
        batch_actions.append(Agent.action)
        ep_rews.append(Agent.reward)
        batch_log_probs.append(Agent.log_prob)

        episode_length += 1
        cum_reward += Agent.reward
        # print(f'info updates : \n batch_observations: {batch_observations} \n batch_actions: {batch_actions} \n ep_rews: {ep_rews}') 

        # TODO: delete eventually .. testing network update (NOT WORKING YET)
        # if len(batch_observations) > 0 and online: 
        #     batch_observations = torch.tensor(batch_observations, dtype = torch.float)
        #     batch_acts = torch.tensor(batch_actions, dtype = torch.float)
        #     batch_log_probs = torch.tensor(batch_log_probs, dtype = torch.float)
        #     print('intial batch_rewards - ', batch_rewards)
        #     batch_rewards.append(ep_rews)
        #     print('final batch_rewards - ', batch_rewards)
        #     batch_rtgs = model.compute_rtgs(batch_rewards) # TODO: should be batch_rewards instead of ep_rews
        #     batch_lens = [600] # TODO: update so correct 
        #     print('converted to torch')
        #     print(f'info updates : \n batch_observations: {batch_observations.shape} \n batch_actions: {batch_acts.shape} \n ep_rews: {ep_rews} \n batch_rtgs: {batch_rtgs}') 
            
            
        #     model.learn_adjusted(batch_observations, batch_acts, batch_log_probs, batch_rtgs, batch_lens, 0) # TODO: update 
        #     print('updated model')
            
        #     # reset each batch 
        #     batch_observations = []
        #     batch_actions = []
        #     ep_rews = []
        #     batch_log_probs = [] 
        #     batch_lens = [] 
        #     curr_index = 0
        #     batch_rewards = []
            
            
        

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
                    
            
            
