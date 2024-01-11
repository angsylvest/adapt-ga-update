from controller import Supervisor, Node, Keyboard, Emitter, Receiver, Field
import math 
from robot_pop import * 
import random

from utils.rl_agent import *
from utils.ppo import * 
from utils.nn import * 
from utils.rl_wrapper import * 

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
Agent.set_id(assigned_r_name)
Agent.set_location((population[given_id].getPosition()[0], population[given_id].getPosition()[1]))

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
curr_action = []
complete = True 

ep_rews = []
ep_actions = []
log_probs = []
batch_observations = []
batch_rewards = []
batch_actions = []
batch_rtgs = []
batch_lens = []



# based off given id of robot assigned 
def find_nearest_robot_genotype(r_index):
    global population 
    global reproduce_list 
    global collected_count 
    global pairs 
    global overall_fitness_scores
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
                other_fitness = overall_fitness_scores[i]
                other_index = i
            elif dis < curr_dist: 
                closest_neigh = str(population[i].getId())
                curr_dist = dis 
                other_fitness = overall_fitness_scores[i]
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

    if receiver.getQueueLength()>0:
        message = receiver.getString()
        ## access to robot id for node acquisition    
        if 'ids' in message: 
            id_msg = message.split(" ")[1:]
            population = []
            
            for id in id_msg: # will convert to nodes to eventual calculation 
                node = robot.getFromId(int(id))
                population.append(node)
                
            receiver.nextPacket() 
            
        elif 'size' in message: 
            population = []
            size = int(message[4:]) 
            receiver.nextPacket()

        elif 'action-request' in message: 
            curr_action = Agent.action
            action, log_prob = model.get_action()

            if complete: log_probs.append(log_prob) 
            curr_action = Agent.action if not complete else action # TODO: must do next action once done with previous

            ep_actions.append(curr_action) 

            receiver.nextPacket()

        elif 'episode-complete' in message: 
            # TODO: reset agent 
            Agent.reset() 
            
            receiver.nextPacket()

        elif 'updating-network' in message: 
            # TODO: make network update pause sim or stop further collection of statistics 
            model.learn_adjusted(batch_observations, ep_actions, log_probs, batch_rtgs, batch_lens) # TODO: update 
            msg = 'update-complete'
            emitter.send(msg.encode('utf-8'))
            
        else: 
            receiver.nextPacket()
            
    if receiver_individual.getQueueLength()>0:  
        # message_individual = receiver_individual.getData().decode('utf-8')
        message_individual = receiver_individual.getString()

        if 'action-complete' in message_individual: 
            obs, rew, done, _ = Agent.observation, Agent.reward, Agent.done
            ep_rews.append(rew)
            receiver_individual.nextPacket()
        else: 
            # print('indiviudal msgs --', message_individual)
            receiver_individual.nextPacket()


# TODO: make adjusted function for batch_obs, back_acts, batch_log_probs, batch_rtgs, batch_lens
     
    
def run_optimization():
    global updated
    global total_found 
    global collected_count
    global population 
    global prev_msg 
    global timestep 
    
    while robot.step(timestep) != -1: 
        message_listener(timestep)        
     
  
def main(): 
    run_optimization()
         
main()
                    
            
            
