from controller import Supervisor, Node, Keyboard, Emitter, Receiver, Field
import math 
from robot_pop import * 
import random

# ensure that we can access utils package to streamline tasks 
import sys 
sys.path.append('../../')
import utils.environment as env_mod 
from utils.rl_agent import *

from math import pi

"""
Main supervisor base 
Optimization algorithm - Collaboration-oriented 
Angel Sylvester 2022
"""

ind_sup = []

# population-level statistics 
batch_observations = []
batch_rewards = []
batch_actions = []
population_array = []
num_agents = 2

# hyperparameters 
timesteps_per_batch = 4800 
max_timesteps_per_episode = 600 
n_updates_per_iteration = 5
lr = 0.005
gamma = 0.95
clip = 0.2
total_timesteps = 200_000_000
act_list = []
updating = True

# columns = 'agent id' + ',time step' + ',fitness' + ',xpos'+ ',ypos' + ',num col' + ',genotype' + ',potential time'

# # global collected_count 
# collected_count = []

# # genetic algorithm-specific parameters 
# num_generations = 10
# simulation_time = 30
# trials = 25
# curr_trial = 0 
robot_population_sizes = [1]
# gene_list = ['control speed 10', 'energy cost 5', 'food energy 30', 'observations thres 5']
# curr_size = robot_population_sizes[0]
env_type = "single source" # "power law"
sim_type = "random"
communication = True
high_dense = True

# # collected counts csv generation 
# overall_f = open(f'../../graph-generation/collection-data/overall-df-{sim_type}-{curr_size}-comm_{communication}-dense_{high_dense}.csv', 'w')
# overall_columns = 'trial' + ',time' + ',objects retrieved' + ',size' + ',type' + ',potential time' + ',total elapsed'
# overall_f.write(str(overall_columns) + '\n')
# overall_f.close()
# overall_f = open(f'../../graph-generation/collection-data/overall-df-{sim_type}-{curr_size}-comm_{communication}-dense_{high_dense}.csv', 'a')

# # for individual robot, statistics about strategy taken over time & individual collision info 
# strategy_f = open(f"../../graph-generation/collision-data/ga-info-{sim_type}-{curr_size}-comm_{communication}-dense_{high_dense}.csv", 'w')
# strategy_f.write('agent id'+ ',time step' + ',straight' + ',alternating-left' + ',alternating-right' + ',true random' + ',time since last block' + ',num encounters' + ',size' + ',fitness'+ ',size'+ ',type' + ',trial' + ',collected' + ',genotype' + ',num better' + ',pos x' + ',pos y' + '\n')
# strategy_f.close()


# # statistics collected 
population = []
# found_list = []
# total_found = 0
block_list = []
# reproduce_list = []
r_pos_to_generate = []
b_pos_to_generate = []
# b_pos_to_generate_alternative = []

# # set-up robot 
TIME_STEP = 32
robot = Supervisor()  # create Supervisor instance
emitter = robot.getDevice("emitter")
emitter.setChannel(1)
receiver = robot.getDevice("receiver") 
receiver.enable(TIME_STEP)
receiver.setChannel(2) 
# taken = False # getting second child assigned 
# updated = False
# fit_update = False 
# start = 0 

# prev_msg = ""
seed_val = 11
# random.seed(seed_val)
# id_msg = ""

emitter_individual = robot.getDevice("emitter_processor")
emitter_individual.setChannel(5)
# assessing = False 
repopulate = False # keep False for now 
# phase_one_times = [620]
central = True

# # generate envs 
curr_env = env_mod.Environment(env_type=env_type, seed = seed_val)

# set up environments 
def generate_robot_central(num_robots):
    global population
    global r_pos_to_generate
    global prev_msg 
    global id_msg
    global population_array

    curr_msg = str("size-" + str(num_robots))
    # if curr_msg != prev_msg: 
    emitter.send(str("size-" + str(num_robots)).encode('utf-8'))
    emitter_individual.send(str("size-" + str(num_robots)).encode('utf-8'))
    prev_msg = curr_msg
    
    if len(population) != 0: 
    
        for r in population: 
            r.remove()
             
    population = []
    fitness_scores = []
    overall_fitness_scores = []
    collected_count = []
    pairs = []
    id_msg = "ids"
        
    for i in range(num_robots):
        rootNode = robot.getRoot()
        rootChildrenField = rootNode.getField('children')
        if i == 0: 
            robot_name = "k0" 
        else: 
            robot_name = "k0(" + str(i) + ")"
        
        # Import the khepera PROTO with a dynamically set robot name
        import_string = 'khepera_pure_rl {{ robotName "{0}" }}'.format(robot_name)
        rootChildrenField.importMFNodeFromString(-1, import_string)
        rec_node = rootChildrenField.getMFNode(-1)
        t_field = rec_node.getField('translation')
        pose = [round(random.uniform(0.3, -0.3),2), round(random.uniform(0.3, -0.3) ,2), 0.02]
        while pose in r_pos_to_generate: # remove any duplicates
            pose = [round(random.uniform(0.3, -0.3),2), round(random.uniform(0.3, -0.3) ,2), 0.02]
        r_pos_to_generate.append(pose)
        t_field.setSFVec3f(pose)

        # create RLAgent class + set information
        agent = RLAgent()
        agent.set_id(robot_name)
        agent.set_location(pose) # TODO: might not be correct 
        population_array.append(agent)

        # sets up metrics 
        population.append(rec_node)
        id_msg += " " + str(rec_node.getId()) 

# set up environments 
def generate_robot_edge(num_robots, right = False):
    global population
    global columns 
    global r_pos_to_generate
    global prev_msg 
    global id_msg
    global population_array
    
    curr_msg = str("size-" + str(num_robots))
    if curr_msg != prev_msg: 
        emitter.send(str("size-" + str(num_robots)).encode('utf-8'))
        emitter_individual.send(str("size-" + str(num_robots)).encode('utf-8'))
        prev_msg = curr_msg
    
    if len(population) != 0: 
    
        for r in population: 
            r.remove()
             
    population = []
    fitness_scores = []
    overall_fitness_scores = []
    collected_count = []
    pairs = []
    id_msg = "ids"
        
    for i in range(num_robots):
        rootNode = robot.getRoot()
        rootChildrenField = rootNode.getField('children')
        if i == 0: 
            robot_name = "k0" 
        else: 
            robot_name = "k0(" + str(i) + ")"
        
        # Import the khepera PROTO with a dynamically set robot name
        import_string = 'khepera_pure_rl {{ robotName "{0}" }}'.format(robot_name)
        rootChildrenField.importMFNodeFromString(-1, import_string)
        rec_node = rootChildrenField.getMFNode(-1)
        
        if right: 
            t_field = rec_node.getField('translation')
            pose = [round(random.uniform(-0.5, -0.9),2), round(random.uniform(-0.9, 0.9),2), 0.02]
            while pose in r_pos_to_generate: # remove any duplicates
                pose = [round(random.uniform(-0.5, -0.9),2), round(random.uniform(-0.9, 0.9),2), 0.02]
            r_pos_to_generate.append(pose)
            t_field.setSFVec3f(pose)
                # print(r_field)

        else: 
            t_field = rec_node.getField('translation')
            pose = [round(random.uniform(0.5, 0.9),2), round(random.uniform(-0.9, 0.9),2), 0.02]
            while pose in r_pos_to_generate: # remove any duplicates
                pose = [round(random.uniform(0.5, 0.9),2), round(random.uniform(-0.9, 0.9),2), 0.02]
            r_pos_to_generate.append(pose)
            t_field.setSFVec3f(pose)
                # print(r_field)            
        
        # create RLAgent class + set information
        agent = RLAgent()
        agent.set_id(robot_name)
        agent.set_location(pose) # TODO: might not be correct 
        population_array.append(agent)

        # sets up metrics 
        population.append(rec_node)
        id_msg += " " + str(rec_node.getId()) 


def regenerate_blocks(seed = None):
    global block_list
    global population 
    global r_pos_to_generate
    global b_pos_to_generate
    global curr_env

    global population 
    
    for obj in block_list: 
        obj.remove()
    
    block_list = []
    assert curr_env

    for i in range(len(r_pos_to_generate)):
        population[i].getField('translation').setSFVec3f(r_pos_to_generate[i])
    
    if seed == 15 and curr_env.seed != 15: 
        curr_env = env_mod.Environment(env_type=env_type, seed = seed)
        b_pos_to_generate = curr_env.generate_blocks()

    if len(b_pos_to_generate) == 0:
        b_pos_to_generate = curr_env.generate_blocks()

    for i in b_pos_to_generate: 
        rootNode = robot.getRoot()
        rootChildrenField = rootNode.getField('children')
        block_name = "red block" + str(i)
        # Import the robot PROTO with a dynamically set block
        import_string = 'block {{ blockName "{0}" }}'.format(block_name)
        rootChildrenField.importMFNodeFromString(-1, import_string)
        # rootChildrenField.importMFNode(-1, '../las_supervisor/cylinder-obj.wbo') 
        rec_node = rootChildrenField.getMFNode(-1)
    
        t_field = rec_node.getField('translation')
        t_field.setSFVec3f(i) 
        
        r_field = rec_node.getField('rotation')
        if r_field.getSFRotation() != [0, 0, -1, 0]:
            r_field.setSFRotation([0, 0, -1, 0])
            
        block_list.append(rec_node)
            
    for rec_node in block_list: # set to be upright
        r_field = rec_node.getField('rotation')
        if r_field.getSFRotation() != [0, 0, -1, 0]:
            r_field.setSFRotation([0, 0, -1, 0])
            
# calculates angle normal to current orientation 
def calc_normal(curr_angle): 

    if (curr_angle + round(pi/2, 2) <= round(pi, 2) and curr_angle <= round(pi, 2) and curr_angle >= 0): 
        return round(curr_angle + round(pi/2, 2), 2)
    
    elif (curr_angle + round(pi/2, 2) > round(pi, 2) and curr_angle < round(pi, 2) and curr_angle > 0): 
        diff = round(pi/2, 2) - (round(pi,2) - curr_angle) 
        return round((-1*round(pi/2, 2) + diff),2)
    
    elif (curr_angle + round(pi/2, 2) < 0 and curr_angle < 0): 
        return round((-1*round(pi, 2) + curr_angle + round(pi/2, 2)),2)
        
    elif (curr_angle + round(pi/2, 2) >= 0 and curr_angle <= 0): 
        diff = abs(round(pi/2, 2) - curr_angle) 
        return round(diff,2) 
        
    elif (curr_angle == round(pi,2)): # handle edge case that seems to only happen w/exactly 3.14 (never broke before because never quite at 3.14????)
        return round(-1*round(pi/2, 2),2)

            
def message_listener():
    global total_found 
    global collected_count 
    global found_list
    global pop_genotypes
    global reproduce_list 
    global population
    global curr_size
    global overall_fitness_scores
    global start 
    global simulation_time
    global prev_msg 
    global updating

    if receiver.getQueueLength()>0 and (robot.getTime() - start < simulation_time):
        # message = receiver.getData().decode('utf-8')
        # print('supervisor msgs --', message) 
        message = receiver.getString()
        
        if message[0] == "$": # handles deletion of objects when grabbed
            obj_node = robot.getFromId(int(message.split("-")[1]))
            
            if obj_node is not None:
                r_node_loc = population[int(message.split("-")[0][1:])].getField('translation').getSFVec3f()
                t_field = obj_node.getField('translation')
                t_node_loc = t_field.getSFVec3f()
                
                if (math.dist(r_node_loc, t_node_loc) < 0.15):
                    if repopulate: 
                        # will be placed somewhere random 
                        side = random.randint(0,1)
                        if side == 1:
                            t_field.setSFVec3f([round(random.uniform(-0.5, -0.9),2), round(random.uniform(-0.9, 0.9),2), 0.02]) 
                        else: 
                            t_field.setSFVec3f([round(random.uniform(0.5, 0.9),2), round(random.uniform(-0.9, 0.9),2), 0.02])   
                    else:
                        t_field.setSFVec3f([-0.9199,-0.92, 0.059]) 
                    # obj_node.remove()
                    # remove redundant requests 
                    if obj_node not in found_list:
                        total_found += 1
                        if not repopulate: 
                            found_list.append(obj_node)
                            
                        collected_count[int(message.split("-")[0][1:])] = collected_count[int(message.split("-")[0][1:])] + 1
                        msg = "%" + message[1:]
                        if prev_msg != msg: 
                            emitter.send(str(msg).encode('utf-8'))
                            prev_msg = msg

        if 'update-complete' in message: 
            updating = False 
            receiver.nextPacket()
            
        else: 
            receiver.nextPacket() 
            
    elif (robot.getTime() - start > simulation_time and prev_msg != 'clean finish'):
    # if over time would want to reset 
        msg = 'cleaning'
        if prev_msg != msg: 
            emitter.send('cleaning'.encode('utf-8'))
            prev_msg = msg
        while receiver.getQueueLength()>0: 
            receiver.nextPacket()
        msg = 'clean finish'
        if prev_msg != msg: 
            emitter.send('clean finish'.encode('utf-8'))
            prev_msg = msg 
        


def rollout():
    global population 
    global max_timesteps_per_episode

    # batch data 
    # batch_obs = []
    # batch_acts = []
    # batch_log_probs = []
    # batch_rews = []
    # batch_rtgs = []
    # batch_lens = []

    # ep_rews = 0 
    t = 0 

    while t < timesteps_per_batch: 
        # ep_rews = []
        # obs = [i.getField('translation') for i in population]
        # done = False

        run_seconds(max_timesteps_per_episode) # gather info from sim for each agent 
        t += max_timesteps_per_episode

        # for ep_t in range(max_timesteps_per_episode): 
        #     # would want to run sim for this time 
        #     t += 1 
            # batch_obs.append[obs] # list of agent obs (positions)

            # # request to calc action (for each agent)
            # msg = 'action-request'
            # emitter.send(msg.encode('utf-8'))

        msg = 'episode-complete' # will reset agent 
        emitter.send(msg.encode('utf-8'))

        # TODO: reset env here correctly 
        regenerate_blocks(seed = 11)

    # return batch_obs, batch_acts, batch_log_probs, batch_lens # TODO: actually use these? 



# normal loop for training 
def run_optimization():
    global robot_population_sizes
    global ind_sup
    global updating 
    
    for size in robot_population_sizes:
        curr_size = size  

        # initialized robots to be placed in the environment 
        if central: 
            generate_robot_central(size)
        else: 
            generate_robot_edge(size) # set True to switch
        # generate blocks for collection task    
        regenerate_blocks(seed = 11) 

        for i in range(len(population)):
            ### generate supervisor for parallelization ####
            
            rootNode = robot.getRoot()
            rootChildrenField = rootNode.getField('children')
            robot_name = "Tinkerbots(" + str(i) + ")"
            # Import the khepera PROTO with a dynamically set robot name
            import_string = 'khepera_individual_pure_rl {{ robotName "{0}" }}'.format(robot_name)
            rootChildrenField.importMFNodeFromString(-1, import_string)
            individual = rootChildrenField.getMFNode(-1)
            ind_sup.append(individual)
            individual.getField('translation').setSFVec3f([0, 2, 0])
            
        emitter_individual.send(id_msg.encode('utf-8'))

        t_so_far = 0 # ts simulated so far 
        i_so_far = 0 # iterations so far 

        while t_so_far < total_timesteps: 

            # perform batch_rollout 
            rollout()

            updating = True 
            msg = 'updating network'
            emitter.send(msg.encode('utf-8'))

            while updating: 
                message_listener()

            # t_so_far += np.sum(batch_lens) # TODO: find way to extract this

            # TODO: need way to pause simulation while this calculation is happening 
    
    return 


# runs simulation for designated amount of time 
def run_seconds(t,waiting=False):
    global pop_genotypes
    global fitness_scores
    global overall_fitness_scores
    global updated
    global fit_update 
    global block_list
    global start 
    global prev_msg 
    
    n = TIME_STEP / 1000*32 # convert ms to s 
    start = robot.getTime()
    new_t = round(t, 1)
    
    while robot.step(TIME_STEP) != -1:
        # run robot simulation for 30 seconds (if t = 30)
        increments = TIME_STEP / 1000
        
        if robot.getTime() - start > new_t: 
            msg = 'return_fitness'
            # if prev_msg != msg: 
            message_listener(robot.getTime()) # will clear out msg until next gen 
            print('requesting fitness')
            emitter.send('return_fitness'.encode('utf-8'))
            prev_msg = msg 
            # print('requesting fitness')
            break 
 
    return 


def main(): 
    run_optimization()
    # save_progress()
         
main()


# not useful code here 
   
# # will use selected partners from each robot and reproduce with that corresponding index, and update population at the end of gen          
# def update_geno_list(genotype_list): 
#     global fitness_scores
#     global overall_fitness_scores
#     global pop_genotypes 
#     global gene_list
#     global taken 
#     global updated 
#     global population 
#     global pairs 
    
#     # only makes executive changes if it's better off to just re-randomize population   
#     # print('getting overall fitness scores --', overall_fitness_scores)
    
#     # if max(overall_fitness_scores) <= 0:
#     cp_genotypes = pop_genotypes.copy()
#     for i in range(len(population)):
#         if i not in pairs: 
#             g = create_individal_genotype(gene_list)
#             new_offspring = reproduce(cp_genotypes[i], cp_genotypes[i])
#             # print('updated genolist --', g)
#             pop_genotypes[i] = new_offspring
                 
#     # update parameters to hopefully improve performance
#     # print('curr population --', population, len(population))
    
#     # fitness_scores = []
#     fs_msg = 'fitness-scores ' + " ".join(fitness_scores)
#     emitter_individual.send(fs_msg.encode('utf-8'))
#     fitness_scores = ["!" for i in range(len(population))]
#     overall_fitness_scores = ["!" for i in range(len(population))]
#     pairs = ["!" for i in range(len(population))]
#     fit_update = False 
#     # print('gene pool updated') 
#     updated = True

# # fitness function for each individual robot 
# def eval_fitness(time_step):
#     global pop_genotypes 
#     global fitness_scores 
#     global fit_update
#     global population 
#     global overall_fitness_scores
            
#     if '!' not in fitness_scores and '!' not in overall_fitness_scores: 
#         # receiver.nextPacket()
#         # print('will update gene pool --')
#         fit_update = True 
#         update_geno_list(pop_genotypes)

# # TODO: send genotype to each individual 
# def reset_genotype():
#     index = 0 
#     global population 
#     global pop_genotypes 
#     global prev_msg 
#     pop_genotypes = []
    
#     for i in range(len(population)):
#         genotype = initial_genotypes[i]
#         pop_genotypes.append(genotype)
#         msg = str("#"+ str(index) + str(genotype))
#         if prev_msg != msg: 
#             emitter.send(str("#"+ str(index) + str(genotype)).encode('utf-8'))
#             prev_msg = msg
#         index +=1 
          
# def initialize_genotypes(size):
#     global initial_genotypes
#     global gene_list 
#     global pop_genotypes 
#     # initial_geno_txt = open('initial_genotype.txt', 'w')
    
#     lines = []
#     pop_genotypes = []
#     for r in range(size):
#         new_geno = create_individal_genotype(gene_list)
#         # print(new_geno)
#         initial_genotypes.append(new_geno)
#         pop_genotypes.append(new_geno)
                
    
# # def save_progress():
#     # global overall_df
#     # overall_f.close()
#     # print('progress saved to csv')
#     # emitter.send('sim-complete'.encode('utf-8'))

#     # for i in range(20): 
#         # rootNode = robot.getRoot()
#         # rootChildrenField = rootNode.getField('children')
#         # rootChildrenField.importMFNode(-1, '../las_supervisor/cylinder-obj.wbo') 
#         # rec_node = rootChildrenField.getMFNode(-1)
    
#         # t_field = rec_node.getField('translation')
#         # t_field.setSFVec3f([round(random.uniform(-0.5, -0.9),2), round(random.uniform(-0.9, 0.9),2), 0.02]) 
#         # block_list.append(rec_node)        
      
#     # for i in range(20): 
#         # rootNode = robot.getRoot()
#         # rootChildrenField = rootNode.getField('children')
#         # rootChildrenField.importMFNode(-1, '../las_supervisor/cylinder-obj.wbo') 
#         # rec_node = rootChildrenField.getMFNode(-1)
        
#         # t_field = rec_node.getField('translation')
#         # t_field.setSFVec3f([round(random.uniform(0.5, 0.9),2), round(random.uniform(-0.9, 0.9),2), 0.02]) 
#         # block_list.append(rec_node)  








        # total_elapsed = 600
            
        # num_generations = total_elapsed // simulation_time
        
        # for i in range(trials): 
        #     print('beginning new trial', i)
        #     msg = 'generation-complete '+ ' '.join(pop_genotypes)
            
        #     emitter_individual.send(str(msg).encode('utf-8'))
            
            
        #     for rec_node in population: 
        #         r_field = rec_node.getField('rotation')
        #         if r_field.getSFRotation() != [0, 0, -1, 0]:
        #             r_field.setSFRotation([0, 0, -1, 0])
                
                
        #     for gen in range(num_generations): 
        #         updated = False 
                
        #         print('number in population', len(population))
        #         print('number of genotypes',  len(pop_genotypes), 'for size: ', size)

        #         run_seconds(simulation_time) 
                
        #         for rec_node in population: 
        #             r_field = rec_node.getField('rotation')
        #             if r_field.getSFRotation() != [0, 0, -1, 0]:
        #                 r_field.setSFRotation([0, 0, -1, 0])
                          
        #         print('found genotypes')
        #         print('new generation starting -')
        #         reproduce_list = []

        #     overall_f.write(str(i) + ',' + str(robot.getTime()) + ',' + str(total_found) + ',' + str(size)+ ',' + 'ga' + ',' + str(20) + ',' + str(total_elapsed) + '\n')    
        #     overall_f.close()
        #     overall_f = open(f'../../graph-generation/collection-data/overall-df-{sim_type}-{curr_size}-comm_{communication}-dense_{high_dense}.csv', 'a')
        #     print('items collected', total_found)
        #     curr_trial = i + 1
        #     if assessing and curr_trial % 2 == 0:
        #         regenerate_blocks(seed = 11)
        #         reset_genotype() 
        #     elif assessing and curr_trial % 2 != 0: 
        #         regenerate_blocks(seed = 15)   
        #     else: 
        #         regenerate_blocks(seed = 11)
        #         reset_genotype() 
            
        #     total_found = 0 
        #     reproduce_list = []
        #     found_list = []
        #     msg = 'trial' + str(i)
        #     emitter.send(msg.encode('utf-8')) 
        #     prev_msg = msg
            
        # for node in ind_sup: 
        #     node.remove() 
            
        # run_seconds(5)  