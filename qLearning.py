''' Reinforcement learning
    @TODO: Gets stuck in same back and forth action sometimes --> tie breaker? best vs. randomng trash early..
    @TODO: Add battery life as its state as well!!! -- tends to run fast to end to avoid collecti
    @TODO: Test with prob of move ranges...look at how fast/slow convergence takes!
    @TODO: Current algo vs. random -- NEXT
    @TODO: Save Test boards... ----> DONE
    @TODO: Experience replay during training... 
    @TODO: Updating learning rate? ----> DONE
    @TODO: Paper summary and concepts?
    @TODO: Save best model..
    @TODO: Notebook for best model and tests
'''
from copy import deepcopy
from itertools import product
import sys

from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd 
from scipy import sparse

import roomba_sim

def positionToState(waterWorld):
    """ Map from waterWorld positions to state number
        number corresponds to [robot_position][trash_index] in binary format
        robot_position is binary number corresponding to robot index
        trash_index maps to binary number where each number maps to presence/absence of trash in specific board index
        Example:
        2*3 board maps to:
        -----
        0 1 2
        3 4 5
        -----
        If robot is in (2,1) and trash in (1,1):
        robot_position  --> 5   --> 101
        trash_index     --> 4   --> 000100
        state_number_binary --> [101][000100] --> return (int)    
    """
    if waterWorld.robot_pos == waterWorld.end_pos:
        # Completed!
        return -1

    x, y = waterWorld.robot_pos
    dim_x = waterWorld.dim_x
    dim_y = waterWorld.dim_y
    dim_xy = dim_x * dim_y

    robot_pos_int = x + y*dim_y
    robot_pos_binary = "{0:b}".format(robot_pos_int)

    trash_positions_index = [x + y*dim_y for (x,y) in waterWorld.trash_positions]
    trash_positions_binary = ['0']*dim_xy
    for index in trash_positions_index:
        trash_positions_binary[index] = '1'
    
    trash_positions_binary = "".join(trash_positions_binary)
    trash_positions_int = int(trash_positions_binary, 2)
    state_pos_binary = robot_pos_binary + trash_positions_binary
    state_int = int(state_pos_binary, 2)
    return state_int

def stateToPositions(state_int, dim_x, dim_y):
    """ Map from waterWorld positions to state number
    """
    dim_xy = dim_x * dim_y
    state_pos_binary = "{0:b}".format(state_int)
    trash_positions_binary = state_pos_binary[-1*dim_xy:]
    robot_pos_binary = state_pos_binary[:-1*dim_xy]
    if not robot_pos_binary:
        robot_pos_binary = "0"
        
    trash_positions_int = int(trash_positions_binary, 2)

    robot_pos_int = int(robot_pos_binary, 2)
    robot_y = robot_pos_int // dim_y
    robot_x = robot_pos_int % dim_y

class QLearning:
    ''' Q-learning algorithm 
    '''
    def __init__(self, 
        discount=0.95, 
        learning_rate=0.75, 
        num_iter=1000, 
        iter_length=10000, 
        #update_iters=100,
        ):
        ''' CLass init
            :param discount: discount factor
            :param learning_rate: learning rate for updating qvalues
        '''
        self.discount = discount
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.iter_length = iter_length
        self.update_iters = roomba_sim.BATTERY_LIFE #update_iters
        self.total_rewards = -np.inf

        self.n_states = -1
        # Map from action --> actionIndex
        self.n_actions = -1
        # Matrix from (stateIndex, actionIndex) --> qvalue
        self.qMap = None
        #np.random.seed(13)


    def init(self, waterWorld, qMap=None):
        ''' Init state/action and qvalues based on input dataframe!
        '''
        self.dim_x = waterWorld.dim_x
        self.dim_y = waterWorld.dim_y
        self.dim_xy = self.dim_x * self.dim_y
        self.dim_trash = 2**(self.dim_xy)
        self.num_trash = waterWorld.num_trash

        self.n_states = self.dim_xy * self.dim_trash * 2

        self.original_trash_positions = deepcopy(waterWorld.trash_positions)
        
        self.n_actions = 4
        #self.qMap = csr_matrix((self.n_states, self.n_actions))#np.zeros((self.n_states, self.n_actions))
        if qMap is not None:
            # Initialize based on input Qmap
            self.qMap = qMap
        else:
            self.qMap = np.zeros((self.n_states, self.n_actions), dtype=np.float16)
            #self.qMap = coo_matrix((self.n_states, self.n_actions))

    def update(self, s, a, r, sp):
        ''' Update Q-values based on 
            :param s: input state
            :param a: input action
            :param r: output reward for s->a
            :param sp: output state for s->a
        '''
        qVal = self.qMap[s, a]
        
        delta = self.learning_rate * (r + self.discount*np.max(self.qMap[sp, :]) - qVal)
        self.qMap[s, a] += delta


    def selectAction(self, s):
        """ Select best action based on current qvalues with tie breaker
        """
        qVals = self.qMap[s, :]
        best_action = np.random.choice(np.flatnonzero(qVals == qVals.max()))
        return best_action


    def train(self, waterWorld, qMap=None):
        ''' Start Q-learning algorithm 
        '''
        self.init(waterWorld, qMap)
        
        rewards = []
        for i in range(self.num_iter):
            epReward = self.runEpisode(waterWorld)
            waterWorld.refresh()
            rewards.append(epReward)
            self.learning_rate = self.discount * self.learning_rate


    def test(self, waterWorld, qMap=None, bPrint=False):
        ''' Start Q-learning algorithm 
        '''
        self.init(waterWorld, qMap)
        
        rewards = []
        for i in range(self.num_iter):
            epReward = self.runEpisode(waterWorld, bUpdate=False, bPrint=bPrint)
            waterWorld.refresh()
            rewards.append(epReward)
        avg_rewards = np.mean(rewards)
        return avg_rewards       

    def updateSteps(self, rewards_collection):
        ''' Update all rewards in reverse!!
        '''
        while rewards_collection:
            (state, action, reward, nextState) = rewards_collection.pop()
            self.update(state, action, reward, nextState)

    
    def runEpisode(self, waterWorld, bUpdate=True, bPrint=False):
        """ Perform one episode from input start state
        """

        discounted_rewards = 0

        rewards_collection = []
        if bPrint:
            print ("<< Running one board >>")
            print ("---------------------------------------------------------------")
        for i in range(self.iter_length):
            if bPrint:
                waterWorld.print_grid()

            state = positionToState(waterWorld)
            if state == -1:
                #print (f"Found all trash in {i} steps with rewards: {discounted_rewards:4f}")
                break
            action = self.selectAction(state)
            reward = waterWorld.do_action(action)
            nextState = positionToState(waterWorld)
            if i==(self.iter_length - 1):
                # Reached the end!
                print ("\tReached max timeout!")
                reward = roomba_sim.REWARD_TIMEOUT
            rewards_collection.append((state, action, reward, nextState))
            discounted_rewards = 1*discounted_rewards + reward 

            if bUpdate and ((i+1) % self.update_iters):
                self.updateSteps(rewards_collection)

        if bPrint:
            print ("---------------------------------------------------------------\n")

        # Update remaining rewards
        if bUpdate:
            self.updateSteps(rewards_collection)

        self.total_rewards = discounted_rewards

        return discounted_rewards