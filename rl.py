''' Reinforcement learning

'''
from copy import deepcopy
from itertools import product
import sys
import time

from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd 
from scipy.sparse import csr_matrix, lil_matrix

import roomba_sim

class QLearning:
    ''' Q-learning algorithm 
    '''
    def __init__(self, 
        discount=0.95, 
        learning_rate=0.1, 
        num_iter=1000, 
        iter_length=10000, 
        update_iters=32,
        ):
        ''' CLass init
            :param discount: discount factor
            :param learning_rate: learning rate for updating qvalues
        '''
        self.discount = discount
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.iter_length = iter_length
        self.update_iters = update_iters
        self.total_rewards = -np.inf

        self.n_states = -1
        # Map from action --> actionIndex
        self.n_actions = -1
        # Matrix from (stateIndex, actionIndex) --> qvalue
        self.qMap = None
        #np.random.seed(13)


    def positionToState(self, waterWorld):
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
        if len(waterWorld.trash_positions) == 0:
            # Completed!
            return -1

        x, y = waterWorld.robot_pos
        robot_pos_int = x + y*self.dim_y
        robot_pos_binary = "{0:b}".format(robot_pos_int)

        trash_positions_index = [x + y*self.dim_y for (x,y) in waterWorld.trash_positions]
        trash_positions_binary = ['0']*self.dim_xy
        for index in trash_positions_index:
            trash_positions_binary[index] = '1'
        
        trash_positions_binary = "".join(trash_positions_binary)
        trash_positions_int = int(trash_positions_binary, 2)
        state_pos_binary = robot_pos_binary + trash_positions_binary
        state_int = int(state_pos_binary, 2)
        return state_int

    def stateToPositions(self, state_int):
        """ Map from waterWorld positions to state number
        """
        state_pos_binary = "{0:b}".format(state_int)
        trash_positions_binary = state_pos_binary[-1*self.dim_xy:]
        robot_pos_binary = state_pos_binary[:-1*self.dim_xy]
        if not robot_pos_binary:
            robot_pos_binary = "0"
            
        trash_positions_int = int(trash_positions_binary, 2)

        robot_pos_int = int(robot_pos_binary, 2)
        robot_y = robot_pos_int // self.dim_y
        robot_x = robot_pos_int % self.dim_y

    def init(self, waterWorld, qMap=None):
        ''' Init state/action and qvalues based on input dataframe!
        '''
        self.dim_x = waterWorld.dim_x
        self.dim_y = waterWorld.dim_y
        self.dim_xy = self.dim_x * self.dim_y
        self.dim_trash = 2**(self.dim_xy)
        self.num_trash = waterWorld.num_trash

        self.n_states = self.dim_xy * self.dim_trash

        self.original_trash_positions = deepcopy(waterWorld.trash_positions)
        
        self.n_actions = 4
        #self.qMap = csr_matrix((self.n_states, self.n_actions))#np.zeros((self.n_states, self.n_actions))
        if qMap is not None:
            # Initialize based on input Qmap
            self.qMap = qMap
        else:
            self.qMap = np.zeros((self.n_states, self.n_actions))

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
        #best_action = qVals.argmax()
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
            #print (f"Reward for {i} iteration: {epReward:.4f}")

    def test(self, waterWorld, qMap=None):
        ''' Start Q-learning algorithm 
        '''
        self.init(waterWorld, qMap)
        
        rewards = []
        for i in range(self.num_iter):
            epReward = self.runEpisode(waterWorld, bUpdate=False)
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

    
    def runEpisode(self, waterWorld, bUpdate=True):
        """ Perform one episode from input start state
        """

        discounted_rewards = 0

        rewards_collection = []
        for i in range(self.iter_length):
            state = self.positionToState(waterWorld)
            if state == -1:
                #print (f"Found all trash in {i} steps with rewards: {discounted_rewards:4f}")
                break
            action = self.selectAction(state)
            reward = waterWorld.do_action(action)
            nextState = self.positionToState(waterWorld)
            rewards_collection.append((state, action, reward, nextState))
            discounted_rewards = 1*discounted_rewards + reward #discount?

            if bUpdate and ((i+1) % self.update_iters):
                self.updateSteps(rewards_collection)

        # Update remaining rewards
        if bUpdate:
            self.updateSteps(rewards_collection)

        self.total_rewards = discounted_rewards

        return discounted_rewards


def TestRoomba(dim_x, dim_y, max_trash_count, qMap, board_count=100):
    total_rewards = 0
    st0 = np.random.get_state()
    
    np.random.seed(1313) # Fix state for fix boards!
    for index in range(board_count):
        trash_num = np.random.randint(low=1, high=max_trash_count)
        env = roomba_sim.State(dim_x, dim_y, trash_num)
        # Init policy
        model = QLearning(num_iter=1, iter_length=1000)
        model.test(env, qMap)
        reward = model.total_rewards / trash_num # scale rewards on total number of trash!
        total_rewards += reward
    mean_rewards = total_rewards / board_count

    print (f"avg_rewards for test: {mean_rewards}\n")

    np.random.set_state(st0) # Set to original state...
    return mean_rewards

def TrainRoomba(dim_x, dim_y, max_trash_count, board_count=10000, test_index=100, test_boards=100):
    """ Train roomba
    """
    qMap = None

    overall_rewards = []
    np.random.seed(13)
    random_seeds = list(range(board_count))
    np.random.shuffle(random_seeds)
    
    for index in range(board_count):
        st = time.time()
        np.random.seed(random_seeds[index])
        trash_num = np.random.randint(low=1, high=max_trash_count)
        env = roomba_sim.State(dim_x, dim_y, trash_num)
        # Init policy
        model = QLearning(num_iter=100, iter_length=1000)
        model.train(env, qMap)
        qMap = model.qMap

        print (f"\nTrained on board {index} with {trash_num} trash!, reward: {model.total_rewards/trash_num} in {time.time()-st:.2f}s")

        if (index+1)%test_index == 0:
            rewards = TestRoomba(dim_x, dim_y, max_trash_count, qMap, board_count=test_boards)
            overall_rewards.append(rewards)

    
    # Final test
    rewards = TestRoomba(dim_x, dim_y, max_trash_count, qMap, board_count=test_boards)
    overall_rewards.append(rewards)

    plt.plot(overall_rewards)
    plt.show()

def main():
    DIM_X = 5
    DIM_Y = 5
    DIM_XY = DIM_X * DIM_Y
    MAX_TRASH_PERC = 0.25
    MAX_TRASH_COUNT = int(DIM_XY * MAX_TRASH_PERC) + 1

    TrainRoomba(DIM_X, DIM_Y, MAX_TRASH_COUNT, board_count=10000, test_index=10)
    

if __name__ == "__main__":
    main() 