''' Reinforcement learning

'''
from copy import deepcopy
from itertools import product
import sys
import time

from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd 

import roomba_sim

class QLearning:
    ''' Q-learning algorithm 
    '''
    def __init__(self, discount=0.5, learning_rate=0.5, num_iter=1000, iter_length=10000):
        ''' CLass init
            :param discount: discount factor
            :param learning_rate: learning rate for updating qvalues
        '''
        self.discount = discount
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.iter_length = iter_length

        self.n_states = -1
        # Map from action --> actionIndex
        self.n_actions = -1
        # Matrix from (stateIndex, actionIndex) --> qvalue
        self.qMap = None
        np.random.seed(13)

    def positionToState(self, waterWorld):
        """ Map from waterWorld positions to state number
        """
        if len(waterWorld.trash_positions) == 0:
            return -1
        x, y = waterWorld.robot_pos
        trash_positions_current = dict.fromkeys(waterWorld.trash_positions, True)

        trash_index_key = tuple([1 if pos in trash_positions_current else 0 for pos in self.original_trash_positions])
        trash_index = self.trash_pos_map[trash_index_key]

        state_int = trash_index*self.dim_xy + (x + y*self.dim_y)
        return state_int

    def init(self, waterWorld):
        ''' Init state/action and qvalues based on input dataframe!
        '''
        self.dim_x = waterWorld.dim_x
        self.dim_y = waterWorld.dim_y
        self.dim_xy = self.dim_x * self.dim_y
        self.num_trash = waterWorld.num_trash

        self.n_states = self.dim_x * self.dim_y * (2**(self.num_trash))

        self.original_trash_positions = deepcopy(waterWorld.trash_positions)
        self.trash_pos_map = {}
        i = 0
        for c in product(*[range(2) for _ in range(len(self.original_trash_positions))]):
            self.trash_pos_map[c] = i
            i+=1

        self.n_actions = 4
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
        return best_action


    def train(self, waterWorld):
        ''' Start Q-learning algorithm 
        '''
        self.init(waterWorld)
        
        rewards = []
        for i in range(self.num_iter):
            epReward = self.runEpisode(waterWorld)
            waterWorld.refresh()
            rewards.append(epReward)
            print (f"Reward for {i} iteration: {epReward:.4f}")
        plt.plot(rewards)
        plt.show()

    def runEpisode(self, waterWorld):
        """ Perform one episode from input start state
        """

        discounted_rewards = 0
        for i in range(self.iter_length):
            state = self.positionToState(waterWorld)
            if state == -1:
                print (f"Found all trash in {i} steps with rewards: {discounted_rewards:4f}")
                break
            action = self.selectAction(state)
            reward = waterWorld.do_action(action)
            nextState = self.positionToState(waterWorld)
            self.update(state, action, reward, nextState)
            discounted_rewards = 1*discounted_rewards + reward #discount?
        return discounted_rewards

def main():
    dim_x = 5
    dim_y = 5
    trash_num = 3

    env = roomba_sim.State(dim_x, dim_y, trash_num)
    # Init policy
    model = QLearning()
    model.train(env)

if __name__ == "__main__":
    main() 