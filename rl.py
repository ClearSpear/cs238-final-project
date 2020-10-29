''' Reinforcement learning

'''
import sys
import time

import numpy as np 
import pandas as pd 

import roomba_sim

class QLearning:
    ''' Q-learning algorithm 
    '''
    def __init__(self, discount=0.5, learning_rate=0.5, bRandom=False):
        ''' CLass init
            :param discount: discount factor
            :param learning_rate: learning rate for updating qvalues
        '''
        self.discount = discount
        self.learning_rate = learning_rate
        self.bRandom = bRandom
        
        self.n_states = -1
        # Map from action --> actionIndex
        self.n_actions = -1
        # Matrix from (stateIndex, actionIndex) --> qvalue
        self.qMap = None
        # RewardMap from state,action pair
        self.rMap = {}
        # Full reward map (state, action, next_state) --> reward
        self.rMapFull = {}

        self.rewardMap = {}

        np.random.seed(13)

    def init(self, df):
        ''' Init state/action and qvalues based on input dataframe!
        '''
        self.n_states = df['s'].max()
        self.n_actions = df['a'].max()

        self.qMap = np.zeros((self.n_states, self.n_actions))

        rCountTotalMap = df.groupby(['s', 'a'])['r'].count().to_dict()
        rCount = df.groupby(['s', 'a', 'sp'])['r'].count()
        
        self.rewardMap = df[['s', 'a', 'sp', 'r']].drop_duplicates().set_index(['s', 'a', 'sp']).to_dict()


    def update(self, s, a, r, sp):
        ''' Update Q-values based on 
            :param s: input state
            :param a: input action
            :param r: output reward for s->a
            :param sp: output state for s->a
        '''
        #Q[s,a]+=α*(r+γ*maximum(Q[s′,:])-Q[s,a])
        stateIndex = s-1
        actionIndex = a-1
        qVal = self.qMap[stateIndex, actionIndex]
        nextStateIndex = sp-1
        
        if not self.bRandom:
            # Select best next policy
            delta = self.learning_rate * (r + self.discount*np.max(self.qMap[nextStateIndex, :]) - qVal)
        else:
            # Select random next action instead of best one!
            delta = self.learning_rate * (r + self.discount*np.random.choice(self.qMap[nextStateIndex, :]) - qVal)
        
        self.qMap[stateIndex, actionIndex] += delta
        self.rMap[(s, a)] = max(self.rMap.get((s, a), -np.inf), r)


    def learn(self, df):
        ''' Start Q-learning algorithm 
            :param: df
        '''
        self.init(df)

        for index, row in df.iterrows():
            s = row['s']
            a = row['a']
            r = row['r']
            sp = row['sp']
            self.update(s, a, r, sp)

    def runEpisode(self, startState):
        """ Perform one episode from input start state
        """
        pass


    def train(self, df, n_eps=1000):
        ''' Start Q-learning algorithm 
            :param: df
        '''
        self.init(df)
        for _ in range(n_eps):
            startState = np.random.randint(self.n_states) + 1


    def getPolicy(self):
        ''' Return policy from internal qMap
        '''
        policy = np.argmax(self.qMap, axis=1) + 1
        return policy

    def getTotalQvalue(self):
        ''' Return current total qvalue
        '''
        avg_rewards = self.qMap.max(axis=1).mean()
        return avg_rewards

    def getTotalRewards(self, policy):
        reward = 0
        for state, action in enumerate(policy):
            state += 1
            reward += self.rMap[(state, action)]
        return reward

def main():
    dim_x = 5
    dim_y = 5
    trash_num = 2

    env = roomba_sim.State(dim_x, dim_y, trash_num)
    # Init policy
    for _ in range(1000):
        pass
        # get optimal action/reward from policy & next state
        # state.print_grid()

if __name__ == "__main__":
    main() 