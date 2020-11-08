import os
import pickle
import time

import numpy as np

from matplotlib import pyplot as plt

from qLearning import QLearning
from qLearning import positionToState
import roomba_sim

TRAIN_ITER = 100

TEST_BOARDS = "test_boards_water"

def TestRoomba(qMap, board_count=100, bPrint=False):
    total_rewards = 0

    for index in range(board_count):
        print (f"Testing board: {index}")
        board_path = os.path.join(TEST_BOARDS, f"board{index}.pkl")
        env = None
        with open(board_path, 'rb') as fb:
            env = pickle.load(fb)
        
        # Init policy
        model = QLearning(num_iter=1, iter_length=roomba_sim.BATTERY_LIFE)
        model.test(env, qMap, bPrint=bPrint)
        reward = model.total_rewards # scale rewards on total number of trash!
        total_rewards += reward

    mean_rewards = total_rewards / board_count

    print (f"avg_rewards for test: {mean_rewards}\n")
    return mean_rewards


def TestRandomRoomba(board_count=100, bPrint=False):
    total_rewards = 0
    policy = roomba_sim.Policy()

    for index in range(board_count):
        board_path = os.path.join(TEST_BOARDS, f"board{index}.pkl")
        env = None
        with open(board_path, 'rb') as fb:
            env = pickle.load(fb)
        
        # Init policy
        for i in range(roomba_sim.BATTERY_LIFE):
            action = policy.get_action(env)
            reward = env.do_action(action)
            
            state = positionToState(env)
            if state == -1:
                break

            if i==(roomba_sim.BATTERY_LIFE - 1):
                reward = roomba_sim.REWARD_TIMEOUT

            total_rewards += reward

    mean_rewards = total_rewards / board_count

    print (f"avg_rewards for test: {mean_rewards}\n")
    return mean_rewards


def TrainRoomba(dim_x, dim_y, max_trash_count, board_count=10000, test_index=100, test_boards=100, learning_decay_index=1000):
    """ Train roomba
    """
    qMap = None

    overall_rewards = []
    random_rewards = []
    np.random.seed(13)
    #random_seeds = list(range(board_count))
    #np.random.shuffle(random_seeds)

    learning_rate = 0.75
    discount = 0.95
    
    for index in range(board_count):
        st = time.time()
        #np.random.seed(random_seeds[index])
        trash_num = np.random.randint(low=1, high=max_trash_count)
        move_prob = np.random.uniform(low=0.5, high=1)

        env = roomba_sim.State(dim_x, dim_y, trash_num, move_prob=move_prob)
        # Init policy
        model = QLearning(num_iter=TRAIN_ITER, iter_length=roomba_sim.BATTERY_LIFE, learning_rate=learning_rate)
        model.train(env, qMap)
        qMap = model.qMap

        print (f"\nTrained on board {index} with {trash_num} trash!, reward: {model.total_rewards/trash_num} in {time.time()-st:.2f}s")

        if (index+1)%test_index == 0:
            rewards = TestRoomba(qMap, board_count=test_boards)
            overall_rewards.append(rewards)

            rewards = TestRandomRoomba(board_count=test_boards)
            random_rewards.append(rewards)

        if (index+1) % learning_decay_index == 0:
            learning_rate = discount * learning_rate

    
    # Final test
    rewards = TestRoomba(qMap, board_count=test_boards)
    overall_rewards.append(rewards)
    rewards = TestRandomRoomba(board_count=test_boards)
    random_rewards.append(rewards)

    # Display board...
    rewards = TestRoomba(qMap, board_count=5, bPrint=True)


    plt.plot(np.arange(0, len(overall_rewards))*test_index, overall_rewards, label='Q-Learning Rewards')
    plt.plot(np.arange(0, len(random_rewards))*test_index, random_rewards, label='Random Rewards')
    plt.xlabel("# Training Iteration")
    plt.ylabel("Average Rewards")
    plt.legend()
    plt.show()

def main():
    DIM_X = 5
    DIM_Y = 5
    DIM_XY = DIM_X * DIM_Y
    MAX_TRASH_COUNT = 7

    TrainRoomba(DIM_X, DIM_Y, MAX_TRASH_COUNT, board_count=10000, test_index=10, learning_decay_index=1000000)
    

if __name__ == "__main__":
    main() 