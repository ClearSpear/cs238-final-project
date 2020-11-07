import time

import numpy as np

from matplotlib import pyplot as plt

from qLearning import QLearning
import roomba_sim

def TestRoomba(dim_x, dim_y, max_trash_count, qMap, board_count=100, bPrint=False):
    total_rewards = 0
    st0 = np.random.get_state()
    
    np.random.seed(1313) # Fix state for fix boards!
    for index in range(board_count):
        trash_num = np.random.randint(low=1, high=max_trash_count)
        env = roomba_sim.State(dim_x, dim_y, trash_num)
        # Init policy
        model = QLearning(num_iter=1, iter_length=1000)
        model.test(env, qMap, bPrint=bPrint)
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

    # Display board...
    rewards = TestRoomba(dim_x, dim_y, max_trash_count, qMap, board_count=5, bPrint=True)


    plt.plot(overall_rewards)
    plt.show()

def main():
    DIM_X = 5
    DIM_Y = 5
    DIM_XY = DIM_X * DIM_Y
    MAX_TRASH_PERC = 0.25
    MAX_TRASH_COUNT = int(DIM_XY * MAX_TRASH_PERC) + 1

    TrainRoomba(DIM_X, DIM_Y, MAX_TRASH_COUNT, board_count=1000, test_index=10)
    

if __name__ == "__main__":
    main() 