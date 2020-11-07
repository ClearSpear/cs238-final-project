""" Generate random boards and save them!
"""
import os
import pickle 

import numpy as np

import roomba_sim

def main(x, y, max_trash_count, board_count=10):
    np.random.seed(1313) # Fix state for fix boards!
    for index in range(board_count):
        trash_num = np.random.randint(low=1, high=max_trash_count)
        env = roomba_sim.State(x, y, trash_num)
        #env.print_grid()
        
        board_path = os.path.join("test_boards", f"board{index}.pkl")
        with open(board_path, 'wb') as output:
            pickle.dump(env, output, pickle.HIGHEST_PROTOCOL)
        
        with open(board_path, 'rb') as fb:
            env2 = pickle.load(fb)
            env2.print_grid()

if __name__ == "__main__":
    main(5, 5, 7, 100) 