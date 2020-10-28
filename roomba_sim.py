
import random
random.seed(94444)
    
REWARD_TRASH = 1.
REWARD_WALLCOLLISION = -.5
REWARD_MOVE = -.1

class State:

    ### Initialization ###

    def __init__(self, dim_x, dim_y, num_trash):
        self.dim_x = dim_x # Width
        self.dim_y = dim_y # Height
        self.num_trash = num_trash
        self.refresh()

    def refresh(self):
        self.init_robot()
        self.init_trash()

    # Robot position currently initialized to a random position
    # in the 0th row
    def init_robot(self):
        self.robot_pos = (random.randint(0, self.dim_x-1), 0)

    def init_trash(self):
        self.trash_positions = [] # List of pairs
        for i in range(self.num_trash):
            # Add trash at random position
            new_pos = (random.randint(0, self.dim_x-1), random.randint(0, self.dim_y-1))
            while new_pos in self.trash_positions:
                new_pos = (random.randint(0, self.dim_x-1), random.randint(0, self.dim_y-1))
            self.trash_positions.append(new_pos)

    ### Methods ###
    
    # Returns REWARD
    def do_action(self, action):
        if action == 0: # Move left
            if self.robot_pos[0] == 0:
                return REWARD_WALLCOLLISION
            self.robot_pos = (self.robot_pos[0] - 1, self.robot_pos[1])
            if self.robot_pos in self.trash_positions:
                self.trash_positions.remove(self.robot_pos)
                return REWARD_TRASH
            else:
                return REWARD_MOVE
        elif action == 1: # Move right
            if self.robot_pos[0] == self.dim_x-1:
                return REWARD_WALLCOLLISION
            self.robot_pos = (self.robot_pos[0] + 1, self.robot_pos[1])
            if self.robot_pos in self.trash_positions:
                self.trash_positions.remove(self.robot_pos)
                return REWARD_TRASH
            else:
                return REWARD_MOVE
        elif action == 2: # Move up
            if self.robot_pos[1] == 0:
                return REWARD_WALLCOLLISION
            self.robot_pos = (self.robot_pos[0], self.robot_pos[1] - 1)
            if self.robot_pos in self.trash_positions:
                self.trash_positions.remove(self.robot_pos)
                return REWARD_TRASH
            else:
                return REWARD_MOVE
        elif action == 3: # Move down
            if self.robot_pos[1] == self.dim_y-1:
                return REWARD_WALLCOLLISION
            self.robot_pos = (self.robot_pos[0], self.robot_pos[1] + 1)
            if self.robot_pos in self.trash_positions:
                self.trash_positions.remove(self.robot_pos)
                return REWARD_TRASH
            else:
                return REWARD_MOVE
        else:
            assert False

    # Returns a doubles vector of the state with fixed size
    # [robot_x, robot_y, dim_x, dim_y, num_trash, trash_0_x, trash_0_y, trash_1_x, trash_1_y, ..., trash_n_x, trash_n_y]
    # If trash is picked up, vector is replaced with -1s
    def return_state_vector(self):
        state_vector = []
        state_vector.append(self.robot_pos[0])
        state_vector.append(self.robot_pos[1])
        state_vector.append(self.dim_x)
        state_vector.append(self.dim_y)
        state_vector.append(self.num_trash)
        for self.trash_pos in self.trash_positions:
            state_vector.append(self.trash_pos[0])
            state_vector.append(self.trash_pos[1])
        for i in range(0, self.num_trash - len(self.trash_positions)):
            state_vector.append(-1)
            state_vector.append(-1)
        return state_vector

    def print_grid(self):
        print("\t  ", end="")
        for x in range(0, self.dim_x):
            print(str(x) + " ", end="") # TODO this will break for numbers over 10
        print("")

        print("\t", end="")
        print("-" * (self.dim_x*2 + 3))

        # Print each row
        for y in range(0, self.dim_y):
            print(y, "\t|", end="")
            for x in range(0, self.dim_x):
                if (x, y) == self.robot_pos:
                    print(" R", end="")
                elif (x, y) in self.trash_positions:
                    print(" t", end="")
                else:
                    print(" .", end="")
            print(" |")

        print("\t", end="")
        print("-" * (self.dim_x*2 + 3))

    def print(self):
        print(self.dim_x, "by", self.dim_y)
        print("Robot at", self.robot_pos)
        print(self.num_trash, "pieces of trash at", self.trash_positions)
        print("State vector:", self.return_state_vector(), "\n")
        self.print_grid()


class Policy:
    def get_action(self, state):
        return random.randint(0, 3)


def main():
    state = State(10, 10, 5)
    policy = Policy()

    while True:
        command = input("> ")
        if command == "print":          # Print out the state
            state.print()
        elif command == "init":         # Initialize the grid to new parameters
            dim_x = int(input("Width: "))
            dim_y = int(input("Height: "))
            trash_num = int(input("Trash num: "))
            state = State(dim_x, dim_y, trash_num)
            print("New grid with dimensions ", state.dim_x, " by ", state.dim_y, " and ", trash_num, " pieces of trash")
        elif command == "refresh":      # Refresh the grid with the same parameters
            state.refresh()
        elif command == "step":         # Step once with the current policy
            action = policy.get_action(state)
            reward = state.do_action(action)
            state.print_grid()
            print("Took action", action, "with reward", reward)
        elif command == "help":         # Print out comamnds
            print("Available commands are: 'print', 'init', 'refresh', 'step', 'help', 'quit'")
        elif command == "quit":         # Quit
            return
        else:
            continue

if __name__ == "__main__":
    main() 

