"""
 Gridworld environment 

 Custom Environment for the GridWorld Problem
 example based on the book Deep Reinforcement Learning

 Author: Ivan Dewerpe
"""

class GridWorldEnv():
    """
    GridWorldEnv encapsulates the environment for the GridWorld
    Problem with custom functions for a reinforcement learning
    using agent.
    """

    def __init__(self, grid_dimension=7, start_state='00', terminal_state=['64'], ditches=['52'],
                ditch_penalty=-10, turn_penalty=-1, win_reward=100, mode="PROD"):
        """
        Construct a new GridWorldEnv object

        :param grid_dimension: n value for a grid size nxn
        :param start_state: Entry point value of the grid cell
        :param terminal_state: List of grid cell values for which there exists a terminal state
        :param ditches: List of grid cell values for which there exists a ditch
        :param ditch_penalty: Penalty if you hit a ditch
        :param turn_penalty: Negative reward for every turn to ensure that agent completes the 
            episode in minimum number of turns.
        :param win_reward: Reward gained for reaching the goal/terminate state.
        :param mode: (PROD/DEBUG) indicating the run mode. Verbosity of messages
        :return: returns nothing

        """
        self.grid_dimension = min(grid_dimension, 9)
        self.start_state = start_state
        self.terminal_state = terminal_state
        self.ditches = ditches
        self.ditch_penalty = ditch_penalty
        self.turn_penalty = turn_penalty
        self.win_reward = win_reward
        self.mode = mode

        self.create_statespace()
        self.actionspace = [0,1,2,3]
        self.action_dict = {0:'UP', 1:'DOWN', 2:'LEFT', 3:'RIGHT'}
        self.state_count = self.get_statespace_len()
        self.action_count = self.get_actionspace_len()
        self.state_dict = {k:v for k,v in zip(self.statespace, range(self.state_count))}
        self.current_state = self.start_state

        if self.mode == "debug":
            #Debug code
            print("DUBGGING")
    def create_statespace(self):
        """
            Create Statespace

            Makes the GridWorld space with desired dimensions.
        """
        self.statespace = []
        # A state is a string "xy" x - row, y - column
        for row in range(self.grid_dimension):
            for column in range(self.grid_dimension):
                self.statespace.append(str(row) + str(column))
    
    def get_statespace(self):
        return self.statespace

    def get_statespace_len(self):
        return len(self.statespace)

    def set_mode(self, mode):
        self.mode = mode
    
    def get_actionspace(self):
        return self.actionspace
    
    def get_actionspace_len(self):
        return len(self.actionspace)

    def get_action_dict(self):
        return self.action_dict
    
    ###############################################################################################
    def next_state(self, current_state, action):
        """
        Next State

        Returns the next state given an action taken in the environment
        :param current_state: The current state at which the agent takes the action 
        :param action: Action taken by the agent
        :return: returns next state
        
        """
        current_row = int(current_state[0])
        current_column = int(current_state[1])

        next_row = current_row
        next_column = current_column

        max_cell_value = self.grid_dimension
        current_action =  self.action_dict[action]

        if 'UP' == current_action: next_row = max(0, current_row - 1)
        elif 'DOWN' == current_action: next_row = min(max_cell_value - 1, current_row + 1)
        elif 'LEFT' == current_action: next_column = max(0, current_column - 1)
        elif 'RIGHT' == current_action: next_column = min(max_cell_value - 1, current_column + 1)

        next_state = str(next_row) + str(next_column)

        if next_state in self.statespace:
            if next_state in self.terminal_state: self.is_game_end = True
            if self.mode == 'DEBUG':
                print("Curent State:{}, Action:{}, NextState:{}"
                .format(current_state, current_action, next_state))
            return next_state
        else:
            return current_state
    
    ###############################################################################################
    def compute_reward(self, state):
        """
            Compute Reward

            Computes the reward for arriving at a given state based on ditches and the end goal
            :param state: State we have arrived in as cell co-ordinate
            :return: returns the reward corresponding to the state
        """
        reward = 0
        reward += self.turn_penalty
        if state in self.ditches: reward += self.ditch_penalty
        if state in self.terminal_state: reward += self.win_reward
        
        return reward
    
    ###############################################################################################
    def reset(self):
        """
            Reset

            Resets map, reward, states to original settings.
            :return: returns fresh entry state for agent
        """
        self.accumulated_reward = 0
        self.current_state = self.start_state
        self.total_turns = 0
        self.is_game_end = False
        
        return self.current_state 

    def step(self, action):
        """
            Step

            Makes the agent take the suggested action

            :param action: Action to be taken by the agent
            :return: returns a tuple of (next_state, instant_reward, done_flag, info) 
        """
        if self.is_game_end:
            raise('Game is Over Exception')
        if action not in self.actionspace:
            raise('Invalid Action Exception')
        
        self.current_state = self.next_state(self.current_state, action)
        observed_state = self.current_state
        reward = self.compute_reward(observed_state)
        self.total_turns += 1
        
        if self.mode == 'DEBUG':
            print("Obs:{}, Reward:{}, Done:{}, Total Turns:{}"
            .format(observed_state, reward, self.is_game_end, self.total_turns))
        
        return observed_state, reward, self.is_game_end, self.total_turns

if __name__ == '__main__':
    """
        Main Function to test the code
    """
    env = GridWorldEnv(mode='DEBUG')
    env.reset()
    env.step(1)
    env.step(3)
    env.step(2)
    env.step(0)