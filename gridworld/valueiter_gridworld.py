from environment.gridworld import GridWorldEnv
import numpy as np

class ValueIteration:

    def __init__(self, env, discount, conv_threshold, num_iteration,
                mode = "PROD", sync_mode = "SYNC"):
        """
            Value Iteration object encapsulates the variables for the bellman 
            equation for optimal value.

            V(s) = R + max Sum(PV(s'))

            :param env: Environment in which the agent is trained
            :param discount: Discount factor - Discount future rewards to the
                present time so that they may be accurately compared.
            :param conv_threshold: Threshold value for determining convergence - 
                changes in the value function should be below a certain threshold
            :param num_iteration: Maximum number of iteration to reach the threshold value.
            :param mode: PROD/DEBUG indicating the run mode, with DEBUG being more verbose
            :param sync_mode: Choice between sync and async
                - SYNC the updates to the value function are made at the end of the iteration.
                - ASYNC the values of the individual states are updateed as and when a change is observed
            :return: retunrs nothing
        """ 
        self.env = env
        self.discount = discount
        self.conv_threshold = conv_threshold
        self.num_iteration = num_iteration
        self.mode = mode
        self.sync_mode = sync_mode
        
        self.state_count = self.env.get_statespace_len()
        self.state_dict = self.env.get_state_dict()
        self.action_count = self.env.get_actionspace_len()
        self.action_dict = self.env.get_action_dict()

        self.uniform_action_prob = 1.0/self.action_count
        self.V = np.zeros(self.stateCount)
        self.Q = [np.zeros(self.state_count) for s in range(self.stateCount)]
        self.Policy = np.zeros(self.state_count)
        self.total_reward = 0
        self.total_steps = 0

        def reset_episode(self):
            """
                Resets the episode
            """
            self.total_reward = 0
            self.total_steps = 0
        
        def iterate_value(self):
            """
                Iterate Values for the training and checking for convergence
            """
            self.V = np.zeros(self.state_count)

            for i in range(self.num_iteration):
                latest_V = np.copy(self.V)
                for state_index in range(self.state_count):
                    current_state = self.env.statespace[state_index]
                    for action in range(self.env.actionspace):
                        next_state = self.env.next_state(current_state, action)
                        reward = self.env.compute_reward(next_state)
                        next_state_index = self.env.state_dict[next_state]

                        #Compute the action-value estimate given an action in a particular state
                        self.Q[state_index][action] = reward + self.discount * latest_V[next_state_index]
                    #Action generating the maximum possible reward
                    self.V[state_index] = max(self.Q[state_index])
                if np.sum(np.abs(latest_V - self.V) <= self.conv_threshold):
                    print("We have achieved convergence on the {} iteration"
                    .format(state_index))
                    break
        
        def run_episode(self):
            """
                Runs a new epsiode
                :return: returns total reward on turn
            """
            self.reset_episode()
            observed_state = self.env.reset()
            running = True
            while running:
                action = self.Policy[self.env.state_dict[observed_state]]
                self.env.step(action)


        def evaluate_policy(self):
            """
                Evaluates the policy
                :return: returns a tuple of mean, standard deviation and mode of rewards.
            """
        
        def extract_policy(self):
            """
                Determines the best policy for any state action pair
            """
            self.Policy = np.argmax(self.Q, axis = 1)
        
        def solve_mdp(self):
            SystemError

        if __name__ == '__main__':
            """
                Main Function 
            """