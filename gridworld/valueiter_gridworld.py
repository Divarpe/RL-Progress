from environment.gridworld import GridWorldEnv
import numpy as np
import sys

class ValueIteration:

    def __init__(self, env, discount=0.9, conv_threshold=0.0001, num_iteration=1000,
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
        self.env = GridWorldEnv()
        self.discount = discount
        self.conv_threshold = conv_threshold
        self.num_iteration = num_iteration
        self.mode = mode
        self.sync_mode = sync_mode
        
        self.state_count = self.env.get_statespace_len()
        self.state_dict = self.env.state_dict
        self.action_count = self.env.get_actionspace_len()
        self.action_dict = self.env.action_dict

        self.uniform_action_prob = 1.0/self.action_count
        self.V = np.zeros(self.state_count)
        self.Q = np.zeros((49, 4), dtype=float)
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
                for action in self.env.actionspace:
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
        done = False
        while not done:
            action = self.Policy[self.env.state_dict[observed_state]]
            if self.mode == "DEBUG":
                print("On turn {} we have action: {}".format(self.total_steps,action))
            new_state, reward, done, debug = self.env.step(action)
            self.total_reward += reward
            self.total_steps += 1
            observed_state = new_state
        return self.total_reward


    def evaluate_policy(self, n_episodes=1000):
        """
            Evaluates the policy
            :return: returns a tuple of mean, standard deviation, median and range of rewards.
        """
        episode_scores = []
        for e in range(n_episodes):
            episode_scores.append(self.run_episode())
        mean = np.mean(episode_scores)
        median = np.median(episode_scores)
        standard_deviation = np.std(episode_scores)
        range_stat = np.max(episode_scores) - np.min(episode_scores)
        
        return mean, standard_deviation, median, range_stat

        
    def extract_policy(self):
        """
            Determines the best policy for any state action pair
        """
        if self.mode == "DEBUG":
            print("Q: ", self.Q)

        self.Policy = np.argmax(self.Q, axis = 1)
        
        if self.mode == "DEBUG":
            print("Optimal Policy: ", np.shape(self.Policy))
    
    def solve_mdp(self, n_episodes=100):
        """
        """
        self.iterate_value()
        self.extract_policy()
        return self.evaluate_policy(n_episodes)

if __name__ == '__main__':
    """
        Main Function 
    """
    valueIteration = ValueIteration(env = GridWorldEnv, mode = "PROD")
    mean_reward, std, median, range = valueIteration.solve_mdp()
    print("This is the mean {}, median {}, range, {} and standard deviation {}"
    .format(mean_reward, median, range, std))