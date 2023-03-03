import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from Environment import Environment
from Parameters import *

np.random.seed(1)


class Q_learning(object):
    def __init__(self, env, learning_rate, gamma, epsilon):
        # Class environment
        self.env = env

        # List of actions
        self.actions = list(range(self.env.n_actions))

        # hyperparameters
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Create full Q-table for all cells
        self.q_table = pd.DataFrame(columns=self.actions)


        # Help to check if the Q-table is converged
        self.history = []

    # Add new states to the Q-table
    def check_state_validation(self, state):
        if state not in self.q_table.index:
            new_row = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = self.q_table.append(new_row)

    def epsilon_greedy_policy(self, observation):
        # check the state is valid or not
        self.check_state_validation(observation)
        # action selection
        if np.random.uniform() > self.epsilon:
            # choose random action
            action = np.random.choice(self.actions)
        else:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action==np.max(state_action)].index)
        return action

    # Learn and update the Q table use the Q learning update rules as :
    # Q(s,a) = Q(s,a) + alpha *(r + gamma * max[Q(s',a)] - Q(s,a))
    def learn(self, state, action, reward, next_state):
        # Check if the next step is in the Q-table
        self.check_state_validation(next_state)

        # Current state in the current position
        q_predict = self.q_table.loc[state, action]

        # Calculate the q target value according to update rules
        q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()

        # Update Q-table
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

        # Add the new Q-value into history
        self.history.append(self.q_table.values)

        return self.q_table.loc[state, action],self.history

    # Train for updating the Q table
    def train(self, num_epoch):
        # Resulted list for the plotting Episodes via Steps
        steps = []
        # Resulted list for the plotting Episodes via cost
        all_costs = []
        # Resulted list for the plotting Episodes via average accuracy
        accuracy = []
        # List for average rewards
        Reward_list = []
        # list for difference between new and old Q-tables
        difference_list = []

        # Initialize variable
        goal_count = 0
        rewards = 0
        positive_count = 0
        negative_count = 0

        for i in range(num_epoch):
            # Initial Observation
            observation = self.env.reset()

            # Initialize step count
            step = 0

            # Initialize cost count
            cost = 0

            # Calculate the success rate for every 50 steps
            if i != 0 and i % 50 == 0:
                goal_count = goal_count / 50
                accuracy += [goal_count]
                goal_count = 0

            while True:

                # chooses action based on epsilon greedy policy
                action = self.epsilon_greedy_policy(str(observation))

                # Takes an action and get the next observation and reward
                next_observation, reward, done, info = self.env.step(action)

                # learns from this transition
                new_Q, self.history = self.learn(str(observation), action, reward, str(next_observation))

                # calculate the cost
                cost += new_Q

                # Swapping the observations - current and next
                observation = next_observation

                # Count the number of Steps in the current Episode
                step += 1

                # Break while loop when it is the end of current Episode
                # When agent reached the goal or hole
                if done:
                    # Record the positive cost and negative cost
                    if reward > 0:
                        positive_count += 1
                    else:
                        negative_count += 1

                    # Record the step
                    steps += [step]
                    all_costs += [cost]
                    # Record the convergence
                    difference = self.check_convergence()
                    difference_list += [difference]

                    # goal count +1, if reaching the goal
                    if reward == 1:
                        goal_count += 1

                    # Record total rewards to calculate average rewards
                    rewards += reward
                    Reward_list += [rewards / (i + 1)]

                    break

            print('episode:{} Diff:{}'.format(i,difference))


        # Record the data to the list
        all_cost_bar = [positive_count, negative_count]

        # Showing the Q-table with values for each action
        # self.print_q_table()

        # Plot the results
        self.plot_results(steps,difference_list, accuracy, all_cost_bar, Reward_list,all_costs)

        return self.q_table, steps,  all_costs, accuracy, all_cost_bar, Reward_list, difference_list

    # Check the Q-value is converge or not
    def check_convergence(self):
        # when the iteration<=2, it is meaningless to check the convergence
        if len(self.history) < 2:
            return 0
        else:
            q1 = pd.DataFrame(self.history[-2], columns=self.q_table.columns)
            q2 = pd.DataFrame(self.history[-1], columns=self.q_table.columns)

            # Merge two DataFrames and select only states and actions that exist in both Q tables
            merged_df = q1.merge(q2, how='inner', left_index=True, right_index=True)

            # Compute the difference between states and actions present in both Q-tables
            diff = np.abs(merged_df.iloc[:, :len(self.env.action_space)].values - merged_df.iloc[:,
                                                                              len(self.env.action_space):].values).mean()
            return diff

    # print q table
    def print_q_table(self, filename):
        with open(filename, 'w') as file:
            file.write('Length of full Q-table =' + str(len(self.q_table.index)) + '\n')
            file.write('Full Q-table:\n')
            file.write(str(self.q_table))

    # plot training results
    def plot_results(self, steps, difference_list, accuracy, all_cost_bar, Reward_list,all_costs):

        #
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(steps)), steps, linewidth=2, color='b')
        plt.grid(True)
        plt.title('Episode vs Steps', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Steps', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)

        #
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(difference_list)), difference_list, linewidth=2, color='b')
        plt.grid(True)
        plt.title('Episode vs Diff', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Diff', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)

        #
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(all_costs)), all_costs, linewidth=2, color='b')
        plt.grid(True)
        plt.title('Episode vs Cost', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Cost', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)


        #
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(accuracy)), accuracy, linewidth=2, color='b')
        plt.grid(True)
        plt.title('Episode vs Accuracy', fontsize=16, fontweight='bold')
        plt.xlabel('Every 50 Episodes', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)

        plt.figure(figsize=(8, 6))
        list = ['Success', 'Fail']
        color_list = ['blue', 'red']
        plt.bar(np.arange(len(all_cost_bar)), all_cost_bar, tick_label=list, color=color_list)
        plt.title('Bar/Success and Fail', fontsize=16, fontweight='bold')
        plt.ylabel('Number')

        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(Reward_list)), Reward_list, linewidth=2, color='b')
        plt.grid(True)
        plt.title('Episode vs Reward', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Reward', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        # Showing the plots
        plt.show(block=False)

    # Test after training
    def test(self):
        # Test for 100 episodes
        num_test = 100

        # Print route
        final_route = {}

        # Initialize count, and data store lists
        num_find_goal = 0
        reward_list = []
        steps_list = []

        # run 100 episode to test the correctness of the method
        for i in range(num_test):
            # reset the environment
            observation = self.env.reset()

            for j in range(NUM_STEPS):

                # Choose the best action based on the optimal_policy
                state_action = self.q_table.loc[str(observation), :]
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)

                # perform action and get a tuple
                next_observation, reward, done, info = self.env.step(action)

                # Coordinate transformation
                y = int(math.floor(next_observation / GRID_SIZE)) * PIXELS
                x = int(next_observation % GRID_SIZE) * PIXELS
                final_route[j] = [x, y]

                if done:
                    # Record the number of goal reaching
                    if reward == 1:
                        num_find_goal += 1

                    # While a episode terminates, record the total reward, step
                    # Then add to the list
                    r = reward
                    step = j + 1
                    reward_list += [r]
                    steps_list += [step]

                    break

                observation = next_observation

        # Print final route
        self.env.final_route_global = final_route
        self.env.final()

        print("correctness:{}".format(num_find_goal / num_test))

        # Plot results
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(steps_list)), steps_list, linewidth=2, color='r')
        plt.grid(True)
        plt.title('Episode vs steps', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('steps', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)


        #
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(reward_list)), reward_list, linewidth=2, color='r')
        plt.grid(True)
        plt.title('Episode vs Reward', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Reward', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)


        # Showing the plots
        plt.show(block=False)


# Commands to be implemented after running this file
if __name__ == "__main__":
    # Create an environment
    env = Environment(grid_size=GRID_SIZE)

    # Create a q learning agent
    Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

    # learn and update
    Q_table = Q_learning.train(num_epoch=NUM_EPISODES)

    # print q table
    Q_learning.print_q_table('Q-table_ql.txt')

    # Test after training
    Q_learning.test()

    # Remain visualization
    env.mainloop()
