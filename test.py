import numpy as np
import matplotlib.pyplot as plt
import argparse

from Environment import Environment
from Parameters import *
from Monte_carlo_control import Monte_carlo
from SARSA import SARSA
from Q_learning import Q_learning

class TEST:
    def __init__(self, task):
        self.task = task

    # Define line plot functions
    # Episodes via steps
    def plot_steps(self, steps, label):
        plt.figure()
        plt.plot(np.arange(len(steps[0])), steps[0], 'r', label=label[0], linewidth=1)
        plt.plot(np.arange(len(steps[1])), steps[1], 'g', label=label[1], linewidth=1)
        plt.plot(np.arange(len(steps[2])), steps[2], 'b', label=label[2], linewidth=1)
        plt.title('Episode via Steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend(loc='best')
        plt.show(block=False)
    #
    def plot_steps_2(self, steps, label):
        plt.figure()
        plt.plot(np.arange(len(steps[0])), steps[0], 'r', label=label[0], linewidth=1)
        plt.plot(np.arange(len(steps[1])), steps[1], 'b', label=label[1], linewidth=1)
        plt.title('Episode via Steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend(loc='best')
        plt.show(block=False)


    # Episodes via Costs
    def plot_all_cost(self, all_cost, label):
        plt.figure()
        plt.plot(np.arange(len(all_cost[0])), all_cost[0], 'r', label=label[0], linewidth=1)
        plt.plot(np.arange(len(all_cost[1])), all_cost[1], 'g', label=label[1], linewidth=1)
        plt.plot(np.arange(len(all_cost[2])), all_cost[2], 'b', label=label[2], linewidth=1)
        plt.title('Episode via Cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        plt.legend(loc='best')
        plt.show(block=False)


    # Episodes via Accuracy
    def plot_accuracy(self, accuracy, label):
        plt.figure()
        plt.plot(np.arange(len(accuracy[0])), accuracy[0], 'r', label=label[0], linewidth=1)
        plt.plot(np.arange(len(accuracy[1])), accuracy[1], 'g', label=label[1], linewidth=1)
        plt.plot(np.arange(len(accuracy[2])), accuracy[2], 'b', label=label[2], linewidth=1)
        plt.title('Episode via Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.show(block=False)

    def plot_accuracy_2(self, accuracy, label):
        plt.figure()
        plt.plot(np.arange(len(accuracy[0])), accuracy[0], 'r', label=label[0], linewidth=1)
        plt.plot(np.arange(len(accuracy[1])), accuracy[1], 'b', label=label[1], linewidth=1)
        plt.title('Episode via Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.show(block=False)


    # Episodes via Average Rewards
    def plot_average_rewards(self, Reward_list, label):
        plt.figure()
        plt.plot(np.arange(len(Reward_list[0])), Reward_list[0], 'r', label=label[0], linewidth=1)
        plt.plot(np.arange(len(Reward_list[1])), Reward_list[1], 'g', label=label[1], linewidth=1)
        plt.plot(np.arange(len(Reward_list[2])), Reward_list[2], 'b', label=label[2], linewidth=1)
        plt.title('Episode via Average rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average rewards')
        plt.legend(loc='best')
        plt.show(block=False)

    def plot_average_rewards_2(self, Reward_list, label):
        plt.figure()
        plt.plot(np.arange(len(Reward_list[0])), Reward_list[0], 'r', label=label[0], linewidth=1)
        plt.plot(np.arange(len(Reward_list[1])), Reward_list[1], 'b', label=label[1], linewidth=1)
        plt.title('Episode via Average rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average rewards')
        plt.legend(loc='best')
        plt.show(block=False)


    # Define scatter plot functions
    def plot_steps_scatter(self, steps, label):
        plt.figure()
        plt.scatter(np.arange(len(steps[0])), steps[0], alpha=0.8, s=1.5, c='r', label=label[0])
        plt.scatter(np.arange(len(steps[1])), steps[1], alpha=0.8, s=1.5, c='g', label=label[1])
        plt.scatter(np.arange(len(steps[2])), steps[2], alpha=0.8, s=1.5, c='b', label=label[2])
        plt.title('Episode via Steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend(loc='best')
        plt.show(block=False)


    def plot_all_cost_scatter(self, all_cost, label):
        plt.figure()
        plt.scatter(np.arange(len(all_cost[0])), all_cost[0], label=label[0], alpha=0.8, s=2, c='r')
        plt.scatter(np.arange(len(all_cost[1])), all_cost[1], label=label[1], alpha=0.8, s=2, c='g')
        plt.scatter(np.arange(len(all_cost[2])), all_cost[2], label=label[2], alpha=0.8, s=2, c='b')
        plt.title('Episode via Cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        plt.legend(loc='best')
        plt.show(block=False)

    def plot_all_cost_scatter_2(self, all_cost, label):
        plt.figure()
        plt.scatter(np.arange(len(all_cost[0])), all_cost[0], label=label[0], alpha=0.8, s=2, c='r')
        plt.scatter(np.arange(len(all_cost[1])), all_cost[1], label=label[1], alpha=0.8, s=2, c='b')
        plt.title('Episode via Cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        plt.legend(loc='best')
        plt.show(block=False)


    def plot_accuracy_scatter(self, accuracy, label):
        plt.figure()
        plt.scatter(np.arange(len(accuracy[0])), accuracy[0], alpha=0.8, s=1.5, c='r', label=label[0])
        plt.scatter(np.arange(len(accuracy[1])), accuracy[1], alpha=0.8, s=1.5, c='g', label=label[1])
        plt.scatter(np.arange(len(accuracy[2])), accuracy[2], alpha=0.8, s=1.5, c='b', label=label[2])
        plt.title('Episode via Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.show(block=False)


    def plot_average_rewards_scatter(self, Reward_list, label):
        plt.figure()
        plt.scatter(np.arange(len(Reward_list[0])), Reward_list[0], alpha=0.8, s=1.5, c='r', label=label[0])
        plt.scatter(np.arange(len(Reward_list[1])), Reward_list[1], alpha=0.8, s=1.5, c='g', label=label[1])
        plt.scatter(np.arange(len(Reward_list[2])), Reward_list[2], alpha=0.8, s=1.5, c='b', label=label[2])
        plt.title('Episode via Average rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average rewards')
        plt.legend(loc='best')
        plt.show(block=False)

    def plot_differnce(self,difference_list, label):
        plt.figure()
        plt.plot(np.arange(len(difference_list[0])), difference_list[0], 'r', label=label[0], linewidth=1)
        plt.plot(np.arange(len(difference_list[2])), difference_list[2], 'b', label=label[2], linewidth=1)
        plt.title('Episode via Diff')
        plt.xlabel('Episode')
        plt.ylabel('Diff')
        plt.legend(loc='best')
        plt.show(block=False)

    def plot_diff(self,difference_list, label):
        plt.figure()
        plt.plot(np.arange(len(difference_list[0])), difference_list[0], 'r', label=label[0], linewidth=1)
        plt.plot(np.arange(len(difference_list[1])), difference_list[1], 'r', label=label[1], linewidth=1)
        plt.plot(np.arange(len(difference_list[2])), difference_list[2], 'b', label=label[2], linewidth=1)
        plt.title('Episode via Diff')
        plt.xlabel('Episode')
        plt.ylabel('Diff')
        plt.legend(loc='best')
        plt.show(block=False)


    def test_agent(self):
        global Monte_carlo, SARSA, Q_learning
        if self.task=='t1':
            env = Environment(GRID_SIZE)
            # Create three agents corresponding to three algorithms
            Monte_carlo = Monte_carlo(env, epsilon=EPSILON, gamma=GAMMA)
            #
            SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)
            #
            Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

            label_1 = ['Monte_carlo', 'SARSA', 'Q_learning']

            Q_1, steps_1, all_cost_1, accuracy_1, all_cost_bar_1, Rewards_list_1 = Monte_carlo.fv_mc_prediction(num_epoch=NUM_EPISODES)

            Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2, difference_list_2 = SARSA.train(num_epoch=NUM_EPISODES)

            Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3, difference_list_3 = Q_learning.train(num_epoch=NUM_EPISODES)

            steps = [steps_1, steps_2, steps_3]

            all_cost = [all_cost_1, all_cost_2, all_cost_3]

            accuracy = [accuracy_1, accuracy_2, accuracy_3]

            Rewards_list = [Rewards_list_1, Rewards_list_2, Rewards_list_3]

            self.plot_steps(steps, label_1)

            self.plot_all_cost_scatter(all_cost, label_1)

            self.plot_accuracy(accuracy, label_1)

            self.plot_average_rewards(Rewards_list, label_1)

            # plot the result
            plt.show()


        # Job 2, comparison test for different learning rate value settings
        elif self.task == 't2':
            env = Environment(grid_size=GRID_SIZE)

            label_2 = ['lr:0.01', 'lr:0.001', 'lr:0.0001']

            Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

            Q_learning.lr = 0.01

            Q_1, steps_1, all_cost_1, accuracy_1, all_cost_bar_1, Rewards_list_1, difference_list_1 = Q_learning.train(num_epoch=NUM_EPISODES)

            Q_learning.lr = 0.001

            Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2, difference_list_2 = Q_learning.train(num_epoch=NUM_EPISODES)

            Q_learning.lr = 0.0001

            Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3, difference_list_3 = Q_learning.train(num_epoch=NUM_EPISODES)

            steps = [steps_1, steps_2, steps_3]

            all_cost = [all_cost_1, all_cost_2, all_cost_3]

            accuracy = [accuracy_1, accuracy_2, accuracy_3]

            Rewards_list = [Rewards_list_1, Rewards_list_2, Rewards_list_3]

            self.plot_steps(steps, label_2)

            self.plot_all_cost_scatter(all_cost, label_2)

            self.plot_accuracy(accuracy, label_2)

            self.plot_average_rewards(Rewards_list, label_2)

            # plot the result
            plt.show()

        elif self.task=='t3':
            env = Environment(GRID_SIZE)
            # Create three agents corresponding to three algorithms
            SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)
            #
            Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

            label_3 = ['SARSA', 'Q_learning']

            Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2, difference_list_2 = SARSA.train(num_epoch=NUM_EPISODES)

            Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3, difference_list_3 = Q_learning.train(num_epoch=NUM_EPISODES)

            steps = [steps_2, steps_3]

            all_cost = [all_cost_2, all_cost_3]

            accuracy = [accuracy_2, accuracy_3]

            Rewards_list = [Rewards_list_2, Rewards_list_3]

            self.plot_steps_2(steps, label_3)

            self.plot_all_cost_scatter_2(all_cost, label_3)

            self.plot_accuracy_2(accuracy, label_3)

            self.plot_average_rewards_2(Rewards_list, label_3)

            # plot the result
            plt.show()

    # Job 3, comparison test for different gamma value settings
        elif self.task == 't4':
            env = Environment(GRID_SIZE)

            label_4 = ['gamma:0.8', 'gamma:0.9', 'gamma:0.99']

            Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

            Q_learning.gamma = 0.8
            Q_1, steps_1, all_cost_1, accuracy_1, all_cost_bar_1, Rewards_list_1, difference_list1 = Q_learning.train(num_epoch=NUM_EPISODES)

            Q_learning.gamma = 0.9
            Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2, difference_list2 = Q_learning.train(num_epoch=NUM_EPISODES)

            Q_learning.gamma = 0.99
            Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3, difference_list3 = Q_learning.train(num_epoch=NUM_EPISODES)

            steps = [steps_1, steps_2, steps_3]

            all_cost = [all_cost_1, all_cost_2, all_cost_3]

            accuracy = [accuracy_1, accuracy_2, accuracy_3]

            Rewards_list = [Rewards_list_1, Rewards_list_2, Rewards_list_3]

            difference_list = [difference_list1, difference_list2, difference_list3]

            self.plot_steps(steps, label_4)

            self.plot_all_cost_scatter(all_cost, label_4)

            self.plot_accuracy(accuracy, label_4)

            self.plot_diff(difference_list, label_4)

            self.plot_average_rewards(Rewards_list, label_4)

            # plot the result
            plt.show()


    #Job 4, comparison test for different epsilon value settings

        elif self.task == 't5':
            env = Environment(GRID_SIZE)

            label_5 = ['epsilon:0.7', 'epsilon:0.8', 'epsilon:0.9']

            Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

            Q_learning.epsilon = 0.7
            Q_1, steps_1, all_cost_1, accuracy_1, all_cost_bar_1, Rewards_list_1, difference_list1 = Q_learning.train(num_epoch=NUM_EPISODES)

            Q_learning.epsilon = 0.8
            Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2, difference_list2 = Q_learning.train(num_epoch=NUM_EPISODES)

            Q_learning.epsilon = 0.9
            Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3, difference_list3 = Q_learning.train(num_epoch=NUM_EPISODES)

            steps = [steps_1, steps_2, steps_3]

            all_cost = [all_cost_1, all_cost_2, all_cost_3]

            accuracy = [accuracy_1, accuracy_2, accuracy_3]

            Rewards_list = [Rewards_list_1, Rewards_list_2, Rewards_list_3]

            difference_list = [difference_list1, difference_list2, difference_list3]

            self.plot_steps(steps, label_5)

            self.plot_all_cost_scatter(all_cost, label_5)

            self.plot_accuracy(accuracy, label_5)

            self.plot_diff(difference_list, label_5)

            self.plot_average_rewards(Rewards_list, label_5)

            # plot the result
            plt.show()


