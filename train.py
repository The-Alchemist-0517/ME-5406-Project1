import matplotlib.pyplot as plt

from Environment import Environment
from Parameters import *
from Monte_carlo_control import Monte_carlo
from SARSA import SARSA
from Q_learning import Q_learning

class TRAIN:
    def __init__(self, agent='ql', grid_size=GRID_SIZE, num_epoch=NUM_EPISODES):
        self.agent = agent
        self.grid_size = grid_size
        self.num_epoch = num_epoch
        self.env = Environment(grid_size=grid_size)

    def train_agent(self):
        if self.agent == 'mc':
            # Create a monte carlo agent
            monte_carlo = Monte_carlo(self.env, epsilon=EPSILON, gamma=GAMMA)

            # Learning and updating Q table
            monte_carlo.fv_mc_prediction(num_epoch=self.num_epoch)

            # write_Q_table(file_name='./Q_table/monte_carlo', Q = Q)

            # Test after training
            monte_carlo.test()

            # Remain visualization
            self.env.mainloop()

        elif self.agent == 'sarsa':
            # Create a SARSA agent
            sarsa = SARSA(self.env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

            # Learning and updating
            sarsa.train(num_epoch=self.num_epoch)

            # Test after training
            sarsa.test()

            # Remain visualization
            self.env.mainloop()

        elif self.agent == 'ql':
            # Create a q learning agent
            q_learning = Q_learning(self.env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

            # Learning and updating
            q_learning.train(num_epoch=self.num_epoch)

            # Test after training
            q_learning.test()

            # plot the result
            plt.show()

            # Remain visualization
            self.env.mainloop()

