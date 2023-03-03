import numpy as np
import matplotlib.pyplot as plt


from Environment import Environment
from Parameters import *
from Monte_carlo_control import Monte_carlo
from SARSA import SARSA
from Q_learning import Q_learning
from train import TRAIN
from test import TEST

if __name__ == '__main__':
    # Train and get the Q-table/training result
    # run_training = TRAIN(agent='mc')
    # run_training.train_agent()

    #Do test and comparison
    run_testing = TEST(task='t1')
    run_testing.test_agent()