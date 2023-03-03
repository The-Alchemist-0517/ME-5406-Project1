import random

# Environment parameters
PIXELS        = 40          # pixels
GRID_SIZE     = 10        # Size of environment
ENV_HEIGHT    = GRID_SIZE   # grid height
ENV_WIDTH     = GRID_SIZE   # grid width


# Training parameters
NUM_STEPS     = 100
NUM_EPISODES  = 20000
LEARNING_RATE = 0.001
GAMMA         = 0.9
EPSILON       = 0.9

random.seed(0)
#Generate hole positions
hole_positions = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
# delete the start point and the end point
hole_positions.remove((0, 0))
hole_positions.remove((GRID_SIZE - 1, GRID_SIZE - 1))
# final hole positions
NUM_HOLES = int(GRID_SIZE * GRID_SIZE * 0.25)
hole_positions_final = random.sample(hole_positions, NUM_HOLES)
x_coords = [coord[0] for coord in hole_positions_final]
y_coords = [coord[1] for coord in hole_positions_final]

