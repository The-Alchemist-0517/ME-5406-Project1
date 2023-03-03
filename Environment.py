import random
import numpy as np  # To deal with data in form of matrices
import tkinter as tk  # To build GUI
import time  # Time is needed to slow down the agent and to see how he runs
from PIL import Image, ImageTk  # For adding images into the canvas widget
from Parameters import *  # import parameters


# Setting the sizes for the environment
pixels = PIXELS  # pixels
env_height = ENV_HEIGHT  # grid height
env_width = ENV_WIDTH  # grid width
hole_positions = hole_positions_final #hole positions


# Creating class for the environment
class Environment(tk.Tk, object):
    def __init__(self, grid_size):
        super(Environment, self).__init__()

        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.n_states = env_width * env_height

        self.title('Frozen lake')
        self.geometry('{0}x{1}'.format(env_height * pixels, env_height * pixels))

        # Dictionaries to draw the final route
        self.final_route_temp = {}
        self.final_route = {}

        # Global variable for dictionary with coordinates for the final route
        self.final_route_global = {}

        # Key for the dictionaries
        self.i = 0

        # Writing the final dictionary first time
        self.c = True

        # Showing the steps for the shortest route
        self.shortest = 0

        self.grid_size = grid_size

        self.generate_map()


        # Function to build the environment

    def generate_map(self):
        self.canvas_widget = tk.Canvas(self, bg='white',
            height=env_height * pixels,
            width=env_width * pixels)

        # Uploading an image for background
        # img_background = Image.open("images/bg.png")
        # self.background = ImageTk.PhotoImage(img_background)
        # # Creating background on the widget
        # self.bg = self.canvas_widget.create_image(0, 0, anchor='nw', image=self.background)

        # creating grid lines
        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')
        for row in range(0, env_height * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_height * pixels, row
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')

        # image process
        ice_image = Image.open("Images/iceflower.png")
        robot_image = Image.open("Images/agent4.png")
        goal_image = Image.open("Images/goal.png")
        hole_image = Image.open("Images/icehole.png")

        self.ice_image = ImageTk.PhotoImage(ice_image)
        self.robot_image = ImageTk.PhotoImage(robot_image)
        self.goal_image = ImageTk.PhotoImage(goal_image)
        self.hole_image = ImageTk.PhotoImage(hole_image)

        # create list to save every ice hole
        self.holes = []
        # Creating map
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # create ice holes
                if (i, j) in hole_positions_final:
                    self.hole = self.canvas_widget.create_image(pixels * i, pixels * j, anchor='nw', image=self.hole_image)
                    self.holes.append(self.hole)
                # create agent
                elif (i,j) == (0,0):
                    self.agent = self.canvas_widget.create_image(0, 0, anchor='nw', image=self.robot_image)
                # create goal
                elif (i,j) == (GRID_SIZE-1,GRID_SIZE-1):
                    self.goal = self.canvas_widget.create_image(pixels * (GRID_SIZE - 1), pixels * (
                            GRID_SIZE - 1), anchor='nw', image=self.goal_image)
                # create ice surface
                else:
                    self.ice = self.canvas_widget.create_image(pixels * i, pixels * j, anchor='nw', image=self.ice_image)

        # Packing everything
        self.canvas_widget.pack()

        # Record the coordinate of the holes and the goal
        self.hole_positions = []
        for hole in self.holes:
            hole_coords = self.canvas_widget.coords(hole)
            self.hole_positions.append(hole_coords)
        # self.hole_positions = self.canvas_widget.coords(self.holes) # need for transformation
        self.goal_position = self.canvas_widget.coords(self.goal)  # need transformation

        print('Generated !')


    # transform the coordinate to index number
    def position_transition(self, x, y):
        width = self.grid_size
        # Coordinate transformation: Coordinate-> Indexed number
        s = int(x / 40) + int(y / 40 * width)
        return s

    # reset the environment and start new Epoch
    def reset(self):
        self.update()
        # time.sleep(0.1)

        # update agent
        self.canvas_widget.delete(self.agent)
        self.agent = self.canvas_widget.create_image(0, 0, anchor='nw', image=self.robot_image)

        # clear the dictionary and the key
        self.final_route_temp = {}
        self.i = 0

        # return observation
        agent_position = self.canvas_widget.coords(self.agent)

        # position transformation(coordinate -> index number)
        agent_position = self.position_transition(agent_position[0], agent_position[1])

        return agent_position

    # get the next state and reward
    def step(self, action):
        # Current state of the agent
        state = self.canvas_widget.coords(self.agent)
        base_action = np.array([0, 0])

        # update next state according to the action
        # Action 'up'
        if action==0:
            if state[1] >= pixels:
                base_action[1] -= pixels
        # Action 'down'
        elif action==1:
            if state[1] < (env_height - 1) * pixels:
                base_action[1] += pixels
        # Action right
        elif action==2:
            if state[0] < (env_width - 1) * pixels:
                base_action[0] += pixels
        # Action left
        elif action==3:
            if state[0] >= pixels:
                base_action[0] -= pixels

        # move the agent
        self.canvas_widget.move(self.agent, base_action[0], base_action[1])

        # write the new coordinate of robot into the route dictionary
        self.final_route_temp[self.i] = self.canvas_widget.coords(self.agent)

        # update next state
        next_state = self.final_route_temp[self.i]

        # update key for the dictionary
        self.i += 1

        # calculate the reward
        # next_state = 'goal'
        if next_state==self.goal_position:
            reward = 1
            done = True

            # Filling the dictionary first time
            if self.c==True:
                for j in range(len(self.final_route_temp)):
                    self.final_route[j] = self.final_route_temp[j]
                self.c = False
                self.shortest = len(self.final_route_temp)

            # check if the current found route is shorter
            if len(self.final_route_temp) < len(self.final_route):
                # Saving the number of steps for the shortest route
                self.shortest = len(self.final_route_temp)
                # Clearing the dictionary for the final route
                self.final_route = {}
                # Reassigning the dictionary
                for j in range(len(self.final_route_temp)):
                    self.final_route[j] = self.final_route_temp[j]

        # next_state = 'hole'
        elif next_state in self.hole_positions:
            reward = -1
            done = True

            # clear the dictionary and the key
            self.final_route_temp = {}
            self.i = 0

        else:
            reward = 0
            done = False

        # position transformation(coordinate -> index number)
        next_state = self.position_transition(next_state[0], next_state[1])

        return next_state, reward, done, {}

    # render the environment
    def render(self):
        time.sleep(0.05)
        self.update()


    # show the shortest route
    def final(self):
        # delete the agent at the end
        self.canvas_widget.delete(self.agent)

        # show the number of step of the shortest route
        print('The shortest route:', self.shortest)

        # create initial point
        origin = np.array([20, 20])
        self.initial_point = self.canvas_widget.create_oval(
            origin[0] - 5, origin[1] - 5,
            origin[0] + 5, origin[1] + 5,
            fill='red', outline='red')

        # fill the route
        for j in range(len(self.final_route)):
            # show the coordinates of the final route
            self.track = self.canvas_widget.create_oval(
                self.final_route[j][0] + origin[0] - 5, self.final_route[j][1] + origin[0] - 5,
                self.final_route[j][0] + origin[0] + 5, self.final_route[j][1] + origin[0] + 5,
                fill='red', outline='red')
            # write the final route in the global variable
            self.final_route_global[j] = self.final_route[j]

    # return the final dictionary with route coordinates
    def final_states(self):
        return self.final_route_global


def update():
    for t in range(100):
        s = env.reset()
        while True:
            env.render()
            a = random.randint(0, 3)
            s_, r, done, info = env.step(a)
            if done:
                break


# see the environment without running full algorithm
if __name__=='__main__':
    # Create environment
    env = Environment(grid_size=GRID_SIZE)

    # update()
    env.mainloop()
