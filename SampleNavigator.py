import gzip
import numpy as np
from random import randint
from scipy.stats import entropy
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork
import time

def get_direction_to_target(loc, target):
    
    x_diff = target[0] - loc[0]
    y_diff = target[1] - loc[1]
    
    if x_diff == 0 and y_diff == 0:
        print("Asking for direction to target I'm already on")
        exit(-1)
    elif abs(x_diff) > abs(y_diff):  
        if loc[0] < target[0]:
            direction = "right"
        elif loc[0] > target[0]:
            direction = "left"
    else:
        if loc[1] < target[1]:
            direction = "down"
        elif loc[1] > target[1]:
            direction = "up"
    
    next_x = loc[0]
    next_y = loc[1]
    if direction == "up":
        next_y -= 1
    elif direction == "down":
        next_y += 1
    elif direction == "left":
        next_x -= 1
    elif direction == "right":
        next_x += 1
        
    next_loc = (next_x, next_y)
        
    if is_corner(next_loc) and not is_corner(target):
        print("Using corner safeguard")
        if next_loc == (0, 27):
            if direction == "left":
                direction = "up"
            elif direction == "down":
                direction = "right"
            else:
                direction = None
        elif next_loc == (27, 27):
            if direction == "right":
                direction = "up"
            elif direction == "down":
                direction = "left"
            else:
                direction = None
        elif next_loc == (27, 0):
            if direction == "right":
                direction = "down"
            elif direction == "up":
                direction = "left"
            else:
                direction = None
    
    return direction

def is_corner(loc):
    
    x = loc[0]
    y = loc[1]
    
    return (x == 0 and y == 27) or (x == 27 and y == 27) or (x == 27 and y == 0)

class SampleNavigator:
    def __init__(self):
        
        self.num_samples = 25
        self.confidence_threshold = 0.85
        self.world_estimator = WorldEstimatingNetwork()
        self.digit_classifier = DigitClassificationNetwork()
        self.mask = np.zeros((28, 28))
        self.path = []
        self.target = None
        self.goals = [(0, 27), (27, 27), (27, 0)]

    def getAction(self, robot, explored_map):
        
        # if we hit a goal and are still here, we messed up
        if is_corner(robot.getLoc()):
            self.goals.remove(self.target)
            self.target = self.goals[0]
        
        self.path.append(robot.getLoc())

        x, y = robot.getLoc()
        
        for row_inc in range(-1, 2):
            for col_inc in range(-1, 2):
                col = x + col_inc
                row = y + row_inc
                # if the square is off the map, do nothing
                if row > 27 or row < 0 or col < 0 or col > 27:
                    continue
                # Otherwise add to the mask
                else:
                    self.mask[row, col] = 1
                    
        if self.target is not None:
            if robot.getLoc()[0] == self.target[0] and robot.getLoc()[1] == self.target[1]:
                self.target = None
            else:
                return get_direction_to_target(robot.getLoc(), self.target)
        
        world_estimate = self.world_estimator.runNetwork(explored_map, self.mask)
        digit_softmax = np.exp(self.digit_classifier.runNetwork(world_estimate))
        
        corner_mask = np.zeros((28, 28))
        corner_mask[0, 27] = 1
        corner_mask[27, 27] = 1
        corner_mask[27, 0] = 1
        
        all_explored = np.all(self.mask + corner_mask)
        if all_explored:
            print("Going with best guess - no more info to gain")
        
        if np.max(digit_softmax) > self.confidence_threshold or all_explored:

            # go towards goal
            digit = np.argmax(digit_softmax)
            
            if digit >= 0 and digit <= 2:
                self.target = (0, 27)
            elif digit >= 3 and digit <= 5:
                self.target = (27, 27)
            elif digit >= 6 and digit <= 9:
                self.target = (27, 0)
            else:
                print("Should be unreachable digit value")
                exit(-1)
            return get_direction_to_target(robot.getLoc(), self.target)
        else:
            starting_entropy = entropy(digit_softmax[0])
            best_objective = -10000
            best_target = None
            
            for _ in range(self.num_samples):
                sample_target = None
                while sample_target is None:

                    rand_row = randint(0, 27)
                    rand_col = randint(0, 27)
                    
                    if self.mask[rand_row, rand_col] == 1 or is_corner((rand_col, rand_row)):
                        continue
                    else:
                        sample_target = (rand_col, rand_row)
                
                new_mask = np.zeros((28, 28))
                sim_x = x
                sim_y = y
                steps = 0
                
                while (sim_x, sim_y) != sample_target:
                    steps += 1
                    sim_direction = get_direction_to_target((sim_x, sim_y), sample_target)
                    
                    if sim_direction == "up":
                        sim_y -= 1
                    elif sim_direction == "down":
                        sim_y += 1
                    elif sim_direction == "left":
                        sim_x -= 1
                    elif sim_direction == "right":
                        sim_x += 1

                    # The robot can see all the squares around it
                    # So iterate through all nearby squares
                    for row_inc in range(-1, 2):
                        for col_inc in range(-1, 2):
                            col = sim_x + col_inc
                            row = sim_y + row_inc
                            # if the square is off the map, do nothing
                            if row > 27 or row < 0 or col < 0 or col > 27:
                                continue
                            # Otherwise add to the new mask if new item
                            else:
                                new_mask[row, col] = 0 if self.mask[row, col] == 1 else 1
                            
                    # make sure we actually take an exploratory step
                    if not np.any(new_mask):
                        continue
                
                    new_world_estimate = self.world_estimator.runNetwork(explored_map + world_estimate * new_mask, self.mask + new_mask)
                    new_digit_softmax = np.exp(self.digit_classifier.runNetwork(new_world_estimate))
                    new_entropy = entropy(new_digit_softmax[0])
                    
                    info_gain = starting_entropy - new_entropy
                    objective = info_gain / steps

                    if objective > best_objective:
                        best_objective = objective
                        best_target = sample_target
                    
                self.target = best_target
        
            return get_direction_to_target(robot.getLoc(), self.target)
    
if __name__ == "__main__":
    
    # Create a Map Class Object
    map = Map()

    # Get the current map from the Map Class
    data = map.map
    
    show_paths = False
    num_runs = 10
    average = 0.0
    for i in range(num_runs):
        start = time.time()
        print("Correct digit:", map.number)
        robot = Robot(0, 0)
        navigator = SampleNavigator()
        game = Game(data, map.number, navigator, robot)
        # This loop runs for at most 15 minutes
        while (time.time() - start) < (15 * 60.0):
            found_goal = game.tick()
            if found_goal:
                print(f"Found goal at time step: {game.getIteration()}!")
                break
        print(f"Final Score: {game.score}\n")
        average += game.score / num_runs
        
        if show_paths:
            for loc in navigator.path:
                col = loc[0]
                row = loc[1]
                
                game.truthMap[row, col] = 128
                
            Image.fromarray(game.truthMap).show()
        
        map.getNewMap()
        data = map.map

    print(average)