import gzip
import numpy as np
from random import randint
from scipy.stats import entropy
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork

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
    
    return direction

def is_corner(loc):
    
    x = loc[0]
    y = loc[1]
    
    return (x == 0 and y == 27) or (x == 27 and y == 27) or (x == 27 and y == 0)

class InfoGainNavigator:
    def __init__(self):
        
        self.confidence_threshold = 0.9
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
                # Otherwise add to the new mask if new item
                else:
                    self.mask[row, col] = 1
                    
        if self.target is not None:
            print("Location:", robot.getLoc(), "Target:", self.target)
            if robot.getLoc()[0] == self.target[0] and robot.getLoc()[1] == self.target[1]:
                self.target = None
            else:
                return get_direction_to_target(robot.getLoc(), self.target)
        
        direction = None
        
        world_estimate = self.world_estimator.runNetwork(explored_map, self.mask)
        digit_softmax = np.exp(self.digit_classifier.runNetwork(world_estimate))
        
        if np.max(digit_softmax) > self.confidence_threshold:

            # go towards goal
            digit = np.argmax(digit_softmax)
            print("Digit is", digit)
            
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
            valid_directions = ["up", "down", "left", "right"]
            starting_entropy = entropy(digit_softmax[0])
            best_info_gain = -1000
            best_direction = None
            
            for step in valid_directions:
                if not robot.checkValidMove(step):
                    continue
                
                new_mask = np.zeros((28, 28))
                new_x = x
                new_y = y
                
                if step == "up":
                    new_y -= 1
                elif step == "down":
                    new_y += 1
                elif step == "left":
                    new_x -= 1
                elif step == "right":
                    new_x += 1
                else:
                    print("Should be unreachable (during exploration calculation)")
                    exit(-1)

                # The robot can see all the squares around it
                # So iterate through all nearby squares
                for row_inc in range(-1, 2):
                    for col_inc in range(-1, 2):
                        col = new_x + col_inc
                        row = new_y + row_inc
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

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_direction = step
                    
            direction = best_direction
        
            if direction is None:
                corner_mask = np.zeros((28, 28))
                corner_mask[0, 27] = 1
                corner_mask[27, 27] = 1
                corner_mask[27, 0] = 1
                    
                if np.all(self.mask + corner_mask):
                    self.target = self.goals[0]
                    direction = get_direction_to_target(robot.getLoc(), self.target)
                while direction is None:
                        
                    print("Looping for direction")

                    rand_row = randint(0, 27)
                    rand_col = randint(0, 27)
                    
                    if self.mask[rand_row, rand_col] == 1 or is_corner((rand_col, rand_row)):
                        continue
                    else:
                        self.target = (rand_col, rand_row)
                        print("Get direction last resort")
                        direction = get_direction_to_target(robot.getLoc(), self.target)
        
        return direction
    
if __name__ == "__main__":
    
    # Create a Map Class Object
    map = Map()

    # Get the current map from the Map Class
    data = map.map
    
    for i in range(100):
        print(map.number)
        robot = Robot(0, 0)
        navigator = InfoGainNavigator()
        game = Game(data, map.number, navigator, robot)
        # This loop runs the game for 1000 ticks, stopping if a goal is found.
        for x in range(0, 1000):
            found_goal = game.tick()
            if found_goal:
                print(f"Found goal at time step: {game.getIteration()}!")
                break
        print(f"Final Score: {game.score}")
        
        map.getNewMap()
        data = map.map
