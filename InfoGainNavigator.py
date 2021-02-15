import gzip
import numpy as np
from random import randint
from scipy.stats import entropy
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork

class InfoGainNavigator:
    def __init__(self):
        
        self.confidence_threshold = 0.75
        self.world_estimator = WorldEstimatingNetwork()
        self.digit_classifier = DigitClassificationNetwork()
        self.mask = np.zeros((28, 28))
        self.path = []

    def getAction(self, robot, explored_map):
        
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
                    
        # print(x, y)
        # print(self.mask, "\n")
        direction = None
        
        world_estimate = self.world_estimator.runNetwork(explored_map, self.mask)
        digit_softmax = np.exp(self.digit_classifier.runNetwork(world_estimate))
        
        if np.max(digit_softmax) > self.confidence_threshold:

            # go towards goal
            digit = np.argmax(digit_softmax)
            print("Digit is", digit)
            
            if digit >= 0 and digit <= 2:
                if x > 0:
                    direction = "left"
                else:
                    direction = "down"
            elif digit >= 3 and digit <= 5:
                if x < 27:
                    direction = "right"
                else:
                    direction = "down"
            elif digit >= 6 and digit <= 9:
                if x < 27:
                    direction = "right"
                else:
                    direction = "up"
            else:
                print("Should be unreachable digit value")
                exit(-1)
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
                while direction is None:

                    randNumb = randint(0, 3)

                    if randNumb == 0:
                        direction = 'left'
                    if randNumb == 1:
                        direction = 'right'
                    if randNumb == 2:
                        direction = 'down'
                    if randNumb == 3:
                        direction = 'up'
                    
                    # If it is not a valid move, reset
                    if not robot.checkValidMove(direction):
                        direction = None
        
        return direction
    
if __name__ == "__main__":
    
    # Create a Map Class Object
    map = Map()

    # Get the current map from the Map Class
    data = map.map
    
    for i in range(10):
        print(map.number)
        robot = Robot(0, 0)
        navigator = InfoGainNavigator()
        game = Game(data, map.number, navigator, robot)
        # This loop runs the game for 1000 ticks, stopping if a goal is found.
        for x in range(0, 1000):
            print(x)
            found_goal = game.tick()
            if found_goal:
                print(f"Found goal at time step: {game.getIteration()}!")
                break
        print(f"Final Score: {game.score}")
        
        map.getNewMap()
        data = map.map
