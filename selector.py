import os
import sys
import numpy as np
import tensorflow as tf
from A2C_runner import A2CRunner
import enum 

class ModelType(enum.Enum): 
    Random = 1
    A2C = 2

class ModelSelector:
    def __init__(self, model_type, *args):
        self.model_type = model_type
        if model_type == ModelType.A2C:
            self.model = A2CRunner(args[0], args[1])
        elif model_type == ModelType.Random:
            pass
        else:
            raise Exception('Invalid Model Type')

        

    def game_action_handler(self, game_state):
        if self.model_type == ModelType.A2C:
            return self.model.calculate_action(game_state)
        elif model_type == ModelType.Random:
            return random.randint(0,350)

    def game_over_handler(self, game_state):
        if self.model_type == ModelType.A2C:
            self.model.print_total_rewards(num_previous_round=50)
            self.model.train()
        elif model_type == ModelType.Random:
            pass

    def save_model(self):
        self.model.save()
    
    def load_model(self):
        self.model.load()