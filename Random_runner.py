import os
import sys
import numpy as np
import tensorflow as tf
from A2C_model import A2CModel
from pylab import *
import random

class RandomRunner:
    def __init__(self):
        self.rewards = []
        self.total_rewards = []
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        metrics_log_dir = 'logs/gradient_tape/' + current_time + '/metrics'
        self.metric_summary_writer = tf.summary.create_file_writer(metrics_log_dir)

    
    def calculate_action(self, game_state):
        level_map = game_state[1171:] #1 + 351 + 819 | ball, bot_x, tile_map
        num_non_block = level_map.count(0) 
        incentive = num_non_block/100.0
        disincentive = 0
        for row in reversed(range(9)): #9 rows in level_map
            for column in range(7): #7 columns in level_map
                value = level_map[row*7 + column]
                if value  != 0:
                    disincentive += row
        disincentive = disincentive/100.0
        self.rewards.append(2 + incentive - disincentive)

        return random.randint(0,25)


    def train(self):
        # Define our metrics
        reward_metric = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        level_metric = tf.keras.metrics.Mean('level', dtype=tf.float32)

        #Log metrics
        reward_metric(sum(self.rewards))
        level_metric(len(self.rewards) + 1)

        #Write metrics to file
        current_epoch = len(self.total_rewards)
        with self.metric_summary_writer.as_default():
            tf.summary.scalar('reward', reward_metric.result(), step=current_epoch)
            tf.summary.scalar('level', level_metric.result(), step=current_epoch)

        print("Episode Total Reward: {}".format(sum(self.rewards)))
        self.total_rewards.append(sum(self.rewards)) 

        #reset episode variables
        self.rewards = []

        reward_metric.reset_states()
        level_metric.reset_states()

    def save(self):
        pass
    
    def load(self):
        pass
