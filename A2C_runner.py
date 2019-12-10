import os
import sys
import numpy as np
import tensorflow as tf
from A2C_model import A2CModel
from pylab import *

class A2CRunner:
    def __init__(self, state_size, action_space):
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_space = list(range(action_space))
        self.model = A2CModel(state_size, action_space)
        self.total_rewards = []

        self.checkpoint_dir = './checkpoints/A2C'
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=5)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        metrics_log_dir = 'logs/gradient_tape/' + current_time + '/metrics'
        self.metric_summary_writer = tf.summary.create_file_writer(metrics_log_dir)
        
        self.avg_grad = None
        self.grad_count = 0

    def visualize_data(self, total_rewards):
        """
        Takes in array of rewards from each episode, visualizes reward over episodes.

        :param rewards: List of rewards from all episodes
        """

        x_values = arange(0, len(total_rewards), 1)
        y_values = total_rewards
        plot(x_values, y_values)
        xlabel('episodes')
        ylabel('cumulative rewards')
        title('Reward by Episode')
        grid(True)
        show()


    def discount(self, rewards, discount_factor=.99):
        """
        Takes in a list of rewards for each timestep in an episode, 
        and returns a list of the sum of discounted rewards for
        each timestep. Refer to the slides to see how this is done.

        :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
        :param discount_factor: Gamma discounting factor to use, defaults to .99
        :return: discounted_rewards: list containing the sum of discounted rewards for each timestep in the original
        rewards list
        """
        discounted_rewards = rewards.copy()
        for i in range(len(discounted_rewards)-1,0,-1):
            index = i - 1
            discounted_rewards[index] = discounted_rewards[index+1] * discount_factor + discounted_rewards[index]

        return discounted_rewards
    
    def calculate_action(self, game_state):
        """
        Generates lists of states, actions, and rewards for one complete episode.

        :param env: The openai gym environment
        :param model: The model used to generate the actions
        :return: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps
        in the episode
        """

        #print(game_state)
        action_probability = self.model.call(tf.convert_to_tensor([game_state]))
        action_probability = action_probability.numpy()[0]
        #print(action_probability)
        action = np.random.choice(self.action_space, p=action_probability)

        self.states.append(game_state)
        self.actions.append(action)
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

        return action


    def train(self):
        """
        This function should train your model for one episode.
        Each call to this function should generate a complete trajectory for one episode (lists of states, action_probs,
        and rewards seen/taken in the episode), and then train on that data to minimize your model loss.
        Make sure to return the total reward for the episode.

        :param env: The openai gym environment
        :param model: The model
        :return: The total reward for the episode
        """

        # Define our metrics
        actor_loss_metric = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
        critic_loss_metric = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)
        total_loss_metric = tf.keras.metrics.Mean('total_loss', dtype=tf.float32)
        reward_metric = tf.keras.metrics.Mean('reward', dtype=tf.float32)
        level_metric = tf.keras.metrics.Mean('level', dtype=tf.float32)

        with tf.GradientTape() as tape:
            discounted_rewards = self.discount(self.rewards)
            #print(discounted_rewards)
            loss, actor_loss, critic_loss = self.model.loss(tf.convert_to_tensor(self.states), tf.convert_to_tensor(self.actions), tf.convert_to_tensor(discounted_rewards))
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if self.grad_count == 0:
            self.avg_grad = gradients
        else:
            self.avg_grad = [tf.add(x[0], x[1]) for x in zip(self.avg_grad, gradients)]
        self.grad_count += 1
        if self.grad_count % 32 == 0:
            print('Applying Gradient')
            self.avg_grad = [x/32 for x in self.avg_grad]
            self.model.optimizer.apply_gradients(zip(self.avg_grad, self.model.trainable_variables))
            self.grad_count = 0

        #Log metrics
        actor_loss_metric(actor_loss)
        critic_loss_metric(critic_loss)
        total_loss_metric(loss)
        reward_metric(sum(self.rewards))
        level_metric(len(self.rewards) + 1)

        #Write metrics to file
        current_epoch = len(self.total_rewards)
        with self.metric_summary_writer.as_default():
            tf.summary.scalar('actor_loss', actor_loss_metric.result(), step=current_epoch)
            tf.summary.scalar('critic_loss', critic_loss_metric.result(), step=current_epoch)
            tf.summary.scalar('total_loss', total_loss_metric.result(), step=current_epoch)
            tf.summary.scalar('reward', reward_metric.result(), step=current_epoch)
            tf.summary.scalar('level', level_metric.result(), step=current_epoch)

        print("Episode Total Reward: {}".format(sum(self.rewards)))
        print("Loss: {}".format(loss))
        self.total_rewards.append(sum(self.rewards)) 

        #reset episode variables
        self.states = []
        self.actions = []
        self.rewards = []

        actor_loss_metric.reset_states()
        critic_loss_metric.reset_states()
        total_loss_metric.reset_states()
        reward_metric.reset_states()
        level_metric.reset_states()

    def print_total_rewards(self, num_previous_round=50):
        average = sum(self.total_rewards[-num_previous_round:]) / num_previous_round
        print("Last {} Average Total Reward: {}".format(num_previous_round, average))

        #Visualize your rewards.
        #visualize_data(self.total_rewards)

    def save(self):
        print('Saving Checkpoint')
        self.manager.save()
    
    def load(self):
        if os.path.exists(self.checkpoint_dir):
            #Check if chkpts exists
            num_of_files = len(os.listdir(self.checkpoint_dir))
            if num_of_files > 0:
                print('Restoring Checkpoint')
                self.checkpoint.restore(self.manager.latest_checkpoint)
                #Sanity check
                #print(self.model.call(tf.convert_to_tensor([range(129)])))
