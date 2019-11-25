import os
import sys
import numpy as np
import tensorflow as tf
from reinforce_with_baseline import ReinforceWithBaseline

class Core:
    def __init__(self, state_size, action_space):
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_space = list(range(action_space))
        self.model = ReinforceWithBaseline(state_size, action_space)
        self.total_rewards = []

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

        #TODO: if game_state.status == gameover, train()

        action_probability = self.model.call(tf.convert_to_tensor([game_state]))
        action_probability = action_probability.numpy()[0]
        action = np.random.choice(self.action_space, p=action_probability)
        print(sum(action_probability))
        self.states.append(game_state)
        self.actions.append(action)
        self.rewards.append(1)

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

        # TODO:
        # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
        # 2) Compute discounted rewards.
        # 3) Compute the loss from the model and run backpropagation on the model.

        with tf.GradientTape() as tape:
            discounted_rewards = self.discount(self.rewards)
            loss = self.model.loss(tf.convert_to_tensor(self.states), tf.convert_to_tensor(self.actions), tf.convert_to_tensor(discounted_rewards))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        print("Episode Total Reward: {}".format(sum(self.rewards)))
        print("Loss: {}".format(loss))
        self.total_rewards.append(sum(self.rewards)) 

        #reset episode variables
        self.states = []
        self.actions = []
        self.rewards = []

    def print_total_rewards(self, num_previous_round=50):
        average = sum(self.total_rewards[-num_previous_round:]) / num_previous_round
        print("Last {} Average Total Reward: {}".format(num_previous_round, average))

        #Visualize your rewards.
        #visualize_data(self.total_rewards)


