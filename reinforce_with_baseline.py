import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions

        #actor network
        self.dense1 = tf.keras.layers.Dense(48, input_dim=state_size, activation='relu', use_bias=True, kernel_initializer=tf.random_uniform_initializer())
        self.dense5 = tf.keras.layers.Dense(48, activation='relu', use_bias=True, kernel_initializer=tf.random_uniform_initializer())
        self.dense6 = tf.keras.layers.Dense(48, activation='relu', use_bias=True, kernel_initializer=tf.random_uniform_initializer())
        self.dense7 = tf.keras.layers.Dense(48, activation='relu', use_bias=True, kernel_initializer=tf.random_uniform_initializer())
        self.dense2 = tf.keras.layers.Dense(num_actions, activation='softmax', use_bias=True, kernel_initializer=tf.random_uniform_initializer())

        #critic network
        self.dense3 = tf.keras.layers.Dense(48, input_dim=state_size, activation='relu', use_bias=True, kernel_initializer=tf.random_uniform_initializer())
        self.dense8 = tf.keras.layers.Dense(48, activation='relu', use_bias=True, kernel_initializer=tf.random_uniform_initializer())
        self.dense4 = tf.keras.layers.Dense(1, use_bias=True, kernel_initializer=tf.random_uniform_initializer())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        print('INIT')

    @tf.function
    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        of each state in the episode
        """
        output = self.dense1(states)
        output = self.dense5(output)
        output = self.dense6(output)
        output = self.dense7(output)
        output = self.dense2(output)

        return output

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode
        :return: A [episode_length] matrix representing the value of each state
        """
        output = self.dense3(states)
        output = self.dense8(output)
        output = self.dense4(output)

        return output

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the handout to see how this is done.

        Remember that the loss is similar to the loss as in reinforce.py, with one specific change.

        1) Instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. Here, advantage is defined as discounted_rewards - state_values, where state_values is calculated by the critic network.
        
        2) In your actor loss, you must set advantage to be tf.stop_gradient(discounted_rewards - state_values). You may need to cast your (discounted_rewards - state_values) to tf.float32. tf.stop_gradient is used here to stop the loss calculated on the actor network from propagating back to the critic network.
        
        3) To calculate the loss for your critic network. Do this by calling the value_function on the states and then taking the sum of the squared advantage.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        probabilities = self.call(states)
        actions_reshape = tf.stack([range(len(actions)), actions], axis=1)
        action_probabilities = tf.gather_nd(probabilities, actions_reshape)
        action_probabilities = tf.math.negative(tf.math.log(action_probabilities))
        actor_loss_element = tf.stop_gradient(tf.cast(tf.subtract(discounted_rewards, self.value_function(states)), dtype=tf.float32))
        element_multiply = tf.multiply(action_probabilities, actor_loss_element)
        actor_loss = tf.reduce_sum(element_multiply)

        critic_loss_element = tf.subtract(discounted_rewards, self.value_function(states))
        critic_loss_element_square = tf.square(critic_loss_element)
        critic_loss = tf.reduce_sum(critic_loss_element_square)

        weighted_actor_loss = tf.multiply(actor_loss, 1)
        weighted_critic_loss = tf.multiply(critic_loss, 0.5)
        return tf.add(weighted_actor_loss, weighted_critic_loss)
