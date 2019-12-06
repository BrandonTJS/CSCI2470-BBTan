import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class A2CModel(tf.keras.Model):
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
        super(A2CModel, self).__init__()
        self.num_actions = num_actions

        #actor network
        self.a1 = tf.keras.layers.Dense(48, activation='relu', use_bias=True, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.a2 = tf.keras.layers.Dense(48, activation='relu', use_bias=True, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.a3 = tf.keras.layers.Dense(48, activation='relu', use_bias=True, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.a4 = tf.keras.layers.Dense(48, activation='relu', use_bias=True, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.a5 = tf.keras.layers.Dense(num_actions, activation='softmax', use_bias=True, kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        #critic network
        self.c1 = tf.keras.layers.Dense(128, activation='relu', use_bias=True, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.c2 = tf.keras.layers.Dense(128, activation='relu', use_bias=True, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.c3 = tf.keras.layers.Dense(1, use_bias=True, kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
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
        output = self.a1(states)
        output = self.a2(output)
        output = self.a3(output)
        output = self.a4(output)
        output = self.a5(output)

        return output

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode
        :return: A [episode_length] matrix representing the value of each state
        """
        output = self.c1(states)
        output = self.c2(output)
        output = self.c3(output)

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
        action_probabilities = -tf.math.log(tf.clip_by_value(action_probabilities, 1e-7, 1))
        test = tf.squeeze(self.value_function(states))
        advantage = tf.subtract(discounted_rewards, test)
        advantage = tf.cast(advantage, dtype=tf.float32)
        advantage = tf.stop_gradient(advantage)
        element_multiply = tf.multiply(action_probabilities, advantage)
        actor_loss = tf.reduce_mean(tf.reduce_sum(element_multiply))

        # advantage = tf.subtract(discounted_rewards, self.value_function(states))
        # advantage = tf.cast(advantage, dtype=tf.float32)
        # advantage = tf.stop_gradient(advantage)

        # action_one_hot = tf.one_hot(actions, self.num_actions, dtype=tf.float32)
        # neg_log_policy = -tf.math.log(tf.clip_by_value(probabilities, 1e-7, 1))
        # actor_loss = tf.reduce_mean(tf.reduce_sum(neg_log_policy * action_one_hot, axis=1) * actor_loss_element)

        critic_loss_element = tf.subtract(discounted_rewards, tf.squeeze(self.value_function(states)))
        critic_loss_element_square = tf.square(critic_loss_element)
        critic_loss = tf.reduce_mean(tf.reduce_sum(critic_loss_element_square))

        print('Actor Loss: ' + str(actor_loss))
        print('Critic Loss: ' + str(critic_loss))

        weighted_actor_loss = tf.multiply(actor_loss, 1)
        weighted_critic_loss = tf.multiply(critic_loss, 0.5)
        return weighted_actor_loss + weighted_critic_loss
