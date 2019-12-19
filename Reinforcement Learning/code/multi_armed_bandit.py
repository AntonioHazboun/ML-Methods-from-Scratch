import numpy as np
from random import randrange
from statistics import mean


class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon
        

    def fit(self, env, steps=1000):
        """
        Trains the MultiArmedBandit on an OpenAI Gym environment
        
        args:
          env - (Env) OpenAI Gym environment with discrete actions and observations
          steps - (int) The number of actions to perform during training

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. shape is
            (number of environment states x number of possible actions)
            rewards - (np.array) 1D sequence of averaged rewards of length 100
        """
        
        
        
        state_action_values = np.zeros((env.observation_space.n,env.action_space.n))
        s = int(np.floor(steps / 100))
        
        
        averaged_rewards = np.zeros(100)
        all_rewards = []
        
        Q = np.zeros(env.action_space.n)
        N = np.zeros(env.action_space.n)
        
        env.reset()
        
        for step_count in range(steps):
            if np.random.random() < self.epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q)
            
            observation, reward, done, info = env.step(action)
            N[action] += 1
            Q[action] += (reward-Q[action])/N[action]
            
            all_rewards.append(reward)
            
            if done == True:
                env.reset()
                
        partitioned_reward_groups = [all_rewards[i * s:(i + 1) * s] for i in range((len(all_rewards) + s - 1) // s )]
            
        averaged_rewards = np.asarray([mean(partition) for partition in partitioned_reward_groups])
            
        for space in range(env.observation_space.n):
            state_action_values[space,:]=Q
        
        return state_action_values, averaged_rewards
            
        
        
        
        
    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode
        
        Args:
          env - (Env) OpenAI Gym environment with discrete actions and observations
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. shape is
            (number of environment states x number of possible actions)

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        
        states = []
        actions = []
        rewards = []
        
        env.reset()
        done = False
        while done == False:
            action = np.argmax(state_action_values[0])
            observation, reward, done, info = env.step(action)
            states.append(observation)
            actions.append(action)
            rewards.append(reward)
        
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        
        return states, actions, rewards
        
