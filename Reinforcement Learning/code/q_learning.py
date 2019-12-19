import numpy as np
from statistics import mean


class QLearning:
    """
    QLearning reinforcement learning agent

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      discount - (float) The discount factor. Controls the perceived value of
        future reward relative to short-term reward.
      adaptive - (bool) Whether to use an adaptive policy for setting
        values of epsilon during training
    """

    def __init__(self, epsilon=0.2, discount=0.95, adaptive=False):
        self.epsilon = epsilon
        self.discount = discount
        self.adaptive = adaptive

    def fit(self, env, steps=1000):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment

        args:
          env - (Env) OpenAI Gym environment with discrete actions and observations
          steps - (int) The number of actions to perform during training

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. shape is
            (number of environment states x number of possible actions)
            rewards - (np.array) 1D sequence of averaged rewards of length 100
        """
        
        # Q(s, a)←(1- a )Q(s, a) + a(r + gmaxa’Q(s’, a’))
        
        state_action_values = np.zeros((env.observation_space.n,env.action_space.n))
        state_action_counts = np.zeros((env.observation_space.n,env.action_space.n))
        
        s = int(np.floor(steps/100))
        
        
        rewards = []
        total_rewards = 0
        
        current_state = env.reset()
        
        for step_count in range(steps):
            progress = step_count/steps
            if (np.random.uniform() < self._get_epsilon(progress)):
                action = env.action_space.sample()
            else:
                if not np.any(state_action_values[current_state]):
                    action = env.action_space.sample()
                else:
                    action = np.argmax(state_action_values[current_state])
            
            observation, reward, done, info = env.step(action)
            
            state_action_counts[current_state,action] += 1
            
            alpha = 1/state_action_counts[current_state,action]
            
            state_action_values[current_state,action] = state_action_values[current_state,action] + alpha * (reward + self.discount * np.amax(state_action_values[observation]) - state_action_values[current_state][action])
            
            current_state = observation
            
            if done == True:
                current_state = env.reset()
                
            total_rewards += reward
            
            if (step_count + 1) % s == 0:
                rewards.append(total_rewards / s)
                total_rewards = 0
                
            
            
        return state_action_values, np.asarray(rewards)

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the QLearning algorithm and the state action values. Predictions are
        run for exactly one episode.

        Arguments:
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
        
        current_state = env.reset()
        done = False
        while done == False:
            action = np.argmax(state_action_values[current_state])
            observation, reward, done, info = env.step(action)
            states.append(observation)
            actions.append(action)
            rewards.append(reward)
            current_state = observation
        
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        
        return states, actions, rewards

    def _get_epsilon(self, progress):
        """
        Retrieves the current value of epsilon. 

        Arguments:
            progress - (float) value between 0 and 1 that indicates the
                training progess. calculated as current_step / steps.
        """
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        """
        Modifies epsilon such that it shrinks with learner progress, controls how 
        much exploratory behaior there is after model has identified tennable strategies

        Arguments:
            progress - (float) value between 0 and 1 that indicates the
                training progess. calculated as current_step / steps.
        """
        return (1 - progress) * self.epsilon
