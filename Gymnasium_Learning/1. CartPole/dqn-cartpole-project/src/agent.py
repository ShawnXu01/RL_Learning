import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential

class DQNAgent:
    def __init__(self, state_size, action_size, main_model, target_model, replay_buffer, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.model = main_model          # Use the passed main model
        self.target_model = target_model # Use the passed target model
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Explore
        # Use the main model for action selection
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit

    def learn(self): # The batch_size argument is not needed here
        if self.replay_buffer.size() < self.batch_size:
            return

        minibatch = self.replay_buffer.sample(self.batch_size)
        
        # Unpack the experiences into separate arrays
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to numpy arrays for vectorized operations
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Predict Q-values for current states using the main model
        current_q_values = self.model.predict(states)
        # Predict Q-values for next states using the target model
        next_q_values = self.target_model.predict(next_states)
        
        # This will be our target for training
        target_q_values = current_q_values.copy()
        
        # Find the row indices for the batch
        batch_indices = np.arange(self.batch_size)
        
        # Calculate the Q-value update using the Bellman equation
        # For terminal states (done=True), the future reward is 0
        q_update = rewards + self.gamma * np.amax(next_q_values, axis=1) * (1 - dones)
        
        # Update the Q-value for the specific action that was taken
        target_q_values[batch_indices, actions] = q_update
        
        # Train the main model on the whole batch at once
        self.model.model.fit(states, target_q_values, epochs=1, verbose=0)

        # Decay epsilon for exploration-exploitation trade-off
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copy weights from main model to target model."""
        self.target_model.model.set_weights(self.model.model.get_weights())