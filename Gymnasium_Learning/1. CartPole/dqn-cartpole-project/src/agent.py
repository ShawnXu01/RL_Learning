import numpy as np
import random
import torch
import torch.nn as nn


class DQNAgent:
    def __init__(self, state_size, action_size, main_model, target_model, replay_buffer,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32,
                 lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.model = main_model          # DQNModel wrapper
        self.target_model = target_model # DQNModel wrapper
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.device = self.model.device
        # optimizer and loss for PyTorch
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Explore
        # Use the main model for action selection
        q_values = self.model.predict(state)
        return int(np.argmax(q_values[0]))  # Exploit

    def learn(self):
        # require at least batch_size experiences before learning
        if self.replay_buffer.size() < self.batch_size:
            return None

        minibatch = self.replay_buffer.sample(self.batch_size)
        # Unpack
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to numpy arrays and then torch tensors
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        states_t = torch.from_numpy(states.astype(np.float32)).to(self.device)
        next_states_t = torch.from_numpy(next_states.astype(np.float32)).to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Predict Q-values (current)
        self.model.model.train()
        current_q_values = self.model.model(states_t)  # (batch, action_size)

        batch_size_actual = states_t.shape[0]
        batch_indices = torch.arange(batch_size_actual, device=self.device)

        # Double DQN target: actions from main network, values from target network
        with torch.no_grad():
            next_q_values_main = self.model.model(next_states_t)
            next_actions = torch.argmax(next_q_values_main, dim=1)
            next_q_values_target = self.target_model.model(next_states_t)
            max_next_q = next_q_values_target[batch_indices, next_actions]
            q_update = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        # Build target Q-values and assign
        target_q_values = current_q_values.clone().detach()
        target_q_values[batch_indices, actions_t] = q_update

        # Compute loss between current predictions and target
        loss = self.loss_fn(current_q_values, target_q_values)

        # Backprop with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Return training metrics for logging
        mean_q = current_q_values.mean().item()
        max_q = current_q_values.max().item()
        return {'loss': float(loss.item()), 'mean_q': mean_q, 'max_q': max_q}

    def update_target_network(self, tau=1.0):
        """Copy weights from main model to target model."""
        for p, tp in zip(self.model.model.parameters(), self.target_model.model.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)