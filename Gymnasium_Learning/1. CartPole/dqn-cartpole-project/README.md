# DQN CartPole Project

This project implements a Deep Q-Network (DQN) algorithm to train an agent to balance a pole on a cart in the "CartPole-v1" environment using the Gymnasium library.

## Project Structure

```
dqn-cartpole-project
├── src
│   ├── agent.py          # Defines the DQNAgent class for implementing the DQN algorithm.
│   ├── model.py          # Defines the DQNModel class for building the DQN model.
│   ├── replay_buffer.py   # Defines the ReplayBuffer class for storing agent experiences.
│   └── train.py          # Entry point for training the agent.
├── requirements.txt      # Lists the required Python libraries.
└── README.md             # Documentation for the project.
```

## Installation

To install the required libraries, run:

```
pip install -r requirements.txt
```

## Usage

To train the DQN agent, execute the following command:

```
python src/train.py
```

This will initialize the environment, agent, and start the training loop.

## DQN Algorithm Overview

The DQN algorithm combines Q-learning with deep neural networks. It uses a replay buffer to store experiences and a target network to stabilize training. The agent learns to predict the best actions to take in the environment based on its experiences.

## License

This project is licensed under the MIT License.