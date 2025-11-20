import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from agent import DQNAgent
from replay_buffer import ReplayBuffer
from model import DQNModel

def train():
    # Configure GPU - Use CUDA for accelerated training
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU available, training will run on GPU.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected, training will run on CPU.")

    # Create a non-rendering environment for training
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    replay_buffer = ReplayBuffer(2000)
    main_model = DQNModel(state_size=state_size, action_size=action_size)
    target_model = DQNModel(state_size=state_size, action_size=action_size)
    target_model.model.set_weights(main_model.model.get_weights())
    agent = DQNAgent(state_size, action_size, main_model, target_model, replay_buffer)
    
    num_episodes = 100
    batch_size = 32
    episode_rewards = []
    
    print("TRAINING Started\n")

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])
            replay_buffer.add(state, action, reward, next_state, done)
            
            if replay_buffer.size() > batch_size:
                agent.learn()
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")
        
        if episode % 10 == 0:
            agent.update_target_network()
    
    env.close()
    print("TRAINING Finished\n")

    # Plot and save the results chart
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    chart_filename = 'cartpole_rewards_chart.png'
    plt.savefig(chart_filename)
    print(f"\nReward Plot saved as '{chart_filename}'")

    # Save model weights
    model_weights_path = 'cartpole_dqn.weights.h5'
    agent.model.model.save_weights(model_weights_path)
    print(f"Model weights saved to '{model_weights_path}'")
    
    # Create a new environment for visualization
    vis_env = gym.make("CartPole-v1", render_mode="human")
    
    # Set epsilon to 0 during testing to avoid random actions
    agent.epsilon = 0.0
    
    state, _ = vis_env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    
    while not done:
        # Let the agent choose actions based on the learned policy
        action = agent.choose_action(state)
        
        # Execute the action
        next_state, _, terminated, truncated, _ = vis_env.step(action)
        done = terminated or truncated
        
        state = np.reshape(next_state, [1, state_size])
    
    vis_env.close()
    print("--- Visualization Finished ---")


if __name__ == "__main__":
    train()