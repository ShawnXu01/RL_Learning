import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from agent import DQNAgent
from replay_buffer import ReplayBuffer
from model import DQNModel


def train():
    # Configure device for PyTorch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        print("GPU available, training will run on GPU.")
    else:
        print("No GPU detected, training will run on CPU.")

    # Ensure files are saved next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    replay_buffer = ReplayBuffer(2000)

    main_model = DQNModel(state_size=state_size, action_size=action_size, device=device)
    target_model = DQNModel(state_size=state_size, action_size=action_size, device=device)
    target_model.model.load_state_dict(main_model.model.state_dict())

    agent = DQNAgent(state_size, action_size, main_model, target_model, replay_buffer)

    num_episodes = 500
    batch_size = 32
    episode_rewards = []
    # per-episode aggregated metrics
    losses = []
    mean_qs = []
    max_qs = []
    epsilons = []

    print("TRAINING Started\n")

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False

        # per-episode metrics
        episode_losses = []
        episode_mean_qs = []
        episode_max_qs = []

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])
            replay_buffer.add(state, action, reward, next_state, done)

            if replay_buffer.size() > batch_size:
                metrics = agent.learn()
                if metrics is not None:
                    episode_losses.append(metrics.get('loss'))
                    episode_mean_qs.append(metrics.get('mean_q'))
                    episode_max_qs.append(metrics.get('max_q'))

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")
        # Decay epsilon once per episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()

        # save per-episode aggregated metrics
        if episode_losses:
            losses.append(float(np.mean(episode_losses)))
            mean_qs.append(float(np.mean(episode_mean_qs)))
            max_qs.append(float(np.mean(episode_max_qs)))
        else:
            losses.append(None)
            mean_qs.append(None)
            max_qs.append(None)

    env.close()
    print("TRAINING Finished\n")

    # Plot and save the results chart (raw + moving average)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_rewards, label='Episode Reward')
    # compute moving average
    window = 10
    if len(episode_rewards) >= window:
        mov_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window, num_episodes + 1), mov_avg, label=f'{window}-episode MA')
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()
    chart_filename = os.path.join(script_dir, 'cartpole_rewards_chart.png')
    plt.savefig(chart_filename)
    print(f"\nReward Plot saved as '{chart_filename}'")

    # Save training history for later analysis/plotting
    history_path = os.path.join(script_dir, 'training_history.npz')
    np.savez(history_path, episode_rewards=np.array(episode_rewards),
             losses=np.array(losses, dtype=object), mean_qs=np.array(mean_qs, dtype=object),
             max_qs=np.array(max_qs, dtype=object), epsilons=np.array(epsilons, dtype=object))
    print(f"Training history saved as '{history_path}'")

    # Save model weights (PyTorch .pt) in script dir
    model_weights_path = os.path.join(script_dir, 'cartpole_dqn.pt')
    agent.model.save(model_weights_path)
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
