import os
import argparse
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys

from model import create_model
from replay_buffer import ReplayBuffer
from utils import preprocess_observation


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Print device info so user can confirm GPU usage
    print(f"Using device: {device}")
    if device.type == 'cuda':
        try:
            print(f"CUDA device count: {torch.cuda.device_count()}, name: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
        except Exception:
            # in case getting device name fails for some reason
            pass

    env = gym.make('CarRacing-v3', continuous=False)

    obs, _ = env.reset()
    in_channels = 3
    num_actions = env.action_space.n

    policy = create_model(device, in_channels=in_channels, num_actions=num_actions)
    target = create_model(device, in_channels=in_channels, num_actions=num_actions)
    target.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    buffer = ReplayBuffer(args.replay_size)

    epsilon = args.epsilon_start
    total_steps = 0

    os.makedirs('checkpoints', exist_ok=True)

    for episode in range(1, args.max_episodes + 1):
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        episode_reward = 0.0
        done = False

        while True:
            total_steps += 1
            # epsilon-greedy
            if random.random() < epsilon:
                action = random.randrange(num_actions)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q = policy(s)
                    action = int(q.argmax(1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done_flag = terminated or truncated
            next_state = preprocess_observation(next_obs)

            buffer.push(state, action, reward, next_state, done_flag)

            state = next_state
            episode_reward += reward

            # learn
            if len(buffer) >= args.batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(args.batch_size)

                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                q_values = policy(states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q = target(next_states_t).max(1)[0].unsqueeze(1)
                    target_q = rewards_t + args.gamma * (1.0 - dones_t) * next_q

                loss = criterion(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % args.target_update == 0:
                soft_update(target, policy, args.tau)

            if total_steps % args.save_every == 0:
                torch.save(policy.state_dict(), f'checkpoints/dqn_{total_steps}.pth')

            if done_flag:
                break

        # decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        print(f"Episode {episode} reward={episode_reward:.2f} steps={total_steps} eps={epsilon:.3f}")

        # record episode reward for plotting
        try:
            episode_rewards.append(episode_reward)
        except NameError:
            episode_rewards = [episode_reward]

        if episode % args.save_episode == 0:
            torch.save(policy.state_dict(), f'checkpoints/dqn_last.pth')

    env.close()

    # save final model and training history
    final_ckpt = 'checkpoints/dqn_final.pth'
    torch.save(policy.state_dict(), final_ckpt)

    # ensure episode_rewards exists
    if 'episode_rewards' not in locals():
        episode_rewards = []

    episodes = np.arange(1, len(episode_rewards) + 1)
    np.savez('training_history.npz', episodes=episodes, rewards=np.array(episode_rewards))

    # plot reward vs episode
    if len(episode_rewards) > 0:
        plt.figure()
        plt.plot(episodes, episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Reward')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('training_reward.png')
        print('Saved training plot to training_reward.png')
        try:
            plt.show()
        except Exception:
            pass

    # automatically run visualization script to show trained policy
    try:
        print('Launching visualization to show trained policy...')
        subprocess.run([sys.executable, 'visualize.py', '--checkpoint', final_ckpt])
    except Exception as e:
        print('Failed to launch visualization:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--replay_size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.02)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--target_update', type=int, default=1000)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--save_episode', type=int, default=50)
    args = parser.parse_args()
    train(args)
