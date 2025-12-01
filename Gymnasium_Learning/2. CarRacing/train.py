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
import glob
import re

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

    # ensure paths are created relative to this script so TrainingHistory
    # is placed inside the CarRacing folder regardless of current cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs('checkpoints', exist_ok=True)
    history_dir = os.path.join(script_dir, 'TrainingHistory')
    os.makedirs(history_dir, exist_ok=True)
    # directory for training reward output (npz + plot)
    output_dir = os.path.join(script_dir, 'Output_TrainingReward')
    os.makedirs(output_dir, exist_ok=True)
    viz_path = os.path.join(script_dir, 'visualize.py')

    # prefer a single overwriteable checkpoint to allow infinite continue-run behavior
    # single checkpoint path (will be overwritten each chunk)
    single_ckpt = f'{history_dir}/dqn_checkpoint.pth'
    latest_ckpt = None
    start_episode = 1
    episode_rewards = []

    if os.path.exists(single_ckpt):
        latest_ckpt = single_ckpt
        try:
            ckpt = torch.load(latest_ckpt, map_location=device)
            epi = ckpt.get('episode', None)
            if isinstance(epi, int) and epi > 0:
                start_episode = epi + 1
            episode_rewards = ckpt.get('episode_rewards', [])
            total_steps = ckpt.get('total_steps', 0)
            # load model and optimizer state into policy/optimizer
            model_state = ckpt.get('model_state_dict', None)
            if model_state is not None:
                try:
                    policy.load_state_dict(model_state)
                    target.load_state_dict(policy.state_dict())
                except Exception:
                    print('Warning: failed to load model_state from single checkpoint')
            opt_state = ckpt.get('optimizer_state_dict', None)
            if opt_state is not None:
                try:
                    optimizer.load_state_dict(opt_state)
                except Exception:
                    print('Warning: failed to load optimizer state from single checkpoint')
            print(f'Found single checkpoint {latest_ckpt}, will resume from episode {start_episode}')
        except Exception as e:
            print('Failed to read single checkpoint metadata:', e)
    else:
        # fallback: legacy chunk files
        ckpt_files = glob.glob(f'{history_dir}/dqn_chunk_*.pth') + glob.glob('checkpoints/dqn_chunk_*.pth')
        if ckpt_files:
            def epi_from_name(name):
                m = re.search(r'dqn_chunk_(\d+)\.pth$', name)
                return int(m.group(1)) if m else -1

            ckpt_files_sorted = sorted(ckpt_files, key=lambda x: epi_from_name(x))
            latest_ckpt = ckpt_files_sorted[-1]
            epi = epi_from_name(latest_ckpt)
            if epi > 0:
                start_episode = epi + 1
                print(f'Found legacy checkpoint {latest_ckpt}, will resume from episode {start_episode}')
                # load model and optimizer state below (after model/optimizer created)

    # start from detected episode (resume) or 1
    # Optional warmup: run some random steps to populate replay buffer before learning
    if args.warmup_steps > 0:
        print(f'Warming up for {args.warmup_steps} random steps to fill replay buffer...')
        warmed = 0
        obs_w, _ = env.reset()
        state_w = preprocess_observation(obs_w)
        while warmed < args.warmup_steps:
            # bias random actions slightly towards gas so car moves during warmup
            a = random.randrange(num_actions)
            # with 40% prob force gas action to ensure movement
            if random.random() < 0.4:
                a = 3
            next_obs_w, reward_w, term_w, trunc_w, _ = env.step(a)
            done_w = term_w or trunc_w
            next_state_w = preprocess_observation(next_obs_w)
            buffer.push(state_w, a, reward_w, next_state_w, done_w)
            warmed += 1
            total_steps += 1
            state_w = next_state_w
            if done_w:
                obs_w, _ = env.reset()
                state_w = preprocess_observation(obs_w)
        print('Warmup finished; starting training')

    # If the detected resume start is already past the requested max_episodes,
    # extend the target so we run at least one more chunk by default. This
    # avoids the script immediately exiting when users re-run with the same
    # `--max_episodes` they used previously.
    end_episode = args.max_episodes
    if start_episode > end_episode:
        # run at least one chunk's worth of episodes from the resume point
        new_end = start_episode + args.chunk_size - 1
        print(f"Resume start ({start_episode}) > requested max_episodes ({end_episode}). ")
        print(f"Extending target to {new_end} so training continues for one chunk.")
        end_episode = new_end

    for episode in range(start_episode, end_episode + 1):
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        episode_reward = 0.0
        done = False

        episode_steps = 0

        while True:
            total_steps += 1
            episode_steps += 1
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
                torch.save(policy.state_dict(), f'{history_dir}/dqn_{total_steps}.pth')

            if done_flag:
                break

        # decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        print(f"Episode {episode} reward={episode_reward:.2f} steps={total_steps} ep_steps={episode_steps} eps={epsilon:.3f}")

        # record episode reward for plotting
        episode_rewards.append(episode_reward)

        if episode % args.save_episode == 0:
            torch.save(policy.state_dict(), f'{history_dir}/dqn_last.pth')

        # if we've reached a chunk boundary, save checkpoint with optimizer + history,
        # run visualization and then exit so user can restart to continue training
        if episode % args.chunk_size == 0:
            ckpt_path = f'{history_dir}/dqn_chunk_{episode}.pth'
            try:
                torch.save({
                    'episode': episode,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_rewards': episode_rewards,
                    'total_steps': total_steps,
                }, ckpt_path)
                # overwrite single checkpoint so next run resumes from here
                single_ckpt = f'{history_dir}/dqn_checkpoint.pth'
                torch.save({
                    'episode': episode,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode_rewards': episode_rewards,
                    'total_steps': total_steps,
                }, single_ckpt)
                print(f'Saved chunk checkpoint to {ckpt_path} and updated {single_ckpt}')
            except Exception as e:
                print('Failed saving checkpoint:', e)

            # save history and plot for this chunk
            episodes = np.arange(1, len(episode_rewards) + 1)
            np.savez(os.path.join(output_dir, 'training_history.npz'), episodes=episodes, rewards=np.array(episode_rewards))
            if len(episode_rewards) > 0:
                try:
                    plt.figure()
                    plt.plot(episodes, episode_rewards)
                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                    plt.title('Training Reward')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'training_reward.png'))
                    plt.close()
                    print(f"Saved training plot to {os.path.join(output_dir, 'training_reward.png')}")
                except Exception:
                    pass

            # launch visualization for this checkpoint
            try:
                print('Launching visualization to show trained policy...')
                subprocess.run([sys.executable, viz_path, '--checkpoint', ckpt_path])
            except Exception as e:
                print('Failed to launch visualization:', e)

            print(f'Chunk up to episode {episode} finished. Exiting so you can restart to continue.')
            return

    env.close()

    # save final model and training history
    final_ckpt = f'{history_dir}/dqn_final.pth'
    torch.save(policy.state_dict(), final_ckpt)

    # ensure episode_rewards exists
    if 'episode_rewards' not in locals():
        episode_rewards = []

    episodes = np.arange(1, len(episode_rewards) + 1)
    np.savez(os.path.join(output_dir, 'training_history.npz'), episodes=episodes, rewards=np.array(episode_rewards))

    # plot reward vs episode
    if len(episode_rewards) > 0:
        plt.figure()
        plt.plot(episodes, episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Reward')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_reward.png'))
        print(f"Saved training plot to {os.path.join(output_dir, 'training_reward.png')}")
        try:
            plt.show()
        except Exception:
            pass

    # automatically run visualization script to show trained policy
    try:
        print('Launching visualization to show trained policy...')
        subprocess.run([sys.executable, viz_path, '--checkpoint', final_ckpt])
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
    parser.add_argument('--chunk_size', type=int, default=100, help='Number of episodes per chunk before saving and visualizing')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of random env steps to run before training to fill replay buffer')
    args = parser.parse_args()
    train(args)
