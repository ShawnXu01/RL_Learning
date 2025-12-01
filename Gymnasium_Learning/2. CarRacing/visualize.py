"""visualize.py
Loads a saved checkpoint and runs the CarRacing environment to visualize results.
Usage:
    python visualize.py --checkpoint checkpoints/dqn_final.pth --episodes 1
"""
import argparse
import time
import gymnasium as gym
import torch
import numpy as np
import cv2

from model import create_model
from utils import preprocess_observation, discrete_to_continuous


def visualize(checkpoint: str, episodes: int = 1, max_steps: int = 2000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # try human render, fallback to rgb_array + cv2.imshow
    use_human = False
    try:
        env = gym.make('CarRacing-v3', continuous=False, render_mode='human')
        use_human = True
    except Exception:
        env = gym.make('CarRacing-v3', continuous=False, render_mode='rgb_array')

    obs, _ = env.reset()
    num_actions = env.action_space.n

    model = create_model(device, in_channels=3, num_actions=num_actions)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            img = preprocess_observation(obs)
            with torch.no_grad():
                s = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
                q = model(s)
                action = int(q.argmax(1).item())

            try:
                next_obs, reward, terminated, truncated, _ = env.step(action)
            except Exception:
                cont = discrete_to_continuous(action)
                next_obs, reward, terminated, truncated, _ = env.step(cont)

            if use_human:
                # when render_mode='human' the window is handled by the env
                pass
            else:
                frame = env.render()
                if frame is not None:
                    cv2.imshow('CarRacing - Trained Policy', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        done = True

            obs = next_obs
            done = terminated or truncated
            step += 1

        print(f'Finished visualization episode {ep+1}')

    env.close()
    if not use_human:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False, default='checkpoints/dqn_final.pth')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=2000)
    args = parser.parse_args()
    visualize(args.checkpoint, args.episodes, args.max_steps)
