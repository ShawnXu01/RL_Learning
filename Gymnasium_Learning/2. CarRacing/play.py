import argparse
import os
import gymnasium as gym
import torch
import imageio
import numpy as np

from model import create_model
from utils import preprocess_observation, discrete_to_continuous


def play(checkpoint: str, output: str, max_frames: int = 5000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CarRacing-v3', continuous=False, render_mode='rgb_array')

    obs, _ = env.reset()
    num_actions = env.action_space.n

    model = create_model(device, in_channels=3, num_actions=num_actions)
    # support both raw state_dict and checkpoint dicts that contain 'model_state_dict'
    ckpt_obj = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt_obj, dict) and 'model_state_dict' in ckpt_obj:
        state_dict = ckpt_obj['model_state_dict']
    else:
        state_dict = ckpt_obj
    model.load_state_dict(state_dict)
    model.eval()

    frames = []
    obs, _ = env.reset()
    done = False
    frame_count = 0

    while not done and frame_count < max_frames:
        img = preprocess_observation(obs)
        with torch.no_grad():
            s = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
            q = model(s)
            action = int(q.argmax(1).item())

        # if the env is discrete, step by int; else map to continuous
        step_action = action
        try:
            next_obs, reward, terminated, truncated, _ = env.step(step_action)
        except Exception:
            cont = discrete_to_continuous(action)
            next_obs, reward, terminated, truncated, _ = env.step(cont)

        frame = env.render()
        frames.append(np.array(frame))
        obs = next_obs
        done = terminated or truncated
        frame_count += 1

    env.close()

    if output:
        imageio.mimwrite(output, frames, fps=30)
        print(f"Saved video to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='play_output.mp4')
    parser.add_argument('--max_frames', type=int, default=5000)
    args = parser.parse_args()
    play(args.checkpoint, args.output, args.max_frames)
