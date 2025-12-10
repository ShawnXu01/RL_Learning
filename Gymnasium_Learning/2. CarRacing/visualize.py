"""visualize.py
Loads a saved checkpoint and runs the CarRacing environment to visualize results.
Usage:
    python visualize.py --checkpoint TrainingHistory/dqn_checkpoint.pth --episodes 1
"""
import argparse
import time
import os
import glob
import gymnasium as gym
import torch
import numpy as np
import cv2

from model import create_model
from utils import preprocess_observation, discrete_to_continuous


def visualize(checkpoint: str, episodes: int = 1, max_steps: int = 2000, seed: int = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # If seed provided, fix RNGs so the chosen track and any sampling is reproducible
    if seed is not None:
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Create environment in continuous mode so we can apply continuous
    # control (steering, gas, brake). For visualization we always map the
    # model's discrete output to a continuous action that includes baseline gas.
    use_human = False
    try:
        env = gym.make('CarRacing-v3', continuous=True, render_mode='human')
        use_human = True
    except Exception:
        env = gym.make('CarRacing-v3', continuous=True, render_mode='rgb_array')

    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    # model was trained with a discrete action space (5 actions). The
    # continuous environment's action_space may not have `.n`, so fall
    # back to 5 if needed.
    if hasattr(env.action_space, 'n'):
        num_actions = env.action_space.n
    else:
        num_actions = 5

    model = create_model(device, in_channels=3, num_actions=num_actions)

    # Resolve checkpoint path: prefer user-supplied path (if exists).
    # Otherwise search common candidate locations relative to this script,
    # including a commonly misspelled folder name `TrainingHistroy`.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def find_latest_from_pattern(patterns):
        candidates = []
        for p in patterns:
            matches = glob.glob(p)
            if matches:
                candidates.extend(matches)
        if not candidates:
            return None
        # choose the newest file by modification time
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    resolved_ckpt = None
    # if user provided a checkpoint and it exists, use it
    if checkpoint and os.path.exists(checkpoint):
        resolved_ckpt = checkpoint
    else:
        # search several likely locations relative to the script
        exact_candidates = [
            os.path.join(script_dir, 'TrainingHistory', 'dqn_checkpoint.pth'),
            os.path.join(script_dir, 'TrainingHistroy', 'dqn_checkpoint.pth'),
            os.path.join(script_dir, '..', 'TrainingHistory', 'dqn_checkpoint.pth'),
            os.path.join(script_dir, '..', 'TrainingHistroy', 'dqn_checkpoint.pth'),
        ]
        for c in exact_candidates:
            if os.path.exists(c):
                resolved_ckpt = c
                break

        if resolved_ckpt is None:
            # look for any matching dqn_*.pth files and pick the newest
            patterns = [
                os.path.join(script_dir, 'TrainingHistory', 'dqn_*.pth'),
                os.path.join(script_dir, 'TrainingHistroy', 'dqn_*.pth'),
                os.path.join(script_dir, '..', 'TrainingHistory', 'dqn_*.pth'),
                os.path.join(script_dir, '..', 'TrainingHistroy', 'dqn_*.pth'),
            ]
            latest = find_latest_from_pattern(patterns)
            if latest:
                resolved_ckpt = latest

    if resolved_ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found. Tried the provided path '{checkpoint}' and looked in:\n"
            f"  {os.path.join(script_dir, 'TrainingHistory')}\n"
            f"  {os.path.join(script_dir, 'TrainingHistroy')}\n"
            "Pass a valid path with --checkpoint or create a 'TrainingHistory' folder inside the CarRacing folder."
        )

    print(f'Loading checkpoint: {resolved_ckpt}')
    ckpt_obj = torch.load(resolved_ckpt, map_location=device)
    if isinstance(ckpt_obj, dict) and 'model_state_dict' in ckpt_obj:
        state_dict = ckpt_obj['model_state_dict']
    else:
        state_dict = ckpt_obj
    model.load_state_dict(state_dict)
    model.eval()

    for ep in range(episodes):
        if seed is not None:
            obs, _ = env.reset(seed=seed)
        else:
            obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            img = preprocess_observation(obs)
            with torch.no_grad():
                s = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
                q = model(s)
                action = int(q.argmax(1).item())

            # map discrete model output to continuous control and always step
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
    parser.add_argument('--checkpoint', type=str, required=False, default='TrainingHistory/dqn_checkpoint.pth')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1028, help='Optional seed to fix track generation and RNGs')
    args = parser.parse_args()
    visualize(args.checkpoint, args.episodes, args.max_steps, seed=args.seed)
