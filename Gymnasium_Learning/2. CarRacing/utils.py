import numpy as np
import cv2


def preprocess_observation(obs):
    # obs: HxWx3 uint8 (96x96x3)
    # normalize to [0,1] and transpose to C,H,W
    img = np.array(obs, dtype=np.float32) / 255.0
    # transpose
    img = np.transpose(img, (2, 0, 1))
    return img


def discrete_to_continuous(action_idx: int):
    # map 5 discrete actions to continuous controls (steering, gas, brake)
    # 0: do nothing
    # 1: steer right
    # 2: steer left
    # 3: gas
    # 4: brake
    # NOTE: steering with zero gas often causes the car to not move.
    # To implement the "always press gas" idea we include a base gas
    # value for most actions so the car keeps moving and the agent
    # only needs to learn steering behavior.
    base_gas = 0.8
    if action_idx == 0:
        # do nothing -> keep a baseline gas so car keeps moving
        return np.array([0.0, base_gas, 0.0], dtype=np.float32)
    elif action_idx == 1:
        # steer right while keeping baseline gas
        return np.array([+1.0, base_gas, 0.0], dtype=np.float32)
    elif action_idx == 2:
        # steer left while keeping baseline gas
        return np.array([-1.0, base_gas, 0.0], dtype=np.float32)
    elif action_idx == 3:
        # explicit gas -> full gas
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elif action_idx == 4:
        # brake (keep gas at 0 to allow braking effect)
        return np.array([0.0, 0.0, 0.8], dtype=np.float32)
    else:
        return np.array([0.0, base_gas, 0.0], dtype=np.float32)
