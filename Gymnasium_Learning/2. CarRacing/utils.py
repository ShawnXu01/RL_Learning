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
    # Provide a small gas value when steering to improve exploration.
    if action_idx == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    elif action_idx == 1:
        return np.array([+1.0, 0.2, 0.0], dtype=np.float32)
    elif action_idx == 2:
        return np.array([-1.0, 0.2, 0.0], dtype=np.float32)
    elif action_idx == 3:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elif action_idx == 4:
        return np.array([0.0, 0.0, 0.8], dtype=np.float32)
    else:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
