import gymnasium as gym
import numpy as np
import os
from model import DQNModel
from agent import DQNAgent
from replay_buffer import ReplayBuffer


def visualize_agent():

    # 设置环境和模型参数
    env = gym.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 权重文件路径（与 train.py 使用相同位置）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, 'cartpole_dqn.pt')

    # 重新创建模型和代理
    main_model = DQNModel(state_size=state_size, action_size=action_size)

    # 尝试加载权重
    try:
        main_model.load(weights_path)
        print(f"Model weights loaded successfully from '{weights_path}'.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        print("Please ensure the 'cartpole_dqn.pt' file is in the same directory as this script.")
        env.close()
        return

    dummy_target_model = DQNModel(state_size=state_size, action_size=action_size)
    dummy_replay_buffer = ReplayBuffer(1)
    agent = DQNAgent(state_size, action_size, main_model, dummy_target_model, dummy_replay_buffer)
    agent.epsilon = 0.0

    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = np.reshape(next_state, [1, state_size])
        total_reward += reward

    print(f"Visualization finished. Final reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    visualize_agent()