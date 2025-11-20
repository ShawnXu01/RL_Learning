import gymnasium as gym
import numpy as np
import os  # 导入 os 模块
from model import DQNModel
from agent import DQNAgent
from replay_buffer import ReplayBuffer # Agent's __init__ needs this, even if we don't use it

def visualize_agent():
    # --- 1. 设置环境和模型参数 ---
    # 确保这些参数与训练时完全一致
    env = gym.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # --- 修改: 创建一个绝对路径来定位权重文件 ---
    # 获取当前脚本文件所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 将目录和文件名拼接成一个完整的、绝对的路径
    weights_path = os.path.join(script_dir, 'cartpole_dqn.weights.h5')

    # --- 2. 重新创建模型和代理 ---
    # 必须创建一个结构完全相同的模型来加载权重
    main_model = DQNModel(state_size=state_size, action_size=action_size)
    
    # 尝试加载权重
    try:
        main_model.model.load_weights(weights_path)
        print(f"模型权重从 '{weights_path}' 加载成功。")
    except Exception as e:
        print(f"加载权重失败: {e}")
        print("请确保 'cartpole_dqn.weights.h5' 文件与此脚本位于同一目录下。")
        env.close()
        return

    # 创建代理。对于可视化，我们不需要目标网络和回放缓冲区，
    # 但为了匹配 DQNAgent 的初始化要求，我们创建虚拟的实例。
    dummy_target_model = DQNModel(state_size=state_size, action_size=action_size)
    dummy_replay_buffer = ReplayBuffer(1)
    
    agent = DQNAgent(state_size, action_size, main_model, dummy_target_model, dummy_replay_buffer)
    
    # --- 3. 设置代理为“测试模式” ---
    # 关闭随机探索 (exploration)，只使用学到的策略 (exploitation)
    agent.epsilon = 0.0

    # --- 4. 运行可视化仿真 ---
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0
    
    print("--- 按 Enter 键开始可视化... ---")
    input()

    while not done:
        # a. 代理根据当前状态选择最佳动作
        action = agent.choose_action(state)
        
        # b. 环境执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # c. 更新状态和总奖励
        state = np.reshape(next_state, [1, state_size])
        total_reward += reward

    print(f"可视化结束。最终奖励: {total_reward}")
    env.close()

if __name__ == "__main__":
    visualize_agent()