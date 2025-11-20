import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from agent import DQNAgent
from replay_buffer import ReplayBuffer
from model import DQNModel

def train():
    # --- 检查并配置 GPU ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU 可用，训练将在 GPU 上运行。")
        except RuntimeError as e:
            print(e)
    else:
        print("未检测到 GPU, 训练将在 CPU 上运行。")
    # -----------------------------

    # 创建用于训练的非渲染环境
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    replay_buffer = ReplayBuffer(2000)
    main_model = DQNModel(state_size=state_size, action_size=action_size)
    target_model = DQNModel(state_size=state_size, action_size=action_size)
    target_model.model.set_weights(main_model.model.get_weights())

    agent = DQNAgent(state_size, action_size, main_model, target_model, replay_buffer)
    
    num_episodes = 100
    batch_size = 32
    episode_rewards = []
    
    print("--- 开始训练 ---")
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])
            replay_buffer.add(state, action, reward, next_state, done)
            
            if replay_buffer.size() > batch_size:
                agent.learn()
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")
        
        if episode % 10 == 0:
            agent.update_target_network()
    
    env.close()
    print("--- 训练结束 ---")

    # --- 绘制并保存结果图表 ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    chart_filename = 'cartpole_rewards_chart.png'
    plt.savefig(chart_filename)
    print(f"\n奖励图表已保存为 '{chart_filename}'")

    # --- 新增: 保存模型权重 ---
    model_weights_path = 'cartpole_dqn.weights.h5'
    agent.model.model.save_weights(model_weights_path)
    print(f"模型权重已保存到 '{model_weights_path}'")

    # --- 新增: 可视化训练好的代理 ---
    print("\n--- 开始可视化 ---")
    
    # 创建一个用于可视化的新环境
    vis_env = gym.make("CartPole-v1", render_mode="human")
    
    # 在测试时，我们不希望有随机行为，所以将 epsilon 设为 0
    agent.epsilon = 0.0
    
    state, _ = vis_env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    
    while not done:
        # 让 agent 根据学到的策略选择动作
        action = agent.choose_action(state)
        
        # 执行动作
        next_state, _, terminated, truncated, _ = vis_env.step(action)
        done = terminated or truncated
        
        state = np.reshape(next_state, [1, state_size])
    
    vis_env.close()
    print("--- 可视化结束 ---")


if __name__ == "__main__":
    train()