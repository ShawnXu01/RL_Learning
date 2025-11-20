import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# --- 第一部分：定义仿真环境 ---
class MiniVRPEnv(gym.Env):
    """
    微型 VRP 环境定义
    """
    def __init__(self):
        self.grid_size = 5
        # 定义 2 个客户的位置 (x, y)
        self.customer_locs = [(1, 2), (3, 3)]
        self.n_customers = len(self.customer_locs)
        
        # 【动作空间】: 4个离散动作 (0:上, 1:下, 2:左, 3:右)
        self.action_space = spaces.Discrete(4)
        
        # 【状态空间】: [车横坐标x, 车纵坐标y, 客户1状态, 客户2状态]
        # x, y 取值范围 0-4 (共5个)
        # 客户状态 取值范围 0-1 (0没去过, 1去过了)
        self.observation_space = spaces.MultiDiscrete([5, 5, 2, 2])
        
    def reset(self, seed=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        self.agent_pos = [0, 0] # 车总是从 (0,0) 出发
        self.visited = [0] * self.n_customers # 所有客户都没被访问
        return tuple(self.agent_pos + self.visited), {}
    
    def step(self, action):
        """执行一步动作"""
        # 1. 移动逻辑
        x, y = self.agent_pos
        if action == 0: x = max(0, x-1)       # 上 (x减小)
        elif action == 1: x = min(self.grid_size-1, x+1) # 下 (x增大)
        elif action == 2: y = max(0, y-1)     # 左 (y减小)
        elif action == 3: y = min(self.grid_size-1, y+1) # 右 (y增大)
        self.agent_pos = [x, y]
        
        # 2. 计算基础奖励 (每走一步扣1分，鼓励走最短路)
        reward = -1 
        terminated = False
        
        # 3. 检查有没有碰到客户
        current_pos = (x, y)
        for i, loc in enumerate(self.customer_locs):
            # 如果到了客户位置，且这个客户还没被访问过
            if current_pos == loc and self.visited[i] == 0:
                self.visited[i] = 1
                reward += 20 # 奖励访问
                print(f"  -> 成功访问客户 {i+1}!")
        
        # 4. 检查任务是否全部完成
        if all(self.visited):
            reward += 50 # 终极大奖
            terminated = True # 游戏结束
            
        return tuple(self.agent_pos + self.visited), reward, terminated, False, {}

# --- 第二部分：Q-Learning 训练主程序 ---

# 1. 实例化环境
env = MiniVRPEnv()

# 2. 初始化 Q 表
# Q表是一个多维数组。维度是: [x维度, y维度, 客户1维度, 客户2维度, 动作维度]
# 大小为: 5 * 5 * 2 * 2 * 4 = 400 个格子的表格
q_table = np.zeros([5, 5, 2, 2, 4]) 

# 3. 超参数设置
alpha = 0.1    # 学习率 (学得有多快)
gamma = 0.9    # 折扣因子 (有多重视未来)
epsilon = 0.1  # 探索率 (有多少概率乱走)

print("=== 开始训练 (Training) ===")

# 训练 1000 个回合
for episode in range(1000):
    state, _ = env.reset()
    done = False
    
    while not done:
        # --- 步骤 A: 选择动作 (Epsilon-Greedy) ---
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # 探索: 随机选动作
        else:
            # 利用: 查表，选 Q 值最大的动作
            action = np.argmax(q_table[state]) 

        # --- 步骤 B: 执行动作 ---
        next_state, reward, done, _, _ = env.step(action)
        
        # --- 步骤 C: 更新 Q 表 (核心公式) ---
        # 1. 预测的未来最大价值 (Target)
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        
        # 2. 更新当前状态的 Q 值
        current_q = q_table[state][action]
        q_table[state][action] = current_q + alpha * (td_target - current_q)
        
        # 3. 状态跳转
        state = next_state

print("\n=== 训练结束！展示成果 ===")

# --- 第三部分：测试 (看看车学会了没) ---
state, _ = env.reset()
done = False
steps = []

print("开始送货路径演示:")
while not done:
    # 这里不再探索，完全贪婪 (只选最大的)
    action = np.argmax(q_table[state]) 
    
    action_name = ['上','下','左','右'][action]
    steps.append(f"{state[:2]}->{action_name}")
    
    state, reward, done, _, _ = env.step(action)

print(" -> ".join(steps))
print("任务完成！")