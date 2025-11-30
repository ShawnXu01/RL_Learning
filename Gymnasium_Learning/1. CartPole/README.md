# 使用 DQN 解决 CartPole 问题

本项目使用深度 Q 网络 (Deep Q-Network, DQN) 算法，训练一个智能体在 Gymnasium 的 "CartPole-v1" 环境中学会如何平衡推车上的杆子。

## 项目结构

本项目包含以下核心文件，均位于 `dqn-cartpole-project/src` 目录下：

```
dqn-cartpole-project/
└── src/
    ├── train.py              # 训练脚本：启动和管理整个训练流程。
    ├── run_visualization.py  # 可视化脚本：加载已训练的模型并展示其表现。
    ├── agent.py              # 智能体：实现 DQN 算法的核心逻辑，包括决策和学习。
    ├── model.py              # 模型：定义神经网络的架构。
    ├── replay_buffer.py      # 经验回放池：存储和采样智能体的经验。
    └── cartpole_dqn.weights.h5 # (训练后生成) 保存模型权重的文件。
    └── cartpole_rewards_chart.png # (训练后生成) 训练奖励曲线图。
```

## 环境设置

1.  确保你已经安装了 Python 3.8 或更高版本。
2.  安装所有必需的 Python 库：
    ```bash
    pip install gymnasium tensorflow matplotlib
    ```
    *（注意：如果你的机器有兼容的 NVIDIA GPU 并已配置好 CUDA，TensorFlow 将会自动使用 GPU 进行加速。）*

## 如何训练智能体

训练是整个项目的第一步，它会从头开始教智能体如何平衡杆子。

1.  打开终端，并确保你的路径位于 `dqn-cartpole-project/src` 文件夹内。
2.  运行以下命令来启动训练：

    ```bash
    python train.py
    ```

### 训练过程与产出

-   **过程**: 终端会开始打印每一轮 (episode) 的训练进度和该轮获得的总奖励，例如：
2.  安装所有必需的 Python 库（使用 PyTorch）：
    ```bash
    pip install gymnasium torch torchvision matplotlib
    ```
    *（注意：如果你的机器有兼容的 NVIDIA GPU 并已配置好 CUDA，PyTorch 将会自动使用 GPU 进行加速。）*
-   **训练产出**: 训练脚本运行结束后（默认 100 轮），会在 `dqn-cartpole-project/src` 文件夹下生成两个关键文件：
    1.  `cartpole_rewards_chart.png`: 一张折线图，可视化了智能体在整个训练过程中的奖励变化趋势。
    2.  `cartpole_dqn.weights.h5`: 一个 HDF5 文件，包含了训练好的神经网络的模型权重。这是智能体“大脑”的最终形态。
## 如何可视化训练成果
    2.  `cartpole_dqn.pt`: 一个 PyTorch 模型文件，包含了训练好的神经网络的模型权重。你可以用该文件来加载模型并运行可视化。
在你成功完成训练并生成 `cartpole_dqn.weights.h5` 文件后，你可以运行可视化脚本来亲眼看看智能体的表现。

1.  确保终端路径仍然在 `dqn-cartpole-project/src` 文件夹内。
在你成功完成训练并生成 `cartpole_dqn.pt` 文件后，你可以运行可视化脚本来亲眼看看智能体的表现。

    ```bash
    python run_visualization.py
    ```

### 可视化过程

-   **加载模型**: 脚本会首先加载 `cartpole_dqn.weights.h5` 文件。
-   **启动仿真**: 终端会提示 `--- Press Enter to start visualization ---`。按下回车键后，**会弹出一个窗口**，实时显示智能体控制小车平衡杆子的动画。
-   **结束**: 当杆子倾斜角度过大或小车移出边界时，一局游戏结束，可视化窗口会自动关闭。

---

## 许可

本项目采用 MIT 许可。