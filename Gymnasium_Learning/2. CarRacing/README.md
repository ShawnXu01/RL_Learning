CarRacing DQN Project
=====================

这是一个基于 `gymnasium` 中 `CarRacing-v3` 环境的最小 DQN 项目模板，放在 `Gymnasium_Learning/2. CarRacing` 目录下。

快速开始（Windows - cmd）：

1) 创建并激活虚拟环境（可选但推荐）：

```
python -m venv .venv
.venv\Scripts\activate
```

2) 安装依赖：

```
pip install -r requirements.txt
```

3) 训练（示例）：

```
python train.py
```

4) 使用训练好的模型播放并保存视频：

```
python play.py --checkpoint TrainingHistory/dqn_checkpoint.pth --output play_output.mp4
```

文件说明：
- `model.py`: 简单的 CNN Q 网络
- `replay_buffer.py`: 经验回放缓冲
- `utils.py`: 预处理与动作映射
- `train.py`: DQN 训练主脚本
- `play.py`: 运行保存的模型并导出视频

说明：此项目为教学与原型用途。像 CarRacing 这样的像素输入任务通常需要大量训练资源与更复杂的算法（PPO/IMPALA/SAC 等），此处提供的是可运行的最小 DQN 框架。
