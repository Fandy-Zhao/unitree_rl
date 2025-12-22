# RL包 - Unitree机器人强化学习控制系统

## 概述

本包是一个基于强化学习的Unitree机器人控制系统，专门为Unitree Go2机器人设计。该系统集成了深度视觉处理、运动控制策略和实时ROS通信，支持复杂的机器人运动任务和导航功能。

## 包结构

```
rl/
├── CMakeLists.txt          # CMake构建配置文件
├── package.xml             # ROS包描述文件
├── Action_sim_335-11_flat.npy  # 预训练的动作策略模型
├── README.md               # 本文档
├── scripts/                # 可执行Python脚本
│   ├── action.py          # 主要的动作控制节点
│   └── depth_process.py   # 深度相机数据处理节点
├── src/                    # 源代码目录
│   └── rl/                # Python模块
│       ├── __init__.py
│       ├── unitree_ros_real.py  # ROS实时通信核心模块
│       └── utils.py       # 工具函数
├── include/               # C++头文件（当前为空）
├── rsl_rl/               # 强化学习核心库
│   ├── rsl_rl/           # 主要模块
│   │   ├── algorithms/   # RL算法实现
│   │   │   └── ppo.py    # PPO算法实现
│   │   ├── modules/      # 神经网络模块
│   │   │   ├── actor_critic.py         # Actor-Critic网络
│   │   │   ├── actor_critic_recurrent.py # 循环Actor-Critic网络
│   │   │   ├── depth_backbone.py       # 深度视觉骨干网络
│   │   │   └── estimator.py            # 状态估计器
│   │   ├── env/          # 环境定义
│   │   ├── runners/      # 训练和推理运行器
│   │   ├── storage/      # 数据存储
│   │   └── utils/        # 工具函数
│   ├── setup.py          # Python包安装配置
│   └── README.md         # RSL库文档
├── model/                # 训练好的模型文件
    ├── base_jit.pt       # 基础运动模型
    ├── my_base_jit.pt    # 自定义基础模型
    ├── vision_weight.pt  # 视觉权重模型
    └── my_vision_weight.pt # 自定义视觉模型

```

## 核心功能模块

### 1. 动作控制系统 (`scripts/action.py`)
**主要参数：**
- `--loop_mode` (str, default='timer'): 循环模式（'timer'或'while'）


**主要功能：**
- 实现机器人的高层动作控制策略
- 集成强化学习模型进行实时决策
- 支持站立策略、跑酷策略和运动模式切换
- 处理视觉信息并生成相应的运动指令

**关键特性：**
- 基于预训练模型的动作生成
- 多策略切换机制（站立、跑酷、运动模式）
- 实时ROS通信接口
- 视觉信息集成处理

### 2. 深度视觉处理 (`scripts/depth_process.py`)

**主要参数：**
- `--loop_mode` (str, default='timer'): 循环模式（'timer'或'while'）
- `--sim_real` (str, default='sim'): 运行模式（'sim'或'real'）

**主要功能：**
- 处理RealSense深度相机或仿真平台提供数据
- 实现实时深度图像处理和分析
- 提供环境感知和障碍物检测
- 支持相机参数配置和标定

**关键特性：**
- Intel RealSense相机集成
- 深度图像滤波和处理
- 实时性能优化
- ROS话题发布

### 3. ROS通信核心 (`src/rl/unitree_ros_real.py`)

**主要功能：**
- 实现与Unitree机器人的底层通信
- 处理机器人状态数据和控制指令
- 提供传感器数据接口
- 实现运动学计算和坐标变换

**关键特性：**
- Unitree低级命令/状态处理
- 实时数据流管理
- 多架构支持（x86_64, aarch64）
- 安全性和错误处理

### 4. 强化学习框架 (`rsl_rl/`)

**算法实现：**
- **PPO算法** (`algorithms/ppo.py`): 近端策略优化算法实现
- **Actor-Critic网络** (`modules/actor_critic.py`): 策略和价值函数网络
- **循环网络** (`modules/actor_critic_recurrent.py`): 支持时序记忆的网络结构
- **深度视觉骨干** (`modules/depth_backbone.py`): 视觉特征提取网络
- **状态估计器** (`modules/estimator.py`): 机器人状态估计模块


## 配置系统

### 主配置文件 (`traced/config.json`)

包含系统的主要配置参数：
- **机器人参数**: 关节配置、运动学限制
- **控制参数**: 控制类型、刚度和阻尼设置
- **环境配置**: 地形类型、障碍物设置
- **传感器配置**: 深度相机参数、视野范围
- **训练参数**: 奖励函数、课程学习设置

## 模型文件

### 预训练模型
- `Action_sim_335-11_flat.npy`: 主要的动作策略模型
- `base_jit.pt`: JIT编译的基础运动模型
- `vision_weight.pt`: 视觉处理网络权重

### 自定义模型
- `my_base_jit.pt`: 用户自定义的基础模型
- `my_vision_weight.pt`: 用户自定义的视觉模型

## 依赖项

### ROS依赖
- `roscpp`: C++ ROS客户端库
- `rospy`: Python ROS客户端库
- `sensor_msgs`: 传感器消息定义
- `geometry_msgs`: 几何消息定义
- `cv_bridge`: OpenCV-ROS图像转换
- `unitree_legged_msgs`: Unitree机器人消息定义

### Python依赖
- `torch`: PyTorch深度学习框架
- `numpy`: 数值计算库
- `opencv-python`: 计算机视觉库
- `pyrealsense2`: Intel RealSense SDK

### 系统依赖
- OpenCV: 图像处理库
- CMake 3.0.2+: 构建系统
- C++14: 编译器支持

## 使用方法

### 1. 启动深度处理节点
```bash
rosrun rl depth_process.py
```

### 2. 启动动作控制节点
```bash
rosrun rl action.py
```

### 3. launch启动（还没有搞好）
```bash
roslaunch rl rl.launch
```
## 注意事项
- 注意修改模型地址在scripts/action.py

## 待完善完善功能
- [ ] 完善launch文件，包括参数配置



