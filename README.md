# LeRobot Model Deploy Test

这是一个面向个人使用的 LeRobot 模型训练、离线评估和真机部署备份仓库。主要用于保存已经验证过的 FR3 + 双相机策略适配代码和运行命令，不以通用框架、完整数据发布或长期维护为目标。

## 当前范围

| 路线 | 训练 | 离线评估 | 实时部署 |
| --- | --- | --- | --- |
| LeRobot ACT | 可用 | 可用 | 可用 |
| LeRobot Diffusion Policy | 可用 | 可用 | 可用 |
| LeRobot PI0 / PI0.5 | 可用 | 可用 | 仅提供 runtime 类，未接入统一部署 CLI |
| `DP_train`（Original DP） | **本仓库不可用** | **本仓库不可用** | **本仓库不可用** |
| `openpi_train` | **本仓库不可用** | **本仓库不可用** | **本仓库不可用** |

仓库不包含数据集、训练 checkpoint、Hugging Face 缓存、模型权重、ROS 2 工作区或 Franka 控制服务。

## 目录

```text
policy_adapters/
  lerobotACT_train/    # ACT adapter、训练、评估、runtime
  lerobotDP_train/     # LeRobot DP adapter、训练、评估、runtime
  lerobotPI_train/     # LeRobot PI0/PI0.5 adapter、训练、评估、runtime
  utils/               # 图像、动作空间和公共数据接口
  deploy_live_policy.py
  live_ros_http_robot_io.py
  replay_lerobot_v3.py
运行命令.md
部署说明.md
dataset_pipeline.md    # 数据链路背景笔记；相关采集/导出程序不在本仓库
```

## 环境

- Python 3.12
- PyTorch 2.8 / CUDA 12.8
- LeRobot 0.5.x 源码位于仓库根目录的 `./lerobot`
- 实时部署另需 ROS 2、`rclpy`、`sensor_msgs`、`std_msgs`、触觉消息包和可用的 Franka HTTP 控制服务

准备 LeRobot：

```bash
git clone --branch v0.5.0 https://github.com/huggingface/lerobot.git lerobot
```

安装本项目环境：

```bash
uv sync --inexact
```

本项目的 `pyproject.toml` 使用 `./lerobot` 作为 editable dependency，因此该目录必须存在。

## 使用

完整但精简的训练、离线评估、回放和部署命令见 [运行命令.md](运行命令.md)。所有命令都应从仓库根目录执行，并使用 `python -m policy_adapters...` 形式。

核心数据约定：

```text
observation.state = [x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper_width]
action = [dx, dy, dz, d_rotvec_x, d_rotvec_y, d_rotvec_z, target_gripper_width]
```

`action` 是逐步相对末端增量，不能直接当作绝对位姿下发。详细说明见 [部署说明.md](部署说明.md) 和 [输入数据说明.md](policy_adapters/输入数据说明.md)。

## 真机安全

RobotIO 默认服务地址使用占位符 `http://xxx:5000`。运行前通过环境变量或参数显式设置：

```bash
export FRANKA_SERVER_URL=http://xxx:5000
```

首次运行必须使用 `--dry-run`，并确认数据语义、workspace、速度、夹爪范围、急停和机器人周围环境。仓库代码仅作为个人实验备份，使用者自行承担真机运行风险。

## 说明

- 数据集和 checkpoint 由使用者单独准备。
- PI0/PI0.5 可能需要 Hugging Face 登录和额外模型文件；token 与缓存不得提交到仓库。
- 本仓库主要用于自用和备份，不承诺通用性、复现支持或持续维护。

