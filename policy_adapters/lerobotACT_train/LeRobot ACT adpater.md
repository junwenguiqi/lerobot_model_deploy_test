# LeRobot ACT Adapter

本文对应当前代码：

```text
policy_adapters/lerobotACT_train/lerobot_act_adapter.py
policy_adapters/lerobotACT_train/train_lerobot_act_minimal.py
policy_adapters/lerobotACT_train/deploy_lerobot_act_runtime.py
policy_adapters/lerobotACT_train/offline_eval_lerobot_act.py
policy_adapters/deploy_live_policy.py
```

公共字段、动作转换、图像预处理工具在：

```text
policy_adapters/utils/
```

## 1. 输入输出语义

当前 ACT 使用本项目的 canonical observation：

```text
observation.images.front: HWC RGB uint8
observation.images.wrist: HWC RGB uint8
observation.state: float32[7]
observation.tactile_left/right: 可存在，但 vanilla ACT 不使用
```

`LeRobotACTAdapter.to_policy_input()` 会转换成 ACTPolicy 输入：

```text
observation.images.front: [B, 3, 256, 256] float32, range [0,1]
observation.images.wrist: [B, 3, 256, 256] float32, range [0,1]
observation.state: [B, 7] float32
```

图像尺寸默认是 `256x256`。adapter 会在训练/部署侧 resize，因此 dataset 原始视频不必须提前变成 256，但 front/wrist shape 和颜色语义要保持可靠。

## 2. 动作语义

ACT 本身只拟合连续向量，不知道动作代表什么。本项目固定为：

```text
action[7] = [
  delta_x, delta_y, delta_z,
  delta_rotvec_x, delta_rotvec_y, delta_rotvec_z,
  target_gripper_width
]
```

这是 step-wise relative end-effector action：

```text
action_t = target_{t+1} relative to state_t
```

部署时不能把 `action[7]` 当成绝对 pose 发给机器人，必须经过：

```python
adapter.action_to_target(current_state7, action7)
```

得到：

```text
target_pose7 = [x, y, z, qx, qy, qz, qw]
target_gripper_width
```

统一部署脚本还会对 `action_raw` 做速度缩放、delta 限幅、workspace/gripper 裁剪，最后发送安全后的 `safe_target`。

## 3. 训练配置

当前训练入口默认值：

```text
chunk_size = 50
n_action_steps = 5
image_size = 256
normalization = mean_std
use_tactile = False
```

LeRobotDataset 使用：

```python
delta_timestamps = adapter.delta_timestamps(fps)
```

ACT 的 action chunk 是：

```text
{"action": [i / fps for i in range(chunk_size)]}
```

训练 batch 进入官方 `ACTPolicy.forward()` 前会先经过：

```python
batch = adapter.to_training_batch(raw_batch, device=device)
```

该步骤负责：

```text
1. 只保留 front / wrist / state / action。
2. 过滤当前 vanilla ACT 不使用的 tactile 字段。
3. 将图像转为 CHW float 并 resize。
4. 检查 action shape 为 [B, chunk_size, 7]。
5. 读取或补齐 action_is_pad: [B, chunk_size]。
```

`action_is_pad` 最终由官方 LeRobot `ACTPolicy.forward()` 用作 loss mask；本项目 adapter 只是确保该字段存在且 shape 正确。

## 4. 官方 LeRobot 与本项目 Adapter 的边界

官方 LeRobot 提供：

```text
ACTConfig
ACTPolicy
LeRobotDataset
```

本项目 `LeRobotACTAdapter` 提供：

```text
canonical observation -> ACTPolicy input
LeRobotDataset batch -> ACTPolicy.forward batch
action_raw[7] -> ActionTarget
本项目 action/state 语义约束
```

所以不是替代官方 LeRobot，而是给官方 policy 接上本项目的数据和机器人控制语义。

## 5. 部署与离线检查

ACT checkpoint 由 `LeRobotACTRuntime.from_checkpoint()` 恢复：

```text
checkpoint -> ACTConfig -> ACTPolicy -> load_state_dict
checkpoint/run_config -> LeRobotACTAdapter config
dataset_stats -> mean/std normalization
```

在线部署推荐统一入口：

```text
policy_adapters/deploy_live_policy.py --policy-kind act
```

离线检查入口：

```text
policy_adapters/lerobotACT_train/offline_eval_lerobot_act.py
```

当前 vanilla ACT 不消费触觉。如果未来要用触觉，需要扩展 input_features、to_training_batch、to_policy_input、normalization 和部署同步策略。
