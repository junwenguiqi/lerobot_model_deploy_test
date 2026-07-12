# Dataset Pipeline 改进方案：Raw 采集、LeRobot v3 导出与部署

## 1. data collect -> load -> model input -> action
架构：
```text
训练pipeline：
实时传感器
  -> SensorBuffer + TimeSync
  -> ObservationBuilder
  -> RawDatasetWriter
  -> raw dataset: PNG + tactile zarr + parquet/jsonl
raw dataset
  -> raw_to_lerobot_v3.py
  -> training dataset: LeRobot v3 风格 parquet + MP4 videos

部署pipeline：
实时传感器
  -> SensorBuffer + TimeSync
  -> ObservationBuilder
  -> OnlinePolicyAdapter
  -> deployed policy
```

核心原则：
raw 采集保持可靠、可调试；离线导出器负责生成 LeRobot v3 训练数据；部署时不要模拟 LeRobot 文件读取，而是直接构造出和 LeRobot DataLoader 读取后等价的内存 observation。


raw dataset = 原始可靠数据源
LeRobot v3 dataset = 训练 / 回放 / 导出格式
online adapter = 部署输入格式


## Raw Dataset 格式
```text
raw/
  command_events.jsonl
  robot_state_events.jsonl
  vr_controller_events.jsonl

image/
  front/episode_000000/frame_000000.png
  wrist/episode_000000/frame_000000.png

tactile/
  tactile_left/data.zarr
  tactile_left/meta.jsonl
  tactile_right/data.zarr
  tactile_right/meta.jsonl

data/
  chunk-000/file-000.parquet
  episodes/episode_000000.parquet

meta/
  info.json
  episodes.jsonl
  config.yaml
```

### raw dataset 中主要训练字段
```text
observation.state: float32[7]
  [current_x, current_y, current_z,
   current_rotvec_x, current_rotvec_y, current_rotvec_z,
   current_gripper_width]

action: float32[7]
  [delta_x, delta_y, delta_z,
   delta_rotvec_x, delta_rotvec_y, delta_rotvec_z,
   target_gripper_width]

action.sent_action8: float32[8]
  实际发给机器人控制接口的原始绝对命令，保留用于调试

image.path:
  front RGB PNG 相对路径

wrist_image.path:
  wrist RGB PNG 相对路径

tactile.left.zarr_index / tactile.right.zarr_index:
  对应左右触觉 zarr 中的帧索引
```

## 共享 ObservationBuilder
ObservationBuilder作用：负责从原始数据采集到抽象为原始数据通用接口
这个模块应该被两条路径共享：
- 数据采集
- 在线部署
原始数据通用接口：后面不管是转化为lerobot dataset v3，还是给模型部署侧使用
```python
obs = {
    "timestamp": t,
    "image": {
        "front": front_rgb_uint8,   # H, W, 3
        "wrist": wrist_rgb_uint8,   # H, W, 3
    },
    "state": state_vec,
    "tactile": {
        "left": tactile_left_vec,
        "right": tactile_right_vec,
    },
    "sync": {
        "front_dt": ...,
        "wrist_dt": ...,
        "state_dt": ...,
        "tactile_left_dt": ...,
        "tactile_right_dt": ...,
    },
}
```

### 共享 ObservationBuilder
职责：
- 复用 nearest-neighbor sync 逻辑。
- 校验 image、wrist image、action、state、VR、tactile 的时间差。
- 从当前 robot state 构造 `observation.state` 7 维。
- 从目标 pose 和当前 pose 构造 `action` 7 维相对动作。
- 保留 sync diagnostics。
- 提供 `camera_info_to_record`、`tactile_row_fields` 等公共辅助函数。


## 离线导出：Raw Dataset -> LeRobot v3
### 共享ObservationBuilder -> RawDatasetWriter
- 创建 raw dataset 目录结构。
- 写 front/wrist RGB PNG。
- 写 tactile zarr 和 tactile meta jsonl。
- 写 per-episode parquet。
- 写 master parquet。
- 写 raw command/state/VR jsonl。
- 写 `meta/info.json`、`meta/config.yaml`、`meta/episodes.jsonl`。

当前 Tashan/Paxini recorder 已经切到公共 `RawDatasetWriter`：
Tashan/Paxini 的差异通过 profile 保留：
```text
TASHAN_RAW_WRITER_PROFILE
PAXINI_RAW_WRITER_PROFILE
```

### Raw Dataset -> LeRobot v3 转换
职责：
- 读取 raw episode parquet。
- 读取 raw PNG。
- 读取 tactile zarr。
- 写 LeRobot v3 风格 parquet、metadata、MP4 videos。
- 分别校验 `tactile_msg_package`，防止 Tashan/Paxini 导出器用错。

逻辑：
   - 读取 raw episode parquet 和 image index。
   - 将 front/wrist PNG 序列编码成 MP4。
   - 写出 LeRobot 风格 metadata 和 parquet。
   - 将 tactile 导出为固定形状 parquet columns。


导出lerobot dataset结构：
```text
data/
  chunk-000/file-000.parquet

meta/
  info.json
  tasks.parquet
  episodes/chunk-000/file-000.parquet
  stats.json

videos/
  observation.images.front/chunk-000/file-000.mp4
  observation.images.wrist/chunk-000/file-000.mp4

tactile/

```

LeRobot feature names：
```text
observation.images.front
observation.images.wrist
observation.state
observation.tactile_left
observation.tactile_right
action
timestamp
frame_index
episode_index
task_index
index
```

### tactile paxini raw dataset and lerobot dataset
raw: float32[] data  raw 里保存为 zarr，dtype 是 float32。
进入 LeRobot dataset: 会把 zarr 里的每一帧 tactile 读出来，然后写进 LeRobot 的主 parquet

### tactile tashan raw dataset and lerobot dataset
raw: uint8[] data  raw 里保存为 zarr，dtype 是 uint8。
进入 LeRobot dataset: 会把 zarr 里的每一帧 tactile 读出来，然后写进 LeRobot 的主 parquet

### Validation / Inspect 工具
检查能力：
- raw dataset 字段、维度、图片、zarr、episode 对齐检查。
- LeRobot dataset metadata、parquet、video、episode range 检查。
- raw 和 LeRobot 的逐帧 state/action/tactile 对齐检查。
- raw PNG 与导出 MP4 解码帧抽样对比。
- RGB/BGR 通道错误检测。
- 从 LeRobot dataset 读取 canonical sample 并打印 shape/dtype/range。


### 大规模采集训练建议
固定流程：
```text
1. 采集 raw dataset
2. check_raw_dataset
3. raw_to_lerobot_v3
4. validate_lerobot_v3 --raw-root
5. inspect_lerobot_v3_sample
6. lerobot-dataset-viz 可视化抽查
```

## 归一化方式：
LeRobot ACT:
  image: [0,255] -> [0,1] -> MEAN_STD，范围不固定
  state/proprio: MEAN_STD，范围不固定
  action: MEAN_STD，训练时归一化，推理后反归一化，范围不固定

diffusion_policy:
  image: [0,255] -> [0,1] -> [-1,1]，可选再 ImageNet normalize
  low_dim/state: 通常 min/max -> [-1,1]
  action: 通常 min/max -> [-1,1]，输出后反归一化
  注意：超出训练集 min/max 时可能超过 [-1,1]

openpi:
  image: [0,255] -> [-1,1]，不走 norm_stats
  state/action:
    PI0 用 mean/std，范围不固定
    PI0.5/FAST 用 q01/q99 -> 约 [-1,1]，但不 clip
  模型输出 action 后再 Unnormalize 回真实动作

总体上，这几类策略都会把视觉、本体感知、动作等连续输入/目标变换到模型友好的相近数值量级，通常集中在 0 附近，常见范围约为 [-1,1] 或 [-几,几]。
视觉图片：基本都被处理到 [-1,1] 或接近标准化后的常见范围。
本体感知/state：LeRobot ACT 是 mean/std，典型值在 -几 到 +几；diffusion_policy 多数是 [-1,1]；openpi 取决于 PI0/PI0.5，但也会控制在接近 0 附近或约 [-1,1]。
动作/action：训练时也会变成类似量级，推理输出后再反归一化回真实动作空间。

## 训练
训练路径：
```text
LeRobot MP4/parquet
  -> DataLoader
  -> decoded images + state + tactile
  -> model adapter
  -> model
```

## 部署
部署时不需要写 MP4 或 parquet 再推理。
部署路径：
```text
live camera/state/tactile
  -> ObservationBuilder
  -> same semantic observation
  -> model adapter
  -> model
```

### OnlinePolicyAdapter
部署 adapter 应该构造出和训练 DataLoader 读完 LeRobot 后等价的内存数据：
```python
{
    "observation.images.front": front_rgb_uint8,
    "observation.images.wrist": wrist_rgb_uint8,
    "observation.state": state_vec,
    "observation.tactile_left": tactile_left_vec,
    "observation.tactile_right": tactile_right_vec,
}
```
然后由 model-specific adapter 转成具体模型需要的输入。


model input adapter
```text
live sensors
  -> ObservationBuilder
  -> OnlinePolicyAdapter
  -> canonical observation
```



## 语义方面
1. 看数据集生成代码
这里决定 action 到底是什么：关节位置、末端绝对 pose、末端相对 delta、夹爪宽度等。
看 raw_to_lerobot_v3 / exporter 里怎么写 observation.state 和 action。

2. 看 policy config
确认它期待哪些键：
observation.images.*
observation.state
action
以及 action dim、state dim、image size、action horizon、observation horizon。

3. 看 DataLoader / transform / normalizer
这里决定数据有没有被 resize、stack、normalize，以及 action chunk 是怎么从未来帧取出来的。

4. 看 policy 代码
确认它真的用了哪些字段。比如 tactile 即使在 batch 里，如果 forward() 没有读它，就不会进模型。

5. 看 deployment/eval 代码
如果输出 action 后直接发给 robot joint controller，那就是 joint action；如果经过 delta -> absolute pose 转换，再发 /pose，那就是相对末端动作。


对当前项目来说，策略语义其实要分两种情况：
  从零训练 ACT/DP：ACT/DP 本身不强制 action 是什么。
    直接使用
    observation.state = 当前末端状态
    action = [dx, dy, dz, d_rotvec_x, d_rotvec_y, d_rotvec_z, target_gripper_width]
    这个定义就好
    唯一要求：训练导出和部署 adapter 完全一致，它就是这个语义。

  如果是用别人的 pretrained checkpoint：
    反查它原来训练数据集的语义。否则 action 维度一样也可能完全不能用。

## 进一步考虑的问题
“能被 LeRobot Dataset 读”不等于“语义上已经天然适配 ACT/DP/OpenPI/StarVLA”。现在要分三层看：数据结构、图像约束、策略语义。

1. dataset带着触觉数据可以，可以在OnlinePolicyAdapter做处理，如不加载这个触觉
或者
训练时导出两个训练版本：tactile 字段可保留但训练不使用和完全没有tactile字段
部署时直接从采集与同步结束的对齐帧，直接处理成OnlinePolicyAdapter后的数据形式；OnlinePolicyAdapter 必须复现训练时 DataLoader/transform 之后、进入 policy 之前的语义，而不是只复现字段名字。
形式兼容不代表语义兼容
形式上已经接近 LeRobot v3，能被数据读取层吃进去。
但策略语义还需要主动对齐

### 视觉RGB图像语义分析
视觉尺寸与多视角限制
| 框架/策略 | 是否强制 224x224 | 多视角限制 | 对你当前 front 540x960 / wrist 480x640 的影响 |
|---|---:|---|---|
| LeRobot ACT | 不强制 224 | 支持多相机；代码里没有严格要求多相机同尺寸 | 大概率能跑，但高分辨率很吃显存，建议导出训练版时统一 resize |
| LeRobot DP / diffusion | 不强制 224 | **要求所有 image feature shape 一致** | 当前 front/wrist 尺寸不同，直接双相机训练大概率会失败 |
| 原版 `diffusion_policy` | 不强制 224 | 由 `shape_meta` 控制；多相机支持更灵活 | 可支持不同相机，但要在 config 里写清楚；共享视觉 backbone 时最好统一尺寸 |
| OpenPI | 默认目标是 **224x224** | 默认期待 `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` 三个键 | 可自动 resize/pad 到 224，但要做 key mapping；缺失相机用 zero image + mask |
| StarVLA | 通常训练配置用 224x224 | 支持 list 多视角，但依赖 modality/config | 建议按 config resize 到 224，并把 front/wrist 映射成它期望的 primary/wrist/image list |


模型input的眼睛视角和腕部视角图像统一：256*256

### 动作语义
训练 label 怎么定义
部署时怎么反变换

推荐：eef相对增量或者绝对joint
这里使用eef相对增量

step-wise delta：
```text
每一行 action_t = target_{t+1} relative to state_t
```
展开为：
```text
action_t = [
  x_{t+1} - x_t,
  y_{t+1} - y_t,
  z_{t+1} - z_t,
  rot_delta(state_t -> target_{t+1}),
  gripper_width_{t+1}
]
```

ACT/DP 虽然会一次输出未来 action chunk，例如 8 步：

```text
chunk_t = [
  action_t,
  action_{t+1},
  action_{t+2},
  ...
  action_{t+7}
]
```

但 chunk 只是模型输出形状和训练采样方式。chunk 里的每个元素仍然是 step-wise delta：
```text
action_t     = target_{t+1} relative to state_t
action_{t+1} = target_{t+2} relative to state_{t+1}
action_{t+2} = target_{t+3} relative to state_{t+2}
...
```

所以模型和监督标签语义并不矛盾：
```text
模型预测语义：基于历史观测，预测未来一段连续局部控制命令
监督标签语义：这一段连续局部控制命令中的每一项都是 step-wise delta
```

部署时如果执行 chunk 的前 2-4 步，也按 step-wise 方式执行：
```text
第 1 步：pred[0] 叠加到当前实际末端 pose
第 2 步：pred[1] 叠加到执行后的当前实际末端 pose
第 3 步：pred[2] 叠加到再下一步当前实际末端 pose
然后重新观测、重新预测
```


### 各策略视觉上和动作维度上语义
```text
策略语义 = dataset/exporter 字段定义
        + DataLoader/transform 如何取历史观测和未来 action chunk
        + policy forward/select_action 实际使用哪些字段
        + deployment adapter 如何把模型输出反变换成机器人命令
```