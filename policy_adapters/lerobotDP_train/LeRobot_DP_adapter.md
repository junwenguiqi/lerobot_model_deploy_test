# LeRobot Diffusion Policy Adapter

本目录只包含 LeRobot 0.5.x 的 `DiffusionPolicy` 路线，不包含 Original Stanford Diffusion Policy。

主要文件：

```text
lerobot_dp_adapter.py            数据和动作适配
train_lerobot_dp_minimal.py      训练入口
offline_eval_lerobot_dp.py       离线预测与 GT 对比
deploy_lerobot_dp_runtime.py     checkpoint runtime
```

默认输入：

```text
observation.images.front
observation.images.wrist
observation.state[7]
```

默认配置：

```text
image_size = 256
n_obs_steps = 2
horizon = 16
n_action_steps = 8
action_dim = 7
```

训练 label 是数据集中未来 action 行组成的 step-wise delta chunk。部署时 runtime 维护 observation history，输出 action chunk，再由统一部署入口执行安全限幅和目标转换。

训练、离线评估和实时部署命令见仓库根目录的 `运行命令.md`。
