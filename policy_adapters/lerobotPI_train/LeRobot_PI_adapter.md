# LeRobot PI0 / PI0.5 Adapter

这里实现的是 LeRobot 0.5.x 内置的 PI0/PI0.5 策略适配，不是独立的 `openpi_train` 训练链路。

主要文件：

```text
lerobot_pi_adapter.py            数据和动作适配
train_lerobot_pi_minimal.py      PI0/PI0.5 训练入口
offline_eval_lerobot_pi.py       离线预测与 GT 对比
deploy_lerobot_pi_runtime.py     checkpoint runtime 类
```

默认输入：

```text
observation.images.front
observation.images.wrist
observation.state[7]
task
```

默认图像大小为 `224×224`，action chunk 默认为 50。动作仍使用本项目统一的 7 维逐步相对末端增量语义。当前 vanilla adapter 不消费触觉字段。

PI0/PI0.5 可能需要 Hugging Face 登录、tokenizer 和预训练模型文件。凭据与缓存必须保存在仓库之外，不得提交。

训练和离线评估命令见仓库根目录的 `运行命令.md`。runtime 类已经存在，但当前尚未接入 `deploy_live_policy.py`，因此本仓库不提供 PI 的统一实时部署命令。
