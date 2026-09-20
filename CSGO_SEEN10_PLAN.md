# Puffin 接入 CSGO Benchmark v2 Seen-10

本次仅实现 Table 1 中 Puffin 负责的 GENERATION 路线：discrete generation 与
continuous generation。数据、地图顺序、样本 identity、pose 归一化与 continuous clip
顺序全部以发布的 `minimal_dataset_report.json`、`benchmark_manifest.json`、
`splits/` 和 `calibration/` 为准，不扫描 `images/` 重建 split，不修改 DATA_ROOT。

## 接入方案

- 保留 Puffin 原生 `Qwen2p5RadioStableDiffusion3HFDynamic` 与 SD3 diffusion loss。Radar
  作为现有 `cam2image` 的 image-condition，继续由 VAE 编码后输入动态 DiT。
- 新增数值 5DoF MLP：对 `[x,y,z,pitch,yaw]` 的 benchmark 归一化值做数值投影，
  加到 Puffin 的 64 个 generation query。文本只使用固定任务 instruction，不把 pose
  坐标仅拼成自然语言。
- 新增 manifest-driven dataset/collator：训练与验证读 radar + pose + target FPV；
  推理 dataset 显式 `include_target=False`，不打开目标帧，也不读历史/未来 GT 帧。
- 复用原生 `scripts/train.py` → `CustomRunner` → MMEngine `TrainLoop`、
  `CustomAdamW`、BF16 AMP、distributed launcher 和 MMEngine checkpoint 格式。新增轻量
  Seen-10 validation metric/artifact hook，总迭代的 1/5 处验证并保存，共五次；
  `late.pth`/`latest.pth` 指向末次保存，`best.pth` 指向 validation loss 最低的
  五个 checkpoint 之一。保留原生 `--resume` 恢复 optimizer/scheduler 状态。
- 训练 hook 记录主 loss 并在结束时写出 loss 曲线。推理使用同一个冻结
  checkpoint 生成 discrete 和 continuous，输出固定为 448×448 RGB JPEG，文件名保留
  `<map>/<file_frame>.jpg`。
- `scripts/run_csgo_seen10.sh` 提供 `smoke/train/infer/eval/all` 和 `--seed`。
  `BUILD_SHARED_EVALUATOR=0`，现有共享评测器已满足接口，本次不修改它。

## 预计文件

- 修改 `Puffin/src/models/puffin/model.py`：数值 pose conditioning 与 validation forward。
- 新增 `Puffin/csgo_seen10/`：dataset/collator、validation metric 与 checkpoint/loss 产物 hook。
- 新增 `Puffin/configs/pipelines/csgo_seen10.py`：Seen-10 模型、训练、验证与五档保存配置。
- 新增 `Puffin/train_seen10.py`、`Puffin/infer_seen10.py`、
  `Puffin/scripts/run_csgo_seen10.sh`。
- 新增 `Puffin/CSGO_SEEN10.md` 和针对数据合同/数值条件/输出 identity 的小型测试。

## 验收边界

`RUN_FULL=0`，因此本轮不启动 50,000 样本的完整训练和 32,800 张正式推理。
最小验收要求是：真实 manifest 样本可组 batch；Puffin 执行一次 forward/backward；
checkpoint 可保存并严格重载；产生至少一张标准 448 RGB JPEG；共享评测器的
smoke 入口能读取该输出。只有完整覆盖离散 20,000 张和连续 12,800 帧后才允许运行正式
metric；smoke 产物不当作 Table 1 结果。
