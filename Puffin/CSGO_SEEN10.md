# Puffin 接入 CSGO Benchmark v2 Seen-10

本适配只实现 Puffin 在 Table 1 负责的 discrete generation 和 continuous generation。数据身份、地图顺序、归一化 pose、radar 路径及 continuous clip 顺序由共享评测器的只读 `protocol.py` 按发布 manifest/splits/calibration 提供；项目适配层把这些行转换成 Puffin 原生 `cam2image` batch。训练和验证打开 target FPV，推理使用 `include_target=False`，只读取 radar 与数值 pose。Pose 作为 5DoF 数值向量经可训练 MLP 加到 generation queries；固定文本 instruction 只包含地图名，不包含 pose 数值。Seen-10 配置只读取 Qwen/RADIO 的小型架构配置与 tokenizer；Base checkpoint 提供 Qwen、RADIO、SD3 transformer 权重，单独的 Puffin VAE 文件以 strict load 载入，避免下载重复的大模型权重。

## 环境与路径

在项目根目录创建独立 Python 环境。新训练需要 `checkpoints/Puffin-Base.pth` 和单独的 `checkpoints/vae.pth`。正式评测继续使用独立的 UniLIP Python 环境和共享 evaluator。

```bash
cd /home/jiahao/task/Puffin/Puffin
/home/jiahao/miniconda3/bin/conda create --yes --prefix .venv python=3.10 pip
source /home/jiahao/miniconda3/etc/profile.d/conda.sh
conda activate /home/jiahao/task/Puffin/Puffin/.venv
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
DS_BUILD_OPS=0 python -m pip install -r requirements_seen10.txt
mkdir -p checkpoints
huggingface-cli download KangLiao/Puffin Puffin-Base.pth --local-dir checkpoints --repo-type model
huggingface-cli download wusize/Puffin vae.pth --local-dir checkpoints --repo-type model

export PUFFIN_PYTHON=/home/jiahao/task/Puffin/Puffin/.venv/bin/python
export UNILIP_PYTHON=/home/jiahao/miniconda3/envs/UniLIP/bin/python
export DATA_ROOT=/home/jiahao/task/UniLIP/data/csgo_benchmark_v2
export SHARED_EVAL_DIR=/home/jiahao/task/csgo_benchmark_v2_eval_general
export OUTPUT_ROOT=/home/jiahao/task/Puffin/Puffin/outputs/csgo_benchmark_v2_seen10/Puffin
export PYTHONPATH=/home/jiahao/task/Puffin/Puffin${PYTHONPATH:+:$PYTHONPATH}
```

## Smoke、训练与推理

`check` 编译适配代码、运行小型数据合同测试，并从真实 Seen-10 train 与 target-free inference split 读取 batch，不加载模型权重：

```bash
bash scripts/run_csgo_seen10.sh check --seed 0
```

默认 `smoke` 完成集成闭环：真实 train batch、单步训练/验证、保存一个 MMEngine checkpoint、严格重载该 checkpoint、为 discrete 与 continuous 各生成一张 448×448 RGB JPEG，再由共享 evaluator smoke 入口读取。输出与正式训练隔离到 `outputs/csgo_benchmark_v2_smoke/Puffin/seed_<seed>/`。此闭环需要本地模型权重、一张 CUDA GPU、HF 小型配置/tokenizer 文件和 UniLIP evaluator metric 权重：

```bash
bash scripts/run_csgo_seen10.sh smoke --seed 0
```

2026-09-20 已在 RTX PRO 6000 上完成该闭环。验收运行使用隔离目录 `outputs/csgo_benchmark_v2_smoke_retry4/Puffin/seed_0/`，10 个地图的发布 manifest 均可读取；smoke 只选择每个任务的首个样本，训练 loss 为 0.2222、验证 loss 为 0.2617，checkpoint 保存/严格重载、两张 RGB 448×448 JPEG 和共享 evaluator 读取全部成功。评测器输出明确标记 `smoke_only=true`、`formal=false`、`official_output_written=false`，这些诊断数值不是 Table 1 正式指标。

从本地 Puffin Base 与独立 VAE checkpoint 训练一个 Seen-10 pass。默认 12,500 次 optimizer iteration、batch size 4；在 2,500、5,000、7,500、10,000、12,500 步验证并保存五个原生 checkpoint。`best.pth`、`late.pth` 和 `latest.pth` 是这五个文件的链接。训练续跑使用 Puffin 原生 `--resume`；完整 checkpoint 已包含 VAE，因此 resume 不需再指定这两份初始化权重：

```bash
python train_seen10.py --seed 0 --init-checkpoint checkpoints/Puffin-Base.pth
python train_seen10.py --seed 0 --resume "$OUTPUT_ROOT/seed_0/checkpoints/iter_7500.pth"
```

使用同一个选定 checkpoint 分别补全 discrete 与 continuous 输出；已有有效 448×448 RGB JPEG 会跳过，缺少样本会继续生成。continuous 按发布 clip/frame 顺序逐样本推理，且每个 sample identity 使用稳定 seed。完整 MMEngine checkpoint 含有 Python pickle 元数据；推理和 `--resume` 只应接受可信的本地 checkpoint：

```bash
python infer_seen10.py --mode both --seed 0 \
  --checkpoint "$OUTPUT_ROOT/seed_0/checkpoints/best.pth" \
  --output-root "$OUTPUT_ROOT"
```

以上原命令保持 eager、batch size 1 和既有 seed/signature 语义，因此可继续 resume 旧 manifest。Seen-10 加速按四个阶段执行：

1. 批量阶段按 manifest 行顺序组成固定 block。某个 block 有缺图时会重算整个 block，但只写入缺少的 JPEG；compiled 模式会把诊断尾批补齐到固定 batch size，再丢弃补齐样本的输出。
2. 缓存阶段每张 radar 只编码一次并缓存 VAE posterior 参数，SD3 pipeline 在模型实例内复用。每个 sample 分别创建 posterior 与 diffusion generator；两者都从原有 `sample_seed(seed, sample_id)` 起步，但分别维护状态、不会相互消耗随机数，以保留 legacy 采样语义。
3. Dense 阶段为等长单 radar tensor 增加 transformer 向量化路径，变长或多 radar 条件继续走通用动态路径。
4. Compile 阶段只编译固定形状 transformer，使用 `mode="reduce-overhead"`、`dynamic=False`、`fullgraph=True`；编译参数和首次 compiled `model.generate` 状态写入 manifest，JPEG 完整性仍由 `complete` 单独记录。

任一加速模式都不能与旧 eager 输出混用。Compiled engine 只支持本 Seen-10 的固定 448×448、单 radar 输入；其他变长或多 radar 调用应使用 eager，其 Transformer 会自动保留动态路径。若 compiled 首次推理失败或不适合当前设备，改用 eager 并指定另一个输出根目录重跑。VAE 解码默认按单图分块以限制显存，也可设置 `--decoder-chunk-size`。

```bash
bash scripts/run_csgo_seen10.sh infer --seed 0 \
  --checkpoint "$OUTPUT_ROOT/seed_0/checkpoints/best.pth" \
  --inference-engine compiled --batch-size 16 --decoder-chunk-size 1 \
  --output-root "${OUTPUT_ROOT}_compiled"
```

加速模式会把 engine、batch size、seed 策略、优化版本与 compile 设置写入 manifest 签名；不能与旧 eager 输出混用。请为加速运行选择独立 `--output-root`。单独启用 `--batch-size N` 可使用 eager 批量推理，不触发 `torch.compile`。

或用 shell runner 执行各阶段：

```bash
bash scripts/run_csgo_seen10.sh train --seed 0
bash scripts/run_csgo_seen10.sh infer --seed 0 --mode both
bash scripts/run_csgo_seen10.sh all --seed 0
```

`--max-samples` 仅用于诊断，不可用于正式评测。正式 inference 会将文件写到：

```text
outputs/csgo_benchmark_v2_seen10/Puffin/seed_<seed>/discrete/gen_imgs/<map>/<file_frame>.jpg
outputs/csgo_benchmark_v2_seen10/Puffin/seed_<seed>/continuous/gen_imgs/<map>/<file_frame>.jpg
```

训练 checkpoint 与主 loss 产物位于 `outputs/csgo_benchmark_v2_seen10/Puffin/seed_<seed>/checkpoints/`：`iter_<step>.pth`、`best.pth`、`late.pth`、`latest.pth`、`train_loss.jsonl`、`seen_validation.jsonl` 和 `loss_curve.svg`。新训练不会覆盖非空 work directory；请选新目录，或从 MMEngine checkpoint resume。

## 统一评测

只有生成覆盖完整且无额外图片时运行正式 metric。共享 evaluator 以 UniLIP Python 执行，不改 evaluator 或 `DATA_ROOT`。runner 的 `eval` 和 `all` 会将 discrete、continuous 输出分别交给共享 `run_eval.py`：

```bash
bash scripts/run_csgo_seen10.sh eval --seed 0

"$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" discrete \
  --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
  --pred-root "$OUTPUT_ROOT/seed_0/discrete/gen_imgs" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT_ROOT/seed_0/evaluation_shared/discrete"

"$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" continuous \
  --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
  --pred-root "$OUTPUT_ROOT/seed_0/continuous/gen_imgs" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT_ROOT/seed_0/evaluation_shared/continuous"
```

结果按 evaluator 约定保存 per-map JSON 与 `summary_equal_map.json`。不完整生成应先用 `infer_seen10.py` 补齐，不能将 smoke 或 `--max-samples` 结果作为正式 metric。
