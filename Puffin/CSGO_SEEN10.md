# Puffin CSGO Benchmark v2：legacy 与 exp32_gen aligned

本文是**运行入口、环境资产、命令和输出约定**的唯一维护位置。设计与三方对比见
[CSGO_SEEN10_PLAN.md](../CSGO_SEEN10_PLAN.md)，本次实际验收与尚未验证项见
[CSGO_SEEN10_VALIDATION.md](CSGO_SEEN10_VALIDATION.md)。旧接入记录保留在本文后半部分；
`csgo_benchmark_v2_start.md` 是历史材料，不覆盖当前配置。文档分工参考 ControlAR 和 OmniGen2，
不复制它们的模型配方、性能或运行状态。

## 实验选择与边界

任务仅为 radar/map + 当前地图名 + 当前 x/y/z/pitch/yaw → FPV。不新增定位、aux_loc_loss、
perception_loss、历史/未来真实帧或上一生成帧条件。主参考是 generation-only `exp32_gen`；
joint `exp32` 是次要对照。优化参数、架构和原生采样计算量是明确披露的模型差异。

| 项目 | legacy（不传 experiment） | aligned（显式选择） |
|---|---|---|
| 配置 | `configs/pipelines/csgo_seen10.py` | `configs/pipelines/csgo_seen10_exp32gen_aligned.py` |
| 选择 | 旧命令不变 | `--experiment csgo_seen10_exp32gen_aligned` |
| radar / FPV | 448 / 448 | 224 / 448 |
| pose | 数值 MLP + 地图文本 | 仅物理5DoF文本，不创建数值模块 |
| 更新 / 有效 batch | 12,500 / 4（旧默认单卡） | 19,500 / 128 |
| 参数效率 | 旧配置行为 | Qwen/SD3 r32；生成桥r16；α64/dropout0.05 |
| 冻结 | 旧配置行为 | RADIO、视觉→语言projector、VAE及非活跃参数 |
| LR | 2e-5 | LoRA1e-4；queries/输出桥5e-6 |
| AdamW | β=(.9,.95)，矩阵decay.05 | 同原生规则；低维参数decay0 |
| scheduler | 旧iteration口径 | 585 optimizer updates warmup；cosine→0 |
| 常规保存/完整验证 | 2500/5000/7500/10000/12500 | 4000/8000/12000/16000/19500 |
| 主/补充结果 | 历史结果原样保留 | late=19500为主，best为补充 |
| 推理 | 原eager默认28步；支持compiled | 官方50步、CFG4.5；支持原生/加速引擎 |
| 默认seed | 0 | 42 |

有效batch只约束 `WORLD_SIZE × micro_batch × accumulation = 128`，不固定其中任意一项。
每次更新消费128条源记录；共2,496,000次曝光。每个打乱epoch用390个完整全局batch
（49,920条，尾部80条舍弃），50个组织epoch恰好19,500updates；曝光等价49.92epoch。
权威终止条件是optimizer updates；scheduler仅在optimizer更新后推进，LR不随执行拓扑缩放。

图像只做确定性resize/归一化，radar由冻结VAE编码。RADIO和`projector`是理解侧视觉分支，
在本生成路径中不使用，仍明确冻结；`connector_1/2`、`llm2connector_1/2`、`projector_1/2`
属于**语言→生成**桥，不能因为名称相似而误冻。训练保留原生flow matching、原生timestep
采样和0.1文本dropout；丢文本时地图/pose文本也丢弃，但radar保留。验证禁用文本dropout。

## 新服务器：环境、官方资产与路径

先把**包含本次新增/未提交文件**的代码复制到新服务器的独立空目录，仅clone上游官方仓库
不会包含aligned接入。下面是需手动替换主机和目标路径的示例，不使用`--delete`：

```bash
# 在源服务器执行；目标/remote/workspace/Puffin应是新目录。
rsync -av \
  --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='outputs/' --exclude='checkpoints/' --exclude='.cache/' \
  --exclude='*.pth' --exclude='*.safetensors' --exclude='Puffin-World/' \
  /home/jiahao/task/Puffin/ USER@NEW_SERVER:/remote/workspace/Puffin/
```

发布数据与共享evaluator单独完整迁移，不重划split、不改校准；本机已有训练结果不随上述命令
复制。需要迁移某个aligned run时，再单独复制完整`seed_42/`到新run root，保持相对链接，
不要把旧legacy预测放入aligned目录。迁移命令仅提供给用户手动执行，本次未连接远程服务器。

从嵌套工作目录执行，例如 `cd /remote/workspace/Puffin/Puffin`。runner也会依据脚本位置
定位工作目录，不依赖调用shell的cwd。不要复制旧服务器`.venv`。代码不包含大权重、
数据、共享评测器和旧结果，须分别准备；无需安装UniLIP模型本体。

```bash
bash scripts/setup_csgo_seen10.sh --env-only
.venv/bin/python scripts/download_csgo_seen10_assets.py --profile aligned
source <(.venv/bin/python scripts/download_csgo_seen10_assets.py --profile aligned --print-env)
```

新环境采用Python3.10、Torch2.7.0/torchvision0.22.0 cu128及`requirements_seen10.txt`。
已有兼容环境保留，不默认升级/降级。下载器只准备必要资产，按固定revision与SHA/blob校验，
复用完整文件和HF缓存；损坏目标报错并保留，不静默覆盖。固定清单见
[`scripts/csgo_seen10_assets.json`](scripts/csgo_seen10_assets.json)。

资产使用**官方Puffin Demo的实际配方**：KangLiao/Puffin的Base，wusize/Puffin的`vae.pth`，
以及作者官方Space的SD3配置。依据是固定revision的
[官方app.py](https://huggingface.co/spaces/KangLiao/Puffin/blob/1984a63b134f39e4197eb25b35ea023e17165d77/app.py)。
两个SD3配置与上游官方文件Git blob完全相同，scheduler shift=3.0。
这不声称打包VAE与受限SD3原生safetensors逐张量相同，也不需要下载受限大骨干。
Base约8.90GB、VAE约335MB；Qwen/RADIO只下载小配置和tokenizer，骨干已在Base内。
训练/推理从本地固定资产加载，不在运行中补下载。

每次新开终端重新执行上面的`source <(... --print-env)`。路径不同的服务器显式设置：

```bash
export CSGO_DATA_ROOT="/remote/workspace/UniLIP/data/csgo_benchmark_v2"
export SHARED_EVAL_DIR="/remote/workspace/csgo_benchmark_v2_eval_general"
# 共享评测器由自己的仓库独立管理；已有环境无需重装。
CSGO_EVAL_TORCH_BACKEND=cu128 bash "$SHARED_EVAL_DIR/setup_env.sh"
export EVAL_PYTHON="$SHARED_EVAL_DIR/.venv/bin/python"

export EXP="csgo_seen10_exp32gen_aligned"
export ALIGNED_ROOT="$PWD/outputs/$EXP/Puffin"
```

aligned数据/评测器路径优先级为CLI→环境变量→外层仓库同级默认目录。
支持`--data-root`、`--shared-eval-dir`/`--eval-root`、`--eval-python`；兼容`DATA_ROOT`、
`UNILIP_PYTHON`。显式错误路径不回退。训练Python默认项目`.venv/bin/python`，可设
`PUFFIN_PYTHON`；评测Python使用显式配置或所选共享评测器的独立环境，绝不回退到训练环境。
aligned不隐式继承可能指向旧结果的`OUTPUT_ROOT`，请显式传`--output-root`。

可选诊断（不需要每次正式运行前重复执行）：

```bash
bash scripts/setup_csgo_seen10.sh --check --profile aligned
bash scripts/setup_csgo_seen10.sh --check-cuda --profile aligned
.venv/bin/python scripts/download_csgo_seen10_assets.py --profile aligned --check --json
```

`--check`不安装、不下载、不初始化CUDA；`--check-cuda`才执行小型GPU检查。
跨服务器应重建环境并运行smoke，不能将本机验收描述为已在远端验证。

## aligned检查、训练与精确恢复

```bash
bash scripts/run_csgo_seen10.sh check --experiment "$EXP" --seed 42

# 独立少量smoke：仍验证effective batch128，不代表正式实验。
bash scripts/run_csgo_seen10.sh smoke --experiment "$EXP" --seed 42 \
  --micro-batch-size 1 --gradient-accumulation-steps 128 \
  --inference-engine eager --batch-size 1 --num-workers 0 \
  --output-root "$PWD/outputs/smoke_$EXP/Puffin"

# 正式示例：4卡×micro4×accum8=128。用户自行选择空闲设备。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash scripts/run_csgo_seen10.sh train --experiment "$EXP" --seed 42 \
  --micro-batch-size 4 --gradient-accumulation-steps 8 \
  --output-root "$ALIGNED_ROOT"

# 相同run续跑，不从旧CSGO权重初始化。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash scripts/run_csgo_seen10.sh train --experiment "$EXP" --seed 42 \
  --micro-batch-size 4 --gradient-accumulation-steps 8 \
  --output-root "$ALIGNED_ROOT" \
  --resume "$ALIGNED_ROOT/seed_42/checkpoints/latest.pth"
```

省略accum时根据实际world和micro计算；不能整除128则报错。单卡micro4→accum32，
2卡micro4→accum16同样合法。`--dry-run`打印将执行的命令而不启动任务。
多节点可显式配置`NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。

新run拒绝非空目录。完整checkpoint包含模型/LoRA、小模块、optimizer、scheduler、
禁用scaler状态、optimizer step、RNG与逻辑sampler游标；只在梯度累计边界保存。
恢复校验内容身份而非服务器绝对路径。同拓扑可验证精确续跑；合法拓扑变化允许恢复，
但随机流/浮点归约变化意味着不再宣称逐位一致。只能加载可信的本地checkpoint。
aligned训练显式启用严格确定性算法、cuDNN deterministic并关闭benchmark，固定
`CUBLAS_WORKSPACE_CONFIG=:4096:8`；显式冲突值报错。此策略写入checkpoint和配方指纹，
用于避免仅恢复RNG却仍被非确定性反向kernel破坏的情况，可能产生额外运行开销。
不同硬件、PyTorch/CUDA版本或DDP拓扑仍不承诺逐位一致。legacy和推理路径不强加这些训练设置。
迁移已有run时，完整保留`seed_42/`目录及相对checkpoint链接，尤其是历史best指向的实际
step文件和日志；仅拷贝latest一个文件不足以保留历史best选择和完整审计记录。
必须同时保留该checkpoint对应的源码版本和固定元数据文件，不能用新源码静默恢复旧配方。

每个常规里程碑完整验证5000条。`best.pth`链接最低validation loss；`late.pth`仅在
19500创建；`latest.pth`链接最近完整保存。主论文比较使用late，对齐UniLIP的final选择；
best是补充，不能把不同模型原生loss直接横比。

## aligned推理与共享评测

```bash
for selection in late best; do
  bash scripts/run_csgo_seen10.sh infer --experiment "$EXP" --seed 42 \
    --selection "$selection" --output-root "$ALIGNED_ROOT" --mode both \
    --steps 50 --cfg-scale 4.5 \
    --inference-engine compiled --batch-size 16 --decoder-chunk-size 1 \
    --prediction-tag native50_compiled_b16
done

for selection in late best; do
  bash scripts/run_csgo_seen10.sh eval --experiment "$EXP" --seed 42 \
    --selection "$selection" --output-root "$ALIGNED_ROOT" --task both \
    --prediction-tag native50_compiled_b16 --eval-python "$EVAL_PYTHON"
done
```

可用`--mode discrete`/`continuous`分别执行；同一selection的两轨必须使用同一checkpoint。
默认官方采样为FlowMatchEuler、50步、CFG4.5、shift3、BF16、seed42、空negative prompt，
不启用thinking、不做best-of-N。CFG以双分支batch计算，不改变源样本计数。

加速四阶段是批量、radar posterior/pipeline缓存、dense向量化、Transformer compile；
不是减少采样步数。新版dense分别处理target latent56×56和radar latent28×28。
`--inference-engine native-eager --batch-size 1`使用原生动态路径作参考；`eager`保留批量/cache/dense；
`compiled`编译固定形状，尾批补齐后只保存真实样本。VAE默认单图解码降低峰值显存。
不同engine/batch使用不同prediction-tag，切换后不能混入原预测。
本次真实50步BF16 smoke确认跨引擎/批次有明显像素差异，不能称为无损加速；具体数值见
验收记录。原生与加速模式遵循同一采样器/步数/CFG配方，但不保证最终像素等价。

每个sample使用identity派生的seed，posterior和diffusion使用两个分开的生成器（均以该seed初始化）；
连续帧独立，不回灌前帧。
随机输入不依赖batch划分，但不同kernel/设备的最终像素不承诺逐位一致。固定block恢复只写
缺图，已有文件必须是完整448×448 RGB JPEG。文件名沿用`<map>/<file_frame>.jpg`与原saver编码。
恢复拒绝checkpoint、协议、采样配置或内容身份不一致的目录；损坏图片不静默覆盖。

```text
outputs/csgo_seen10_exp32gen_aligned/Puffin/seed_42/
  checkpoints/{step_00004000.pth,...,step_00019500.pth,best.pth,late.pth,latest.pth}
  predictions/{late,best}/native50_compiled_b16/{discrete,continuous}/gen_imgs/<map>/<file_frame>.jpg
  evaluation_shared/{late,best}/native50_compiled_b16/{discrete,continuous}/
```

完整推理为20,000离散图、12,800连续图。正式评测拒绝smoke/不完整/混合checkpoint结果，
且只调用共享evaluator。离散指标为PSNR/SSIM/LPIPS/Boundary_F1/FID，连续为
PSNR/SSIM/LPIPS/TWE/TDE/FVD；equal-map macro、clip16/stride16、FVD224和tracking参数
以共享`benchmark_v2.yaml`为准。默认smoke仅验证两轨各一帧的连通性，连续frame-only
不验证完整时序指标，也不产生正式论文结果。

## 实施与正式运行边界

本次只实施和隔离smoke，`RUN_FORMAL=0`：不自动运行19,500步或全量32,800图。
具体通过项和资源限制见验收文档；正式训练/推理/评测由用户手动执行。
预计五个完整checkpoint需约50–60GB，建议另预留资产、环境和输出后共80–100GB（不含数据）。
训练/全量推理ETA必须依据目标服务器实测，不能套用ControlAR的9小时记录。

## Legacy 首次接入记录（以下不是 aligned 配方）

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
  --pred-root "${OUTPUT_ROOT}_compiled/seed_0/discrete/gen_imgs" \
  --data-root "$DATA_ROOT" \
  --output "${OUTPUT_ROOT}_compiled/seed_0/evaluation_shared/discrete"

"$UNILIP_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" continuous \
  --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
  --pred-root "${OUTPUT_ROOT}_compiled/seed_0/continuous/gen_imgs" \
  --data-root "$DATA_ROOT" \
  --output "${OUTPUT_ROOT}_compiled/seed_0/evaluation_shared/continuous"
```

结果按 evaluator 约定保存 per-map JSON 与 `summary_equal_map.json`。不完整生成应先用 `infer_seen10.py` 补齐，不能将 smoke 或 `--max-samples` 结果作为正式 metric。
