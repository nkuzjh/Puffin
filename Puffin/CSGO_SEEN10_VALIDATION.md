# Puffin aligned 实施验收记录

日期：2026-09-27。实验：`csgo_seen10_exp32gen_aligned`。边界：`RUN_FORMAL=0`。
本文只记录实际执行证据；操作命令见 [CSGO_SEEN10.md](CSGO_SEEN10.md)，
设计和三方横向对照见 [CSGO_SEEN10_PLAN.md](../CSGO_SEEN10_PLAN.md)。

## 2026-10-02：首次正式启动的 NCCL 故障与通信修复

- 首次 2 卡 × micro64 × accumulation1 正式训练于 08:17 启动，参数审计文件于
  08:19:22 写入；08:26:39 双 rank 的 NCCL watchdog 报 CUDA illegal memory access。
  首个日志没有 Python 模型异常栈，也没有任何 optimizer update loss 或 checkpoint。
  失败产物和原日志已隔离到
  `outputs/launch_control/failed_attempts/attempt_01_20261002_0835/`。
- 独立双卡 NCCL 探针不加载 Puffin：默认配置下小 all-reduce 和对象广播通过，
  256 MiB BF16 广播在两个 rank 均触发 CUDA 700。`NCCL_P2P_DISABLE=1`、
  `NCCL_PROTO=Simple`、`NCCL_CUMEM_HOST_ENABLE=0` 的单变量对照均未修复。
  仅设 `NCCL_SHM_DISABLE=1` 的对照通过相同广播，NCCL INFO 显示改走 NET/Socket。
  两卡短 DDP 追加探针在此设置下还完成 3 次更新，两个 rank 的权重和一致。
  探针脚本、日志保存在 `outputs/launch_control/`；这些是通信验收，尚不能代替完整 Puffin 训练验收。
- 本机是两张 RTX PRO 6000 Blackwell Server Edition，Torch 2.7.0+cu128、
  NCCL 2.26.2+cuda12.2，GPU 间拓扑为 NODE。NVIDIA 的
  [NCCL issue #2418](https://github.com/NVIDIA/nccl/issues/2418)
  报告了高度相似的双卡、版本与错误，并在其环境中由 NCCL 2.31.2 消除；
  这提供外部佐证，不等于已证明本机的 NCCL 内部缺陷位置。
- 当前服务器启动器在运行环境中传入 `NCCL_SHM_DISABLE=1`，正式训练进程继承该值；保持
  seed42、2×64×1=128、19500 updates、2496000 次曝光、严格确定性及模型/数据配方不变。
  通信改走 NET/Socket 后吞吐需重新实测，不声称跨通信实现逐位一致。

### 08:52：修复后的正式双卡启动验收通过

- 启动前完整 CPU 检查通过 87 项测试；08:43:55 从官方初始化重新启动正式训练，
  仍使用原命令的 2×64×1 拓扑，仅新增上述 NCCL 环境设置。
- 08:46:52 已完成 step2，08:48:38 达到 step6；08:50:35 主代理核查连续 step1–11，
  最终保存验收证据时已推进至 step14。所有 loss 有限，step1=0.3246151358、
  step10=0.3560837507、step14=0.3137413412；当前日志没有 traceback 或致命错误。
- 父进程 PID 3103915、torchrun PID 3104023；rank0/1 PID 3104090/3104091，分别映射
  GPU0/1，WORLD_SIZE=2，均继承正确数据根目录和 `NCCL_SHM_DISABLE=1`。
  该阶段测得约 26.5 秒/update，仅为启动阶段吞吐，不含后续完整验证/保存时间。
- 启动验收证据：`outputs/launch_control/startup_acceptance_20261002_0850.json`；
  当前 stdout/stderr 仍为根目录 `puffin_aligned.nohup.out`，逐步 loss 为正式
  `outputs/csgo_seen10_exp32gen_aligned/Puffin/seed_42/train_loss.jsonl`。
- 已取消本对话尚未触发的启动检查，并结束完成职责的启动控制器；正式训练为独立 session，
  确认训练父子进程继续存活。验收只覆盖成功启动，未完成 19500 步训练或全量推理/评测。

## 2026-10-02：当前服务器启动预检与等待状态

本次用户已授权在 UniLIP `exp32_1` 成功结束、Puffin 准备完成后启动正式训练；
上方及历史记录的 `RUN_FORMAL=0` 是此前实施边界。本节记录的是启动准备，
尚未取得本服务器正式双卡训练通过的证据。

- `/home/jiahao/task/Puffin/Puffin` 与 `/data/jiahao/task/Puffin/Puffin` 指向同一目录。
  项目环境的 CPU 导入、原生库隔离及 `pip check` 均通过；未初始化 CUDA。
- 现有官方资产下载进程继续运行，没有重复下载。初始证据保存于
  `outputs/launch_control/readiness_initial.json`，不能将下载中的资产视作完整性验证通过。
- 03:31 初检时共享评测器目录缺失，缺 `protocol.py`、`run_eval.py`；发布数据缺
  `calibration/z_calibration.json`。runner `check` 在评测器路径检查处失败。
  本机 execution-minimal 归档 SHA256 与迁移清单一致，但不含 calibration，未修改数据合同。
  待补 calibration 的发布 SHA256 为
  `67436a888e0f79520bf1ab156009f7384b3a1ade7638b44513f31b2a9b6c14ee`。
- `scripts/watch_csgo_aligned_start.py` 已作为独立后台进程运行，状态与事件记录分别为
  `outputs/launch_control/state.json` 和 `events.jsonl`。启动要求前序训练进程退出、
  根 TrainerState 达到 19550/19550 且含训练总结、最终权重完整，以及 Puffin CPU 预检通过。
  排他锁、已有正式目录和启动事务记录防止重复启动。
- 控制器保留用户的 `2 × micro64 × accumulation1 = 128`、seed42、19500 updates，
  每次实际启动重新 source 激活脚本，stdout/stderr 固定写入
  `puffin_aligned.nohup.out`；已有同名日志先归档。不会自动改变实验配方或盲目重试错误。
- 7 项控制器模拟测试通过，覆盖未完成/异常退出、PID 复用、缺依赖拒绝启动、
  不完整权重、已有输出、防重复及真实启动调用的环境/日志参数；另实测第二个控制器被锁拒绝。
  这些是调度检查，不代表 GPU 训练验收。实际启动后须进程仍存活且至少完成 10 次有限 loss
  更新才记录 `started_healthy`；启动失败记录 `failed_needs_repair`，需依据错误继续修复。

### 04:08：依赖补齐后的完整 CPU 预检

- 官方资产下载结束，独立校验进程于 03:40:59 完成，11/11 资产通过，退出码 0。
  证据：`outputs/launch_control/assets_final_check.meta.json` 与 `.stdout.json`。
- 用户已补齐共享评测器和 calibration，后者及 extrema 文件均匹配发布 SHA256。
  原 calibration 外部软链接备份为 `calibration.external_symlink_backup_20261002_0402`，
  项目数据目录中的 calibration 已放入相同内容的实际文件，外部源文件保持不变。
- 共享协议也拒绝 images/radars 指向根目录外的软链接，故正式使用完整实际 bundle：
  `CSGO_DATA_ROOT=/data/jiahao/data/csgo_benchmark_v2`。两根目录共 73 份发布清单、split、
  calibration 文件哈希完全一致；控制器预检和正式启动共同使用该路径。
- 环境检查通过；完整 aligned check 退出码 0、80 项测试通过；train/validation/discrete/
  continuous 数量为 50000/5000/20000/12800，连续轨为 200 clips × 64 frames。
  数据身份仍为 `7f6cb5b01c7103a0906e857cbb39eea59d1d70ab66150e8557f26093a0fea6c8`，
  target isolation 检查通过。正式 2×64×1 命令 dry-run 通过；未运行 GPU 训练。
- 控制器于 04:08 仅重启自身以加载数据路径配置；新 PID 为 2994288。
  8 项控制器测试通过，包含预检与启动使用同一实际数据根目录、CPU 检查隐藏 CUDA。
  当前仍等待 UniLIP exp32_1 成功结束，正式启动前会再次预检。
- 完整 stdout/stderr、退出码、路径修复和数据比较证据保存于
  `outputs/launch_control/preflight_20261002_0402.log` 和 `.json`。

### 04:22：本对话定时检查与智能体修复

用户已授权在原对话中定时检查，并在启动失败时由智能体根据项目文档修复后重试。
`scripts/schedule_csgo_aligned_check.py` 用本机定时器调用已验证的 `codex queue --thread`
向原线程 `01a0f8e6-b496-74a2-b507-d1981d9ba1e5` 投递，不创建独立桌面任务。

- 首次时间为 2026-10-02 07:50:00 +08:00；worker PID 3001268，状态 `armed`。
  当前服务器和 VS Code/Codex 会话需保持可用。
- 实际同线程投递探针退出码 0，返回排队消息 ID；证据为
  `outputs/launch_control/same_thread_queue_probe.json`。这验证消息入队，不代表未来检查已经执行。
- 7 项定时器测试通过，覆盖到时只发一次、字面传递 prompt、未到时不发送、取消、
  错误不盲重试、崩溃后的不确定投递及 thread 回执核对。
- 检查指令保存于 `outputs/launch_control/puffin_scheduled_check_prompt.txt`，包含：
  等待前序任务、查实际错误并修复、按文档保持有效 batch/更新预算/数据协议、归档失败产物、
  防止重复训练、启动健康后停止检查。仍需等待时由智能体按实测 ETA 重设下一次检查。
- 定时器本身不修改训练或启动模型，只投递检查任务；实际诊断修复由本对话智能体执行。
  状态与投递记录为 `outputs/launch_control/puffin_scheduled_check.json`。

## 2026-09-30：无 sudo 环境修复补充验收

- 新环境统一由 Conda 创建在项目 `.venv`，包括 Python 3.10、`libgl`、`libglib`；
  PyTorch 与训练依赖版本保持原配方，不改模型、数据或训练配置。
- 默认不修改已有环境；显式 `--repair` 才安装依赖。拒绝 Conda base、危险路径、
  非 Conda 原地修复和仍被进程使用的环境。权重和下载缓存保留。
- 安装/运行入口隔离用户级包、外部 Python 路径和其他环境动态库；共享评测子进程
  使用评测器自身的库路径，不继承 Puffin 的库路径。
- 全部自动测试 **72/72** 通过，包含新建/修复的模拟安装流程、旧环境只读保留、
  模块来源、原生库来源及评测器隔离。新 shell 与文档代码块语法检查通过。
- 在独立临时目录 `/tmp/puffin-env-probe.vXBhUB/env` 实际创建了最小 Conda 环境，
  安装 NumPy 2.2.5、opencv-python 5.0.0.93，OpenCV 导入与 `pip check` 通过；
  `/proc/self/maps` 确认 `libGL.so.1`、GLib、GThread 均来自该前缀。
- 复用本机已有 Puffin 环境完成只读导入检查与入口 dry-run，没有在该环境执行安装。
  未进行远端实机验证、完整全新 PyTorch 环境安装或 GPU 训练/推理；没有启动正式实验。

以下保留 2026-09-27 原验收记录。

## 环境、资产与隔离

- 工作目录：`/home/jiahao/task/Puffin/Puffin`。
- 本机环境：Python 3.10.21，Torch 2.7.0+cu128，torchvision 0.22.0+cu128。
- GPU：RTX PRO 6000 Blackwell；本次真实模型 smoke 显式限制 allocator 为单卡容量的24%。
  同卡已有 ControlAR/openpi 任务，没有停止或修改这些任务。
- 复用已有官方 Base 和 VAE，只补齐两个小型SD3配置，没有下载新的大权重或重装本机环境。
- Base SHA256：`4045661c81b29adc8aa1cc22079ddbbf86d353d2e0f35c0ffec310f193504257`。
- VAE SHA256：`9c9bdf9aabbd3efce5450eb628cf8d965c37f3fa4210a7fef151a94a939db211`。
- VAE遵循作者官方Demo的wusize/Puffin来源；scheduler shift3.0和VAE配置的Git blob与
  上游SD3配置一致，不声称打包VAE与上游受限safetensors逐张量相同。
- 固定版本资产11/11校验通过；包括恢复指纹使用的Qwen generation_config.json。
- 本次输出只写 `outputs/aligned_smoke_20260927_OgjlCp/`。旧训练、预测、评测和checkpoint链接保留。

## 配置、数据与CPU检查

`bash scripts/run_csgo_seen10.sh check --experiment csgo_seen10_exp32gen_aligned --seed 42`
成功完成配置解析、发布数据合同和target隔离检查。加入迁移元数据、确定性和正式评测门禁回归后，
完整自动测试53/53通过。

| 检查 | 实测结果 |
|---|---|
| split样本数 | train50000 / validation5000 / discrete20000 / continuous12800 |
| 地图 | 发布Seen-10，四个split均equal-map |
| 连续身份 | 200clips，每clip64frames |
| 数据/calibration内容身份 | `7f6cb5b01c7103a0906e857cbb39eea59d1d70ab66150e8557f26093a0fea6c8` |
| target隔离 | 两条测试轨各一条真实样本；拦截Image.open，只允许radar文件；无FPV读取 |
| 条件 | radar tensor3×224×224；pose_mode=text；batch不存在pose_values |
| 文本长度 | 完整50000训练指令的最长序列127tokens，低于256上限，无截断 |
| DDP累计 | 两个真实CPU/Gloo进程，world2×micro4×accum16，与global128参考更新一致 |
| 防误操作 | 错误路径/损坏资产/已有环境/旧目录混入等负例按预期拒绝 |
| Legacy入口 | 原`check --seed 0`通过19项测试及真实448/448数值pose数据检查 |
| 多卡命令解析 | 2卡×micro4×accum16、2节点×4卡×micro2×accum8 dry-run通过；未启动GPU训练 |

CPU/Gloo测试不是多GPU Puffin验收。资产`--check`不下载、不安装、不初始化CUDA。

## 真实完整模型参数审计

使用官方原始权重构建实际Puffin图，而非只读config推算。完整逐参数shape、dtype、
requires_grad、LR、weight decay和optimizer归属保存在每个新run的`seed_42/parameter_audit.json`。

| 可训练职能 | 参数量 | LR |
|---|---:|---:|
| Qwen LoRA r32 | 36,929,536 | 1e-4 |
| SD3 LoRA r32 | 41,877,504 | 1e-4 |
| 生成connector LoRA r16 | 3,620,864 | 1e-4 |
| meta_queries / projector_1 / projector_2全训 | 6,395,904 | 5e-6 |
| 合计 | 88,823,808 | — |

555个LoRA target，unexpected trainable=0。独立总参数4,560,857,827，冻结4,472,034,019，
可训练比例1.947524%。RADIO、视觉→语言projector和VAE均冻结且不进入optimizer。
全部可训练master参数为FP32，训练用BF16 autocast。3个非空optimizer组：
LoRA矩阵82,427,904 / full矩阵6,389,760 / full低维6,144；weight decay分别.05/.05/0。

静态方案中总量曾包含SD3位置buffer56,623,104元素；实际审计排除buffer并去重共享参数，
不改变批准的可训练模块、LoRA数量或模型结构。

## GPU smoke

真实完整模型连续短训练已完成：world1×micro1×accum128，2个optimizer updates，
256条源样本；训练loss分别0.2961328、0.3386057，单样本validation loss分别
0.2170223、0.2164345。用时573.54秒（含模型加载、审计和两次完整checkpoint保存），
max_memory_allocated为10.705GiB。smoke使用缩短的2-update scheduler，并明确标记smoke_only，
不是正式19500-update配方的收敛测试或吞吐benchmark。
逐参数对比step1/step2，冻结参数变化数为0；各允许训练的模块族均有参数更新。
具体记录在smoke根目录`gradient_coverage_smoke.json`（检查的是参数变化，包含AdamW decay，
不能把变化数量等同于非零数据梯度数量）。

首次精确resume未通过：第2步train loss完整数值相同，scheduler、sampler、各RNG状态一致，
但1044个模型tensor和2098个optimizer tensor有差异，validation loss为0.2160254而非0.2164345。
原始证据保留在`resume_comparison.json`，不把它写成精确恢复通过。

根因排查复现了本机Flash SDPA反向非确定性：BF16 `[1,24,1044,64]`相同q/k/v连续执行，
forward相同，q梯度最大差0.0009765625；启用严格deterministic后q/k/v梯度差均为0。
masked SDPA backend2在该小测试中两种模式均无差异。该复现只定位算子风险，不能代替完整恢复验收。
相关机制见[PyTorch 2.7复现性说明](https://docs.pytorch.org/docs/2.7/notes/randomness.html)。
最终版本的严格确定性真实训练/恢复已通过：

- `det_continuous/`：world1×micro4×accum32，连续2个updates。
- `det_resume/`：仅复制第1步的完整状态、日志与best/latest相对链接，然后`--resume auto`至第2步。
  checkpoint1使用只读用途的硬链接复用已完成文件，没有复制/覆盖旧实验。
- **6,318个tensor逐位相同，完整payload无差异**，包括模型、optimizer、scheduler、scaler、RNG、
  sampler和配方/数据身份；训练/验证日志、parameter_audit.json及best/latest链接也完全一致。
- 最终train loss0.3421717，validation loss0.21677184104919434；best loss0.21641772985458374
  来自step1，因此best仍指向step1，latest指向step2。smoke不会创建late=19500别名。
- 连续运行353.73秒，恢复运行353.07秒，含初始化、审计和完整checkpoint I/O；
  峰值GPU allocated分别11.360/11.367GiB。磁盘等待和其他任务共享GPU使这些时间不能用作正式ETA。
- 每个完整checkpoint为10,921,168,901bytes。逐位比较结果保存在
  `deterministic_resume_comparison.json`；首次失败证据保留，不与修复后产物混合。

### 真实完整模型推理与共享评测

同一个`det_continuous/seed_42/checkpoints/step_00000002.pth`，seed42，原生
FlowMatchEuler50步、CFG4.5、shift3、BF16、decoder chunk1。离散/连续各2个真实样本：

| 引擎 | 执行batch | 离散2图用时 | 连续2图用时 | 结果 |
|---|---:|---:|---:|---|
| native-eager（原生动态列表路径） | 1 | 31.17秒 | 34.84秒 | 完成 |
| eager（posterior缓存+dense） | 1 | 11.70秒 | 11.34秒 | 完成 |
| compiled（缓存+dense+compile） | 16 | 93.58秒，含首次编译 | 38.50秒 | 完成 |

compiled用2条真实样本加14条尾部padding，**不是16个不同样本的稳态吞吐测试**，不能直接
据此给全量ETA或计算公平算力。总体推理检查346.68秒（含完整模型加载），峰值allocated10.794GiB。
三种引擎各输出4张，共12张JPEG，全部RGB448×448；6份manifest均为target_read=false、
pose_mode=text、同一checkpoint、smoke_only=true。compiled两轨再次执行均generated=0、
skipped_existing=2。详情在`inference_smoke_summary.json`。

**数值限制：通过的是功能与数据合同检查，不是跨引擎像素等价。**
50步BF16放大了不同kernel/batch的数值差异：eager b1对native b1，四图的[0,1]空间RMSE
为0.0164–0.0497（两输出之间PSNR26.06–35.70dB）；eager b1对compiled b16，RMSE
为0.0595–0.1676（两输出之间PSNR15.51–24.52dB）。后者同时改变batch与compile，不能把全部
差异归因于单一因素。**这些是输出之间的比较，不是对GT的benchmark指标。**
不能将加速宣称为像素无损或声称改变batch不改变最终图像；只保证随机输入按sample identity派生、
固定引擎/批次恢复不混写。原生路径保留用于参考，不根据GT指标在这些引擎中挑选结果。

共享evaluator的两个CPU smoke命令均exit0，直接读取compiled输出：离散`--limit 1`，
连续`--max-clips 1 --frame-only`。确认`smoke_only=true / formal=false /
official_output_written=false`，记录在`evaluator_{discrete,continuous}_smoke_diagnostic.json`。
实际执行离散PSNR/SSIM/LPIPS/Boundary_F1及连续PSNR/SSIM/LPIPS；smoke明确跳过FID、
TWE、TDE和FVD，不能将本次结果表述为完整指标验收。

当前smoke根目录约62GiB，包含失败排查与最终通过的完整checkpoint；未自动删除这些证据。

另外，已独立完成随机1-block SD3+LoRA的不同尺寸kernel诊断（不是完整Puffin）：
BF16 target latent56×56 / radar28×28，batch4含尾部padding，峰值137.5MiB。
native→dense max/mean绝对差为0.0078125/0.0000161；dense→compiled为
0.01953125/0.0022888；重复padding输出差0。BF16不同kernel不要求逐位相等。

## 尚未执行与使用边界

- 没有启动19,500步正式训练，没有启动20,000/12,800条全量推理或正式评测。
- 没有在远程服务器重新安装环境，也没有执行真实多GPU Puffin训练；这些不能由CPU测试替代。
- 未做完整5000条验证的耗时验收；正式代码保留五次完整验证，smoke只验证一条。
- 未运行FID、完整clip TWE/TDE/FVD smoke；少量frame-only检查不覆盖这些指标。
- 训练或推理时间必须在目标服务器按实际设备、batch和资源占用测量，不套用ControlAR的9小时。
- smoke checkpoint及不完整预测均不能通过正式评测入口，不能作为论文结果。
