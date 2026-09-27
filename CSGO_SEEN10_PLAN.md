# Puffin exp32_gen aligned：已批准设计与验收边界

操作命令集中在 [Puffin/CSGO_SEEN10.md](Puffin/CSGO_SEEN10.md)，实施证据在
[Puffin/CSGO_SEEN10_VALIDATION.md](Puffin/CSGO_SEEN10_VALIDATION.md)。本文记录设计，
不把未运行的正式实验写成完成状态。旧首次接入设计保留在末尾。

## 对齐原则与三方对照

主对照是generation-only exp32_gen，joint exp32仅为次要对照。对齐发布数据、输入信息、
generation样本曝光和评测口径；LoRA/优化/原生采样差异明确披露，不声称参数量或FLOPs相等。
不增加定位、aux_loc_loss、perception_loss或额外时序信息。

| 项目 | Puffin官方 | UniLIP exp32_gen实际 | Puffin aligned |
|---|---|---|---|
| 原生任务 | 理解/生成多任务；stage4下游微调 | Seen-10 generation-only | Seen-10 generation-only |
| 图像尺寸 | 最长边≤512、32对齐动态尺寸；生成默认512 | radar224、目标/评测FPV448 | radar224、直接生成448，原生结构支持 |
| Pose | 原生相机文本及camera map | 物理5DoF文本 | 物理5DoF文本，无数值adapter |
| 视觉侧 | stage4冻RADIO，不单独冻projector | 视觉塔/VAE冻结，inactive loc冻结 | RADIO+视觉语言projector+VAE始终冻结 |
| LoRA | 官方训练配置未启用 | 语言/生成r32，生成connector r16，α64/d.05 | 按同职能r32/r16，α64/d.05 |
| 全训小模块 | 原生生成模块 | queries/生成projector | meta_queries/projector_1/2 |
| 学习率 | stage4 5e-6；stage2另有累计相关配方 | 生成主组1e-4 | LoRA1e-4；全训小模块5e-6 |
| AdamW | β(.9,.95)，矩阵wd.05、低维0 | β(.9,.999)，wd0 | 保留Puffin规则 |
| Scheduler | 3% linear warmup，cosine→0 | .3% warmup，cosine→1e-5 | 585updates warmup，cosine→0 |
| 有效batch | 随阶段/启动拓扑 | 128 | 只校验world×micro×accum=128 |
| 更新/曝光 | 非同一CSGO预算 | 19500/2496000 | 19500/2496000 |
| 保存/完整验证 | stage4每5000保存 | eval=no；正式final | 4000/8000/12000/16000/19500 |
| 主选点 | 无此benchmark约定 | final | late主、best补充 |
| 目标 | 原生flow matching | 原生flow matching | 保留Puffin timestep/noise/flow定义 |
| 推理 | 官方FlowMatchEuler，50步CFG4.5 | 实际DPM-Solver20步CFG4.5 | 官方50步CFG4.5，兼容四阶段加速 |

依据：`Puffin/configs/pipelines/stage_4_instruction_tuning.py`，
`Puffin/configs/datasets/processors.py`，`Puffin/src/datasets/base_datasets.py`，
`Puffin/scripts/demo/generation.py`；参考项目的`csgo_configs/exp32_gen.yaml`、
`outputs/csgo_1b/exp32_gen/trainer_state.json`与实际日志/代码。不得把exp32_gen配置里
inactive localization的action projector LR5e-4当作generation projector LR。

## 条件与职能映射

当前radar经冻结VAE直接进入SD3；文本经Qwen、queries和生成桥进入SD3。RADIO和视觉→语言
`projector`在本生成任务不活跃，但仍显式冻结。地图名和pose只用文本，不创建pose_mlp，
不新增embedding/token参数。文本使用发布物理坐标；x/y .1f、z .3f、pitch/yaw转度后.1f，
模板来自exp32_gen短指令，z范围来自发布frozen calibration。训练/验证/推理共用模板，
不得截断五个值或地图名。

| 职能 | 精确模块前缀 | 状态 | LR | 实测trainable | 占独立总参数约 |
|---|---|---|---|---:|---:|
| 视觉塔 | visual_encoder | frozen | — | 0 | 0% |
| 视觉→语言连接器 | projector | frozen | — | 0 | 0% |
| 图像tokenizer | vae | frozen | — | 0 | 0% |
| 条件语言骨干 | llm.model.layers.* | r32/α64/d.05 | 1e-4 | 36929536 | .810% |
| 生成DiT | transformer.transformer_blocks.* | r32/α64/d.05 | 1e-4 | 41877504 | .918% |
| 生成桥 | connector_1/2、llm2connector_1/2 | r16/α64/d.05 | 1e-4 | 3620864 | .079% |
| 输出桥 | projector_1/2 | full | 5e-6 | 6297600 | .138% |
| 生成查询 | meta_queries | full | 5e-6 | 98304 | .002% |

实际GPU构建审计trainable88,823,808；独立总参数4,560,857,827，比例约1.9475%。Qwen共享
embedding与lm_head只计一次，buffer不计参数。审计修正了方案静态估算：SD3的56,623,104个
位置编码buffer元素不属于参数；可训练模块及其数量与批准方案一致。

精确LoRA targets（其余预训练参数冻结，bias不训）：

```text
llm.model.layers.*.self_attn.{q_proj,k_proj,v_proj,o_proj}
llm.model.layers.*.mlp.{gate_proj,up_proj,down_proj}
transformer.transformer_blocks.*.attn.{to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out}
transformer.transformer_blocks.*.ff.net.{0.proj,2}
transformer.transformer_blocks.*.ff_context.net.{0.proj,2}
connector_{1,2}.layers.*.self_attn.{q_proj,k_proj,v_proj,out_proj}
connector_{1,2}.layers.*.mlp.{fc1,fc2}
llm2connector_{1,2}
```

共196/285/74个Linear，最后一个SD3块不存在部分context输出，不虚构缺失层。
无官方LoRA配方，故参考exp32_gen的低秩强度与LoRA LR；预训练全训小模块使用官方stage4
较低LR。AdamW β(.9,.95)、矩阵wd.05/低维0、clip1、3%warmup→cosine0保持Puffin特性。
这是新的混合配方，不冒称官方LoRA或已验证最优配方。

## 预算、恢复与结果选择

每update128条真实源样本，19500updates，共2496000曝光=49.92个50000样本等价epoch。
全局乱序每epoch截成390个完整batch，尾部80不计入预算；50个组织epoch并非终止依据。
DDP累计loss正确缩放，no_sync同时覆盖forward/backward；scheduler只按update推进。
保存与完整5000条验证只有4000/8000/12000/16000/19500五个常规里程碑。
相对链接best选native val loss最低，late仅19500，latest用于恢复。只用late对比UniLIP final。

checkpoint记录完整state、optimizer/scheduler、禁用scaler状态、各rank RNG、sampler逻辑
update游标、梯度累计边界和内容身份。路径可迁移；有效batch拓扑可改变但不声称跨拓扑逐位
一致。正式和smoke不能互相续跑；旧实验任何目录、链接、checkpoint都不覆盖。
实施时真实resume检查暴露Flash SDPA反向非确定性，因此aligned训练显式启用PyTorch严格
deterministic、cuDNN deterministic、benchmark=False与固定CUBLAS workspace；策略加入
checkpoint身份。不更改原生注意力、目标、采样或优化配方；额外确定性开销作为执行差异记录。

## 原生采样与加速

官方作者Space明确从wusize/Puffin加载VAE，故固定此作者发布权重，而非下载受限SD3整个仓库。
Space的SD3配置和上游官方Git blob一致，shift3.0已经有实际证据；不声称两个不同打包的VAE
权重逐张量相同。资产revision、hash、证据链接写在scripts/csgo_seen10_assets.json。

采样50步FlowMatchEuler、CFG4.5、BF16、seed42、空negative、无thinking；不默认降到20步。
四阶段为batch、posterior/pipeline缓存、独立patch不同尺寸latent的dense路径、固定shape
Transformer compile。原生动态eager留作数值参考；batch与编译浮点差异单独验收，不改采样目标。
每sample一个输出；独立连续帧；目标图不读。输出448 RGB JPEG及manifest identity与原saver对齐。
共享evaluator原样使用，不复制指标代码。

## 文件组织与验收

新配置/PEFT与训练loop独立于旧MMEngine路径；数据层增加text/尺寸显式选项；推理新增aligned
分流和不同latent几何；runner只在显式experiment时选择新版。新增portable setup、固定资产
下载与路径解析；训练环境和共享evaluator环境独立。操作命令只在主文档维护。

验收包括：完整split/calibration、target打开拦截、完整frozen/optimizer参数审计、LoRA覆盖、
多种batch拓扑、DDP累计和scheduler计数、短真实训练/精确resume、离散和连续少量输出、共享
evaluator smoke、不同尺寸dense/compile比较、旧命令回归、迁移dry-run/坏路径失败测试。
真实多卡/远端服务器未实测时必须明确记录，不用CPU或tiny模型测试冒充真实模型验收。
实施边界RUN_FORMAL=0；完成后交由用户手动启动正式实验。

## 历史：首次接入设计（数值pose，非当前aligned方案）

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
