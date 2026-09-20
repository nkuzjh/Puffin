"""Puffin generation training for the CSGO Benchmark v2 Seen-10 split."""

import os

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from mmengine.config import read_base
from mmengine.dataset import DefaultSampler, InfiniteSampler
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.runner import IterBasedTrainLoop, ValLoop
from src.models.stable_diffusion3.transformer_sd3_dynamic import SD3Transformer2DModel
from csgo_seen10.model_builders import EmptyQwenFromConfig, EmptyRadioFromConfig
from src.optimisers.custom_adamw import CustomAdamW
from csgo_seen10.dataset import CollateSeen10, CsgoSeen10Dataset
from csgo_seen10.hooks import Seen10TrainingArtifactHook

with read_base():
    from ..models.qwen2_5_1_5b_radio_sd3_dynamic_puffin import model


DATA_ROOT = os.environ.get(
    "DATA_ROOT", "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2"
)
SHARED_EVAL_DIR = os.environ.get(
    "SHARED_EVAL_DIR", "/home/jiahao/task/csgo_benchmark_v2_eval_general"
)
IMAGE_SIZE = 448
smoke_mode = os.environ.get("PUFFIN_SEEN10_SMOKE", "0") == "1"
LLM_NAME_OR_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
RADIO_NAME_OR_PATH = "nvidia/C-RADIOv3-H"

# Construct architecture from small config files only. The combined Puffin
# checkpoint supplies Qwen, RADIO, and the SD3 transformer weights; its VAE is
# distributed separately as wusize/Puffin/vae.pth.
# Keep the native Puffin architecture, optimizers and MMEngine checkpoint format.
# SDPA avoids making flash-attn an additional requirement for this configuration.
model.freeze_visual_encoder = True
model.freeze_llm = True
model.freeze_transformer = False
model.use_activation_checkpointing = True
model.pose_conditioning = True
model.val_autocast_dtype = "bfloat16"
model.connector_1._attn_implementation = "sdpa"
model.connector_2._attn_implementation = "sdpa"
model.pretrained_pth = os.environ.get("PUFFIN_INIT_CHECKPOINT") or None
model.pretrained_vae_pth = os.environ.get("PUFFIN_VAE_CHECKPOINT") or None
del os  # Do not leave a module object in MMEngine's serializable config namespace.
model.strict_pretrained_loading = True
model.generation_dtype = "bfloat16"
model.llm = dict(
    type=EmptyQwenFromConfig,
    model_name_or_path=LLM_NAME_OR_PATH,
    dtype="bfloat16",
    attn_implementation="sdpa",
)
model.visual_encoder = dict(
    type=EmptyRadioFromConfig,
    model_name_or_path=RADIO_NAME_OR_PATH,
    dtype="bfloat16",
    amp_dtype="bfloat16",
)
model.transformer = dict(
    type=SD3Transformer2DModel,
    sample_size=128,
    patch_size=2,
    in_channels=16,
    num_layers=24,
    attention_head_dim=64,
    num_attention_heads=24,
    joint_attention_dim=4096,
    caption_projection_dim=1536,
    pooled_projection_dim=2048,
    out_channels=16,
    pos_embed_max_size=192,
)
model.vae = dict(
    type=AutoencoderKL,
    act_fn="silu",
    block_out_channels=[128, 256, 512, 512],
    down_block_types=["DownEncoderBlock2D"] * 4,
    force_upcast=True,
    in_channels=3,
    latent_channels=16,
    layers_per_block=2,
    norm_num_groups=32,
    out_channels=3,
    sample_size=1024,
    scaling_factor=1.5305,
    shift_factor=0.0609,
    up_block_types=["UpDecoderBlock2D"] * 4,
    use_post_quant_conv=False,
    use_quant_conv=False,
)
model.train_scheduler = dict(
    type=FlowMatchEulerDiscreteScheduler,
    num_train_timesteps=1000,
    shift=3.0,
)
model.test_scheduler = dict(
    type=FlowMatchEulerDiscreteScheduler,
    num_train_timesteps=1000,
    shift=3.0,
)

max_iters = 1 if smoke_mode else 12500  # formal run: one pass over 50,000 examples
milestone_count = 1 if smoke_mode else 5
save_steps = max_iters // milestone_count
if not smoke_mode and max_iters % 5:
    raise ValueError("Formal CSGO Seen-10 max_iters must be divisible by five")
batch_size = 1 if smoke_mode else 4
dataloader_num_workers = 0 if smoke_mode else 4
lr = 2e-5
betas = (0.9, 0.95)
weight_decay = 0.05
max_norm = 1.0
warmup_ratio = 0.03

train_dataset = dict(
    type=CsgoSeen10Dataset,
    data_root=DATA_ROOT,
    shared_eval_dir=SHARED_EVAL_DIR,
    split="seen_train",
    include_target=True,
    image_size=IMAGE_SIZE,
    max_samples=1 if smoke_mode else None,
)
val_dataset = dict(
    type=CsgoSeen10Dataset,
    data_root=DATA_ROOT,
    shared_eval_dir=SHARED_EVAL_DIR,
    split="seen_validation",
    include_target=True,
    image_size=IMAGE_SIZE,
    max_samples=1 if smoke_mode else None,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=train_dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=CollateSeen10),
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    drop_last=False,
    dataset=val_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=CollateSeen10),
)

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=CustomAdamW, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    # BF16 does not need FP16 loss scaling. MMEngine's GradScaler otherwise
    # attempts CUDA non-finite checks on BF16 gradients during unscale_().
    loss_scale=dict(enabled=False),
    dtype="bfloat16",
)
if smoke_mode:
    param_scheduler = []
else:
    param_scheduler = [
        dict(
            type=LinearLR,
            start_factor=1e-5,
            by_epoch=False,
            begin=0,
            end=int(warmup_ratio * max_iters),
        ),
        dict(
            type=CosineAnnealingLR,
            eta_min=0.0,
            by_epoch=False,
            begin=int(warmup_ratio * max_iters),
            end=max_iters,
        ),
    ]

train_cfg = dict(type=IterBasedTrainLoop, max_iters=max_iters, val_interval=save_steps)
val_cfg = dict(type=ValLoop)
val_evaluator = []  # ValLoop supplies val_loss from the model's final loss element.

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=milestone_count,
        save_last=True,
        save_best=None,
    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
)
custom_hooks = [
    dict(
        type=Seen10TrainingArtifactHook,
        max_iters=max_iters,
        milestone_count=milestone_count,
        validation_seed=20260827,
    )
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
visualizer = None
log_level = "INFO"
work_dir = "./work_dirs/csgo_seen10"
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
