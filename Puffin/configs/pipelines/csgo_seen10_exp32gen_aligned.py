"""Aligned Seen-10 generation architecture and optimizer-step recipe.

This config deliberately contains no run-specific absolute asset identity.
Official weights and small architecture metadata are selected by the caller's
PUFFIN_* environment variables and checked separately by the training runner.
"""

import os

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from mmengine.config import read_base

with read_base():
    from .csgo_seen10 import model


EXPERIMENT = "csgo_seen10_exp32gen_aligned"
RADAR_SIZE = 224
TARGET_SIZE = 448
POSE_MODE = "text"
GLOBAL_BATCH = 128
TRAIN_EXAMPLES = 50000
TRAIN_EXAMPLES_PER_EPOCH = 49920
UPDATES_PER_EPOCH = 390
MAX_UPDATES = 19500
WARMUP_UPDATES = 585
MILESTONES = (4000, 8000, 12000, 16000, 19500)
VALIDATION_EXAMPLES = 5000
VALIDATION_SEED = 20260827

# Keep the original architecture and native Puffin loss. Official Base and
# VAE are loaded before adapter injection in csgo_seen10.aligned.
model.freeze_visual_encoder = True
model.freeze_llm = True
model.freeze_transformer = True
model.pose_conditioning = False
model.unconditional = 0.1
model.unconditional_cross_view = 0.1
model.use_activation_checkpointing = True
model.val_autocast_dtype = "bfloat16"
model.generation_dtype = "bfloat16"
model.strict_pretrained_loading = True
model.pretrained_pth = None
model.pretrained_vae_pth = None
model.llm.model_name_or_path = os.environ.get("PUFFIN_QWEN_PATH", "")
model.llm.local_files_only = True
model.visual_encoder.model_name_or_path = os.environ.get("PUFFIN_RADIO_PATH", "")
model.visual_encoder.local_files_only = True
model.tokenizer.pretrained_model_name_or_path = os.environ.get("PUFFIN_QWEN_PATH", "")
model.tokenizer.local_files_only = True
model.vae = dict(
    type=AutoencoderKL.from_config,
    pretrained_model_name_or_path=os.environ.get("PUFFIN_SD3_PATH", ""),
    subfolder="vae",
    local_files_only=True,
)
model.train_scheduler = dict(
    type=FlowMatchEulerDiscreteScheduler.from_config,
    pretrained_model_name_or_path=os.environ.get("PUFFIN_SD3_PATH", ""),
    subfolder="scheduler",
    local_files_only=True,
)
model.test_scheduler = dict(model.train_scheduler)
del os
