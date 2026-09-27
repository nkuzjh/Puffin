"""Text-only 5DoF conditioning, matching exp32_gen's physical short prompt."""

from __future__ import annotations

import hashlib
import math

PROMPT_VERSION = "physical_pose_exp32gen_short_v1"
PROMPT_TEMPLATE = (
    "Generate a CS2 FPV image on map '{map_name}' from the radar map and camera pose: "
    "x={x:.1f}, y={y:.1f}, z={z:.3f}, pitch={pitch:.1f}, yaw={yaw:.1f}. "
    "Coordinates use 1024x1024 map pixels; yaw 0=east clockwise; "
    "pitch 0=down, 180=up; z range [{z_min:.2f}, {z_max:.2f}]."
)
PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()


def build_pose_prompt(row) -> str:
    raw, calibration = row["pose_raw"], row["z_calibration"]
    numbers = {key: float(raw[key]) for key in ("x", "y", "z", "pitch", "yaw")}
    numbers.update({key: float(calibration[key]) for key in ("z_min", "z_max")})
    if not all(math.isfinite(value) for value in numbers.values()):
        raise ValueError("Non-finite physical pose/calibration in published row")
    if numbers["z_max"] <= numbers["z_min"]:
        raise ValueError("Invalid frozen z calibration")
    numbers["pitch"] = math.degrees(numbers["pitch"])
    numbers["yaw"] = math.degrees(numbers["yaw"])
    return PROMPT_TEMPLATE.format(map_name=row["map_name"], **numbers)


def check_prompt_lengths(tokenizer, rows, max_length=256):
    """Fail before training instead of silently truncating condition information."""
    longest = 0
    for row in rows:
        # Include Puffin's native generation wrapper and reserve queries separately.
        text = "<|im_start|>user\nGenerate an image: " + build_pose_prompt(row) + "<|im_end|>\n<|im_start|>assistant\n"
        length = len(tokenizer.encode(text, add_special_tokens=False))
        longest = max(longest, length)
        if length > max_length:
            raise ValueError(f"Pose prompt would be truncated: {row['sample_id']} has {length}>{max_length} tokens")
    return longest
