"""Serve options this server or a family has no path for, refused before any weight is downloaded."""

from __future__ import annotations

import argparse
from typing import Any


def check(args: argparse.Namespace, family: Any, backend: str, config_dir: Any = None) -> None:
    """Refuse image and share options the family can't serve (``backend`` is always ``mlx`` in PLI)."""

    if getattr(args, "vision_urls", False) and not getattr(args, "vision", False):
        raise ValueError("--vision-urls needs --vision")
    images = getattr(args, "vision_max_images", None)
    if images is not None:
        if not isinstance(images, int) or isinstance(images, bool) or images < 1:
            raise ValueError("--vision-max-images must be a positive integer")
        if not getattr(args, "vision", False):
            raise ValueError("--vision-max-images needs --vision")
    if getattr(args, "vision", False):             # only --vision reads the config here
        from tensorfold.families import read_config
        from tensorfold.vision.config import validate_vision_config

        validate_vision_config(read_config(config_dir) if config_dir else {}, family.model_type)
        if family.model_type == "qwen4_exp":
            raise ValueError("--vision: Flash Next has no image tower on Apple Silicon yet")
    share = getattr(args, "decode_share", None)
    if share is not None and share < 0:
        raise ValueError(f"--decode-share is 0 (whole prompts first) or more, not {share}")


def vision_options(args: argparse.Namespace) -> dict[str, Any]:
    """``--vision`` and ``--vision-urls`` as a family's load options."""

    if not getattr(args, "vision", False):
        return {}
    return {"vision": True, "vision_urls": bool(getattr(args, "vision_urls", False))}


__all__ = ["check", "vision_options"]
