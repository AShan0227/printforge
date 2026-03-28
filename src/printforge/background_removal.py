"""
Background Removal — multi-backend抠图模块
==============================================
支持三档 fallback 链：

  1. BiRefNet HF Inference API  (CAAI AIR 2024 SOTA，免费，需 HF_TOKEN)
  2. rembg                    (本地，需 pip install rembg)
  3. 简单阈值法               (零依赖 fallback)

用法
----
    from printforge.background_removal import BackgroundRemover

    remover = BackgroundRemover()          # auto-priority
    # 或指定后端：
    remover = BackgroundRemover(backend="rembg")

    result = remover.remove(image)         # PIL.Image → PIL.Image (RGBA)
"""

from __future__ import annotations

import io
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from env or openclaw token file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".openclaw" / "workspace" / ".hf_token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def _smooth_alpha(alpha: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """
    对 alpha 通道做轻微高斯平滑 + 自适应阈值，输出 [0,1] float32。

    零硬依赖（fallback 到 PIL-only 实现如果 scipy 不可用）。
    """
    try:
        from scipy import ndimage
        smoothed = ndimage.gaussian_filter(alpha, sigma=sigma)
    except ImportError:
        # PIL-based box blur fallback
        from PIL import Image as PILImage, ImageFilter
        img_pil = PILImage.fromarray((alpha * 255).astype(np.uint8), mode="L")
        blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        smoothed = np.array(blurred).astype(np.float32) / 255.0

    # 自适应阈值
    threshold = np.mean(smoothed)
    std = np.std(smoothed) + 1e-6
    result = np.clip((smoothed - threshold) / std * 0.5 + 0.5, 0, 1)
    return result.astype(np.float32)


# --------------------------------------------------------------------------- #
# Backend enum
# --------------------------------------------------------------------------- #


class Backend(str, Enum):
    """Background-removal backend identifiers."""
    AUTO = "auto"          # 尝试 BiRefNet → rembg → threshold
    BIREFNET_HF = "birefnet-hf"   # HuggingFace Inference API
    REMBG = "rembg"        # 本地 rembg 包
    THRESHOLD = "threshold"  # 简单 alpha 阈值，零依赖


# --------------------------------------------------------------------------- #
# Core class
# --------------------------------------------------------------------------- #


class BackgroundRemover:
    """
    多后端背景移除器。

    Args
    ----
    backend : str | Backend
        要使用的主后端。"auto" 按 BiRefNet-HF → rembg → threshold 顺序尝试。
    foreground_ratio : float
        将前景缩放至此比例（0-1），默认 0.85。
    """

    # HF Inference API endpoint
    HF_API_URL = "https://api-inference.huggingface.co/models/ZhengPeng7/BiRefNet"

    def __init__(
        self,
        backend: str | Backend = "auto",
        foreground_ratio: float = 0.85,
    ):
        self.backend = Backend(backend) if isinstance(backend, str) else backend
        self.foreground_ratio = foreground_ratio

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def remove(self, image: Image.Image) -> Image.Image:
        """
        移除输入图像的背景，返回带透明通道的 RGBA PIL.Image。

        Args
        ----
        image : PIL.Image
            输入图像，任意模式（会自动 convert 到 RGB）。

        Returns
        -------
        PIL.Image
            RGBA 图像，背景透明（alpha=0），前景保留原色（alpha=255）。
        """
        if self.backend == Backend.AUTO:
            return self._remove_auto(image)
        elif self.backend == Backend.BIREFNET_HF:
            return self._remove_birefnet_hf(image)
        elif self.backend == Backend.REMBG:
            return self._remove_rembg(image)
        elif self.backend == Backend.THRESHOLD:
            return self._remove_threshold(image)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------ #
    # Auto chain
    # ------------------------------------------------------------------ #

    def _remove_auto(self, image: Image.Image) -> Image.Image:
        """Try BiRefNet-HF → rembg → threshold."""
        # 1. BiRefNet via HuggingFace Inference API
        result = self._try_birefnet_hf(image)
        if result is not None:
            return result

        # 2. rembg (本地)
        result = self._try_rembg(image)
        if result is not None:
            return result

        # 3. 简单阈值（零依赖）
        logger.warning("All backends unavailable — using threshold fallback")
        return self._remove_threshold(image)

    # ------------------------------------------------------------------ #
    # Backend: BiRefNet via HuggingFace Inference API
    # ------------------------------------------------------------------ #

    def _try_birefnet_hf(self, image: Image.Image) -> Optional[Image.Image]:
        """Return RGBA image or None if unavailable."""
        try:
            return self._remove_birefnet_hf(image)
        except Exception as e:
            logger.warning("BiRefNet HF API failed: %s", e)
            return None

    def _remove_birefnet_hf(self, image: Image.Image) -> Image.Image:
        """
        使用 BiRefNet (CAAI AIR 2024) 通过 HuggingFace Inference API 移除背景。

        BiRefNet 输出的是 [1, 1, H, W] 的 sigmoid 概率图（0=背景，1=前景）。
        我们把它转成 alpha mask，合成 RGBA 图像。

        行为：
        - 自动 resize 到模型接受的尺寸
        - 对 mask 做轻微 blur + threshold 得到干净的 alpha 通道
        - 保持原始颜色，alpha=0 表示背景，alpha=255 表示前景
        """
        hf_token = _get_hf_token()
        if not hf_token:
            raise RuntimeError(
                "BiRefNet HF API 需要 HF_TOKEN。"
                "请设置环境变量 HF_TOKEN，或在 ~/.openclaw/workspace/.hf_token 放置 token。"
            )

        # ---- 预处理 ----
        img_rgb = image.convert("RGB")
        orig_w, orig_h = img_rgb.size

        # API 预处理为长边 1024
        input_size = 1024
        ratio = input_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        img_in = img_rgb.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img_in.save(buf, format="PNG")
        payload = buf.getvalue()

        # ---- API 调用 ----
        logger.info("BiRefNet HF API: querying ZhengPeng7/BiRefNet...")
        resp = requests.post(
            self.HF_API_URL,
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/octet-stream",
            },
            data=payload,
            timeout=120,
        )

        if resp.status_code == 503:
            raise RuntimeError(
                f"BiRefNet HF API 模型正在加载中 (503)。"
                "请稍后重试，或使用 rembg/threshold 后端。"
            )
        resp.raise_for_status()

        # ---- 解析响应 ----
        # 路径 A: API 直接返回 RGBA PNG
        try:
            result_rgba = Image.open(io.BytesIO(resp.content)).convert("RGBA")
            logger.info("BiRefNet HF API: received RGBA output directly")
            return result_rgba
        except Exception:
            pass

        # 路径 B: API 返回 float32 mask bytes [H*W]
        try:
            mask_bytes = np.frombuffer(resp.content, dtype=np.float32)
            side = int(np.sqrt(mask_bytes.size))
            mask = mask_bytes.reshape(side, side)
        except Exception:
            # 路径 C: PNG/L mask 图像
            try:
                mask_img = Image.open(io.BytesIO(resp.content)).convert("L")
                mask = np.array(mask_img).astype(np.float32) / 255.0
            except Exception as e:
                raise RuntimeError(
                    f"BiRefNet HF API 返回格式无法解析: {e}"
                )

        # ---- 后处理 mask ----
        # 缩放回原图尺寸并做平滑
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        mask_resized = mask_img.resize((orig_w, orig_h), Image.LANCZOS)
        alpha = np.array(mask_resized).astype(np.float32) / 255.0
        alpha = _smooth_alpha(alpha)

        # 合成 RGBA: RGB(H,W,3) + alpha(H,W,1)
        rgb = np.array(img_rgb)
        alpha_3ch = alpha[:, :, np.newaxis]
        rgba_np = np.concatenate(
            [rgb, (alpha_3ch * 255).astype(np.uint8)],
            axis=2,
        )
        return Image.fromarray(rgba_np, mode="RGBA")

    # ------------------------------------------------------------------ #
    # Backend: rembg (本地)
    # ------------------------------------------------------------------ #

    def _try_rembg(self, image: Image.Image) -> Optional[Image.Image]:
        """Return RGBA result or None if rembg is not available."""
        try:
            from rembg import remove as _rembg_remove
        except ImportError:
            return None
        try:
            return self._remove_rembg(image)
        except Exception as e:
            logger.warning("rembg failed: %s", e)
            return None

    def _remove_rembg(self, image: Image.Image) -> Image.Image:
        """
        使用 rembg 移除背景。

        rembg.remove() 返回 RGBA PIL.Image，背景透明。
        """
        from rembg import remove as rembg_remove

        logger.info("Using rembg for background removal")
        result = rembg_remove(image)
        # 确保 RGBA 返回
        if result.mode != "RGBA":
            result = result.convert("RGBA")
        return result

    # ------------------------------------------------------------------ #
    # Backend: 简单阈值（零依赖 fallback）
    # ------------------------------------------------------------------ #

    def _remove_threshold(self, image: Image.Image) -> Image.Image:
        """
        零依赖 alpha 阈值背景移除。

        策略：
        1. HSV 分解，取 S 通道（S 低 = 背景/阴影）
        2. V 通道辅助（V 极端 = 黑/白 背景）
        3. 组合评分 + 高斯平滑 → alpha mask

        效果远不如深度模型，但对纯色或高对比度背景有效。
        """
        img_rgb = image.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32) / 255.0  # (H, W, 3) float

        # HSV 分解
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin + 1e-8

        # Saturation: 低 S → 可能是背景
        s = np.where(cmax > 0, delta / (cmax + 1e-8), 0.0)

        # Value: 极低或极高 → 可能是背景
        v = cmax
        v_score = np.abs(v - 0.5) * 2  # 0=中等灰, 1=极黑/极白

        # 前景得分 = 高饱和 或 高对比度
        fg_score = s * 0.6 + v_score * 0.4
        alpha = (fg_score > 0.25).astype(np.float32)

        # 平滑 + 自适应阈值
        alpha = _smooth_alpha(alpha)

        # RGB(H,W,3) + alpha(H,W) → RGB(H,W,3) + alpha(H,W,1)
        rgb_uint8 = (arr * 255).astype(np.uint8)
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        rgba_np = np.concatenate([rgb_uint8, alpha_uint8[:, :, np.newaxis]], axis=2)
        return Image.fromarray(rgba_np, mode="RGBA")

    # ------------------------------------------------------------------ #
    # Utility: scale foreground to target ratio
    # ------------------------------------------------------------------ #

    def scale_foreground(self, rgba: Image.Image) -> Image.Image:
        """
        将前景 bounding box 缩放至 foreground_ratio 比例，
        保持长宽比，输出 512×512（TripoSR 默认输入尺寸）。

        Args
        ----
        rgba : PIL.Image (RGBA)

        Returns
        -------
        PIL.Image (RGB) — 前景合成到灰色背景
        """
        arr = np.array(rgba)
        alpha = arr[:, :, 3].astype(np.float32) / 255.0

        # 找前景 bounding box
        rows = np.any(alpha > 0.1, axis=1)
        cols = np.any(alpha > 0.1, axis=0)
        if not (rows.any() and cols.any()):
            # 全透明，resize 回 512×512 灰色背景
            gray = Image.new("RGB", (512, 512), (128, 128, 128))
            return gray

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h, w = rmax - rmin + 1, cmax - cmin + 1
        size = max(h, w)
        scale = 1.0 / self.foreground_ratio

        # margin-preserving crop + resize
        pad = int(size * (scale - 1) / 2)
        pad_top = max(0, rmin - pad)
        pad_bottom = min(arr.shape[0], rmax + 1 + pad)
        pad_left = max(0, cmin - pad)
        pad_right = min(arr.shape[1], cmax + 1 + pad)

        cropped = arr[pad_top:pad_bottom, pad_left:pad_right]
        resized = np.array(
            Image.fromarray(cropped).resize((512, 512), Image.LANCZOS)
        )

        # 合成到灰色背景（前景 alpha=1 → 保留，背景 → 0.5 灰）
        rgb = resized[:, :, :3].astype(np.float32) / 255.0
        a = resized[:, :, 3].astype(np.float32) / 255.0
        a = a[:, :, np.newaxis]  # (H, W) → (H, W, 1) for broadcast
        out_rgb = (rgb * a + (1 - a) * 0.5) * 255
        return Image.fromarray(out_rgb.astype(np.uint8), mode="RGB")
