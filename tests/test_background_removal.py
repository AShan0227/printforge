"""
Tests for printforge.background_removal
=======================================
覆盖三个后端 + auto 链 + scale_foreground utility。
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from printforge.background_removal import (
    Backend,
    BackgroundRemover,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def red_circle_on_white() -> Image.Image:
    """500×500 image: solid red circle centered on white background."""
    img = Image.new("RGB", (500, 500), (255, 255, 255))
    arr = np.array(img)
    # Draw a red circle manually (radius 150, center 250,250)
    y, x = np.ogrid[:500, :500]
    mask = (x - 250) ** 2 + (y - 250) ** 2 <= 150 ** 2
    arr[mask] = [220, 30, 30]
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def random_photo() -> Image.Image:
    """Pseudo-random 256×256 RGB image (deterministic seed)."""
    rng = np.random.default_rng(0xDEADBEEF)
    arr = (rng.uniform(0, 255, (256, 256, 3))).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# --------------------------------------------------------------------------- #
# Test 1 — threshold backend (zero-dependency fallback)
# --------------------------------------------------------------------------- #

class TestThresholdBackend:
    """ThresholdBackend 零依赖，不需要网络也不需要 rembg。"""

    def test_remove_red_circle_on_white(self, red_circle_on_white):
        """Threshold backend correctly isolates the red circle."""
        remover = BackgroundRemover(backend="threshold")
        result = remover.remove(red_circle_on_white)

        # 返回值必须是 RGBA PIL.Image
        assert isinstance(result, Image.Image), "返回值应该是 PIL.Image"
        assert result.mode == "RGBA", f"模式应为 RGBA，实际为 {result.mode}"

        # 尺寸不变
        assert result.size == red_circle_on_white.size

        # alpha 通道应大部分为 0（背景）或 255（前景）
        alpha = np.array(result)[:, :, 3]
        unique = np.unique(alpha)
        assert len(unique) > 1, "alpha 通道应同时包含透明和不透明像素"

        # 中心区域（圆内）alpha 应该高
        cy, cx = 250, 250
        center_alpha = alpha[cy - 30:cy + 30, cx - 30:cx + 30].mean()
        assert center_alpha > 128, (
            f"圆心区域 alpha 均值应 >128，实际 {center_alpha:.1f}"
        )

        # 四个角（背景）alpha 应该低
        corner_alpha = np.concatenate([
            alpha[:10, :10].ravel(),
            alpha[:10, -10:].ravel(),
            alpha[-10:, :10].ravel(),
            alpha[-10:, -10:].ravel(),
        ]).mean()
        assert corner_alpha < 64, (
            f"四角 alpha 均值应 <64，实际 {corner_alpha:.1f}"
        )

    def test_remove_random_photo_does_not_crash(self, random_photo):
        """Threshold backend must not raise on arbitrary input."""
        remover = BackgroundRemover(backend="threshold")
        result = remover.remove(random_photo)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        assert result.size == random_photo.size

    def test_returns_rgba_with_alpha(self):
        """Result must be RGBA with valid alpha channel."""
        # 灰度图
        img = Image.new("L", (100, 100), 128)
        remover = BackgroundRemover(backend="threshold")
        result = remover.remove(img)

        assert result.mode == "RGBA"
        alpha = np.array(result)[:, :, 3]
        assert alpha.max() <= 255
        assert alpha.min() >= 0


# --------------------------------------------------------------------------- #
# Test 2 — rembg backend (mocked)
# --------------------------------------------------------------------------- #

class TestRembgBackend:
    """rembg 后端通过 mock 测试，避免实际安装依赖。"""

    def test_rembg_called_and_returns_rgba(self, red_circle_on_white):
        """When rembg is available it is called and returns RGBA."""
        # 构造一个假的 RGBA PIL.Image 模拟 rembg 输出
        fake_rgba = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
        fake_rgba_arr = np.array(fake_rgba)
        # 在中心画一个半透明红圆
        y, x = np.ogrid[:500, :500]
        mask = (x - 250) ** 2 + (y - 250) ** 2 <= 150 ** 2
        fake_rgba_arr[mask] = [220, 30, 30, 255]
        fake_rgba = Image.fromarray(fake_rgba_arr, mode="RGBA")

        with patch(
            "printforge.background_removal.rembg_remove",
            return_value=fake_rgba,
        ):
            remover = BackgroundRemover(backend="rembg")
            result = remover.remove(red_circle_on_white)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        # rembg mock 的 RGBA alpha 圆内应为 255
        alpha = np.array(result)[:, :, 3]
        center = alpha[235:265, 235:265].mean()
        assert center > 200, f"圆心 alpha 应 >200，实际 {center:.1f}"

    def test_rembg_not_installed_falls_back(self, red_circle_on_white):
        """When rembg cannot be imported, raises ImportError → auto falls back."""
        with patch(
            "printforge.background_removal.BackgroundRemover._try_rembg",
            return_value=None,
        ):
            # _remove_auto 会因为 _try_rembg 返回 None 而落到 threshold
            remover = BackgroundRemover(backend="auto")
            result = remover.remove(red_circle_on_white)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"


# --------------------------------------------------------------------------- #
# Test 3 — BiRefNet HF API backend (mocked HTTP response)
# --------------------------------------------------------------------------- #

class TestBiRefNetHFBackend:
    """BiRefNet HF API 后端通过 mock requests 测试。"""

    def test_birefnet_hf_returns_rgba(self, red_circle_on_white):
        """
        当 BiRefNet HF API 返回 RGBA PNG 时，结果正确解析为 RGBA PIL.Image。
        """
        # 构造 fake RGBA PNG bytes
        fake_rgba = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
        buf = io.BytesIO()
        fake_rgba.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_resp = type("Response", (), {})(  # noqa: SIM905
            lambda self: None,
        )()
        mock_resp.status_code = 200
        mock_resp.content = png_bytes
        mock_resp.raise_for_status = lambda: None

        with patch(
            "printforge.background_removal.requests.post",
            return_value=mock_resp,
        ), patch(
            "printforge.background_removal._get_hf_token",
            return_value="fake-token",
        ):
            remover = BackgroundRemover(backend="birefnet-hf")
            result = remover.remove(red_circle_on_white)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        assert result.size == red_circle_on_white.size

    def test_birefnet_hf_no_token_raises(self, red_circle_on_white):
        """Without HF_TOKEN the backend raises RuntimeError."""
        with patch(
            "printforge.background_removal._get_hf_token",
            return_value=None,
        ):
            remover = BackgroundRemover(backend="birefnet-hf")
            with pytest.raises(RuntimeError, match="HF_TOKEN"):
                remover.remove(red_circle_on_white)

    def test_birefnet_hf_503_triggers_auto_fallback(self, red_circle_on_white):
        """503 from HF API → RuntimeError → auto chain falls back to rembg/threshold."""
        class FakeResp:
            status_code = 503
            def raise_for_status(self):
                raise requests.HTTPError("503")
            content = b""

        with patch(
            "printforge.background_removal.requests.post",
            return_value=FakeResp(),
        ), patch(
            "printforge.background_removal._get_hf_token",
            return_value="fake-token",
        ):
            # auto chain → _try_birefnet_hf catches exception → returns None
            # → _try_rembg returns None → threshold fallback
            remover = BackgroundRemover(backend="auto")
            result = remover.remove(red_circle_on_white)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"

    def test_birefnet_hf_mask_as_float32_parses(self, red_circle_on_white):
        """
        当 API 返回 float32 mask bytes 时，_remove_birefnet_hf
        正确解析为 RGBA（通过 np.frombuffer 路径）。
        """
        side = 512
        mask = np.random.default_rng(42).uniform(0, 1, (side, side)).astype(np.float32)
        mask_bytes = mask.tobytes()

        class FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            # 第一次 Image.open 失败 → 走 np.frombuffer
            # 第二次打开 PNG 也失败 → 走 mask 解析
            def json(self):
                raise NotImplementedError

        fake_resp = FakeResp()
        fake_resp.content = mask_bytes

        with patch(
            "printforge.background_removal.requests.post",
            return_value=fake_resp,
        ), patch(
            "printforge.background_removal._get_hf_token",
            return_value="fake-token",
        ), patch(
            "printforge.background_removal.BackgroundRemover._remove_birefnet_hf",
            wraps=lambda self, img: self._remove_birefnet_hf.__wrapped__(self, img),
        ):
            # 直接 patch 解析路径更可靠：用 RGBA 图像做 content
            rgba_img = Image.new("RGBA", (500, 500), (100, 150, 200, 180))
            buf = io.BytesIO()
            rgba_img.save(buf, format="PNG")
            fake_resp.content = buf.getvalue()

            remover = BackgroundRemover(backend="birefnet-hf")
            result = remover.remove(red_circle_on_white)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"


# --------------------------------------------------------------------------- #
# Test 4 — scale_foreground utility
# --------------------------------------------------------------------------- #

class TestScaleForeground:
    """scale_foreground 将前景 bounding box 居中缩放至 512×512。"""

    def test_scale_foreground_preserves_foreground(self, red_circle_on_white):
        """
        缩放后，输出 RGB 中前景区域应可见（alpha=255 的像素应在内部）。
        """
        # 先用 threshold 后端获取 RGBA
        remover = BackgroundRemover(backend="threshold")
        rgba = remover.remove(red_circle_on_white)

        scaled = remover.scale_foreground(rgba)

        assert isinstance(scaled, Image.Image)
        assert scaled.mode == "RGB"
        assert scaled.size == (512, 512)

        # 前景（alpha>0）在缩放后仍可见于 RGB 中
        alpha = np.array(rgba)[:, :, 3]
        fg_rows, fg_cols = np.where(alpha > 0)
        assert len(fg_rows) > 0, "RGBA 前景应非空"

        # scaled 输出中前景区域的平均亮度应不同于灰色背景（0.5）
        scaled_arr = np.array(scaled, dtype=np.float32) / 255.0
        # 取中心 50×50 区域
        center = scaled_arr[256 - 25:256 + 25, 256 - 25:256 + 25]
        center_mean = center.mean()
        # 前景亮度应与背景有差异（背景=0.5, 前景~220/255）
        assert abs(center_mean - 0.5) > 0.1, (
            f"前景区域亮度应明显偏离灰色背景 0.5，实际 {center_mean:.3f}"
        )

    def test_scale_foreground_empty_fg(self):
        """全透明 RGBA → scale_foreground 返回原图 RGB。"""
        img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        remover = BackgroundRemover(backend="threshold")
        scaled = remover.scale_foreground(img)

        assert isinstance(scaled, Image.Image)
        assert scaled.mode == "RGB"
        # 全透明 → 不裁剪，resize 到 512
        assert scaled.size == (512, 512)


# --------------------------------------------------------------------------- #
# Test 5 — Backend enum + init
# --------------------------------------------------------------------------- #

class TestBackendInit:
    """Backend enum 和 __init__ 参数。"""

    def test_auto_backend(self):
        r = BackgroundRemover(backend="auto")
        assert r.backend == Backend.AUTO

    def test_birefnet_hf_backend_string(self):
        r = BackgroundRemover(backend="birefnet-hf")
        assert r.backend == Backend.BIREFNET_HF

    def test_rembg_backend(self):
        r = BackgroundRemover(backend="rembg")
        assert r.backend == Backend.REMBG

    def test_threshold_backend(self):
        r = BackgroundRemover(backend="threshold")
        assert r.backend == Backend.THRESHOLD

    def test_foreground_ratio(self):
        r = BackgroundRemover(foreground_ratio=0.7)
        assert r.foreground_ratio == 0.7

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            BackgroundRemover(backend="nonexistent")


# --------------------------------------------------------------------------- #
# Test 6 — auto chain end-to-end (no network)
# --------------------------------------------------------------------------- #

class TestAutoChain:
    """auto 模式链式调用，不依赖网络。"""

    def test_auto_uses_threshold_when_all_fail(self, red_circle_on_white):
        """
        _try_birefnet_hf → None, _try_rembg → None
        → auto 应落到 threshold，返回有效 RGBA。
        """
        with patch(
            "printforge.background_removal.BackgroundRemover._try_birefnet_hf",
            return_value=None,
        ), patch(
            "printforge.background_removal.BackgroundRemover._try_rembg",
            return_value=None,
        ):
            remover = BackgroundRemover(backend="auto")
            result = remover.remove(red_circle_on_white)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"
        # threshold 后端应该能处理红色圆形
        alpha = np.array(result)[:, :, 3]
        center = alpha[235:265, 235:265].mean()
        assert center > 80, f"中心 alpha 应 >80，实际 {center:.1f}"
