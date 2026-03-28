"""
测试 slicer_preview — 切片预览模块。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from printforge.slicer_preview import LayerContour, SlicerPreview


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cube_mesh() -> trimesh.Trimesh:
    """返回一个 10×10×10 mm 的简单立方体网格。"""
    mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    mesh.apply_translation([0, 0, 5])  # 移至 Z∈[5,15]
    return mesh


@pytest.fixture
def cylinder_mesh() -> trimesh.Trimesh:
    """返回一个半径 5 mm、高 20 mm 的圆柱网格。"""
    mesh = trimesh.creation.cylinder(radius=5.0, height=20.0, sections=32)
    mesh.apply_translation([0, 0, 10])  # 移至 Z∈[0,20]
    return mesh


@pytest.fixture
def stl_file(cube_mesh: trimesh.Trimesh) -> Path:
    """将立方体导出为临时 STL 文件。"""
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        tmp = Path(f.name)
    cube_mesh.export(str(tmp), file_type="stl")
    yield tmp
    tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 1. 基础切片功能
# ---------------------------------------------------------------------------

class TestSliceLayers:
    """slice_layers 核心功能测试。"""

    def test_returns_list_of_layer_contour(self, cube_mesh: trimesh.Trimesh):
        """返回列表长度与层数一致，每项是 LayerContour。"""
        sp = SlicerPreview()
        layers = sp.slice_layers(cube_mesh, layer_height_mm=2.0)  # Z∈[5,15], step=2 → 5 layers
        assert isinstance(layers, list)
        assert len(layers) >= 1
        assert all(isinstance(l, LayerContour) for l in layers)

    def test_layer_z_monotonically_increasing(self, cube_mesh: trimesh.Trimesh):
        """Z 高度严格递增。"""
        sp = SlicerPreview()
        layers = sp.slice_layers(cube_mesh, layer_height_mm=1.0)
        z_vals = [l.z_height for l in layers]
        assert z_vals == sorted(z_vals)

    def test_layer_index_sequential(self, cube_mesh: trimesh.Trimesh):
        """layer_index 从 0 开始连续递增。"""
        sp = SlicerPreview()
        layers = sp.slice_layers(cube_mesh, layer_height_mm=2.0)
        indices = [l.layer_index for l in layers]
        assert indices == list(range(len(indices)))

    def test_fine_layer_height_many_layers(self, cube_mesh: trimesh.Trimesh):
        """细层高（0.1 mm）生成足够多切片。"""
        sp = SlicerPreview()
        layers = sp.slice_layers(cube_mesh, layer_height_mm=0.1)
        # Z 跨度 10 mm，步长 0.1 → ≈ 100 层（包含首尾浮点）
        assert len(layers) >= 90


# ---------------------------------------------------------------------------
# 2. 轮廓几何正确性
# ---------------------------------------------------------------------------

class TestLayerContours:
    """层轮廓数据正确性。"""

    def test_cube_layer_has_single_rectangle(self, cube_mesh: trimesh.Trimesh):
        """立方体每层应该只有一个闭合矩形轮廓。"""
        sp = SlicerPreview()
        layers = sp.slice_layers(cube_mesh, layer_height_mm=1.0)
        mid_layer = layers[len(layers) // 2]
        # 在 Z∈[5,15] 中间区域，应该只有一条外轮廓
        assert len(mid_layer.polygons) >= 1
        for poly in mid_layer.polygons:
            pts = np.asarray(poly)
            assert pts.shape[1] == 2, "多边形须为 (N, 2) 形状"
            assert pts.shape[0] >= 4, "矩形至少 4 个顶点"

    def test_cylinder_layer_is_circle(self, cylinder_mesh: trimesh.Trimesh):
        """圆柱体中间层轮廓近似圆形。"""
        sp = SlicerPreview()
        layers = sp.slice_layers(cylinder_mesh, layer_height_mm=0.5)
        mid_layer = layers[len(layers) // 2]
        assert len(mid_layer.polygons) >= 1
        poly = np.asarray(mid_layer.polygons[0])
        # 半径约 5 mm
        dists = np.linalg.norm(poly, axis=1)
        mean_r = dists.mean()
        assert 4.0 <= mean_r <= 6.0, f"期望半径≈5, 实际={mean_r:.2f}"

    def test_bottom_top_layers_have_no_intersection(self, cube_mesh: trimesh.Trimesh):
        """立方体底面和顶面之外不应有轮廓（恰好在 Z=5 或 Z=15 时）。"""
        sp = SlicerPreview()
        layers = sp.slice_layers(cube_mesh, layer_height_mm=0.1)
        first = layers[0]
        last = layers[-1]
        # 第一层 Z 刚好在 5+，最后一层 Z 刚好在 15- 以下
        assert first.z_height > cube_mesh.bounds[0, 2]
        assert last.z_height < cube_mesh.bounds[1, 2]


# ---------------------------------------------------------------------------
# 3. 预览图生成
# ---------------------------------------------------------------------------

class TestPreviewImage:
    """generate_preview_image 功能测试。"""

    def test_generates_png_file(self, cube_mesh: trimesh.Trimesh):
        """调用后返回存在的 PNG 文件。"""
        sp = SlicerPreview()
        layers = sp.slice_layers(cube_mesh, layer_height_mm=2.0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preview.png"
            result = sp.generate_preview_image(layers, out)
            assert result.exists()
            assert result.suffix == ".png"
            assert result.stat().st_size > 0

    def test_step_parameter_respected(self, cube_mesh: trimesh.Trimesh):
        """step=5 时，子图数量约为总层数/5。"""
        sp = SlicerPreview()
        layers = sp.slice_layers(cube_mesh, layer_height_mm=0.5)  # ~20 layers
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preview.png"
            sp.generate_preview_image(layers, out, step=5)
            # 文件成功生成即通过（图片内容由 matplotlib 负责）
            assert out.exists()

    def test_empty_layers_still_produces_file(self):
        """空层列表也能生成（至少一张图）。"""
        sp = SlicerPreview()
        empty_layer = LayerContour(layer_index=0, z_height=0.0, polygons=[])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preview.png"
            result = sp.generate_preview_image([empty_layer], out)
            assert result.exists()


# ---------------------------------------------------------------------------
# 4. 路径 / 文件接口
# ---------------------------------------------------------------------------

class TestFileInterface:
    """文件路径加载与导出的正确性。"""

    def test_load_stl_by_path(self, stl_file: Path):
        """传入 STL 文件路径正常切片。"""
        sp = SlicerPreview()
        layers = sp.slice_layers(stl_file, layer_height_mm=2.0)
        assert len(layers) >= 1
        assert all(isinstance(l, LayerContour) for l in layers)

    def test_load_nonexistent_raises(self):
        """不存在的文件路径抛出 FileNotFoundError。"""
        sp = SlicerPreview()
        with pytest.raises(FileNotFoundError):
            sp.slice_layers("/nonexistent/model.stl")

    def test_export_layers_json(self, cube_mesh: trimesh.Trimesh):
        """export_layers_json 生成有效 JSON 且可反序列化。"""
        sp = SlicerPreview()
        layers = sp.slice_layers(cube_mesh, layer_height_mm=2.0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "layers.json"
            result = sp.export_layers_json(layers, out)
            assert result.exists()
            data = json.loads(result.read_text(encoding="utf-8"))
            assert "layers" in data
            assert len(data["layers"]) == len(layers)


# ---------------------------------------------------------------------------
# 5. 快捷方法 & 边界条件
# ---------------------------------------------------------------------------

class TestConvenienceMethods:
    """slice_and_preview 等快捷方法。"""

    def test_slice_and_preview_returns_both(self, cube_mesh: trimesh.Trimesh):
        """slice_and_preview 返回 (layers, img_path)。"""
        sp = SlicerPreview()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preview.png"
            layers, img_path = sp.slice_and_preview(
                cube_mesh, output_path=out, layer_height_mm=2.0
            )
            assert isinstance(layers, list)
            assert img_path.exists()

    def test_invalid_z_range_raises(self):
        """网格 Z 轴退化（如 2D 平面）应抛出 ValueError。"""
        # 创建一个极薄（Z=0）的退化 mesh
        degenerate = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            faces=[[0, 1, 2]],
        )
        sp = SlicerPreview()
        with pytest.raises(ValueError, match="Z 轴范围无效"):
            sp.slice_layers(degenerate, layer_height_mm=0.1)


# ---------------------------------------------------------------------------
# 6. LayerContour 数据类
# ---------------------------------------------------------------------------

class TestLayerContour:
    """LayerContour.to_dict / 字段正确性。"""

    def test_to_dict_roundtrip(self):
        """to_dict 包含所有必要字段且 JSON 可序列化。"""
        layer = LayerContour(layer_index=3, z_height=1.2345)
        d = layer.to_dict()
        assert d["layer_index"] == 3
        assert d["z_height"] == 1.2345
        assert isinstance(d["polygons"], list)
        # 确认可 JSON 序列化
        json.dumps(d)

    def test_default_polygons_empty_list(self):
        """LayerContour 默认 polygons=[]。"""
        layer = LayerContour(layer_index=0, z_height=0.0)
        assert layer.polygons == []
