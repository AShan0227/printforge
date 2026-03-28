"""
切片预览模块 — 将 STL/3MF 网格按层切片并生成可视化预览图。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from trimesh import Trimesh


@dataclass
class LayerContour:
    """单层切片轮廓数据。"""

    layer_index: int
    z_height: float  # mm
    polygons: List[Any] = field(default_factory=list)
    # 每个 polygon 是一个 (N, 2) 的 numpy array，表示该闭合轮廓上的点

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "z_height": self.z_height,
            "polygons": [p.tolist() if isinstance(p, np.ndarray) else p for p in self.polygons],
        }


class SlicerPreview:
    """
    将 3D 网格切片并生成预览图的工具类。

    示例
    ------
    >>> sp = SlicerPreview()
    >>> mesh = trimesh.load("model.stl")
    >>> layers = sp.slice_layers(mesh, layer_height_mm=0.2)
    >>> sp.generate_preview_image(layers, "preview.png")
    """

    def __init__(self, mesh: Optional[Trimesh] = None) -> None:
        """
        Args:
            mesh: 预加载的三角网格对象。若为 None，后续调用 slice_layers 时再传入。
        """
        self._mesh: Optional[Trimesh] = mesh

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def slice_layers(
        self,
        mesh: Union[Trimesh, str, Path],
        layer_height_mm: float = 0.2,
    ) -> List[LayerContour]:
        """
        将输入网格按给定层高切片，返回每层轮廓坐标列表。

        Args:
            mesh: trimesh.Trimesh 对象或文件路径（支持 STL / 3MF 等 trimesh 可读格式）。
            layer_height_mm: 切片层高，单位 mm。

        Returns:
            List[LayerContour]: 每层的轮廓列表。

        Raises:
            FileNotFoundError: 传入的路径不存在。
            ValueError: 网格数据无效。
        """
        tri_mesh = self._load_mesh(mesh)
        self._mesh = tri_mesh

        bounds = tri_mesh.bounds  # shape (2, 3): [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        z_min, z_max = float(bounds[0, 2]), float(bounds[1, 2])

        if z_max <= z_min:
            raise ValueError(f"网格 Z 轴范围无效: z_min={z_min}, z_max={z_max}")

        layers: List[LayerContour] = []
        z = z_min + layer_height_mm
        layer_idx = 0

        while z <= z_max:
            # 用 trimesh.section 取 z 高度的横截面
            try:
                slice_geom, to_4x4 = tri_mesh.section(
                    plane_normal=(0, 0, 1),
                    plane_origin=(0, 0, z),
                )
            except Exception as exc:  # pragma: no cover — 某些 z 高度可能无交点
                slice_geom = None

            contour = LayerContour(layer_index=layer_idx, z_height=round(z, 4))

            if slice_geom is not None:
                # section 返回的 geometry 可能是多段线集合，展开所有实体
                entities, transform = self._extract_entities(slice_geom, to_4x4)
                contour.polygons = self._entities_to_polygons(entities, transform)

            layers.append(contour)
            z += layer_height_mm
            layer_idx += 1

        return layers

    def generate_preview_image(
        self,
        layers: List[LayerContour],
        output_path: Union[str, Path],
        step: int = 10,
        figsize: Tuple[int, int] = (12, 16),
        dpi: int = 150,
    ) -> Path:
        """
        按指定间隔抽取切片层，绘制切片预览图。

        每 10 层（step=10）画一张子图，X/Y 轴为实际 mm 坐标，
        子图标题注明层号与 Z 高度。

        Args:
            layers: slice_layers 返回的层列表。
            output_path: 输出图片路径（支持 .png / .pdf / .svg 等 matplotlib 支持格式）。
            step: 间隔多少层取一张。
            figsize: 单张子图的宽高（英寸）。
            dpi: 输出分辨率。

        Returns:
            Path: 输出文件的绝对路径。
        """
        output_path = Path(output_path)
        sampled = layers[::step]
        if not sampled:
            sampled = layers[:1]

        n = len(sampled)
        cols = 3
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
        axes = np.atleast_2d(axes)
        axes = axes.flatten()

        for ax, layer in zip(axes, sampled):
            ax.set_title(f"Layer {layer.layer_index}  |  Z={layer.z_height:.2f} mm", fontsize=9)
            ax.set_aspect("equal")
            ax.invert_yaxis()  # 3D 打印习惯：Z 向上，但图片 Y 轴向下

            for poly in layer.polygons:
                pts = np.asarray(poly)
                if pts.shape[0] < 2:
                    continue
                # 闭合多边形
                closed = np.vstack([pts, pts[0]])
                ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=0.6)
                ax.fill(closed[:, 0], closed[:, 1], alpha=0.15, color="gray")

            # 自动范围
            if layer.polygons:
                all_pts = np.vstack([np.asarray(p) for p in layer.polygons])
                pad = 1.0
                ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
                ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.grid(True, alpha=0.3)

        # 隐藏多余的空子图
        for ax in axes[n:]:
            ax.axis("off")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi)
        plt.close(fig)
        return output_path

    def export_layers_json(
        self,
        layers: List[LayerContour],
        output_path: Union[str, Path],
    ) -> Path:
        """将切片数据导出为 JSON 文件，便于后续 Cura / PrusaSlicer 等导入。"""
        output_path = Path(output_path)
        data = {"layers": [layer.to_dict() for layer in layers]}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_mesh(self, mesh: Union[Trimesh, str, Path]) -> Trimesh:
        """统一加载接口。"""
        if isinstance(mesh, Trimesh):
            return mesh
        path = Path(mesh)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        loaded = trimesh.load(str(path), force="mesh")
        if not isinstance(loaded, Trimesh):
            # 3MF 可能返回 Scene，取主 mesh
            if hasattr(loaded, "geometry") and loaded.geometry:
                loaded = next(iter(loaded.geometry.values()))
            else:
                raise ValueError(f"无法从 {path} 提取三角网格")
        return loaded

    def _extract_entities(
        self, geom: Any, transform: np.ndarray
    ) -> Tuple[List[Any], np.ndarray]:
        """从 slice_geom 中抽取所有多段线实体（discrete segments）。"""
        entities = []

        # trimesh >= 4.x: geom.entities / geom.vertices
        try:
            vert = np.array(geom.vertices)
            for ent in geom.entities:
                # 每个 entity 是一段折线索引
                indices = ent.points
                entities.append(vert[indices])
        except AttributeError:
            # fallback: 旧版本 trimesh
            if hasattr(geom, "entities"):
                for ent in geom.entities:
                    entities.append(np.array(ent.points))
            else:
                # geom 本身是 lines / sequences
                entities.append(np.array(geom))

        return entities, transform

    def _entities_to_polygons(
        self, entities: List[np.ndarray], transform: np.ndarray
    ) -> List[np.ndarray]:
        """将线段实体组合成闭合多边形 (N, 2)。"""
        polygons: List[np.ndarray] = []

        # 先把 transform 应用到 XY
        def apply_transform(pts: np.ndarray) -> np.ndarray:
            ones = np.ones((pts.shape[0], 1))
            pts4 = np.hstack([pts, ones])
            transformed = (transform @ pts4.T).T
            return transformed[:, :2]  # 只保留 XY

        for seg in entities:
            seg_xy = apply_transform(seg)
            # 过滤掉过短的线段（可能是浮点噪声）
            if seg_xy.shape[0] < 2:
                continue
            polygons.append(seg_xy)

        return polygons

    # ------------------------------------------------------------------
    # 快捷方法
    # ------------------------------------------------------------------

    def slice_and_preview(
        self,
        mesh: Union[Trimesh, str, Path],
        output_path: Union[str, Path] = "slice_preview.png",
        layer_height_mm: float = 0.2,
    ) -> Tuple[List[LayerContour], Path]:
        """
        一步完成切片 + 生成预览图。

        Returns:
            (layers, output_path)
        """
        layers = self.slice_layers(mesh, layer_height_mm=layer_height_mm)
        img_path = self.generate_preview_image(layers, output_path)
        return layers, img_path
