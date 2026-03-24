"""
PrintForge Core Pipeline
========================
Image → TripoSR Inference → SDF Conversion → Marching Cubes → Watertight Mesh → Print Optimization → 3MF

Each stage is a standalone function, composable into the full pipeline.
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the PrintForge pipeline."""
    # TripoSR inference
    model_name: str = "stabilityai/TripoSR"
    device: str = "cuda"  # "cuda" or "cpu"
    
    # SDF / Marching Cubes
    mc_resolution: int = 256  # Marching cubes grid resolution
    
    # Print optimization
    min_wall_thickness_mm: float = 0.4  # FDM minimum
    max_faces: int = 200_000  # Simplify if exceeding
    add_base: bool = False  # Add flat base for easier printing
    base_height_mm: float = 2.0
    
    # Output
    output_format: str = "3mf"  # "3mf" or "stl"
    scale_mm: float = 50.0  # Default size in mm


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    mesh_path: str
    vertices: int
    faces: int
    is_watertight: bool
    wall_thickness_ok: bool
    duration_ms: float
    stages: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


class PrintForgePipeline:
    """Main pipeline: Image → 3D printable mesh."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._model = None
    
    def run(self, image_path: str, output_path: str) -> PipelineResult:
        """Execute the full pipeline."""
        start = time.time()
        stages = {}
        warnings = []
        
        # Stage 1: Load and preprocess image
        logger.info("Stage 1: Loading image...")
        t0 = time.time()
        image = self._load_image(image_path)
        stages["load_image"] = time.time() - t0
        
        # Stage 2: TripoSR inference → raw mesh
        logger.info("Stage 2: TripoSR inference...")
        t0 = time.time()
        raw_mesh = self._infer_3d(image)
        stages["inference"] = time.time() - t0
        
        # Stage 3: SDF conversion → watertight mesh
        logger.info("Stage 3: SDF watertight conversion...")
        t0 = time.time()
        watertight_mesh = self._make_watertight(raw_mesh)
        stages["watertight"] = time.time() - t0
        
        # Stage 4: Print optimization
        logger.info("Stage 4: Print optimization...")
        t0 = time.time()
        optimized_mesh, opt_warnings = self._optimize_for_print(watertight_mesh)
        warnings.extend(opt_warnings)
        stages["optimization"] = time.time() - t0
        
        # Stage 5: Export
        logger.info("Stage 5: Exporting...")
        t0 = time.time()
        self._export(optimized_mesh, output_path)
        stages["export"] = time.time() - t0
        
        total_ms = (time.time() - start) * 1000
        
        result = PipelineResult(
            mesh_path=output_path,
            vertices=len(optimized_mesh.vertices) if optimized_mesh else 0,
            faces=len(optimized_mesh.faces) if optimized_mesh else 0,
            is_watertight=optimized_mesh.is_watertight if optimized_mesh else False,
            wall_thickness_ok=len([w for w in warnings if "wall" in w.lower()]) == 0,
            duration_ms=total_ms,
            stages=stages,
            warnings=warnings,
        )
        
        logger.info(f"Pipeline complete: {result.vertices} vertices, {result.faces} faces, "
                     f"watertight={result.is_watertight}, {total_ms:.0f}ms")
        return result
    
    def _load_image(self, image_path: str):
        """Load and preprocess image for TripoSR."""
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        # TripoSR expects 512x512
        img = img.resize((512, 512), Image.LANCZOS)
        return img
    
    def _infer_3d(self, image):
        """Run TripoSR inference to get raw 3D mesh."""
        try:
            # Try to use TripoSR
            from tsr.system import TSR
            if self._model is None:
                logger.info(f"Loading TripoSR model: {self.config.model_name}")
                self._model = TSR.from_pretrained(
                    self.config.model_name,
                    config_name="config.yaml",
                    weight_name="model.ckpt",
                )
                self._model.to(self.config.device)
            
            with __import__("torch").no_grad():
                scene = self._model(image, device=self.config.device)
            
            mesh = scene.get_mesh(
                resolution=self.config.mc_resolution,
            )
            return mesh
            
        except ImportError:
            logger.warning("TripoSR not installed. Using placeholder mesh for testing.")
            return self._create_placeholder_mesh()
    
    def _create_placeholder_mesh(self):
        """Create a simple cube mesh for testing without TripoSR."""
        import trimesh
        return trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    
    def _make_watertight(self, mesh):
        """Convert mesh to watertight using SDF → Marching Cubes."""
        import trimesh
        
        if hasattr(mesh, "is_watertight") and mesh.is_watertight:
            logger.info("Mesh is already watertight, skipping SDF conversion")
            return mesh
        
        # Convert to trimesh if not already
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                mesh = trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                )
            else:
                raise ValueError("Cannot convert mesh to trimesh format")
        
        # Method: Voxelize → Marching Cubes → guaranteed watertight
        try:
            # Voxelize the mesh
            pitch = mesh.bounding_box.extents.max() / self.config.mc_resolution
            voxel_grid = mesh.voxelized(pitch)
            
            # Marching cubes to extract watertight surface
            watertight = voxel_grid.marching_cubes
            
            logger.info(f"SDF conversion: {len(mesh.faces)} → {len(watertight.faces)} faces, "
                        f"watertight={watertight.is_watertight}")
            return watertight
            
        except Exception as e:
            logger.warning(f"SDF conversion failed ({e}), attempting repair instead")
            # Fallback: use trimesh repair
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fill_holes(mesh)
            return mesh
    
    def _optimize_for_print(self, mesh):
        """Optimize mesh for 3D printing."""
        import trimesh
        warnings = []
        
        # 1. Scale to target size
        current_size = mesh.bounding_box.extents.max()
        if current_size > 0:
            scale_factor = self.config.scale_mm / current_size
            mesh.apply_scale(scale_factor)
        
        # 2. Simplify if too many faces
        if len(mesh.faces) > self.config.max_faces:
            ratio = self.config.max_faces / len(mesh.faces)
            mesh = mesh.simplify_quadric_decimation(self.config.max_faces)
            logger.info(f"Simplified: {len(mesh.faces)} faces (ratio={ratio:.2f})")
        
        # 3. Fix normals
        trimesh.repair.fix_normals(mesh)
        
        # 4. Check wall thickness (approximate)
        if hasattr(mesh, "bounding_box"):
            min_extent = mesh.bounding_box.extents.min()
            if min_extent < self.config.min_wall_thickness_mm:
                warnings.append(f"Wall thickness warning: min extent {min_extent:.2f}mm < {self.config.min_wall_thickness_mm}mm")
        
        # 5. Add base if requested
        if self.config.add_base:
            base = trimesh.creation.box(
                extents=[
                    mesh.bounding_box.extents[0] * 1.1,
                    mesh.bounding_box.extents[1] * 1.1,
                    self.config.base_height_mm,
                ]
            )
            base.apply_translation([
                mesh.bounding_box.centroid[0],
                mesh.bounding_box.centroid[1],
                mesh.bounds[0][2] - self.config.base_height_mm / 2,
            ])
            mesh = trimesh.util.concatenate([mesh, base])
        
        # 6. Final watertight check
        if not mesh.is_watertight:
            warnings.append("Final mesh is not watertight. May cause slicing issues.")
            trimesh.repair.fill_holes(mesh)
        
        return mesh, warnings
    
    def _export(self, mesh, output_path: str):
        """Export mesh in the configured format."""
        output_path = Path(output_path)
        
        if self.config.output_format == "3mf" or output_path.suffix == ".3mf":
            # 3MF export
            try:
                mesh.export(str(output_path), file_type="3mf")
            except Exception:
                # Fallback to STL if 3MF export fails
                stl_path = output_path.with_suffix(".stl")
                mesh.export(str(stl_path), file_type="stl")
                logger.warning(f"3MF export failed, saved as STL: {stl_path}")
        else:
            mesh.export(str(output_path), file_type="stl")
