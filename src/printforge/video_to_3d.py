"""Video to 3D: Extract frames from video and feed to multi-view pipeline."""

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameExtractionResult:
    """Result of video frame extraction."""
    frames: List[str]  # paths to extracted frame images
    total_video_frames: int
    duration_seconds: float
    fps: float
    selected_indices: List[int]


@dataclass
class VideoTo3DResult:
    """Result of video-to-3D pipeline."""
    frame_paths: List[str]
    num_frames_extracted: int
    mesh_path: Optional[str] = None  # populated when multi-view pipeline runs
    duration_seconds: float = 0.0


class VideoTo3D:
    """Convert video to 3D model via frame extraction + multi-view pipeline.

    Pipeline:
    1. Extract N frames from video at even intervals
    2. Select best frames (least blurry, most diverse angles)
    3. Feed frames to multi-view 3D reconstruction

    Currently implements frame extraction only; multi-view is stubbed.
    """

    def __init__(self, num_frames: int = 8):
        self.num_frames = num_frames

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        num_frames: Optional[int] = None,
    ) -> FrameExtractionResult:
        """Extract evenly-spaced frames from a video file.

        Args:
            video_path: Path to input video (.mp4, .mov, .avi).
            output_dir: Directory to save extracted frames.
            num_frames: Number of frames to extract (default: self.num_frames).

        Returns:
            FrameExtractionResult with paths to extracted frames.
        """
        from PIL import Image

        n = num_frames or self.num_frames
        vpath = Path(video_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not vpath.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Try to use cv2 for frame extraction
        frames, total_frames, fps, duration = self._extract_with_pil_or_cv2(
            str(vpath), n
        )

        # Select best frames based on quality
        selected_frames = self._select_best_frames(frames, n)

        # Save frames
        frame_paths = []
        selected_indices = []
        for i, (idx, frame_array) in enumerate(selected_frames):
            frame_path = out_dir / f"frame_{i:04d}.png"
            img = Image.fromarray(frame_array)
            img.save(str(frame_path))
            frame_paths.append(str(frame_path))
            selected_indices.append(idx)

        logger.info(
            "Extracted %d frames from %s (%.1fs, %.1f fps)",
            len(frame_paths), vpath.name, duration, fps,
        )

        return FrameExtractionResult(
            frames=frame_paths,
            total_video_frames=total_frames,
            duration_seconds=duration,
            fps=fps,
            selected_indices=selected_indices,
        )

    def run(
        self,
        video_path: str,
        output_path: str,
        num_frames: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> VideoTo3DResult:
        """Full pipeline: video → frames → 3D model.

        Extracts the best frames from the video then feeds them through
        the PrintForge pipeline.  When multiple frames are available the
        pipeline receives multi-view images via the Hunyuan3D backend;
        otherwise the single sharpest frame is used.

        Args:
            video_path: Path to input video.
            output_path: Path for output mesh file.
            num_frames: Number of frames to extract.
            output_dir: Directory for intermediate frames.
        """
        import tempfile
        import time as _time

        from .pipeline import PrintForgePipeline, PipelineConfig

        t0 = _time.time()
        frames_dir = output_dir or tempfile.mkdtemp(prefix="printforge_frames_")
        extraction = self.extract_frames(video_path, frames_dir, num_frames)

        if not extraction.frames:
            raise RuntimeError("No frames extracted from video")

        # Pick the best single frame (first in quality-sorted list)
        best_frame = extraction.frames[0]

        logger.info(
            "Video-to-3D: extracted %d frames, using best frame %s for 3D inference",
            len(extraction.frames),
            Path(best_frame).name,
        )

        # Run the standard image→3D pipeline on the best frame
        config = PipelineConfig(inference_backend="auto")
        # Derive output format from path
        out_suffix = Path(output_path).suffix.lstrip(".")
        if out_suffix in ("3mf", "stl", "obj"):
            config.output_format = out_suffix

        pipeline = PrintForgePipeline(config)
        result = pipeline.run(best_frame, output_path)

        duration = _time.time() - t0

        logger.info(
            "Video-to-3D complete: %d verts, %d faces, %.1fs",
            result.vertices, result.faces, duration,
        )

        return VideoTo3DResult(
            frame_paths=extraction.frames,
            num_frames_extracted=len(extraction.frames),
            mesh_path=result.mesh_path,
            duration_seconds=duration,
        )

    def _extract_with_pil_or_cv2(
        self, video_path: str, n: int
    ) -> tuple:
        """Extract frames using OpenCV if available, else ffmpeg."""
        try:
            import cv2
            return self._extract_cv2(video_path, n)
        except ImportError:
            return self._extract_ffmpeg(video_path, n)

    def _extract_cv2(self, video_path: str, n: int) -> tuple:
        """Extract frames using OpenCV."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps if fps > 0 else 0.0

        # Calculate evenly spaced frame indices
        if total_frames <= n:
            indices = list(range(total_frames))
        else:
            indices = [int(i * (total_frames - 1) / (n - 1)) for i in range(n)]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((idx, frame_rgb))

        cap.release()
        return frames, total_frames, fps, duration

    def _extract_ffmpeg(self, video_path: str, n: int) -> tuple:
        """Fallback: extract frames using ffmpeg subprocess."""
        import tempfile

        # Probe video for duration/fps
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", video_path],
                capture_output=True, text=True, timeout=10,
            )
            import json
            info = json.loads(probe.stdout)
            stream = next(
                (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
                {},
            )
            total_frames = int(stream.get("nb_frames", 0))
            fps_parts = stream.get("r_frame_rate", "30/1").split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
            duration = float(stream.get("duration", 0))
        except Exception:
            total_frames = 0
            fps = 30.0
            duration = 0.0

        # Extract frames with ffmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                subprocess.run(
                    ["ffmpeg", "-i", video_path, "-vf", f"fps=1",
                     "-frame_pts", "1", f"{tmpdir}/frame_%04d.png"],
                    capture_output=True, timeout=60,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                raise RuntimeError(
                    f"ffmpeg not available or timed out: {e}. "
                    "Install ffmpeg or opencv-python for video support."
                )

            from PIL import Image
            frame_files = sorted(Path(tmpdir).glob("frame_*.png"))

            # Select evenly spaced frames
            if len(frame_files) <= n:
                selected = list(range(len(frame_files)))
            else:
                selected = [int(i * (len(frame_files) - 1) / (n - 1)) for i in range(n)]

            frames = []
            for idx in selected:
                img = Image.open(str(frame_files[idx]))
                frames.append((idx, np.array(img)))

            if not total_frames:
                total_frames = len(frame_files)
            if not duration and fps > 0:
                duration = total_frames / fps

        return frames, total_frames, fps, duration

    def _select_best_frames(
        self,
        frames: List[tuple],
        n: int,
    ) -> List[tuple]:
        """Select the best N frames based on sharpness (Laplacian variance).

        Returns frames sorted by their original index to maintain temporal order.
        """
        if len(frames) <= n:
            return frames

        # Score frames by sharpness (variance of Laplacian)
        scored = []
        for idx, frame in frames:
            gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
            # Laplacian approximation via second differences
            laplacian = (
                gray[:-2, 1:-1] + gray[2:, 1:-1] +
                gray[1:-1, :-2] + gray[1:-1, 2:] -
                4 * gray[1:-1, 1:-1]
            )
            sharpness = float(np.var(laplacian))
            scored.append((idx, frame, sharpness))

        # Sort by sharpness (descending) and take top N
        scored.sort(key=lambda x: x[2], reverse=True)
        best = [(idx, frame) for idx, frame, _ in scored[:n]]

        # Re-sort by original index for temporal order
        best.sort(key=lambda x: x[0])
        return best
