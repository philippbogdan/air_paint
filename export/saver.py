"""Background session saver with progress reporting."""

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np

from drawing.stroke import Stroke
from config.settings import get_output_dir


class SaveStage(Enum):
    """Stages of the save process."""
    PREPARING = "Preparing..."
    GENERATING_MESH = "Generating mesh"
    SAVING_JSON = "Saving JSON"
    SAVING_STL = "Saving STL"
    SAVING_OBJ = "Saving OBJ"
    CONVERTING_USDZ = "Converting to USDZ"
    SAVING_SCREENSHOT = "Saving screenshot"
    COMPLETE = "Complete!"
    FAILED = "Failed"


@dataclass
class SaveProgress:
    """Current save progress."""
    stage: SaveStage
    progress: float  # 0.0 to 1.0
    message: str
    is_complete: bool = False
    is_failed: bool = False
    error: Optional[str] = None
    output_dir: Optional[Path] = None


class SessionSaver:
    """Handles session saving in background with progress updates."""

    # Weight of each stage in total progress
    STAGE_WEIGHTS = {
        SaveStage.PREPARING: 0.05,
        SaveStage.GENERATING_MESH: 0.40,  # Most expensive
        SaveStage.SAVING_JSON: 0.05,
        SaveStage.SAVING_STL: 0.15,
        SaveStage.SAVING_OBJ: 0.10,
        SaveStage.CONVERTING_USDZ: 0.20,
        SaveStage.SAVING_SCREENSHOT: 0.05,
    }

    def __init__(self):
        self._progress = SaveProgress(
            stage=SaveStage.PREPARING,
            progress=0.0,
            message="Ready"
        )
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._is_saving = False

    @property
    def is_saving(self) -> bool:
        """Check if save is in progress."""
        return self._is_saving

    @property
    def progress(self) -> SaveProgress:
        """Get current progress (thread-safe)."""
        with self._lock:
            return SaveProgress(
                stage=self._progress.stage,
                progress=self._progress.progress,
                message=self._progress.message,
                is_complete=self._progress.is_complete,
                is_failed=self._progress.is_failed,
                error=self._progress.error,
                output_dir=self._progress.output_dir
            )

    def _update_progress(
        self,
        stage: SaveStage,
        stage_progress: float = 0.0,
        message: Optional[str] = None
    ):
        """Update progress (internal)."""
        # Calculate cumulative progress
        cumulative = 0.0
        for s, weight in self.STAGE_WEIGHTS.items():
            if s == stage:
                cumulative += weight * stage_progress
                break
            cumulative += weight

        with self._lock:
            self._progress.stage = stage
            self._progress.progress = min(cumulative, 1.0)
            self._progress.message = message or stage.value

    def save_async(
        self,
        strokes: List[Stroke],
        stats_dict: dict,
        screenshot: Optional[np.ndarray] = None,
        session_start: Optional[datetime] = None,
        coordinate_frame: str = "camera_a",
        marker_metadata: Optional[dict] = None
    ) -> bool:
        """
        Start saving session in background.

        Returns:
            True if save started, False if already saving
        """
        if self._is_saving:
            return False

        self._is_saving = True
        with self._lock:
            self._progress = SaveProgress(
                stage=SaveStage.PREPARING,
                progress=0.0,
                message="Starting save..."
            )

        self._thread = threading.Thread(
            target=self._save_worker,
            args=(strokes, stats_dict, screenshot, session_start,
                  coordinate_frame, marker_metadata),
            daemon=True
        )
        self._thread.start()
        return True

    def _save_worker(
        self,
        strokes: List[Stroke],
        stats_dict: dict,
        screenshot: Optional[np.ndarray],
        session_start: Optional[datetime],
        coordinate_frame: str,
        marker_metadata: Optional[dict]
    ):
        """Background save worker."""
        try:
            # Create session directory
            self._update_progress(SaveStage.PREPARING, 0.5, "Creating directory...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = get_output_dir() / f"session_{timestamp}"
            session_dir.mkdir(parents=True, exist_ok=True)

            self._update_progress(SaveStage.PREPARING, 1.0)

            # Generate mesh
            self._update_progress(SaveStage.GENERATING_MESH, 0.0, "Generating mesh (0%)")
            vertices, faces = self._generate_mesh_with_progress(strokes)
            self._update_progress(SaveStage.GENERATING_MESH, 1.0, "Mesh complete")

            # Save JSON
            self._update_progress(SaveStage.SAVING_JSON, 0.0)
            self._save_json(
                session_dir, strokes, stats_dict, session_start,
                coordinate_frame, marker_metadata
            )
            self._update_progress(SaveStage.SAVING_JSON, 1.0)

            # Save STL
            if len(vertices) > 0:
                self._update_progress(SaveStage.SAVING_STL, 0.0)
                self._save_stl(vertices, faces, session_dir / "drawing.stl")
                self._update_progress(SaveStage.SAVING_STL, 1.0)

                # Save OBJ
                self._update_progress(SaveStage.SAVING_OBJ, 0.0)
                self._save_obj(vertices, faces, session_dir / "drawing.obj")
                self._update_progress(SaveStage.SAVING_OBJ, 1.0)

                # Convert to USDZ
                self._update_progress(SaveStage.CONVERTING_USDZ, 0.0, "Converting to USDZ...")
                self._convert_usdz(session_dir / "drawing.obj")
                self._update_progress(SaveStage.CONVERTING_USDZ, 1.0)
            else:
                # Skip mesh stages
                self._update_progress(SaveStage.SAVING_STL, 1.0, "No mesh to save")
                self._update_progress(SaveStage.SAVING_OBJ, 1.0)
                self._update_progress(SaveStage.CONVERTING_USDZ, 1.0)

            # Save screenshot
            if screenshot is not None:
                self._update_progress(SaveStage.SAVING_SCREENSHOT, 0.0)
                import cv2
                cv2.imwrite(str(session_dir / "screenshot.png"), screenshot)
            self._update_progress(SaveStage.SAVING_SCREENSHOT, 1.0)

            # Done!
            with self._lock:
                self._progress.stage = SaveStage.COMPLETE
                self._progress.progress = 1.0
                self._progress.message = f"Saved to {session_dir.name}"
                self._progress.is_complete = True
                self._progress.output_dir = session_dir

        except Exception as e:
            with self._lock:
                self._progress.stage = SaveStage.FAILED
                self._progress.message = f"Error: {str(e)[:50]}"
                self._progress.is_failed = True
                self._progress.error = str(e)

        finally:
            self._is_saving = False

    def _generate_mesh_with_progress(
        self,
        strokes: List[Stroke]
    ) -> tuple:
        """Generate mesh with progress updates."""
        from export.mesh import generate_tube_vertices
        from config.settings import EXPORT

        all_vertices = []
        all_faces = []
        vertex_offset = 0

        valid_strokes = [s for s in strokes if not s.is_empty and s.num_points >= 2]
        total = len(valid_strokes)

        for i, stroke in enumerate(valid_strokes):
            # Update progress
            pct = (i / total) if total > 0 else 0
            self._update_progress(
                SaveStage.GENERATING_MESH,
                pct,
                f"Generating mesh ({int(pct * 100)}%)"
            )

            points = stroke.to_array()
            vertices, faces = generate_tube_vertices(
                points,
                radius=EXPORT.TUBE_RADIUS_MM,
                segments=EXPORT.TUBE_SEGMENTS
            )

            if len(vertices) > 0:
                all_vertices.append(vertices)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(vertices)

        if not all_vertices:
            return np.empty((0, 3)), np.empty((0, 3), dtype=int)

        return np.vstack(all_vertices), np.vstack(all_faces)

    def _save_json(
        self,
        session_dir: Path,
        strokes: List[Stroke],
        stats_dict: dict,
        session_start: Optional[datetime],
        coordinate_frame: str,
        marker_metadata: Optional[dict]
    ):
        """Save strokes and stats JSON."""
        # Strokes JSON
        frames = set(s.coordinate_frame for s in strokes) if strokes else {"camera_a"}
        primary_frame = "world" if "world" in frames else "camera_a"

        data = {
            "session_start": session_start.isoformat() if session_start else None,
            "export_time": datetime.now().isoformat(),
            "coordinate_frame": primary_frame,
            "marker_metadata": marker_metadata,
            "stats": stats_dict,
            "strokes": [s.to_dict() for s in strokes]
        }

        with open(session_dir / "strokes.json", 'w') as f:
            json.dump(data, f, indent=2)

        # Stats JSON
        with open(session_dir / "stats.json", 'w') as f:
            json.dump(stats_dict, f, indent=2)

    def _save_stl(self, vertices: np.ndarray, faces: np.ndarray, path: Path):
        """Save STL file."""
        from export.mesh import save_stl
        save_stl(vertices, faces, path)

    def _save_obj(self, vertices: np.ndarray, faces: np.ndarray, path: Path):
        """Save OBJ file."""
        from export.mesh import save_obj
        save_obj(vertices, faces, path)

    def _convert_usdz(self, obj_path: Path):
        """Convert to USDZ."""
        from export.usdz import convert_to_usdz
        convert_to_usdz(obj_path)

    def reset(self):
        """Reset progress state after completion acknowledged."""
        with self._lock:
            self._progress = SaveProgress(
                stage=SaveStage.PREPARING,
                progress=0.0,
                message="Ready"
            )
