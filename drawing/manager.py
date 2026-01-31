"""Drawing state management."""

import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from .stroke import Point3D, Stroke
from config.settings import DRAWING, get_output_dir


@dataclass
class DrawingStats:
    """Statistics about a drawing session."""
    total_strokes: int
    total_points: int
    total_length_mm: float
    session_duration_s: float
    bounding_box: tuple

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "total_strokes": self.total_strokes,
            "total_points": self.total_points,
            "total_length_mm": self.total_length_mm,
            "session_duration_s": self.session_duration_s,
            "bounding_box": {
                "min": list(self.bounding_box[0]),
                "max": list(self.bounding_box[1])
            }
        }


class DrawingManager:
    """
    Manages drawing state including strokes and toggle logic.

    Handles:
    - Starting/stopping strokes with toggle
    - Adding points to current stroke
    - Managing stroke history
    - Exporting drawing data
    """

    def __init__(
        self,
        min_point_distance: float = DRAWING.MIN_POINT_DISTANCE_MM,
        stroke_color: tuple = DRAWING.STROKE_COLOR,
        stroke_thickness: float = DRAWING.STROKE_THICKNESS
    ):
        """
        Initialize the drawing manager.

        Args:
            min_point_distance: Minimum distance between consecutive points in mm
            stroke_color: Default stroke color (BGR)
            stroke_thickness: Default stroke thickness
        """
        self.min_point_distance = min_point_distance
        self.stroke_color = stroke_color
        self.stroke_thickness = stroke_thickness

        self._strokes: List[Stroke] = []
        self._current_stroke: Optional[Stroke] = None
        self._session_start = datetime.now()

        # World coordinate frame support
        self._world_transform: Optional[np.ndarray] = None  # 4x4 transform matrix
        self._coordinate_frame: str = "camera_a"
        self._marker_metadata: Optional[dict] = None

    @property
    def is_drawing(self) -> bool:
        """Check if currently drawing."""
        return self._current_stroke is not None

    @property
    def strokes(self) -> List[Stroke]:
        """Get all completed strokes."""
        return self._strokes

    @property
    def current_stroke(self) -> Optional[Stroke]:
        """Get the current active stroke."""
        return self._current_stroke

    @property
    def all_strokes(self) -> List[Stroke]:
        """Get all strokes including current one."""
        if self._current_stroke is not None:
            return self._strokes + [self._current_stroke]
        return self._strokes

    @property
    def coordinate_frame(self) -> str:
        """Get current coordinate frame ('camera_a' or 'world')."""
        return self._coordinate_frame

    @property
    def has_world_transform(self) -> bool:
        """Check if world transform is currently active."""
        return self._world_transform is not None

    def set_world_transform(
        self,
        transform: np.ndarray,
        marker_id: Optional[int] = None,
        marker_size_mm: Optional[float] = None
    ) -> None:
        """
        Set the world transform for converting camera coordinates to world coordinates.

        Args:
            transform: 4x4 transformation matrix (camera → world)
            marker_id: Optional marker ID for metadata
            marker_size_mm: Optional marker size for metadata
        """
        self._world_transform = transform
        self._coordinate_frame = "world"
        if marker_id is not None or marker_size_mm is not None:
            self._marker_metadata = {
                "marker_id": marker_id,
                "marker_size_mm": marker_size_mm
            }

    def clear_world_transform(self) -> None:
        """Clear world transform and revert to camera coordinates."""
        self._world_transform = None
        self._coordinate_frame = "camera_a"
        # Keep marker_metadata for reference

    def _transform_to_world(self, point_camera: np.ndarray) -> np.ndarray:
        """
        Transform a point from camera coordinates to world coordinates.

        Args:
            point_camera: 3D point in camera frame [X, Y, Z]

        Returns:
            3D point in world frame [X, Y, Z]
        """
        if self._world_transform is None:
            return point_camera

        # Convert to homogeneous coordinates
        point_h = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])

        # Transform
        point_world_h = self._world_transform @ point_h

        return point_world_h[:3]

    def toggle_drawing(self) -> bool:
        """
        Toggle drawing state.

        Returns:
            New drawing state (True if now drawing)
        """
        if self.is_drawing:
            self.stop_drawing()
        else:
            self.start_drawing()
        return self.is_drawing

    def start_drawing(self) -> None:
        """Start a new stroke."""
        if self.is_drawing:
            return

        self._current_stroke = Stroke(
            color=self.stroke_color,
            thickness=self.stroke_thickness,
            coordinate_frame=self._coordinate_frame
        )

    def stop_drawing(self) -> Optional[Stroke]:
        """
        Stop drawing and finalize current stroke.

        Returns:
            The completed stroke, or None if not drawing
        """
        if not self.is_drawing:
            return None

        stroke = self._current_stroke
        stroke.finalize()

        # Only keep non-empty strokes
        if not stroke.is_empty:
            self._strokes.append(stroke)

        self._current_stroke = None
        return stroke

    def add_point(self, point_3d: np.ndarray) -> bool:
        """
        Add a 3D point to the current stroke.

        Points are automatically transformed to world coordinates if world transform is active.

        Args:
            point_3d: 3D point as numpy array [X, Y, Z] in camera coordinates

        Returns:
            True if point was added, False if skipped (not drawing or too close)
        """
        if not self.is_drawing:
            return False

        # Transform to world coordinates if available
        if self._world_transform is not None:
            point_3d = self._transform_to_world(point_3d)
            frame = "world"
        else:
            frame = "camera_a"

        point = Point3D.from_array(point_3d, frame=frame)

        # Check minimum distance from last point
        if self._current_stroke.last_point is not None:
            distance = point.distance_to(self._current_stroke.last_point)
            if distance < self.min_point_distance:
                return False

        self._current_stroke.add_point(point)
        return True

    def clear(self) -> None:
        """Clear all strokes."""
        self._strokes = []
        self._current_stroke = None

    def undo(self) -> Optional[Stroke]:
        """
        Remove the last stroke.

        Returns:
            The removed stroke, or None if no strokes
        """
        # If currently drawing, cancel current stroke
        if self.is_drawing:
            stroke = self._current_stroke
            self._current_stroke = None
            return stroke

        # Otherwise remove last completed stroke
        if self._strokes:
            return self._strokes.pop()

        return None

    def get_stats(self) -> DrawingStats:
        """Get statistics about the drawing."""
        all_strokes = self.all_strokes

        total_points = sum(s.num_points for s in all_strokes)
        total_length = sum(s.length for s in all_strokes)

        # Calculate bounding box
        if not all_strokes or all(s.is_empty for s in all_strokes):
            bbox = ((0, 0, 0), (0, 0, 0))
        else:
            all_points = np.vstack([s.to_array() for s in all_strokes if not s.is_empty])
            min_pt = tuple(all_points.min(axis=0))
            max_pt = tuple(all_points.max(axis=0))
            bbox = (min_pt, max_pt)

        return DrawingStats(
            total_strokes=len(all_strokes),
            total_points=total_points,
            total_length_mm=total_length,
            session_duration_s=(datetime.now() - self._session_start).total_seconds(),
            bounding_box=bbox
        )

    def get_all_points_array(self) -> np.ndarray:
        """
        Get all points from all strokes as a single array.

        Returns:
            (N, 3) array of all points
        """
        all_strokes = self.all_strokes
        if not all_strokes:
            return np.empty((0, 3))

        arrays = [s.to_array() for s in all_strokes if not s.is_empty]
        if not arrays:
            return np.empty((0, 3))

        return np.vstack(arrays)

    def export_json(self, path: Path) -> None:
        """
        Export strokes to JSON file.

        Args:
            path: Path to save JSON file
        """
        # Determine primary coordinate frame from strokes
        all_strokes = self.all_strokes
        frames = set(s.coordinate_frame for s in all_strokes) if all_strokes else {"camera_a"}
        primary_frame = "world" if "world" in frames else "camera_a"

        data = {
            "session_start": self._session_start.isoformat(),
            "export_time": datetime.now().isoformat(),
            "coordinate_frame": primary_frame,
            "marker_metadata": self._marker_metadata,
            "stats": self.get_stats().to_dict(),
            "strokes": [s.to_dict() for s in all_strokes]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_session(self, screenshot: Optional[np.ndarray] = None) -> Path:
        """
        Save the current session to the output directory.

        Args:
            screenshot: Optional screenshot to save

        Returns:
            Path to the session directory
        """
        import cv2

        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = get_output_dir() / f"session_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save strokes JSON
        self.export_json(session_dir / "strokes.json")

        # Save stats JSON
        stats = self.get_stats()
        with open(session_dir / "stats.json", 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)

        # Save screenshot if provided
        if screenshot is not None:
            cv2.imwrite(str(session_dir / "screenshot.png"), screenshot)

        print(f"Session saved to {session_dir}")
        return session_dir

    @classmethod
    def load_session(cls, path: Path) -> "DrawingManager":
        """
        Load a session from JSON file.

        Args:
            path: Path to strokes.json file

        Returns:
            DrawingManager with loaded strokes
        """
        with open(path, 'r') as f:
            data = json.load(f)

        manager = cls()
        for stroke_data in data.get("strokes", []):
            stroke = Stroke.from_dict(stroke_data)
            if not stroke.is_empty:
                manager._strokes.append(stroke)

        return manager
