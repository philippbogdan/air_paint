"""Stroke data structures for 3D drawing."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class Point3D:
    """A single 3D point with timestamp and coordinate frame."""
    x: float
    y: float
    z: float
    timestamp: float = field(default_factory=time.time)
    frame: str = "camera_a"  # "camera_a" or "world"

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z])

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "timestamp": self.timestamp,
            "frame": self.frame
        }

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        timestamp: Optional[float] = None,
        frame: str = "camera_a"
    ) -> "Point3D":
        """Create from numpy array."""
        return cls(
            x=float(arr[0]),
            y=float(arr[1]),
            z=float(arr[2]),
            timestamp=timestamp if timestamp is not None else time.time(),
            frame=frame
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Point3D":
        """Create from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            timestamp=data.get("timestamp", time.time()),
            frame=data.get("frame", "camera_a")
        )

    def distance_to(self, other: "Point3D") -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )


@dataclass
class Stroke:
    """A single stroke consisting of multiple 3D points."""
    points: List[Point3D] = field(default_factory=list)
    color: tuple = (0, 255, 0)  # BGR
    thickness: float = 2.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    coordinate_frame: str = "camera_a"  # "camera_a" or "world"

    def add_point(self, point: Point3D) -> None:
        """Add a point to the stroke."""
        self.points.append(point)

    def finalize(self) -> None:
        """Mark the stroke as complete."""
        self.end_time = time.time()

    @property
    def is_empty(self) -> bool:
        """Check if stroke has no points."""
        return len(self.points) == 0

    @property
    def num_points(self) -> int:
        """Get number of points in stroke."""
        return len(self.points)

    @property
    def duration(self) -> float:
        """Get stroke duration in seconds."""
        if self.is_empty:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def length(self) -> float:
        """Calculate total length of the stroke."""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.points)):
            total += self.points[i-1].distance_to(self.points[i])
        return total

    def to_array(self) -> np.ndarray:
        """Convert all points to numpy array of shape (N, 3)."""
        if self.is_empty:
            return np.empty((0, 3))
        return np.array([[p.x, p.y, p.z] for p in self.points])

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "points": [p.to_dict() for p in self.points],
            "color": list(self.color),
            "thickness": self.thickness,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_points": self.num_points,
            "length_mm": self.length,
            "duration_s": self.duration,
            "coordinate_frame": self.coordinate_frame
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Stroke":
        """Create from dictionary."""
        stroke = cls(
            color=tuple(data.get("color", (0, 255, 0))),
            thickness=data.get("thickness", 2.0),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time"),
            coordinate_frame=data.get("coordinate_frame", "camera_a")
        )
        for point_data in data.get("points", []):
            stroke.add_point(Point3D.from_dict(point_data))
        return stroke

    def get_bounding_box(self) -> tuple:
        """
        Get bounding box of the stroke.

        Returns:
            ((min_x, min_y, min_z), (max_x, max_y, max_z))
        """
        if self.is_empty:
            return ((0, 0, 0), (0, 0, 0))

        arr = self.to_array()
        min_pt = arr.min(axis=0)
        max_pt = arr.max(axis=0)
        return (tuple(min_pt), tuple(max_pt))

    @property
    def last_point(self) -> Optional[Point3D]:
        """Get the last point in the stroke."""
        if self.is_empty:
            return None
        return self.points[-1]
