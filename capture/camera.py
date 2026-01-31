"""Camera abstraction for video capture."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraFrame:
    """Container for a captured frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_number: int


class Camera:
    """Abstraction for a video capture device."""

    def __init__(
        self,
        index: int,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        name: str = "Camera"
    ):
        """
        Initialize a camera.

        Args:
            index: Camera device index
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
            name: Human-readable name for this camera
        """
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0

    def open(self) -> bool:
        """
        Open the camera for capture.

        Returns:
            True if camera opened successfully
        """
        self._cap = cv2.VideoCapture(self.index)

        if not self._cap.isOpened():
            return False

        # Set camera properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Verify settings (cameras may not support requested resolution)
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        if actual_width != self.width or actual_height != self.height:
            print(f"{self.name}: Requested {self.width}x{self.height}, "
                  f"got {actual_width}x{actual_height}")
            self.width = actual_width
            self.height = actual_height

        print(f"{self.name}: Opened at {self.width}x{self.height} @ {actual_fps:.1f} fps")
        return True

    def read(self) -> Optional[CameraFrame]:
        """
        Read a frame from the camera.

        Returns:
            CameraFrame if successful, None otherwise
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        self._frame_count += 1

        return CameraFrame(
            image=frame,
            timestamp=timestamp,
            frame_number=self._frame_count
        )

    def release(self) -> None:
        """Release the camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame_count = 0

    def is_opened(self) -> bool:
        """Check if camera is open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        """Get the number of frames captured."""
        return self._frame_count

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
