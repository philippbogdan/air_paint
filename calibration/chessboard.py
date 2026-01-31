"""Chessboard detection utilities for camera calibration."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from config.settings import CALIBRATION


@dataclass
class ChessboardCorners:
    """Container for detected chessboard corners."""
    corners: np.ndarray  # Shape: (N, 1, 2) where N = rows * cols
    image_size: Tuple[int, int]  # (width, height)

    @property
    def num_corners(self) -> int:
        """Get the number of detected corners."""
        return len(self.corners)


class ChessboardDetector:
    """Detects chessboard patterns for camera calibration."""

    def __init__(
        self,
        board_size: Tuple[int, int] = CALIBRATION.CHESSBOARD_SIZE,
        square_size: float = CALIBRATION.SQUARE_SIZE_MM
    ):
        """
        Initialize the chessboard detector.

        Args:
            board_size: Number of inner corners (columns, rows)
            square_size: Size of each square in mm
        """
        self.board_size = board_size
        self.square_size = square_size

        # Prepare object points (3D points in real world space)
        # These are the same for all calibration images
        self.object_points = np.zeros(
            (board_size[0] * board_size[1], 3),
            dtype=np.float32
        )
        self.object_points[:, :2] = np.mgrid[
            0:board_size[0],
            0:board_size[1]
        ].T.reshape(-1, 2) * square_size

        # Subpixel refinement criteria
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,  # Max iterations
            0.001  # Epsilon
        )

    def detect(self, image: np.ndarray) -> Optional[ChessboardCorners]:
        """
        Detect chessboard corners in an image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            ChessboardCorners if found, None otherwise
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find chessboard corners
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK
        )
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, flags)

        if not ret:
            return None

        # Refine corner positions to subpixel accuracy
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), self.criteria
        )

        return ChessboardCorners(
            corners=corners_refined,
            image_size=(gray.shape[1], gray.shape[0])
        )

    def draw_corners(
        self,
        image: np.ndarray,
        corners: ChessboardCorners,
        found: bool = True
    ) -> np.ndarray:
        """
        Draw detected corners on an image.

        Args:
            image: Input image (will be copied)
            corners: Detected corners
            found: Whether detection was successful (affects color)

        Returns:
            Image with corners drawn
        """
        result = image.copy()
        cv2.drawChessboardCorners(
            result, self.board_size, corners.corners, found
        )
        return result

    def get_object_points(self) -> np.ndarray:
        """Get the 3D object points for calibration."""
        return self.object_points.copy()


def collect_calibration_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    detector: ChessboardDetector
) -> Optional[Tuple[ChessboardCorners, ChessboardCorners]]:
    """
    Detect chessboard in both stereo frames.

    Args:
        frame_a: Frame from camera A
        frame_b: Frame from camera B
        detector: ChessboardDetector instance

    Returns:
        Tuple of (corners_a, corners_b) if found in both, None otherwise
    """
    corners_a = detector.detect(frame_a)
    corners_b = detector.detect(frame_b)

    if corners_a is None or corners_b is None:
        return None

    return corners_a, corners_b
