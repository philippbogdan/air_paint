"""3D triangulation from stereo camera pair."""

import cv2
import numpy as np
from typing import Optional, Tuple
from calibration.stereo import StereoCalibration
from config.settings import TRACKING


class PointSmoother:
    """Exponential moving average for 3D point stabilization."""

    def __init__(self, alpha: float = TRACKING.SMOOTHING_ALPHA):
        """
        Initialize the smoother.

        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother.
        """
        self.alpha = alpha
        self._last_point: Optional[np.ndarray] = None

    def smooth(self, point: np.ndarray) -> np.ndarray:
        """
        Apply exponential moving average smoothing to a 3D point.

        Args:
            point: 3D point [X, Y, Z]

        Returns:
            Smoothed 3D point
        """
        if self._last_point is None:
            self._last_point = point.copy()
            return point
        smoothed = self.alpha * point + (1 - self.alpha) * self._last_point
        self._last_point = smoothed.copy()
        return smoothed

    def reset(self):
        """Reset the smoother state (call when drawing stops)."""
        self._last_point = None


class StereoTriangulator:
    """
    Performs 3D triangulation from stereo camera observations.

    Uses cv2.triangulatePoints with calibrated projection matrices.
    """

    def __init__(self, calibration: StereoCalibration):
        """
        Initialize the triangulator.

        Args:
            calibration: StereoCalibration containing projection matrices
        """
        self.calibration = calibration

    def triangulate(
        self,
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
        undistort: bool = True,
        max_reprojection_error: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Triangulate a 3D point from two 2D observations.

        Args:
            point_a: (x, y) point in camera A image
            point_b: (x, y) point in camera B image
            undistort: Whether to undistort points before triangulation
            max_reprojection_error: Maximum allowed reprojection error in pixels.
                                    If None, uses TRACKING.MAX_REPROJECTION_ERROR.

        Returns:
            3D point as numpy array [X, Y, Z] in mm (camera A coordinate system),
            or None if triangulation fails or reprojection error too high
        """
        if max_reprojection_error is None:
            max_reprojection_error = TRACKING.MAX_REPROJECTION_ERROR

        # Convert to numpy arrays with correct shape for OpenCV
        pt_a = np.array([[point_a]], dtype=np.float32)
        pt_b = np.array([[point_b]], dtype=np.float32)

        if undistort:
            # Undistort points to remove lens distortion
            pt_a = cv2.undistortPoints(
                pt_a,
                self.calibration.K1,
                self.calibration.D1,
                P=self.calibration.K1
            )
            pt_b = cv2.undistortPoints(
                pt_b,
                self.calibration.K2,
                self.calibration.D2,
                P=self.calibration.K2
            )

        # Reshape for triangulatePoints: expects (2, N)
        pt_a = pt_a.reshape(2, 1)
        pt_b = pt_b.reshape(2, 1)

        # Triangulate
        points_4d = cv2.triangulatePoints(
            self.calibration.P1,
            self.calibration.P2,
            pt_a,
            pt_b
        )

        # Convert from homogeneous to 3D coordinates
        point_3d = points_4d[:3, 0] / points_4d[3, 0]

        # Sanity check - point should be in front of cameras
        if point_3d[2] <= 0:
            return None

        # Outlier rejection via reprojection error
        error_a, error_b = self.compute_reprojection_error(point_a, point_b, point_3d)
        if error_a > max_reprojection_error or error_b > max_reprojection_error:
            return None  # Reject outlier

        return point_3d

    def triangulate_multiple(
        self,
        points_a: np.ndarray,
        points_b: np.ndarray,
        undistort: bool = True
    ) -> np.ndarray:
        """
        Triangulate multiple 3D points.

        Args:
            points_a: (N, 2) array of points in camera A
            points_b: (N, 2) array of points in camera B
            undistort: Whether to undistort points

        Returns:
            (N, 3) array of 3D points
        """
        assert points_a.shape == points_b.shape
        assert points_a.shape[1] == 2

        n_points = points_a.shape[0]
        results = np.zeros((n_points, 3))

        for i in range(n_points):
            pt_3d = self.triangulate(
                tuple(points_a[i]),
                tuple(points_b[i]),
                undistort
            )
            if pt_3d is not None:
                results[i] = pt_3d

        return results

    def reproject_to_camera_a(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """
        Reproject a 3D point back to camera A image coordinates.

        Args:
            point_3d: 3D point [X, Y, Z]

        Returns:
            (x, y) in camera A image coordinates
        """
        # Project using P1
        point_h = np.append(point_3d, 1)  # Homogeneous
        projected = self.calibration.P1 @ point_h
        x = projected[0] / projected[2]
        y = projected[1] / projected[2]
        return (x, y)

    def reproject_to_camera_b(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """
        Reproject a 3D point back to camera B image coordinates.

        Args:
            point_3d: 3D point [X, Y, Z]

        Returns:
            (x, y) in camera B image coordinates
        """
        # Project using P2
        point_h = np.append(point_3d, 1)  # Homogeneous
        projected = self.calibration.P2 @ point_h
        x = projected[0] / projected[2]
        y = projected[1] / projected[2]
        return (x, y)

    def compute_reprojection_error(
        self,
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
        point_3d: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute reprojection error for a triangulated point.

        Args:
            point_a: Original 2D point in camera A
            point_b: Original 2D point in camera B
            point_3d: Triangulated 3D point

        Returns:
            (error_a, error_b) reprojection errors in pixels
        """
        reproj_a = self.reproject_to_camera_a(point_3d)
        reproj_b = self.reproject_to_camera_b(point_3d)

        error_a = np.sqrt(
            (point_a[0] - reproj_a[0])**2 +
            (point_a[1] - reproj_a[1])**2
        )
        error_b = np.sqrt(
            (point_b[0] - reproj_b[0])**2 +
            (point_b[1] - reproj_b[1])**2
        )

        return (error_a, error_b)
