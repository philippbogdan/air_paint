"""Dynamic stereo triangulation with ArUco-based pose estimation.

This module provides triangulation for a moving camera (Meta glasses) relative to
a fixed camera (Mac webcam) using ArUco markers for dynamic pose estimation.

Two-Phase Calibration:
1. CALIBRATION PHASE: Both cameras see the marker. Webcam's position relative to
   the marker is cached.
2. OPERATION PHASE: Only glasses needs to see the marker. Webcam uses cached position.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from config.settings import TRACKING
from tracking.aruco import MarkerPose


class DynamicStereoTriangulator:
    """
    Performs 3D triangulation with dynamic stereo geometry.

    Unlike StereoTriangulator which assumes fixed camera positions,
    this class handles a moving camera (glasses) by using ArUco marker
    pose estimation to compute the relative transform on each frame.

    The calibration is done in two phases:
    - Phase 1 (calibration): Both cameras see the marker, webcam pose is cached
    - Phase 2 (operation): Only glasses sees marker, webcam uses cached pose
    """

    def __init__(
        self,
        K1: np.ndarray,
        D1: np.ndarray,
        K2: np.ndarray,
        D2: np.ndarray
    ):
        """
        Initialize the dynamic triangulator with camera intrinsics only.

        Args:
            K1: 3x3 intrinsic matrix for glasses (Camera A)
            D1: Distortion coefficients for glasses
            K2: 3x3 intrinsic matrix for webcam (Camera B)
            D2: Distortion coefficients for webcam
        """
        self.K1 = K1.astype(np.float64)
        self.K2 = K2.astype(np.float64)
        self.D1 = D1.astype(np.float64) if D1 is not None else np.zeros(5)
        self.D2 = D2.astype(np.float64) if D2 is not None else np.zeros(5)

        # Glasses projection matrix (at origin)
        # P1 = K1 @ [I | 0]
        self.P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])

        # Webcam projection matrix (computed dynamically)
        self.P2: Optional[np.ndarray] = None

        # Cached webcam pose from calibration (T_marker_to_webcam)
        self.webcam_pose_cached: Optional[np.ndarray] = None
        self.is_calibrated = False

        # Current relative transform
        self.current_R: Optional[np.ndarray] = None
        self.current_T: Optional[np.ndarray] = None

    def calibrate_webcam(self, webcam_marker_pose: MarkerPose) -> None:
        """
        PHASE 1: Cache the webcam's position relative to the marker.

        Call this when BOTH cameras can see the marker. The webcam's pose
        is cached and will be reused during the operation phase.

        Args:
            webcam_marker_pose: MarkerPose from webcam's ArUco detection
        """
        # Store T_WC (marker → webcam transform)
        # This is the webcam's pose expressed in marker coordinates
        self.webcam_pose_cached = webcam_marker_pose.T_WC.copy()
        self.is_calibrated = True
        print(f"[DynamicTriangulator] Webcam calibrated! "
              f"Distance to marker: {webcam_marker_pose.distance_mm:.0f}mm")

    def update_from_glasses_pose(self, glasses_marker_pose: MarkerPose) -> bool:
        """
        PHASE 2: Update the relative transform using only glasses marker detection.

        Call this every frame with the glasses' marker pose. Uses the cached
        webcam pose to compute the glasses→webcam transform for triangulation.

        Args:
            glasses_marker_pose: MarkerPose from glasses' ArUco detection

        Returns:
            True if update successful, False if not calibrated
        """
        if not self.is_calibrated or self.webcam_pose_cached is None:
            return False

        # Glasses sees marker → we have T_marker_to_glasses (T_WC from glasses perspective)
        T_marker_to_glasses = glasses_marker_pose.T_WC

        # Webcam's cached pose: T_marker_to_webcam
        T_marker_to_webcam = self.webcam_pose_cached

        # Compute glasses→webcam transform:
        # T_glasses_to_webcam = T_marker_to_webcam @ T_glasses_to_marker
        #                     = T_marker_to_webcam @ inv(T_marker_to_glasses)
        T_glasses_to_marker = np.linalg.inv(T_marker_to_glasses)
        T_glasses_to_webcam = T_marker_to_webcam @ T_glasses_to_marker

        # Extract R and T for the projection matrix
        # This describes where the webcam is relative to the glasses
        R_AB = T_glasses_to_webcam[:3, :3]
        T_AB = T_glasses_to_webcam[:3, 3:4]

        self.current_R = R_AB
        self.current_T = T_AB

        # Update webcam projection matrix: P2 = K2 @ [R | T]
        self.P2 = self.K2 @ np.hstack([R_AB, T_AB])

        return True

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
            point_a: (x, y) point in glasses image (pixel coordinates)
            point_b: (x, y) point in webcam image (pixel coordinates)
            undistort: Whether to undistort points before triangulation
            max_reprojection_error: Maximum allowed reprojection error in pixels

        Returns:
            3D point as numpy array [X, Y, Z] in mm (glasses coordinate system),
            or None if triangulation fails or not calibrated
        """
        if self.P2 is None:
            return None

        if max_reprojection_error is None:
            max_reprojection_error = TRACKING.MAX_REPROJECTION_ERROR

        # Convert to numpy arrays with correct shape for OpenCV
        pt_a = np.array([[point_a]], dtype=np.float32)
        pt_b = np.array([[point_b]], dtype=np.float32)

        if undistort:
            # Undistort points to remove lens distortion
            pt_a = cv2.undistortPoints(
                pt_a,
                self.K1,
                self.D1,
                P=self.K1
            )
            pt_b = cv2.undistortPoints(
                pt_b,
                self.K2,
                self.D2,
                P=self.K2
            )

        # Reshape for triangulatePoints: expects (2, N)
        pt_a = pt_a.reshape(2, 1)
        pt_b = pt_b.reshape(2, 1)

        # Triangulate
        points_4d = cv2.triangulatePoints(
            self.P1,
            self.P2,
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
            return None

        return point_3d

    def reproject_to_camera_a(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """
        Reproject a 3D point back to glasses image coordinates.

        Args:
            point_3d: 3D point [X, Y, Z]

        Returns:
            (x, y) in glasses image coordinates
        """
        point_h = np.append(point_3d, 1)
        projected = self.P1 @ point_h
        x = projected[0] / projected[2]
        y = projected[1] / projected[2]
        return (x, y)

    def reproject_to_camera_b(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """
        Reproject a 3D point back to webcam image coordinates.

        Args:
            point_3d: 3D point [X, Y, Z]

        Returns:
            (x, y) in webcam image coordinates
        """
        if self.P2 is None:
            return (0.0, 0.0)

        point_h = np.append(point_3d, 1)
        projected = self.P2 @ point_h
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
            point_a: Original 2D point in glasses image
            point_b: Original 2D point in webcam image
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

    def get_baseline_mm(self) -> Optional[float]:
        """
        Get the current baseline (distance between cameras) in mm.

        Returns:
            Baseline distance or None if not calibrated
        """
        if self.current_T is None:
            return None
        return float(np.linalg.norm(self.current_T))

    def reset_calibration(self) -> None:
        """Reset calibration state to enter calibration mode."""
        self.webcam_pose_cached = None
        self.is_calibrated = False
        self.P2 = None
        self.current_R = None
        self.current_T = None
        print("[DynamicTriangulator] Calibration reset. Ready for recalibration.")
