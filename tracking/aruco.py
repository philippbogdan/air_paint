"""ArUco marker detection and pose estimation for world coordinate frame."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from config.settings import ARUCO


@dataclass
class MarkerPose:
    """Pose of a detected ArUco marker."""
    marker_id: int
    corners: np.ndarray  # 4 corner points in image
    rvec: np.ndarray  # Rotation vector (3,)
    tvec: np.ndarray  # Translation vector (3,) in same units as marker_size
    T_CW: np.ndarray  # 4x4 transform: Camera frame → World (marker) frame
    T_WC: np.ndarray  # 4x4 transform: World (marker) frame → Camera frame

    @property
    def distance_mm(self) -> float:
        """Distance from camera to marker in mm."""
        return float(np.linalg.norm(self.tvec))


class ArucoDetector:
    """
    Detects ArUco markers and estimates their pose.

    The marker defines a "world" coordinate frame:
    - Origin at marker center
    - X-axis pointing right (along marker)
    - Y-axis pointing up (along marker)
    - Z-axis pointing out of the marker (towards camera)
    """

    def __init__(
        self,
        marker_size_mm: float = ARUCO.MARKER_SIZE_MM,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        dictionary_id: int = ARUCO.DICTIONARY_ID,
        target_marker_id: int = ARUCO.MARKER_ID
    ):
        """
        Initialize the ArUco detector.

        Args:
            marker_size_mm: Physical size of the marker in mm
            camera_matrix: 3x3 camera intrinsic matrix (K)
            dist_coeffs: Distortion coefficients
            dictionary_id: ArUco dictionary ID (default: DICT_6X6_250)
            target_marker_id: Which marker ID to look for
        """
        self.marker_size_mm = marker_size_mm
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.target_marker_id = target_marker_id

        # Initialize ArUco detector
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def detect(self, image: np.ndarray) -> Optional[MarkerPose]:
        """
        Detect the target ArUco marker and estimate its pose.

        Args:
            image: BGR image from camera

        Returns:
            MarkerPose if target marker found, None otherwise
        """
        if self.camera_matrix is None:
            return None

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None

        # Find target marker
        target_idx = None
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == self.target_marker_id:
                target_idx = i
                break

        if target_idx is None:
            return None

        # Estimate pose
        marker_corners = corners[target_idx]

        # Use solvePnP for pose estimation
        # Define 3D points of marker corners in marker coordinate system
        half_size = self.marker_size_mm / 2.0
        obj_points = np.array([
            [-half_size,  half_size, 0],  # Top-left
            [ half_size,  half_size, 0],  # Top-right
            [ half_size, -half_size, 0],  # Bottom-right
            [-half_size, -half_size, 0],  # Bottom-left
        ], dtype=np.float32)

        # Reshape corners for solvePnP
        img_points = marker_corners.reshape(-1, 2).astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            self.camera_matrix,
            self.dist_coeffs
        )

        if not success:
            return None

        rvec = rvec.flatten()
        tvec = tvec.flatten()

        # Build transformation matrices
        T_CW, T_WC = self._build_transforms(rvec, tvec)

        return MarkerPose(
            marker_id=self.target_marker_id,
            corners=marker_corners,
            rvec=rvec,
            tvec=tvec,
            T_CW=T_CW,
            T_WC=T_WC
        )

    def _build_transforms(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build 4x4 transformation matrices from rotation and translation vectors.

        Args:
            rvec: Rotation vector (3,)
            tvec: Translation vector (3,)

        Returns:
            (T_CW, T_WC) where:
            - T_CW transforms points from camera frame to world (marker) frame
            - T_WC transforms points from world (marker) frame to camera frame
        """
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # T_WC: World → Camera (what solvePnP gives us)
        # This transforms a point in marker coordinates to camera coordinates
        T_WC = np.eye(4)
        T_WC[:3, :3] = R
        T_WC[:3, 3] = tvec

        # T_CW: Camera → World (inverse)
        # This transforms a point in camera coordinates to marker coordinates
        T_CW = np.eye(4)
        T_CW[:3, :3] = R.T
        T_CW[:3, 3] = -R.T @ tvec

        return T_CW, T_WC

    def transform_point_to_world(
        self,
        point_camera: np.ndarray,
        pose: MarkerPose
    ) -> np.ndarray:
        """
        Transform a 3D point from camera coordinates to world (marker) coordinates.

        Args:
            point_camera: 3D point in camera frame [X, Y, Z]
            pose: MarkerPose with transformation matrices

        Returns:
            3D point in world (marker) frame [X, Y, Z]
        """
        # Convert to homogeneous coordinates
        point_h = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])

        # Transform
        point_world_h = pose.T_CW @ point_h

        return point_world_h[:3]

    def transform_point_to_camera(
        self,
        point_world: np.ndarray,
        pose: MarkerPose
    ) -> np.ndarray:
        """
        Transform a 3D point from world (marker) coordinates to camera coordinates.

        Args:
            point_world: 3D point in world frame [X, Y, Z]
            pose: MarkerPose with transformation matrices

        Returns:
            3D point in camera frame [X, Y, Z]
        """
        point_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
        point_camera_h = pose.T_WC @ point_h
        return point_camera_h[:3]

    def draw_marker(
        self,
        image: np.ndarray,
        pose: MarkerPose,
        axis_length: float = 50.0
    ) -> np.ndarray:
        """
        Draw detected marker and coordinate axes on image.

        Args:
            image: BGR image to draw on
            pose: MarkerPose to visualize
            axis_length: Length of coordinate axes in mm

        Returns:
            Image with marker and axes drawn
        """
        result = image.copy()

        # Draw marker outline
        corners_int = pose.corners.reshape(-1, 2).astype(int)
        cv2.polylines(result, [corners_int], True, (0, 255, 0), 2)

        # Draw marker ID
        center = corners_int.mean(axis=0).astype(int)
        cv2.putText(
            result, f"ID:{pose.marker_id}",
            (center[0] - 20, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        # Draw coordinate axes
        if self.camera_matrix is not None:
            cv2.drawFrameAxes(
                result,
                self.camera_matrix,
                self.dist_coeffs,
                pose.rvec,
                pose.tvec,
                axis_length
            )

        return result
