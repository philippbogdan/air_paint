"""Stereo calibration and save/load utilities."""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from .chessboard import ChessboardDetector, ChessboardCorners
from config.settings import CALIBRATION, get_calibration_path


@dataclass
class StereoCalibration:
    """
    Container for stereo calibration data.

    Contains all matrices needed for stereo triangulation.
    """
    # Camera A (left) intrinsics
    K1: np.ndarray  # 3x3 camera matrix
    D1: np.ndarray  # Distortion coefficients

    # Camera B (right) intrinsics
    K2: np.ndarray  # 3x3 camera matrix
    D2: np.ndarray  # Distortion coefficients

    # Stereo extrinsics (B relative to A)
    R: np.ndarray   # 3x3 rotation matrix
    T: np.ndarray   # 3x1 translation vector

    # Projection matrices
    P1: np.ndarray  # 3x4 projection matrix for camera A
    P2: np.ndarray  # 3x4 projection matrix for camera B

    # Image size
    image_size: Tuple[int, int]  # (width, height)

    # Calibration quality
    rms_error: float

    @property
    def baseline_mm(self) -> float:
        """Get the baseline distance between cameras in mm."""
        return float(np.linalg.norm(self.T))

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            'K1': self.K1,
            'D1': self.D1,
            'K2': self.K2,
            'D2': self.D2,
            'R': self.R,
            'T': self.T,
            'P1': self.P1,
            'P2': self.P2,
            'image_size': np.array(self.image_size),
            'rms_error': np.array([self.rms_error]),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StereoCalibration':
        """Create from loaded dictionary."""
        return cls(
            K1=data['K1'],
            D1=data['D1'],
            K2=data['K2'],
            D2=data['D2'],
            R=data['R'],
            T=data['T'],
            P1=data['P1'],
            P2=data['P2'],
            image_size=tuple(data['image_size']),
            rms_error=float(data['rms_error'][0]),
        )


def calibrate_stereo(
    corners_a_list: List[ChessboardCorners],
    corners_b_list: List[ChessboardCorners],
    detector: ChessboardDetector,
    image_size: Tuple[int, int]
) -> Optional[StereoCalibration]:
    """
    Perform stereo calibration from collected corner pairs.

    Args:
        corners_a_list: List of corners from camera A
        corners_b_list: List of corners from camera B
        detector: ChessboardDetector with object points
        image_size: Image size (width, height)

    Returns:
        StereoCalibration if successful, None otherwise
    """
    if len(corners_a_list) < CALIBRATION.MIN_CALIBRATION_FRAMES:
        print(f"Need at least {CALIBRATION.MIN_CALIBRATION_FRAMES} frames, "
              f"got {len(corners_a_list)}")
        return None

    if len(corners_a_list) != len(corners_b_list):
        print("Mismatched number of corner sets")
        return None

    # Prepare calibration data
    object_points = [detector.get_object_points() for _ in corners_a_list]
    image_points_a = [c.corners for c in corners_a_list]
    image_points_b = [c.corners for c in corners_b_list]

    # Calibrate each camera individually first
    print("Calibrating Camera A...")
    ret_a, K1, D1, _, _ = cv2.calibrateCamera(
        object_points, image_points_a, image_size, None, None
    )
    print(f"Camera A RMS: {ret_a:.4f}")

    print("Calibrating Camera B...")
    ret_b, K2, D2, _, _ = cv2.calibrateCamera(
        object_points, image_points_b, image_size, None, None
    )
    print(f"Camera B RMS: {ret_b:.4f}")

    # Stereo calibration
    print("Performing stereo calibration...")
    flags = (
        cv2.CALIB_FIX_INTRINSIC  # Use intrinsics from individual calibration
    )

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        image_points_a,
        image_points_b,
        K1, D1, K2, D2,
        image_size,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    )

    print(f"Stereo RMS error: {ret:.4f} pixels")

    if ret > CALIBRATION.MAX_RMS_ERROR:
        print(f"WARNING: RMS error {ret:.4f} exceeds threshold {CALIBRATION.MAX_RMS_ERROR}")

    # Compute projection matrices
    # P1 = K1 @ [I | 0] (camera A at origin)
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])

    # P2 = K2 @ [R | T] (camera B relative to A)
    P2 = K2 @ np.hstack([R, T])

    return StereoCalibration(
        K1=K1, D1=D1,
        K2=K2, D2=D2,
        R=R, T=T,
        P1=P1, P2=P2,
        image_size=image_size,
        rms_error=ret
    )


def save_calibration(
    calibration: StereoCalibration,
    path: Optional[Path] = None
) -> Path:
    """
    Save calibration to file.

    Args:
        calibration: StereoCalibration to save
        path: Path to save to (default: from settings)

    Returns:
        Path where calibration was saved
    """
    if path is None:
        path = get_calibration_path()

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(path, **calibration.to_dict())
    print(f"Calibration saved to {path}")
    return path


def load_calibration(path: Optional[Path] = None) -> Optional[StereoCalibration]:
    """
    Load calibration from file.

    Args:
        path: Path to load from (default: from settings)

    Returns:
        StereoCalibration if file exists and is valid, None otherwise
    """
    if path is None:
        path = get_calibration_path()

    if not path.exists():
        print(f"Calibration file not found: {path}")
        return None

    try:
        data = np.load(path)
        calibration = StereoCalibration.from_dict(data)
        print(f"Loaded calibration from {path}")
        print(f"  Image size: {calibration.image_size}")
        print(f"  RMS error: {calibration.rms_error:.4f}")
        print(f"  Baseline: {calibration.baseline_mm:.1f} mm")
        return calibration
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None
