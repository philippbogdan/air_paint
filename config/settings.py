"""Global configuration constants for the 3D air painting application."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraSettings:
    """Camera capture settings."""
    WIDTH: int = 1280
    HEIGHT: int = 720
    FPS: int = 30
    CAMERA_A_INDEX: int = 0  # Mac webcam
    CAMERA_B_INDEX: int = 1  # iPhone via USB-C


@dataclass(frozen=True)
class CalibrationSettings:
    """Calibration parameters."""
    CHESSBOARD_SIZE: tuple[int, int] = (9, 6)  # Inner corners
    SQUARE_SIZE_MM: float = 25.0  # Size of each square in mm
    MIN_CALIBRATION_FRAMES: int = 15
    MAX_RMS_ERROR: float = 1.0  # Maximum acceptable RMS error in pixels
    CALIBRATION_FILE: str = "data/calibration/stereo_calib.npz"


@dataclass(frozen=True)
class TrackingSettings:
    """Hand tracking settings."""
    INDEX_FINGER_TIP_LANDMARK: int = 8  # MediaPipe landmark for index finger tip
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.5
    MAX_NUM_HANDS: int = 1
    DETECTION_SCALE: float = 0.35  # Downscale factor for faster hand detection (smaller = faster)
    # 3D point quality settings
    MIN_POINT_CONFIDENCE: float = 0.4  # Minimum hand confidence to accept 3D point
    MAX_REPROJECTION_ERROR: float = 50.0  # Maximum reprojection error in pixels (relaxed due to distortion mismatch)
    SMOOTHING_ALPHA: float = 0.3  # EMA smoothing factor (higher = more responsive, lower = smoother)


@dataclass(frozen=True)
class DrawingSettings:
    """Drawing behavior settings."""
    MIN_POINT_DISTANCE_MM: float = 2.0  # Minimum distance between consecutive points
    STROKE_COLOR: tuple[int, int, int] = (0, 255, 0)  # Green in BGR
    STROKE_THICKNESS: int = 2
    POINT_RADIUS: int = 4


@dataclass(frozen=True)
class UISettings:
    """UI display settings."""
    WINDOW_NAME: str = "3D Air Painting"
    CAMERA_PREVIEW_WIDTH: int = 640
    CAMERA_PREVIEW_HEIGHT: int = 360
    VIEW_3D_SIZE: tuple[int, int] = (640, 480)
    AUTO_ROTATE_SPEED: float = 0.5  # Degrees per frame
    SKELETON_COLOR: tuple[int, int, int] = (255, 0, 0)  # Blue in BGR
    SKELETON_THICKNESS: int = 2


@dataclass(frozen=True)
class ExportSettings:
    """Export settings."""
    OUTPUT_DIR: str = "output"
    TUBE_RADIUS_MM: float = 2.0
    TUBE_SEGMENTS: int = 8


@dataclass(frozen=True)
class ArucoSettings:
    """ArUco marker settings for world coordinate frame."""
    ENABLED: bool = True  # Enable world-frame mode with ArUco marker
    MARKER_ID: int = 0  # Which marker ID to look for
    MARKER_SIZE_MM: float = 160.0  # Physical size of printed marker
    DICTIONARY_ID: int = 10  # cv2.aruco.DICT_6X6_250
    # Policy when marker not visible: "pause" (don't record) or "continue" (use last known)
    MISSING_MARKER_POLICY: str = "continue"  # "pause" or "continue" when marker not visible


# Global instances
CAMERA = CameraSettings()
CALIBRATION = CalibrationSettings()
TRACKING = TrackingSettings()
DRAWING = DrawingSettings()
UI = UISettings()
EXPORT = ExportSettings()
ARUCO = ArucoSettings()


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_calibration_path() -> Path:
    """Get the full path to the calibration file."""
    return get_project_root() / CALIBRATION.CALIBRATION_FILE


def get_output_dir() -> Path:
    """Get the output directory path."""
    return get_project_root() / EXPORT.OUTPUT_DIR
