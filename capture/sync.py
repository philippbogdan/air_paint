"""Synchronized dual-camera capture."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .camera import Camera, CameraFrame
from config.settings import CAMERA
from typing import Dict


@dataclass
class StereoFrame:
    """Container for synchronized stereo frames."""
    frame_a: CameraFrame
    frame_b: CameraFrame

    @property
    def images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get both images as a tuple."""
        return self.frame_a.image, self.frame_b.image


class SynchronizedCapture:
    """
    Manages synchronized capture from two cameras.

    Note: True hardware sync is not possible with consumer cameras.
    This class captures frames as close together as possible and
    accepts ~33ms lag as acceptable for slow hand motion.
    """

    def __init__(
        self,
        camera_a_index: int = CAMERA.CAMERA_A_INDEX,
        camera_b_index: int = CAMERA.CAMERA_B_INDEX,
        width: int = CAMERA.WIDTH,
        height: int = CAMERA.HEIGHT,
        fps: int = CAMERA.FPS
    ):
        """
        Initialize synchronized capture.

        Args:
            camera_a_index: Index for camera A (usually Mac webcam)
            camera_b_index: Index for camera B (usually iPhone)
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        # Get camera names for display
        camera_names = get_camera_names()
        name_a = camera_names.get(camera_a_index, f"Camera {camera_a_index}")
        name_b = camera_names.get(camera_b_index, f"Camera {camera_b_index}")

        self.camera_a = Camera(
            index=camera_a_index,
            width=width,
            height=height,
            fps=fps,
            name=f"Camera A ({name_a})"
        )
        self.camera_b = Camera(
            index=camera_b_index,
            width=width,
            height=height,
            fps=fps,
            name=f"Camera B ({name_b})"
        )
        self._is_open = False

    def open(self) -> bool:
        """
        Open both cameras.

        Returns:
            True if both cameras opened successfully
        """
        if not self.camera_a.open():
            print("Failed to open Camera A")
            return False

        if not self.camera_b.open():
            print("Failed to open Camera B")
            self.camera_a.release()
            return False

        self._is_open = True
        return True

    def read(self) -> Optional[StereoFrame]:
        """
        Read synchronized frames from both cameras.

        Captures from both cameras as quickly as possible.

        Returns:
            StereoFrame if both captures succeed, None otherwise
        """
        if not self._is_open:
            return None

        # Capture from both cameras in quick succession
        frame_a = self.camera_a.read()
        frame_b = self.camera_b.read()

        if frame_a is None or frame_b is None:
            return None

        return StereoFrame(frame_a=frame_a, frame_b=frame_b)

    def release(self) -> None:
        """Release both cameras."""
        self.camera_a.release()
        self.camera_b.release()
        self._is_open = False

    def is_opened(self) -> bool:
        """Check if both cameras are open."""
        return self._is_open and self.camera_a.is_opened() and self.camera_b.is_opened()

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the capture resolution (assuming both cameras match)."""
        return self.camera_a.width, self.camera_a.height

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def list_available_cameras(max_index: int = 5) -> list[int]:
    """
    List available camera indices.

    Args:
        max_index: Maximum index to check

    Returns:
        List of available camera indices
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def get_camera_names() -> dict[int, str]:
    """
    Get camera names from macOS AVFoundation.

    IMPORTANT ASSUMPTION: We assume OpenCV (AVFoundation backend on macOS) sorts
    cameras by their unique device ID (string sort). This was determined empirically
    with a small sample size (3 cameras: USB webcam, iPhone Continuity, FaceTime).

    If camera detection breaks in the future, this assumption may be wrong.
    In that case, use --interactive or --camera-a/--camera-b flags as fallback.

    Discovery process:
    - AVFoundation returns devices in connection/discovery order
    - OpenCV returns devices in a different order
    - Sorting AVFoundation devices by uniqueID() matched OpenCV's indices
    - uniqueID examples: "0x12000032e69221...", "89B1B5F5-...", "FDF90FEB-..."
      These sort as: 0x... < 8... < F... (ASCII string sort)

    Returns:
        Dict mapping camera index to name
    """
    try:
        import AVFoundation as AVF

        # IMPORTANT: First query to AVFoundation can return unstable order
        # (devices may still be "settling" after connection). Query twice
        # and use the second result for stability.
        AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)  # Warm-up (discard)
        devices = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)  # Stable

        # Sort by unique ID to match OpenCV's ordering (empirically determined)
        # WARNING: This assumption may not hold for all camera configurations
        sorted_devices = sorted(devices, key=lambda d: d.uniqueID())

        return {i: device.localizedName() for i, device in enumerate(sorted_devices)}
    except ImportError:
        # Fallback to system_profiler if AVFoundation not available
        return _get_camera_names_system_profiler()
    except Exception:
        return {}


def _get_camera_names_system_profiler() -> dict[int, str]:
    """Fallback: get camera names from system_profiler (less reliable)."""
    import subprocess

    try:
        result = subprocess.run(
            ['system_profiler', 'SPCameraDataType'],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout

        cameras = []
        for line in output.split('\n'):
            line = line.strip()
            if line.endswith(':') and not line.startswith('Model ID') and not line.startswith('Unique ID'):
                current_name = line[:-1]
                if current_name and current_name != "Camera":
                    cameras.append(current_name)

        return {i: name for i, name in enumerate(cameras)}
    except Exception:
        return {}


def is_iphone_camera(name: str) -> bool:
    """Check if camera name indicates an iPhone."""
    name_lower = name.lower()
    return 'iphone' in name_lower or 'ipad' in name_lower


def find_non_iphone_cameras(max_index: int = 5) -> list[int]:
    """
    Find camera indices that are NOT iPhone cameras.

    Prioritizes Mac built-in webcam and USB webcams.

    Returns:
        List of non-iPhone camera indices
    """
    available = list_available_cameras(max_index)
    camera_names = get_camera_names()

    non_iphone = []
    for idx in available:
        name = camera_names.get(idx, "")
        if not is_iphone_camera(name):
            non_iphone.append(idx)

    return non_iphone


def auto_select_cameras(max_index: int = 5) -> Tuple[int, int]:
    """
    Automatically select two cameras, preferring non-iPhone cameras.

    Note: Name-to-index mapping may be unreliable. Use interactive_select_cameras()
    or manual --camera-a/--camera-b if wrong cameras are selected.

    Returns:
        (camera_a_index, camera_b_index) tuple
    """
    camera_names = get_camera_names()
    available = list_available_cameras(max_index)

    if len(available) < 2:
        # Not enough cameras, return defaults
        return (0, 1)

    # Separate iPhone and non-iPhone cameras
    non_iphone = []
    iphone = []

    for idx in available:
        name = camera_names.get(idx, "")
        if is_iphone_camera(name):
            iphone.append(idx)
        else:
            non_iphone.append(idx)

    # Prefer non-iPhone cameras
    if len(non_iphone) >= 2:
        return (non_iphone[0], non_iphone[1])
    elif len(non_iphone) == 1 and len(iphone) >= 1:
        # Use one non-iPhone and one iPhone as fallback
        return (non_iphone[0], iphone[0])
    else:
        # Use whatever is available
        return (available[0], available[1])


def interactive_select_cameras(max_index: int = 5) -> Tuple[int, int]:
    """
    Interactively select cameras by showing preview of each.

    Shows each camera feed briefly and asks user to identify them.

    Returns:
        (camera_a_index, camera_b_index) tuple
    """
    available = list_available_cameras(max_index)

    if len(available) < 2:
        print("ERROR: Need at least 2 cameras")
        return (0, 1)

    print("\n=== Interactive Camera Selection ===")
    print("Each camera will be shown. Press any key to continue.\n")

    camera_descriptions = {}

    for idx in available:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Add text overlay
                display = frame.copy()
                cv2.putText(
                    display, f"Camera {idx}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
                )
                cv2.putText(
                    display, "Press any key to continue",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                cv2.imshow("Camera Preview", display)
                cv2.waitKey(0)
            cap.release()

    cv2.destroyAllWindows()

    # Ask user to select
    print(f"\nAvailable cameras: {available}")
    try:
        cam_a = int(input("Enter Camera A index (main camera): "))
        cam_b = int(input("Enter Camera B index (secondary camera): "))

        if cam_a not in available or cam_b not in available:
            print("Invalid selection, using defaults")
            return (available[0], available[1])

        return (cam_a, cam_b)
    except (ValueError, EOFError):
        print("Invalid input, using defaults")
        return (available[0], available[1])
