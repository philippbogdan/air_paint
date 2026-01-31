#!/usr/bin/env python3
"""
Automatic stereo calibration script.

Usage:
    python -m calibration.calibrate

The script automatically captures frames when chessboard is detected
in both cameras for 0.5 seconds continuously. Captures 30 samples,
then auto-calibrates and saves.

Controls:
    Q - Quit early
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from capture.sync import SynchronizedCapture, list_available_cameras, get_camera_names, auto_select_cameras, interactive_select_cameras
from calibration.chessboard import ChessboardDetector, collect_calibration_pair
from calibration.stereo import calibrate_stereo, save_calibration, StereoCalibration
from config.settings import CALIBRATION, CAMERA


def draw_status(
    frame: np.ndarray,
    text: str,
    color: tuple = (0, 255, 0)
) -> np.ndarray:
    """Draw status text on frame."""
    result = frame.copy()
    cv2.putText(
        result, text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2
    )
    return result


def run_calibration(camera_a: int = None, camera_b: int = None, interactive: bool = False):
    """
    Run the interactive calibration process.

    Args:
        camera_a: Camera A index (None for auto-select)
        camera_b: Camera B index (None for auto-select)
        interactive: Use interactive camera selection
    """
    print("=" * 60)
    print("Stereo Camera Calibration")
    print("=" * 60)

    # List available cameras
    available = list_available_cameras()
    camera_names = get_camera_names()

    print(f"\nDetected cameras:")
    for idx, name in camera_names.items():
        is_iphone = "iPhone" in name or "iPad" in name
        marker = " [EXCLUDED]" if is_iphone else ""
        print(f"  {idx}: {name}{marker}")

    if len(available) < 2:
        print("\nERROR: Need at least 2 cameras for stereo calibration")
        return

    # Select cameras
    if camera_a is not None and camera_b is not None:
        cam_a, cam_b = camera_a, camera_b
    elif interactive:
        cam_a, cam_b = interactive_select_cameras()
    else:
        cam_a, cam_b = auto_select_cameras()
    print(f"\nUsing Camera A={cam_a} ({camera_names.get(cam_a, 'Unknown')})")
    print(f"       Camera B={cam_b} ({camera_names.get(cam_b, 'Unknown')})")
    print(f"\nChessboard size: {CALIBRATION.CHESSBOARD_SIZE}")
    print(f"Square size: {CALIBRATION.SQUARE_SIZE_MM} mm")
    print(f"\nAutomatic capture mode:")
    print("  - Hold chessboard visible to BOTH cameras")
    print("  - Keep steady for 0.5s to auto-capture")
    print("  - Move to different position/angle, repeat")
    print("  - 30 samples will be captured automatically")
    print("  - Press Q to quit early")
    print()

    # Auto-capture settings
    TARGET_FRAMES = 30
    HOLD_TIME = 0.5  # seconds to hold steady before capture
    MIN_MOVE_TIME = 0.3  # minimum time between captures to ensure different poses

    # Initialize components
    detector = ChessboardDetector()
    capture = SynchronizedCapture(
        camera_a_index=cam_a,
        camera_b_index=cam_b
    )
    corners_a_list = []
    corners_b_list = []
    calibration: StereoCalibration | None = None
    last_message = "Hold chessboard steady in view..."
    message_color = (0, 255, 0)

    # Auto-capture state
    detection_start_time: float | None = None
    last_capture_time: float = 0
    capturing_done = False

    if not capture.open():
        print("ERROR: Failed to open cameras")
        return

    try:
        while True:
            # Capture frames
            stereo_frame = capture.read()
            if stereo_frame is None:
                continue

            frame_a, frame_b = stereo_frame.images
            current_time = time.time()

            # Try to detect chessboard in both frames
            pair = collect_calibration_pair(frame_a, frame_b, detector)

            # Auto-capture logic
            if pair is not None and not capturing_done:
                if detection_start_time is None:
                    # Started detecting
                    detection_start_time = current_time

                time_detected = current_time - detection_start_time
                time_since_last = current_time - last_capture_time

                # Check if held steady long enough and enough time since last capture
                if time_detected >= HOLD_TIME and time_since_last >= MIN_MOVE_TIME:
                    corners_a_list.append(pair[0])
                    corners_b_list.append(pair[1])
                    last_capture_time = current_time
                    detection_start_time = None  # Reset for next capture
                    last_message = f"Captured {len(corners_a_list)}/{TARGET_FRAMES} - move to new position"
                    message_color = (0, 255, 0)
                    print(last_message)

                    # Check if we have enough frames
                    if len(corners_a_list) >= TARGET_FRAMES:
                        capturing_done = True
                        last_message = "Running calibration..."
                        print(f"\n{last_message}")

                        calibration = calibrate_stereo(
                            corners_a_list,
                            corners_b_list,
                            detector,
                            capture.resolution
                        )

                        if calibration is not None:
                            last_message = f"RMS: {calibration.rms_error:.4f}"
                            if calibration.rms_error < 1.0:
                                message_color = (0, 255, 0)
                                save_calibration(calibration)
                                print(f"Calibration saved! RMS: {calibration.rms_error:.4f}")
                                last_message = f"Saved! RMS: {calibration.rms_error:.4f} - Press Q to quit"
                            else:
                                message_color = (0, 165, 255)
                                save_calibration(calibration)
                                print(f"Calibration saved (high RMS: {calibration.rms_error:.4f})")
                                last_message = f"Saved (high RMS: {calibration.rms_error:.4f}) - Press Q"
                        else:
                            last_message = "Calibration failed!"
                            message_color = (0, 0, 255)
            else:
                # Lost detection, reset timer
                detection_start_time = None

            # Create display frames
            if pair is not None:
                corners_a, corners_b = pair
                display_a = detector.draw_corners(frame_a, corners_a, True)
                display_b = detector.draw_corners(frame_b, corners_b, True)
                status_a = "Chessboard FOUND"
                status_b = "Chessboard FOUND"

                # Show hold progress
                if detection_start_time is not None and not capturing_done:
                    progress = min(1.0, (current_time - detection_start_time) / HOLD_TIME)
                    bar_width = int(200 * progress)
                    cv2.rectangle(display_a, (10, 130), (10 + bar_width, 145), (0, 255, 0), -1)
                    cv2.rectangle(display_a, (10, 130), (210, 145), (255, 255, 255), 2)
                    cv2.putText(display_a, "Hold steady...", (10, 125),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                display_a = frame_a.copy()
                display_b = frame_b.copy()
                # Check individual detection
                ca = detector.detect(frame_a)
                cb = detector.detect(frame_b)
                status_a = "FOUND" if ca else "Searching..."
                status_b = "FOUND" if cb else "Searching..."

            # Add status text
            display_a = draw_status(display_a, f"Camera A: {status_a}")
            display_b = draw_status(display_b, f"Camera B: {status_b}")

            # Add frame count
            cv2.putText(
                display_a, f"Frames: {len(corners_a_list)}/{TARGET_FRAMES}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

            if last_message:
                cv2.putText(
                    display_a, last_message,
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, message_color, 2
                )

            if calibration is not None:
                cv2.putText(
                    display_a, f"Calibration RMS: {calibration.rms_error:.4f}",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

            # Resize for display
            scale = 0.5
            display_a = cv2.resize(display_a, None, fx=scale, fy=scale)
            display_b = cv2.resize(display_b, None, fx=scale, fy=scale)

            # Combine side by side
            combined = np.hstack([display_a, display_b])
            cv2.imshow("Stereo Calibration", combined)

            # Handle input (only Q to quit)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_calibration()
