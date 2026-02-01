#!/usr/bin/env python3
"""
3D Air Painting Application

A macOS Python app for "3D air painting" using stereo vision from two cameras
(Mac webcam + iPhone via USB-C). User draws in 3D by moving index finger,
toggled by spacebar or Bluetooth shutter button.

Usage:
    python main.py                           # Run server (default, for iPhone AR viewer)
    python main.py --calibrate               # Calibrate cameras
    python main.py --standalone              # Mac-only mode (no iPhone)

Controls:
    SPACE - Toggle drawing on/off
    Z     - Undo last stroke
    C     - Clear all strokes
    S     - Save session
    Q/ESC - Quit and save
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from config.settings import CAMERA, UI, ARUCO, TRACKING, get_calibration_path
from capture.sync import SynchronizedCapture, list_available_cameras, get_camera_names, auto_select_cameras, interactive_select_cameras
from calibration.stereo import load_calibration, StereoCalibration
from tracking.hands import HandTracker, HandLandmarks
from tracking.triangulate import StereoTriangulator, PointSmoother
from tracking.aruco import ArucoDetector, MarkerPose
from drawing.manager import DrawingManager
from ui.input import InputHandler, InputAction
from ui.window import MainWindow
from export.saver import SessionSaver


class AirPaintingApp:
    """Main application orchestrating all components."""

    def __init__(
        self,
        camera_a_index: int = CAMERA.CAMERA_A_INDEX,
        camera_b_index: int = CAMERA.CAMERA_B_INDEX
    ):
        """
        Initialize the application.

        Args:
            camera_a_index: Index for camera A (Mac webcam)
            camera_b_index: Index for camera B (iPhone)
        """
        self.camera_a_index = camera_a_index
        self.camera_b_index = camera_b_index

        # Components (initialized in setup)
        self.capture: Optional[SynchronizedCapture] = None
        self.calibration: Optional[StereoCalibration] = None
        self.triangulator: Optional[StereoTriangulator] = None
        self.point_smoother: Optional[PointSmoother] = None
        self.aruco_detector: Optional[ArucoDetector] = None
        self.hand_tracker_a: Optional[HandTracker] = None
        self.hand_tracker_b: Optional[HandTracker] = None
        self.drawing_manager: Optional[DrawingManager] = None
        self.input_handler: Optional[InputHandler] = None
        self.window: Optional[MainWindow] = None
        self.saver: Optional[SessionSaver] = None

        # State
        self._running = False
        self._save_complete_time: Optional[float] = None  # For auto-dismiss
        self._last_composed_frame: Optional[np.ndarray] = None
        self._fps = 0.0
        self._frame_times = []
        self._marker_pose: Optional[MarkerPose] = None
        self._executor = ThreadPoolExecutor(max_workers=2)  # For parallel hand detection

        # Two-phase state machine
        # Phase 1: "setup" - waiting for ArUco + HID click to establish world center
        # Phase 2: "drawing" - HID clicks toggle drawing
        self._phase = "setup"
        self._world_locked = False

    def setup(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if setup successful
        """
        print("=" * 60)
        print("3D Air Painting")
        print("=" * 60)

        # Check available cameras
        available = list_available_cameras()
        print(f"\nAvailable cameras: {available}")

        if len(available) < 2:
            print("\nWARNING: Less than 2 cameras detected.")
            print("For stereo 3D, you need both Mac webcam and iPhone connected.")
            print("Continuing in single-camera demo mode...\n")

        # Load calibration
        print("\nLoading calibration...")
        self.calibration = load_calibration()

        if self.calibration is None:
            print("\nNo calibration found!")
            print("Run calibration first: python -m calibration.calibrate")
            print("\nContinuing without calibration (2D tracking only)...")
        else:
            self.triangulator = StereoTriangulator(self.calibration)
            self.point_smoother = PointSmoother(alpha=TRACKING.SMOOTHING_ALPHA)
            print(f"Calibration loaded (RMS: {self.calibration.rms_error:.4f})")
            print(f"Point smoothing enabled (alpha={TRACKING.SMOOTHING_ALPHA})")
            print(f"Outlier rejection enabled (max reproj error={TRACKING.MAX_REPROJECTION_ERROR}px)")

            # Initialize ArUco detector for world coordinate frame
            if ARUCO.ENABLED:
                print(f"\nInitializing ArUco detector (marker ID: {ARUCO.MARKER_ID}, size: {ARUCO.MARKER_SIZE_MM}mm)...")
                self.aruco_detector = ArucoDetector(
                    marker_size_mm=ARUCO.MARKER_SIZE_MM,
                    camera_matrix=self.calibration.K1,
                    dist_coeffs=self.calibration.D1,
                    target_marker_id=ARUCO.MARKER_ID
                )
                print("ArUco detector ready - place marker in view for world coordinates")

        # Initialize capture
        print("\nInitializing cameras...")
        self.capture = SynchronizedCapture(
            camera_a_index=self.camera_a_index,
            camera_b_index=self.camera_b_index
        )

        if not self.capture.open():
            print("Failed to open cameras!")
            # Try to at least open camera A
            self.capture.camera_a.open()
            if not self.capture.camera_a.is_opened():
                print("Cannot open any camera. Exiting.")
                return False
            print("Running with Camera A only.")

        # Initialize hand trackers (one per camera for parallel processing)
        print("Initializing hand tracking...")
        self.hand_tracker_a = HandTracker()
        self.hand_tracker_b = HandTracker()

        # Initialize drawing manager
        self.drawing_manager = DrawingManager()

        # Initialize input handler
        self.input_handler = InputHandler()
        self._setup_input_callbacks()

        # Initialize window
        self.window = MainWindow()

        # Initialize background saver
        self.saver = SessionSaver()

        print("\nSetup complete!")
        print("\n" + "=" * 40)
        print("FLOW:")
        print("  1. Show ArUco marker to camera")
        print("  2. Click HID (Volume Up) to lock world center")
        print("  3. Click HID to toggle drawing on/off")
        print("\nKeyboard shortcuts:")
        print("  SPACE - Same as HID click")
        print("  Z     - Undo last stroke")
        print("  C     - Clear all strokes")
        print("  S     - Save session")
        print("  Q/ESC - Quit")
        print("=" * 40)
        print("\n>>> Show ArUco marker and click HID to start <<<\n")

        return True

    def _setup_input_callbacks(self) -> None:
        """Set up input action callbacks."""
        self.input_handler.register_callback(
            InputAction.HID_CLICK,
            self._on_hid_click
        )
        self.input_handler.register_callback(
            InputAction.UNDO,
            self._on_undo
        )
        self.input_handler.register_callback(
            InputAction.CLEAR,
            self._on_clear
        )
        self.input_handler.register_callback(
            InputAction.SAVE,
            self._on_save
        )

    def _on_hid_click(self) -> None:
        """
        Handle HID button click (Volume Up or Spacebar).

        Two-phase behavior:
        1. Setup phase: If ArUco visible, lock world center and switch to drawing phase
        2. Drawing phase: Toggle drawing on/off
        """
        if self._phase == "setup":
            # In setup phase - try to lock world center
            if self._marker_pose is not None:
                # ArUco is visible - lock world center
                self._world_locked = True
                self._phase = "drawing"
                print("=" * 40)
                print("WORLD CENTER LOCKED!")
                print("HID click now toggles drawing.")
                print("=" * 40)
            else:
                print("ArUco marker not visible - show marker and click again")
        else:
            # In drawing phase - toggle drawing
            self._on_toggle_drawing()

    def _on_toggle_drawing(self) -> None:
        """Handle drawing toggle."""
        was_drawing = self.drawing_manager.is_drawing
        self.drawing_manager.toggle_drawing()
        state = "ON" if self.drawing_manager.is_drawing else "OFF"
        print(f"Drawing: {state}")

        # Reset smoother when drawing stops to avoid lag on next stroke
        if was_drawing and not self.drawing_manager.is_drawing:
            if self.point_smoother is not None:
                self.point_smoother.reset()

    def _on_undo(self) -> None:
        """Handle undo."""
        stroke = self.drawing_manager.undo()
        if stroke:
            print(f"Undid stroke ({stroke.num_points} points)")
        else:
            print("Nothing to undo")

    def _on_clear(self) -> None:
        """Handle clear."""
        self.drawing_manager.clear()
        print("Cleared all strokes")

    def _on_save(self) -> None:
        """Handle save."""
        self._save_session()

    def _save_session(self, blocking: bool = False) -> None:
        """
        Save the current session.

        Args:
            blocking: If True, wait for save to complete (used during cleanup)
        """
        if self.saver.is_saving:
            print("Save already in progress...")
            return

        stats = self.drawing_manager.get_stats()
        if stats.total_strokes == 0:
            print("No strokes to save")
            return

        # Start async save
        started = self.saver.save_async(
            strokes=self.drawing_manager.all_strokes,
            stats_dict=stats.to_dict(),
            screenshot=self._last_composed_frame,
            session_start=self.drawing_manager._session_start,
            coordinate_frame=self.drawing_manager.coordinate_frame,
            marker_metadata=self.drawing_manager._marker_metadata
        )

        if started:
            print("Saving session in background...")
            self._save_complete_time = None

        # If blocking, wait for completion
        if blocking and started:
            while self.saver.is_saving:
                time.sleep(0.1)
            progress = self.saver.progress
            if progress.is_complete:
                print(f"Session saved to: {progress.output_dir}")
            elif progress.is_failed:
                print(f"Save failed: {progress.error}")

    def _update_fps(self) -> None:
        """Update FPS calculation."""
        now = time.time()
        self._frame_times.append(now)

        # Keep only last 30 frames for FPS calculation
        if len(self._frame_times) > 30:
            self._frame_times = self._frame_times[-30:]

        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._fps = (len(self._frame_times) - 1) / elapsed

    def run(self) -> None:
        """Run the main application loop."""
        self._running = True

        while self._running:
            # Capture frames
            stereo_frame = self.capture.read()

            if stereo_frame is None:
                # Try single camera mode
                frame_a = self.capture.camera_a.read()
                if frame_a is None:
                    continue
                frame_a_img = frame_a.image
                frame_b_img = np.zeros_like(frame_a_img)
            else:
                frame_a_img, frame_b_img = stereo_frame.images

            # Detect ArUco marker for world coordinates
            self._marker_pose = None
            if self.aruco_detector is not None:
                self._marker_pose = self.aruco_detector.detect(frame_a_img)

                if self._marker_pose is not None:
                    # Update world transform
                    self.drawing_manager.set_world_transform(
                        self._marker_pose.T_CW,
                        marker_id=self._marker_pose.marker_id,
                        marker_size_mm=ARUCO.MARKER_SIZE_MM
                    )
                    # Draw marker on frame
                    frame_a_img = self.aruco_detector.draw_marker(frame_a_img, self._marker_pose)
                else:
                    # Marker not visible - apply policy
                    if ARUCO.MISSING_MARKER_POLICY == "pause":
                        self.drawing_manager.clear_world_transform()

            # Detect hands in both frames (parallel for speed)
            future_a = self._executor.submit(self.hand_tracker_a.detect, frame_a_img)
            future_b = self._executor.submit(self.hand_tracker_b.detect, frame_b_img)
            hands_a = future_a.result()
            hands_b = future_b.result()

            hand_a = hands_a[0] if hands_a else None
            hand_b = hands_b[0] if hands_b else None

            # Triangulate if we have both hands and calibration
            if (hand_a is not None and hand_b is not None and
                self.triangulator is not None):

                # Confidence gate - skip if either hand has low confidence
                if (hand_a.confidence < TRACKING.MIN_POINT_CONFIDENCE or
                    hand_b.confidence < TRACKING.MIN_POINT_CONFIDENCE):
                    # Low confidence - skip this frame
                    pass
                else:
                    point_a = hand_a.index_finger_tip
                    point_b = hand_b.index_finger_tip

                    # Triangulate with outlier rejection (reprojection error check)
                    point_3d = self.triangulator.triangulate(point_a, point_b)

                    # Only add point if drawing AND (no ArUco OR marker visible OR policy is "continue")
                    can_record = (
                        self.aruco_detector is None or
                        self._marker_pose is not None or
                        ARUCO.MISSING_MARKER_POLICY == "continue"
                    )

                    if point_3d is not None and self.drawing_manager.is_drawing and can_record:
                        # Apply temporal smoothing
                        if self.point_smoother is not None:
                            point_3d = self.point_smoother.smooth(point_3d)
                        self.drawing_manager.add_point(point_3d)

            # Prepare projection function for 2D overlay
            project_func = None
            if self.triangulator is not None:
                project_func = self.triangulator.reproject_to_camera_a

            # Compose display
            composed = self.window.compose(
                frame_a=frame_a_img,
                frame_b=frame_b_img,
                strokes=self.drawing_manager.all_strokes,
                is_drawing=self.drawing_manager.is_drawing,
                hand_a=hand_a,
                hand_b=hand_b,
                project_func=project_func,
                fps=self._fps,
                marker_visible=self._marker_pose is not None,
                coordinate_frame=self.drawing_manager.coordinate_frame,
                phase=self._phase
            )

            self._last_composed_frame = composed

            # Show save progress overlay if saving
            if self.saver.is_saving or self._save_complete_time is not None:
                progress = self.saver.progress

                # Check if just completed
                if progress.is_complete and self._save_complete_time is None:
                    self._save_complete_time = time.time()
                    print(f"Session saved to: {progress.output_dir}")

                # Auto-dismiss after 2 seconds
                if self._save_complete_time is not None:
                    elapsed = time.time() - self._save_complete_time
                    if elapsed > 2.0:
                        self._save_complete_time = None
                        self.saver.reset()
                    else:
                        # Still showing completion
                        composed = self.window.draw_save_progress(
                            composed,
                            progress.progress,
                            progress.message,
                            is_complete=progress.is_complete,
                            is_failed=progress.is_failed
                        )
                elif progress.is_failed:
                    # Show failure until dismissed
                    composed = self.window.draw_save_progress(
                        composed,
                        progress.progress,
                        progress.message,
                        is_complete=False,
                        is_failed=True
                    )
                    # Auto-dismiss failure after 3 seconds
                    if self._save_complete_time is None:
                        self._save_complete_time = time.time()
                else:
                    # Show progress
                    composed = self.window.draw_save_progress(
                        composed,
                        progress.progress,
                        progress.message
                    )

            self.window.show(composed)

            # Handle input
            event = self.input_handler.wait_key(1)
            if event.action == InputAction.QUIT:
                self._running = False

            # Update FPS
            self._update_fps()

    def cleanup(self) -> None:
        """Clean up resources."""
        print("\nCleaning up...")

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=False)

        # Stop HID listener
        if self.input_handler:
            self.input_handler.cleanup()

        # Offer to save if there are unsaved strokes
        if self.drawing_manager and self.drawing_manager.get_stats().total_strokes > 0:
            self._save_session(blocking=True)

        # Release resources
        if self.capture:
            self.capture.release()

        if self.hand_tracker_a:
            self.hand_tracker_a.close()

        if self.hand_tracker_b:
            self.hand_tracker_b.close()

        if self.window:
            self.window.close()

        cv2.destroyAllWindows()
        print("Done!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="3D Air Painting with stereo vision"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration instead of painting"
    )
    parser.add_argument(
        "--camera-a",
        type=int,
        default=CAMERA.CAMERA_A_INDEX,
        help=f"Camera A index (default: {CAMERA.CAMERA_A_INDEX})"
    )
    parser.add_argument(
        "--camera-b",
        type=int,
        default=CAMERA.CAMERA_B_INDEX,
        help=f"Camera B index (default: {CAMERA.CAMERA_B_INDEX})"
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively select cameras with visual preview"
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Run in standalone Mac-only mode (no iPhone AR viewer)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)"
    )

    args = parser.parse_args()

    if args.list_cameras:
        cameras = list_available_cameras(10)
        camera_names = get_camera_names()
        print(f"Available cameras: {cameras}")
        for i in cameras:
            cap = cv2.VideoCapture(i)
            # Set target resolution to see actual achievable fps
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA.FPS)
            # Read back actual values
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            name = camera_names.get(i, "Unknown")
            is_iphone = "iPhone" in name or "iPad" in name
            marker = " [iPhone - excluded by default]" if is_iphone else ""
            print(f"  Camera {i}: {name} ({w}x{h} @ {fps:.1f} fps){marker}")

        # Show auto-selected cameras
        auto_a, auto_b = auto_select_cameras()
        print(f"\nAuto-selected: Camera A={auto_a}, Camera B={auto_b}")
        return

    if args.calibrate:
        from calibration.calibrate import run_calibration
        # Pass camera settings if specified
        cam_a = args.camera_a if args.camera_a != CAMERA.CAMERA_A_INDEX else None
        cam_b = args.camera_b if args.camera_b != CAMERA.CAMERA_B_INDEX else None
        run_calibration(camera_a=cam_a, camera_b=cam_b, interactive=args.interactive)
        return

    # Camera selection (used by both modes)
    cam_a = args.camera_a
    cam_b = args.camera_b
    manually_specified = not (cam_a == CAMERA.CAMERA_A_INDEX and cam_b == CAMERA.CAMERA_B_INDEX)

    # Show available cameras
    camera_names = get_camera_names()
    print("\nDetected cameras:")
    for idx, name in camera_names.items():
        is_iphone = "iPhone" in name or "iPad" in name
        marker = " [EXCLUDED]" if is_iphone else ""
        print(f"  {idx}: {name}{marker}")

    # Select cameras
    if args.interactive:
        cam_a, cam_b = interactive_select_cameras()
    elif not manually_specified:
        cam_a, cam_b = auto_select_cameras()

    print(f"\nUsing: Camera A={cam_a} ({camera_names.get(cam_a, 'Unknown')})")
    print(f"       Camera B={cam_b} ({camera_names.get(cam_b, 'Unknown')})")

    if args.standalone:
        # Run standalone Mac-only mode (no iPhone)
        app = AirPaintingApp(
            camera_a_index=cam_a,
            camera_b_index=cam_b
        )

        try:
            if app.setup():
                app.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            app.cleanup()
        return

    # Default: Run WebSocket server for iPhone AR viewer
    import asyncio
    from server.local_camera_server import LocalCameraServer

    server = LocalCameraServer(
        port=args.port,
        camera_a_index=cam_a,
        camera_b_index=cam_b
    )

    if not server.setup():
        print("Server setup failed!")
        return

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        server.stop()
        server.cleanup()


if __name__ == "__main__":
    main()
