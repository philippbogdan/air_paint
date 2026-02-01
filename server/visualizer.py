#
#  visualizer.py
#  Real-time visualization for Air Paint server
#

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class HandData:
    """Hand tracking data for visualization."""
    index_tip: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    landmarks: Optional[list] = None


class ServerVisualizer:
    """Displays camera feeds and tracking data in OpenCV windows."""

    def __init__(self, window_name: str = "Air Paint Server"):
        self.window_name = window_name
        self.window_created = False

        # Stroke storage for 2D projection
        self.strokes: Dict[int, List[np.ndarray]] = {}  # stroke_id -> list of 3D points
        self.current_marker_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (rvec, tvec)

    def setup(self):
        """Create the display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 480)
        self.window_created = True

    def add_stroke_point(self, point_3d: np.ndarray, stroke_id: int) -> None:
        """
        Store a 3D point for visualization.

        Args:
            point_3d: 3D point as numpy array [X, Y, Z]
            stroke_id: ID of the stroke this point belongs to
        """
        if stroke_id not in self.strokes:
            self.strokes[stroke_id] = []
        self.strokes[stroke_id].append(point_3d.copy())

    def set_marker_pose(self, rvec: np.ndarray, tvec: np.ndarray) -> None:
        """
        Set current marker pose for projection.

        Args:
            rvec: Rotation vector (3,) from ArUco detection
            tvec: Translation vector (3,) from ArUco detection
        """
        self.current_marker_pose = (rvec.copy(), tvec.copy())

    def draw_strokes_2d(
        self,
        frame: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
        original_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Project 3D strokes onto 2D camera frame.

        Args:
            frame: Frame to draw on (may be resized for display)
            K: Camera intrinsic matrix
            D: Distortion coefficients
            original_size: (width, height) of original frame for coordinate scaling

        Returns:
            Frame with strokes drawn
        """
        if not self.strokes or self.current_marker_pose is None:
            return frame

        rvec, tvec = self.current_marker_pose
        frame = frame.copy()
        display_h, display_w = frame.shape[:2]

        # Scale factor if frame was resized
        if original_size is not None:
            orig_w, orig_h = original_size
            scale_x = display_w / orig_w
            scale_y = display_h / orig_h
        else:
            scale_x = 1.0
            scale_y = 1.0

        for stroke_id, points in self.strokes.items():
            if len(points) < 2:
                continue

            # Convert to numpy array for projectPoints
            points_3d = np.array(points, dtype=np.float32)

            # Project all points at once
            try:
                points_2d, _ = cv2.projectPoints(
                    points_3d,
                    rvec,
                    tvec,
                    K,
                    D
                )
                points_2d = points_2d.reshape(-1, 2)

                # Scale to display size
                points_2d[:, 0] *= scale_x
                points_2d[:, 1] *= scale_y

                # Convert to int for drawing
                points_2d = points_2d.astype(np.int32)

                # Draw polyline
                cv2.polylines(frame, [points_2d], False, (0, 0, 255), 3)

                # Draw small circles at each point
                for pt in points_2d:
                    cv2.circle(frame, tuple(pt), 4, (0, 0, 255), -1)
            except cv2.error:
                # Skip if projection fails
                pass

        return frame

    def clear_strokes(self) -> None:
        """Clear all stored strokes."""
        self.strokes.clear()

    def draw_hand_overlay(
        self,
        frame: np.ndarray,
        hand,
        color: Tuple[int, int, int] = (0, 255, 0),
        original_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Draw hand tracking overlay on frame.

        Args:
            frame: Frame to draw on (may be resized for display)
            hand: HandLandmarks object with pixel coordinates
            color: Color for the overlay
            original_size: (width, height) of original frame that hand was detected in.
                          If provided, coordinates are scaled to match display frame.
        """
        if hand is None:
            return frame

        frame = frame.copy()
        display_h, display_w = frame.shape[:2]

        # Hand landmarks are in PIXEL coordinates from original frame
        # Scale to display size if original size provided
        if original_size is not None:
            orig_w, orig_h = original_size
            scale_x = display_w / orig_w
            scale_y = display_h / orig_h
        else:
            # Assume landmarks are already in display coordinates
            scale_x = 1.0
            scale_y = 1.0

        # Draw index finger tip
        if hasattr(hand, 'index_finger_tip') and hand.index_finger_tip is not None:
            tip = hand.index_finger_tip
            # tip is in PIXEL coords, scale to display size
            x = int(tip[0] * scale_x)
            y = int(tip[1] * scale_y)

            # Clamp to frame bounds
            x = max(0, min(x, display_w - 1))
            y = max(0, min(y, display_h - 1))

            # Draw crosshair
            cv2.circle(frame, (x, y), 15, color, 2)
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.line(frame, (x - 20, y), (x + 20, y), color, 2)
            cv2.line(frame, (x, y - 20), (x, y + 20), color, 2)

            # Show confidence
            conf_text = f"{hand.confidence:.0%}"
            cv2.putText(frame, conf_text, (x + 20, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw all landmarks if available
        if hasattr(hand, 'landmarks') and hand.landmarks is not None:
            for i, lm in enumerate(hand.landmarks):
                x = int(lm[0] * scale_x)
                y = int(lm[1] * scale_y)
                # Clamp to frame bounds
                x = max(0, min(x, display_w - 1))
                y = max(0, min(y, display_h - 1))
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

        return frame

    def draw_marker_overlay(
        self,
        frame: np.ndarray,
        marker_visible: bool,
        marker_corners: Optional[np.ndarray] = None,
        camera_label: str = "MARKER"
    ) -> np.ndarray:
        """Draw ArUco marker detection overlay."""
        frame = frame.copy()

        if marker_visible and marker_corners is not None:
            # Draw marker outline
            corners = marker_corners.astype(int)
            cv2.polylines(frame, [corners], True, (0, 255, 255), 3)

            # Draw corner points
            for i, corner in enumerate(corners):
                color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][i]
                cv2.circle(frame, tuple(corner), 8, color, -1)

        # Status indicator
        status_color = (0, 255, 0) if marker_visible else (0, 0, 255)
        status_text = f"{camera_label}: OK" if marker_visible else f"{camera_label}: NOT FOUND"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return frame

    def draw_calibration_overlay(
        self,
        frame: np.ndarray,
        calibration_mode: bool,
        dynamic_calibrated: bool
    ) -> np.ndarray:
        """Draw calibration mode status overlay."""
        frame = frame.copy()
        h, w = frame.shape[:2]

        if calibration_mode:
            # Big calibration mode indicator
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 100), -1)
            cv2.putText(frame, "CALIBRATION MODE", (w // 2 - 150, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, "Point marker at BOTH cameras", (w // 2 - 180, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        elif dynamic_calibrated:
            # Small calibrated indicator
            cv2.putText(frame, "CALIBRATED", (w - 130, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def draw_status_overlay(
        self,
        frame: np.ndarray,
        label: str,
        tracking: bool,
        fps: float = 0.0
    ) -> np.ndarray:
        """Draw status information on frame."""
        frame = frame.copy()
        h, w = frame.shape[:2]

        # Label
        cv2.putText(frame, label, (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Tracking status
        tracking_color = (0, 255, 0) if tracking else (0, 0, 255)
        tracking_text = "HAND: TRACKING" if tracking else "HAND: LOST"
        cv2.putText(frame, tracking_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracking_color, 2)

        return frame

    def update(
        self,
        frame_glasses: Optional[np.ndarray],
        frame_webcam: Optional[np.ndarray],
        hand_glasses,
        hand_webcam,
        marker_visible_glasses: bool = False,
        marker_visible_webcam: bool = False,
        fps: float = 0.0,
        is_drawing: bool = False,
        calibration_mode: bool = False,
        dynamic_calibrated: bool = False,
        K_glasses: Optional[np.ndarray] = None,
        D_glasses: Optional[np.ndarray] = None
    ):
        """
        Update the display with current frames and tracking data.

        Args:
            frame_glasses: Frame from glasses camera
            frame_webcam: Frame from webcam
            hand_glasses: Hand detection from glasses
            hand_webcam: Hand detection from webcam
            marker_visible_glasses: Whether marker is visible in glasses
            marker_visible_webcam: Whether marker is visible in webcam
            fps: Current FPS
            is_drawing: Whether currently drawing
            calibration_mode: Whether in calibration mode
            dynamic_calibrated: Whether dynamic calibration is complete
            K_glasses: Camera intrinsic matrix for glasses (for stroke projection)
            D_glasses: Distortion coefficients for glasses (for stroke projection)

        Returns:
            False if window was closed (q/ESC pressed)
            "recalibrate" if 'c' key pressed
            True otherwise
        """
        if not self.window_created:
            self.setup()

        # Create placeholder if no frame
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Waiting for frames...", (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        # Process glasses frame
        if frame_glasses is not None:
            orig_h, orig_w = frame_glasses.shape[:2]
            display_glasses = cv2.resize(frame_glasses.copy(), (640, 480))
            display_glasses = self.draw_hand_overlay(
                display_glasses, hand_glasses, (0, 255, 0),
                original_size=(orig_w, orig_h)
            )
            display_glasses = self.draw_marker_overlay(
                display_glasses, marker_visible_glasses, camera_label="MARKER A"
            )
            # Draw 2D stroke projection on glasses frame
            if K_glasses is not None and D_glasses is not None:
                display_glasses = self.draw_strokes_2d(
                    display_glasses, K_glasses, D_glasses,
                    original_size=(orig_w, orig_h)
                )
            display_glasses = self.draw_status_overlay(
                display_glasses, "CAMERA A",
                hand_glasses is not None, fps
            )
            display_glasses = self.draw_calibration_overlay(
                display_glasses, calibration_mode, dynamic_calibrated
            )
        else:
            display_glasses = placeholder.copy()
            cv2.putText(display_glasses, "CAMERA A", (250, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            if calibration_mode:
                display_glasses = self.draw_calibration_overlay(
                    display_glasses, calibration_mode, dynamic_calibrated
                )

        # Process webcam frame
        if frame_webcam is not None:
            orig_h, orig_w = frame_webcam.shape[:2]
            display_webcam = cv2.resize(frame_webcam.copy(), (640, 480))
            display_webcam = self.draw_hand_overlay(
                display_webcam, hand_webcam, (255, 0, 255),
                original_size=(orig_w, orig_h)
            )
            display_webcam = self.draw_marker_overlay(
                display_webcam, marker_visible_webcam, camera_label="MARKER B"
            )
            display_webcam = self.draw_status_overlay(
                display_webcam, "CAMERA B",
                hand_webcam is not None, fps
            )
        else:
            display_webcam = placeholder.copy()
            cv2.putText(display_webcam, "CAMERA B", (250, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        # Drawing indicator
        if is_drawing:
            cv2.circle(display_glasses, (620, 20), 10, (0, 0, 255), -1)
            cv2.putText(display_glasses, "REC", (580, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Combine frames side by side
        combined = np.hstack([display_glasses, display_webcam])

        # Add point count display
        stroke_count = sum(len(pts) for pts in self.strokes.values())
        cv2.putText(combined, f"Points: {stroke_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Add help text at bottom
        help_text = "SPACE: Toggle Draw | c: Recalibrate | q: Quit"
        cv2.putText(combined, help_text, (10, combined.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Show
        cv2.imshow(self.window_name, combined)

        # Handle key events (1ms wait)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            return False
        if key == ord('c'):  # Recalibrate
            return "recalibrate"
        if key == ord(' '):  # SPACE - toggle drawing
            return "toggle_drawing"

        return True

    def cleanup(self):
        """Clean up windows."""
        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False
