"""WebSocket server for real-time AR air painting with local dual cameras.

This server uses two local cameras (Mac webcam + USB webcam) for stereo triangulation,
and sends 3D points to iPhone for AR visualization. The iPhone acts as a pure viewer.

Usage:
    python main.py --server-local --camera-a 0 --camera-b 1 --port 8765
"""

import asyncio
import time
import cv2
import numpy as np
from typing import Optional, Set
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    print("Please install websockets: pip install websockets>=12.0")
    raise

from .protocol import (
    DrawingToggleMessage, PointMessage,
    StrokeStartMessage, StrokeEndMessage, WorldAnchorMessage, StatusMessage,
    PingMessage, PongMessage, ErrorMessage, parse_message, serialize_message
)
from config.settings import CAMERA, TRACKING, ARUCO
from capture.sync import SynchronizedCapture
from calibration.stereo import load_calibration, StereoCalibration
from tracking.hands import HandTracker
from tracking.triangulate import StereoTriangulator, PointSmoother
from tracking.aruco import ArucoDetector, MarkerPose
from ui.input import HIDListener
from .visualizer import ServerVisualizer


@dataclass
class LocalServerState:
    """Mutable state for the local camera server."""
    # Two-phase state machine
    # Phase 1: "setup" - waiting for ArUco + HID click to establish world center
    # Phase 2: "drawing" - HID clicks toggle drawing
    phase: str = "setup"

    # Drawing state
    is_drawing: bool = False
    current_stroke_id: int = 0
    total_strokes: int = 0
    points_in_current_stroke: int = 0

    # Tracking state
    hand_tracked: bool = False
    marker_visible_a: bool = False
    marker_visible_b: bool = False
    marker_pose: Optional[MarkerPose] = None

    # World coordinate state - locked transform from ArUco
    world_locked: bool = False
    locked_T_CW: Optional[np.ndarray] = None  # Camera-to-World transform when locked

    # Performance metrics
    fps: float = 0.0
    frame_times: list = field(default_factory=list)

    # Last known positions
    last_point_3d: Optional[np.ndarray] = None


class LocalCameraServer:
    """
    WebSocket server using dual local cameras for stereo triangulation.

    Architecture:
    - Camera A: Mac built-in webcam (or USB webcam)
    - Camera B: External USB webcam
    - iPhone: Pure AR viewer (receives 3D points via WebSocket)

    The cameras are fixed, so we use the standard StereoTriangulator
    (not the dynamic triangulator that handles moving cameras).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        camera_a_index: int = 0,
        camera_b_index: int = 1,
        show_visualization: bool = True,
    ):
        """
        Initialize the server.

        Args:
            host: Host to bind to
            port: Port to listen on
            camera_a_index: Index of camera A (primary)
            camera_b_index: Index of camera B (secondary)
            show_visualization: Whether to show camera feeds in OpenCV window
        """
        self.host = host
        self.port = port
        self.camera_a_index = camera_a_index
        self.camera_b_index = camera_b_index
        self.show_visualization = show_visualization

        # Components (initialized in setup)
        self.calibration: Optional[StereoCalibration] = None
        self.triangulator: Optional[StereoTriangulator] = None
        self.point_smoother: Optional[PointSmoother] = None
        self.aruco_detector_a: Optional[ArucoDetector] = None
        self.aruco_detector_b: Optional[ArucoDetector] = None
        self.hand_tracker_a: Optional[HandTracker] = None
        self.hand_tracker_b: Optional[HandTracker] = None
        self.capture: Optional[SynchronizedCapture] = None
        self.visualizer: Optional[ServerVisualizer] = None
        self.hid_listener: Optional[HIDListener] = None

        # State
        self.state = LocalServerState()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._clients: Set[WebSocketServerProtocol] = set()
        self._running = False

        # Last frames for visualization
        self._last_frame_a: Optional[np.ndarray] = None
        self._last_frame_b: Optional[np.ndarray] = None
        self._last_hand_a = None
        self._last_hand_b = None

        # Pending actions from sync handlers (processed in async frame loop)
        self._pending_world_anchor = False
        self._pending_stroke_start = False
        self._pending_stroke_end = False

    def setup(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if setup successful
        """
        print("=" * 60)
        print("Air Paint Local Camera Server")
        print("=" * 60)

        # Load calibration
        print("\nLoading calibration...")
        self.calibration = load_calibration()

        if self.calibration is None:
            print("ERROR: No calibration found!")
            print("Run calibration first: python main.py --calibrate")
            return False

        self.triangulator = StereoTriangulator(self.calibration)
        self.point_smoother = PointSmoother(alpha=TRACKING.SMOOTHING_ALPHA)
        print(f"Calibration loaded (RMS: {self.calibration.rms_error:.4f})")

        # Initialize ArUco detectors for both cameras
        if ARUCO.ENABLED:
            print(f"\nInitializing ArUco detectors (marker ID: {ARUCO.MARKER_ID})...")
            self.aruco_detector_a = ArucoDetector(
                marker_size_mm=ARUCO.MARKER_SIZE_MM,
                camera_matrix=self.calibration.K1,
                dist_coeffs=self.calibration.D1,
                target_marker_id=ARUCO.MARKER_ID
            )
            self.aruco_detector_b = ArucoDetector(
                marker_size_mm=ARUCO.MARKER_SIZE_MM,
                camera_matrix=self.calibration.K2,
                dist_coeffs=self.calibration.D2,
                target_marker_id=ARUCO.MARKER_ID
            )
            print("ArUco detectors ready (both cameras)")

        # Initialize hand trackers
        print("\nInitializing hand tracking...")
        self.hand_tracker_a = HandTracker()
        self.hand_tracker_b = HandTracker()
        print("Hand trackers initialized")

        # Open dual cameras
        print(f"\nOpening cameras (A={self.camera_a_index}, B={self.camera_b_index})...")
        self.capture = SynchronizedCapture(
            camera_a_index=self.camera_a_index,
            camera_b_index=self.camera_b_index
        )

        if not self.capture.open():
            print("ERROR: Cannot open cameras")
            return False

        print(f"Cameras opened: {self.capture.resolution}")

        # Initialize visualizer
        if self.show_visualization:
            print("\nInitializing visualizer...")
            self.visualizer = ServerVisualizer()
            self.visualizer.setup()
            print("Visualizer ready (press 'q' to quit)")

        # Initialize HID listener for Bluetooth shutter button
        print("\nInitializing HID listener...")
        self.hid_listener = HIDListener(on_click=self._on_hid_button)
        self.hid_listener.start()

        print("\n" + "=" * 60)
        print("Setup complete!")
        print("=" * 60)
        print("\nFLOW:")
        print("  1. Show ArUco marker to camera A")
        print("  2. Press SPACE/HID to lock world origin")
        print("  3. Remove ArUco marker")
        print("  4. Press SPACE/HID to toggle drawing on/off")
        print("\nControls:")
        print("  SPACE or HID button: Lock origin / Toggle drawing")
        print("  Q/ESC: Quit")
        print("=" * 60)
        print("\n>>> Show ArUco marker and press SPACE to lock origin <<<\n")

        return True

    def _on_hid_button(self) -> None:
        """
        Called when Bluetooth HID button (Volume Up) or SPACE is pressed.

        Two-phase behavior:
        1. Setup phase: If ArUco visible, lock world center and switch to drawing phase
        2. Drawing phase: Toggle drawing on/off
        """
        if self.state.phase == "setup":
            # In setup phase - try to lock world center
            if self.state.marker_pose is not None:
                # ArUco is visible - lock world center
                self.state.world_locked = True
                self.state.locked_T_CW = self.state.marker_pose.T_CW.copy()
                self.state.phase = "drawing"
                self._pending_world_anchor = True  # Send to iPhone
                print("=" * 60)
                print("WORLD ORIGIN LOCKED!")
                print("You can now remove the ArUco marker.")
                print("Press SPACE/HID to toggle drawing on/off.")
                print("=" * 60)
            else:
                print("ArUco marker not visible - show marker and press again")
        else:
            # In drawing phase - toggle drawing
            self._toggle_drawing()

    def cleanup(self) -> None:
        """Release all resources."""
        print("\nCleaning up...")

        if self._executor:
            self._executor.shutdown(wait=False)

        if self.capture:
            self.capture.release()

        if self.hand_tracker_a:
            self.hand_tracker_a.close()

        if self.hand_tracker_b:
            self.hand_tracker_b.close()

        if self.visualizer:
            self.visualizer.cleanup()

        if self.hid_listener:
            self.hid_listener.stop()

        cv2.destroyAllWindows()
        print("Cleanup complete")

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a connected client."""
        client_addr = websocket.remote_address
        print(f"Client connected: {client_addr}")
        self._clients.add(websocket)

        try:
            # Send initial status
            status_msg = StatusMessage(
                tracking=False,
                drawing=self.state.is_drawing,
                marker_visible=self.state.marker_visible_a,
                fps=0.0,
                latency_ms=0.0,
                stroke_id=self.state.current_stroke_id,
                total_strokes=self.state.total_strokes,
            )
            await websocket.send(serialize_message(status_msg))
            print(f"Sent initial status to {client_addr}")

            async for raw_message in websocket:
                await self._process_message(websocket, raw_message)

        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {client_addr}")
        finally:
            self._clients.discard(websocket)

    async def _process_message(
        self,
        websocket: WebSocketServerProtocol,
        raw_message: str
    ) -> None:
        """Process an incoming WebSocket message."""
        message = parse_message(raw_message)

        if message is None:
            await self._send_error(websocket, "PARSE_ERROR", "Failed to parse message")
            return

        if isinstance(message, DrawingToggleMessage):
            await self._handle_drawing_toggle(websocket)
        elif isinstance(message, PingMessage):
            await self._handle_ping(websocket, message)

    async def _handle_drawing_toggle(self, websocket: WebSocketServerProtocol) -> None:
        """Handle drawing toggle from client."""
        await self._toggle_drawing_and_notify()

    async def _handle_ping(
        self,
        websocket: WebSocketServerProtocol,
        message: PingMessage
    ) -> None:
        """Handle ping for latency measurement."""
        pong = PongMessage(ping_timestamp=message.timestamp)
        await websocket.send(serialize_message(pong))

    async def _send_error(
        self,
        websocket: WebSocketServerProtocol,
        code: str,
        msg: str
    ) -> None:
        """Send an error message."""
        error = ErrorMessage(code=code, message=msg)
        await websocket.send(serialize_message(error))

    def _toggle_drawing(self) -> None:
        """Toggle drawing state (synchronous, from keyboard/HID)."""
        self.state.is_drawing = not self.state.is_drawing

        if self.state.is_drawing:
            self.state.current_stroke_id += 1
            self.state.points_in_current_stroke = 0
            if self.point_smoother:
                self.point_smoother.reset()
            self._pending_stroke_start = True  # Send to iPhone
            print(f"Drawing ON - Stroke #{self.state.current_stroke_id}")
        else:
            self.state.total_strokes += 1
            self._pending_stroke_end = True  # Send to iPhone
            print(f"Drawing OFF - Stroke #{self.state.current_stroke_id} "
                  f"({self.state.points_in_current_stroke} points)")

    async def _toggle_drawing_and_notify(self) -> None:
        """Toggle drawing state and notify all clients."""
        self._toggle_drawing()
        await self._broadcast_stroke_state()

    async def _broadcast_stroke_state(self) -> None:
        """Broadcast stroke start/end to all clients."""
        if self.state.is_drawing:
            msg = StrokeStartMessage(
                stroke_id=self.state.current_stroke_id,
                color=[1.0, 0.0, 0.0],  # Red
            )
        else:
            msg = StrokeEndMessage(
                stroke_id=self.state.current_stroke_id,
                point_count=self.state.points_in_current_stroke,
            )
        await self._broadcast(serialize_message(msg))

    async def _broadcast(self, message: str) -> None:
        """Broadcast a message to all connected clients."""
        if self._clients:
            await asyncio.gather(
                *[client.send(message) for client in self._clients],
                return_exceptions=True
            )

    def _update_fps(self, frame_time: float) -> None:
        """Update FPS calculation."""
        self.state.frame_times.append(frame_time)

        if len(self.state.frame_times) > 30:
            self.state.frame_times = self.state.frame_times[-30:]

        if len(self.state.frame_times) >= 2:
            elapsed = self.state.frame_times[-1] - self.state.frame_times[0]
            if elapsed > 0:
                self.state.fps = (len(self.state.frame_times) - 1) / elapsed

    async def _process_frame(self) -> None:
        """Capture and process a stereo frame pair."""
        frame_time = time.time()

        # Capture from both cameras
        stereo_frame = self.capture.read()
        if stereo_frame is None:
            return

        frame_a, frame_b = stereo_frame.images

        # Detect ArUco markers
        marker_pose_a = None
        marker_pose_b = None

        if self.aruco_detector_a is not None:
            marker_pose_a = self.aruco_detector_a.detect(frame_a)
        if self.aruco_detector_b is not None:
            marker_pose_b = self.aruco_detector_b.detect(frame_b)

        self.state.marker_visible_a = marker_pose_a is not None
        self.state.marker_visible_b = marker_pose_b is not None
        self.state.marker_pose = marker_pose_a

        # Update visualizer marker pose for 2D stroke projection (only in setup phase)
        if marker_pose_a is not None and self.visualizer is not None and self.state.phase == "setup":
            self.visualizer.set_marker_pose(marker_pose_a.rvec, marker_pose_a.tvec)

        # Handle pending messages from sync HID handler
        if self._pending_world_anchor and self.state.locked_T_CW is not None:
            # Send locked world anchor to iPhone
            # T_WC is the inverse of T_CW (World-to-Camera)
            T_WC = np.linalg.inv(self.state.locked_T_CW)
            anchor_msg = WorldAnchorMessage.from_matrix(
                T_WC,
                marker_id=ARUCO.MARKER_ID,
                marker_size_mm=ARUCO.MARKER_SIZE_MM,
                visible=True
            )
            await self._broadcast(serialize_message(anchor_msg))
            self._pending_world_anchor = False

        if self._pending_stroke_start:
            msg = StrokeStartMessage(
                stroke_id=self.state.current_stroke_id,
                color=[1.0, 0.0, 0.0],  # Red
            )
            await self._broadcast(serialize_message(msg))
            self._pending_stroke_start = False

        if self._pending_stroke_end:
            msg = StrokeEndMessage(
                stroke_id=self.state.current_stroke_id,
                point_count=self.state.points_in_current_stroke,
            )
            await self._broadcast(serialize_message(msg))
            self._pending_stroke_end = False

        # Run hand detection on both frames (parallel)
        loop = asyncio.get_event_loop()

        future_a = loop.run_in_executor(
            self._executor,
            self.hand_tracker_a.detect,
            frame_a
        )
        future_b = loop.run_in_executor(
            self._executor,
            self.hand_tracker_b.detect,
            frame_b
        )

        hands_a, hands_b = await asyncio.gather(future_a, future_b)

        hand_a = hands_a[0] if hands_a else None
        hand_b = hands_b[0] if hands_b else None

        self.state.hand_tracked = hand_a is not None and hand_b is not None

        # Triangulate if we have both hands
        if self.state.hand_tracked and self.triangulator is not None:
            # Confidence gate
            if (hand_a.confidence >= TRACKING.MIN_POINT_CONFIDENCE and
                hand_b.confidence >= TRACKING.MIN_POINT_CONFIDENCE):

                point_a = hand_a.index_finger_tip
                point_b = hand_b.index_finger_tip

                # Triangulate using stereo calibration
                point_3d = self.triangulator.triangulate(point_a, point_b)

                if point_3d is not None:
                    # Transform to world coordinates using LOCKED transform
                    # (not the current marker pose, since marker may be removed)
                    if self.state.world_locked and self.state.locked_T_CW is not None:
                        # Apply locked camera-to-world transform
                        point_h = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
                        point_world_h = self.state.locked_T_CW @ point_h
                        point_3d = point_world_h[:3]

                    print(f"[POINT] 3D: {point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f} mm")

                    # Apply smoothing
                    if self.point_smoother is not None:
                        point_3d = self.point_smoother.smooth(point_3d)

                    self.state.last_point_3d = point_3d

                    # Send point if in drawing phase AND drawing is ON
                    if self.state.phase == "drawing" and self.state.is_drawing:
                        error_a, error_b = self.triangulator.compute_reprojection_error(
                            point_a, point_b, point_3d
                        )
                        max_error = max(error_a, error_b)
                        confidence = max(0.0, 1.0 - (max_error / TRACKING.MAX_REPROJECTION_ERROR))

                        point_msg = PointMessage.from_mm(
                            point_3d[0], point_3d[1], point_3d[2],
                            stroke_id=self.state.current_stroke_id,
                            confidence=confidence
                        )
                        await self._broadcast(serialize_message(point_msg))
                        self.state.points_in_current_stroke += 1

                        # Add point to visualizer for 2D stroke display
                        if self.visualizer is not None:
                            self.visualizer.add_stroke_point(
                                point_3d, self.state.current_stroke_id
                            )

        # Update FPS
        self._update_fps(frame_time)

        # Periodic status update
        if len(self.state.frame_times) % 30 == 0:
            status_msg = StatusMessage(
                tracking=self.state.hand_tracked,
                drawing=self.state.is_drawing,
                marker_visible=self.state.marker_visible_a,
                fps=self.state.fps,
                latency_ms=0.0,
                stroke_id=self.state.current_stroke_id,
                total_strokes=self.state.total_strokes,
            )
            await self._broadcast(serialize_message(status_msg))

        # Store frames for visualization
        self._last_frame_a = frame_a
        self._last_frame_b = frame_b
        self._last_hand_a = hand_a
        self._last_hand_b = hand_b

    def _update_visualization(self) -> bool:
        """Update the visualization window. Returns False if window closed."""
        if not self.visualizer:
            return True

        result = self.visualizer.update(
            frame_glasses=self._last_frame_a,
            frame_webcam=self._last_frame_b,
            hand_glasses=self._last_hand_a,
            hand_webcam=self._last_hand_b,
            marker_visible_glasses=self.state.marker_visible_a,
            marker_visible_webcam=self.state.marker_visible_b,
            fps=self.state.fps,
            is_drawing=self.state.is_drawing,
            calibration_mode=(self.state.phase == "setup"),  # Show "point ArUco" message
            dynamic_calibrated=self.state.world_locked,  # Show "CALIBRATED" when locked
            K_glasses=self.calibration.K1 if self.calibration else None,
            D_glasses=self.calibration.D1 if self.calibration else None
        )

        if result == "toggle_drawing":
            self._on_hid_button()  # Use two-phase logic
            return True

        return result if isinstance(result, bool) else True

    async def run(self) -> None:
        """Run the WebSocket server with frame processing loop."""
        self._running = True

        print(f"\nStarting WebSocket server on ws://{self.host}:{self.port}")
        print("Waiting for iPhone connection...")

        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=60,
        ) as server:
            print(f"Server listening on port {self.port}")

            while self._running:
                # Process frame from local cameras
                await self._process_frame()

                # Update visualization on main thread (required for macOS OpenCV)
                if self.visualizer:
                    should_continue = self._update_visualization()
                    if not should_continue:
                        print("\nVisualization window closed")
                        break

                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.001)

    def stop(self) -> None:
        """Stop the server."""
        self._running = False


async def main(port: int = 8765, camera_a: int = 0, camera_b: int = 1) -> None:
    """Main entry point for standalone server."""
    server = LocalCameraServer(
        port=port,
        camera_a_index=camera_a,
        camera_b_index=camera_b
    )

    if not server.setup():
        print("Setup failed!")
        return

    try:
        await server.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        server.stop()
        server.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Air Paint Local Camera Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--camera-a", type=int, default=0, help="Camera A index")
    parser.add_argument("--camera-b", type=int, default=1, help="Camera B index")

    args = parser.parse_args()

    asyncio.run(main(port=args.port, camera_a=args.camera_a, camera_b=args.camera_b))
