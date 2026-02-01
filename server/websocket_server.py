"""WebSocket server for real-time AR air painting.

This server receives camera frames from an iPhone (which gets them from Meta glasses),
runs MediaPipe hand detection on both the glasses frame and local Mac webcam,
triangulates 3D points using stereo vision, and sends the points back to the iPhone
for AR visualization.

Usage:
    python -m server.websocket_server [--port 8765] [--camera 0]

Or via main.py:
    python main.py --server
"""

import asyncio
import json
import time
import cv2
import numpy as np
from typing import Optional, Set, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    print("Please install websockets: pip install websockets>=12.0")
    raise

from .protocol import (
    MessageType, FrameMessage, DrawingToggleMessage, PointMessage,
    StrokeStartMessage, StrokeEndMessage, WorldAnchorMessage, StatusMessage,
    PingMessage, PongMessage, ErrorMessage, parse_message, serialize_message
)
from config.settings import CAMERA, TRACKING, ARUCO, get_calibration_path
from calibration.stereo import load_calibration, StereoCalibration
from tracking.hands import HandTracker
from tracking.triangulate import StereoTriangulator, PointSmoother
from tracking.dynamic_triangulate import DynamicStereoTriangulator
from tracking.aruco import ArucoDetector, MarkerPose
from ui.input import HIDListener
from .visualizer import ServerVisualizer


@dataclass
class ServerState:
    """Mutable state for the WebSocket server."""
    # Drawing state
    is_drawing: bool = False
    current_stroke_id: int = 0
    total_strokes: int = 0
    points_in_current_stroke: int = 0

    # Tracking state
    hand_tracked: bool = False
    marker_visible: bool = False
    marker_visible_webcam: bool = False  # Webcam marker visibility
    marker_pose: Optional[MarkerPose] = None
    marker_pose_webcam: Optional[MarkerPose] = None  # Webcam marker pose

    # Calibration state (for dynamic triangulation)
    calibration_mode: bool = True  # Start in calibration mode
    dynamic_calibrated: bool = False

    # Performance metrics
    fps: float = 0.0
    frame_times: list = field(default_factory=list)
    last_frame_time: float = 0.0

    # Last known positions (for smoothing)
    last_point_3d: Optional[np.ndarray] = None


class AirPaintServer:
    """
    WebSocket server for real-time AR air painting.

    Architecture:
    - Receives JPEG frames from iPhone (originally from Meta glasses)
    - Captures frames from Mac webcam locally
    - Runs MediaPipe hand detection on both feeds
    - Triangulates 3D points using calibrated stereo
    - Sends points back to iPhone for AR rendering

    The glasses are treated as "Camera A" and Mac webcam as "Camera B"
    in the stereo calibration.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        camera_index: int = 0,  # Mac webcam index
        show_visualization: bool = True,  # Show OpenCV windows
    ):
        """
        Initialize the server.

        Args:
            host: Host to bind to
            port: Port to listen on
            camera_index: Index of Mac webcam (Camera B in stereo pair)
            show_visualization: Whether to show camera feeds in OpenCV window
        """
        self.host = host
        self.port = port
        self.camera_index = camera_index
        self.show_visualization = show_visualization

        # Components (initialized in setup)
        self.calibration: Optional[StereoCalibration] = None
        self.triangulator: Optional[StereoTriangulator] = None
        self.dynamic_triangulator: Optional[DynamicStereoTriangulator] = None
        self.point_smoother: Optional[PointSmoother] = None
        self.aruco_detector: Optional[ArucoDetector] = None
        self.hand_tracker_a: Optional[HandTracker] = None  # For glasses frame
        self.hand_tracker_b: Optional[HandTracker] = None  # For Mac webcam
        self.webcam: Optional[cv2.VideoCapture] = None
        self.visualizer: Optional[ServerVisualizer] = None
        self.aruco_detector_b: Optional[ArucoDetector] = None  # For Mac webcam
        self.hid_listener: Optional[HIDListener] = None  # For Bluetooth HID button

        # State
        self.state = ServerState()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._clients: Set[WebSocketServerProtocol] = set()
        self._running = False

        # Last frames for visualization (updated in frame handler)
        self._last_frame_glasses: Optional[np.ndarray] = None
        self._last_frame_webcam: Optional[np.ndarray] = None
        self._last_hand_glasses = None
        self._last_hand_webcam = None

        # Callbacks for UI updates (optional)
        self._on_frame_callback: Optional[Callable] = None
        self._on_point_callback: Optional[Callable] = None

    def setup(self) -> bool:
        """
        Initialize all components.

        Returns:
            True if setup successful
        """
        print("=" * 60)
        print("Air Paint WebSocket Server")
        print("=" * 60)

        # Load calibration
        print("\nLoading calibration...")
        self.calibration = load_calibration()

        if self.calibration is None:
            print("ERROR: No calibration found!")
            print("Run calibration first: python main.py --calibrate")
            return False

        self.triangulator = StereoTriangulator(self.calibration)
        self.dynamic_triangulator = DynamicStereoTriangulator(
            K1=self.calibration.K1,
            D1=self.calibration.D1,
            K2=self.calibration.K2,
            D2=self.calibration.D2
        )
        self.point_smoother = PointSmoother(alpha=TRACKING.SMOOTHING_ALPHA)
        print(f"Calibration loaded (RMS: {self.calibration.rms_error:.4f})")
        print("Dynamic triangulator initialized (calibration mode)")

        # Initialize ArUco detectors for both cameras
        if ARUCO.ENABLED:
            print(f"\nInitializing ArUco detectors (marker ID: {ARUCO.MARKER_ID})...")
            # Detector for glasses (Camera A)
            self.aruco_detector = ArucoDetector(
                marker_size_mm=ARUCO.MARKER_SIZE_MM,
                camera_matrix=self.calibration.K1,
                dist_coeffs=self.calibration.D1,
                target_marker_id=ARUCO.MARKER_ID
            )
            # Detector for webcam (Camera B)
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

        # Open Mac webcam
        print(f"\nOpening webcam (index {self.camera_index})...")
        self.webcam = cv2.VideoCapture(self.camera_index)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.WIDTH)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.HEIGHT)
        self.webcam.set(cv2.CAP_PROP_FPS, CAMERA.FPS)

        if not self.webcam.isOpened():
            print(f"ERROR: Cannot open webcam at index {self.camera_index}")
            return False

        actual_w = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam opened: {actual_w}x{actual_h}")

        # Initialize visualizer
        if self.show_visualization:
            print("\nInitializing visualizer...")
            self.visualizer = ServerVisualizer()
            self.visualizer.setup()
            print("Visualizer ready (press 'q' to quit)")

        # Initialize HID listener for Bluetooth shutter button (Volume Up)
        print("\nInitializing HID listener...")
        self.hid_listener = HIDListener(on_click=self._on_hid_button)
        self.hid_listener.start()

        print("\nSetup complete!")
        print("=" * 60)
        print("Press Bluetooth HID button (Volume Up) to toggle drawing")
        print("=" * 60)
        return True

    def _on_hid_button(self) -> None:
        """Called when Bluetooth HID button (Volume Up) is pressed."""
        self._toggle_drawing_local()

    def cleanup(self) -> None:
        """Release all resources."""
        print("\nCleaning up...")

        if self._executor:
            self._executor.shutdown(wait=False)

        if self.webcam:
            self.webcam.release()

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
            # Send initial status message so client knows we're ready
            status_msg = StatusMessage(
                tracking=False,
                drawing=self.state.is_drawing,
                marker_visible=self.state.marker_visible,
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

        if isinstance(message, FrameMessage):
            await self._handle_frame(websocket, message)
        elif isinstance(message, DrawingToggleMessage):
            await self._handle_drawing_toggle(websocket, message)
        elif isinstance(message, PingMessage):
            await self._handle_ping(websocket, message)

    async def _handle_frame(
        self,
        websocket: WebSocketServerProtocol,
        message: FrameMessage
    ) -> None:
        """Handle an incoming frame from Meta glasses (via iPhone)."""
        frame_receive_time = time.time()

        # Decode JPEG from glasses
        try:
            jpeg_bytes = message.decode_image()
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame_glasses = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame_glasses is None:
                return
        except Exception as e:
            print(f"Frame decode error: {e}")
            return

        # Capture frame from Mac webcam
        ret, frame_webcam = self.webcam.read()
        if not ret or frame_webcam is None:
            return

        # Detect ArUco marker in both cameras
        marker_pose_a = None
        marker_pose_b = None

        if self.aruco_detector is not None:
            marker_pose_a = self.aruco_detector.detect(frame_glasses)

        if self.aruco_detector_b is not None:
            marker_pose_b = self.aruco_detector_b.detect(frame_webcam)

        # Update state
        self.state.marker_visible = marker_pose_a is not None
        self.state.marker_visible_webcam = marker_pose_b is not None
        self.state.marker_pose = marker_pose_a
        self.state.marker_pose_webcam = marker_pose_b

        # Update visualizer marker pose for 2D stroke projection
        if marker_pose_a is not None and self.visualizer is not None:
            self.visualizer.set_marker_pose(marker_pose_a.rvec, marker_pose_a.tvec)

        if marker_pose_a is not None:
            # Send world anchor update (from glasses perspective)
            anchor_msg = WorldAnchorMessage.from_matrix(
                marker_pose_a.T_WC,  # World to Camera transform
                marker_id=marker_pose_a.marker_id,
                marker_size_mm=ARUCO.MARKER_SIZE_MM,
                visible=True
            )
            await websocket.send(serialize_message(anchor_msg))

        # Dynamic stereo calibration (two-phase)
        if self.dynamic_triangulator is not None:
            if self.state.calibration_mode:
                # CALIBRATION PHASE: Need both cameras to see marker
                if marker_pose_a is not None and marker_pose_b is not None:
                    self.dynamic_triangulator.calibrate_webcam(marker_pose_b)
                    self.state.calibration_mode = False
                    self.state.dynamic_calibrated = True
                    print("=" * 40)
                    print("CALIBRATION COMPLETE!")
                    print("Glasses can now move freely.")
                    print("Only glasses needs to see the marker.")
                    print("Press 'c' to recalibrate if needed.")
                    print("=" * 40)
            else:
                # OPERATION PHASE: Only glasses needs to see marker
                if marker_pose_a is not None:
                    self.dynamic_triangulator.update_from_glasses_pose(marker_pose_a)

        # Run hand detection on both frames (parallel)
        loop = asyncio.get_event_loop()

        future_a = loop.run_in_executor(
            self._executor,
            self.hand_tracker_a.detect,
            frame_glasses
        )
        future_b = loop.run_in_executor(
            self._executor,
            self.hand_tracker_b.detect,
            frame_webcam
        )

        hands_a, hands_b = await asyncio.gather(future_a, future_b)

        hand_a = hands_a[0] if hands_a else None
        hand_b = hands_b[0] if hands_b else None

        self.state.hand_tracked = hand_a is not None and hand_b is not None

        # Triangulate if we have both hands and dynamic calibration is complete
        can_triangulate = (
            self.state.hand_tracked and
            self.state.dynamic_calibrated and
            not self.state.calibration_mode and
            self.dynamic_triangulator is not None and
            self.dynamic_triangulator.P2 is not None  # Have valid projection matrix
        )

        if can_triangulate:
            # Confidence gate
            if (hand_a.confidence >= TRACKING.MIN_POINT_CONFIDENCE and
                hand_b.confidence >= TRACKING.MIN_POINT_CONFIDENCE):

                point_a = hand_a.index_finger_tip
                point_b = hand_b.index_finger_tip

                # Triangulate using dynamic triangulator
                point_3d = self.dynamic_triangulator.triangulate(point_a, point_b)

                if point_3d is not None:
                    print(f"[POINT] 3D: {point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f} mm")
                else:
                    print(f"[POINT] Triangulation rejected (reprojection error)")

                if point_3d is not None:
                    # Transform to world coordinates if marker visible
                    if self.state.marker_pose is not None and self.aruco_detector is not None:
                        point_3d = self.aruco_detector.transform_point_to_world(
                            point_3d, self.state.marker_pose
                        )

                    # Apply smoothing
                    if self.point_smoother is not None:
                        point_3d = self.point_smoother.smooth(point_3d)

                    self.state.last_point_3d = point_3d

                    # Send point if drawing
                    if self.state.is_drawing:
                        # Calculate confidence based on reprojection error
                        error_a, error_b = self.dynamic_triangulator.compute_reprojection_error(
                            point_a, point_b, point_3d
                        )
                        max_error = max(error_a, error_b)
                        confidence = max(0.0, 1.0 - (max_error / TRACKING.MAX_REPROJECTION_ERROR))

                        point_msg = PointMessage.from_mm(
                            point_3d[0], point_3d[1], point_3d[2],
                            stroke_id=self.state.current_stroke_id,
                            confidence=confidence
                        )
                        await websocket.send(serialize_message(point_msg))
                        self.state.points_in_current_stroke += 1

                        # Add point to visualizer for 2D stroke display
                        if self.visualizer is not None:
                            self.visualizer.add_stroke_point(
                                point_3d, self.state.current_stroke_id
                            )

                        # Callback for local visualization
                        if self._on_point_callback:
                            self._on_point_callback(point_3d)

        # Update FPS
        self._update_fps(frame_receive_time)

        # Send periodic status update (every ~30 frames)
        if len(self.state.frame_times) % 30 == 0:
            status_msg = StatusMessage(
                tracking=self.state.hand_tracked,
                drawing=self.state.is_drawing,
                marker_visible=self.state.marker_visible,
                fps=self.state.fps,
                latency_ms=(time.time() - message.timestamp) * 1000,
                stroke_id=self.state.current_stroke_id,
                total_strokes=self.state.total_strokes,
            )
            await websocket.send(serialize_message(status_msg))

        # Store frames for visualization
        self._last_frame_glasses = frame_glasses
        self._last_frame_webcam = frame_webcam
        self._last_hand_glasses = hand_a
        self._last_hand_webcam = hand_b

        # Callback for frame visualization
        if self._on_frame_callback:
            self._on_frame_callback(frame_glasses, frame_webcam, hand_a, hand_b)

    async def _handle_drawing_toggle(
        self,
        websocket: WebSocketServerProtocol,
        message: DrawingToggleMessage
    ) -> None:
        """Handle drawing toggle from Bluetooth button."""
        was_drawing = self.state.is_drawing
        self.state.is_drawing = not self.state.is_drawing

        if self.state.is_drawing:
            # Starting new stroke
            self.state.current_stroke_id += 1
            self.state.points_in_current_stroke = 0

            # Reset smoother for new stroke
            if self.point_smoother:
                self.point_smoother.reset()

            stroke_msg = StrokeStartMessage(
                stroke_id=self.state.current_stroke_id,
                color=[1.0, 0.0, 0.0],  # Red
            )
            await websocket.send(serialize_message(stroke_msg))
            print(f"Drawing ON - Stroke #{self.state.current_stroke_id}")

        else:
            # Ending stroke
            self.state.total_strokes += 1

            stroke_msg = StrokeEndMessage(
                stroke_id=self.state.current_stroke_id,
                point_count=self.state.points_in_current_stroke,
            )
            await websocket.send(serialize_message(stroke_msg))
            print(f"Drawing OFF - Stroke #{self.state.current_stroke_id} ({self.state.points_in_current_stroke} points)")

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

    def _update_fps(self, frame_time: float) -> None:
        """Update FPS calculation."""
        self.state.frame_times.append(frame_time)

        # Keep only last 30 frames
        if len(self.state.frame_times) > 30:
            self.state.frame_times = self.state.frame_times[-30:]

        if len(self.state.frame_times) >= 2:
            elapsed = self.state.frame_times[-1] - self.state.frame_times[0]
            if elapsed > 0:
                self.state.fps = (len(self.state.frame_times) - 1) / elapsed

    async def _broadcast(self, message: str) -> None:
        """Broadcast a message to all connected clients."""
        if self._clients:
            await asyncio.gather(
                *[client.send(message) for client in self._clients],
                return_exceptions=True
            )

    async def run(self) -> None:
        """Run the WebSocket server."""
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

            # Keep running until stopped
            while self._running:
                # Update visualization on main thread (required for macOS OpenCV)
                # Do NOT use run_in_executor - cv2.waitKey() must run on main thread
                if self.visualizer:
                    should_continue = self._update_visualization()
                    if not should_continue:
                        print("\nVisualization window closed")
                        break

                await asyncio.sleep(0.033)  # ~30 FPS for visualization

    def stop(self) -> None:
        """Stop the server."""
        self._running = False

    def _update_visualization(self) -> bool:
        """Update the visualization window. Returns False if window closed."""
        if not self.visualizer:
            return True

        result = self.visualizer.update(
            frame_glasses=self._last_frame_glasses,
            frame_webcam=self._last_frame_webcam,
            hand_glasses=self._last_hand_glasses,
            hand_webcam=self._last_hand_webcam,
            marker_visible_glasses=self.state.marker_visible,
            marker_visible_webcam=self.state.marker_visible_webcam,
            fps=self.state.fps,
            is_drawing=self.state.is_drawing,
            calibration_mode=self.state.calibration_mode,
            dynamic_calibrated=self.state.dynamic_calibrated,
            K_glasses=self.calibration.K1 if self.calibration else None,
            D_glasses=self.calibration.D1 if self.calibration else None
        )

        # Check for keyboard commands
        if result == "recalibrate":
            self._enter_calibration_mode()
            return True
        if result == "toggle_drawing":
            self._toggle_drawing_local()
            return True

        return result if isinstance(result, bool) else True

    def _toggle_drawing_local(self) -> None:
        """Toggle drawing state from keyboard (SPACE key)."""
        self.state.is_drawing = not self.state.is_drawing

        if self.state.is_drawing:
            self.state.current_stroke_id += 1
            self.state.points_in_current_stroke = 0
            if self.point_smoother:
                self.point_smoother.reset()
            print(f"Drawing ON - Stroke #{self.state.current_stroke_id} (keyboard)")
        else:
            self.state.total_strokes += 1
            print(f"Drawing OFF - Stroke #{self.state.current_stroke_id} "
                  f"({self.state.points_in_current_stroke} points) (keyboard)")

    def _enter_calibration_mode(self) -> None:
        """Reset calibration and enter calibration mode."""
        if self.dynamic_triangulator is not None:
            self.dynamic_triangulator.reset_calibration()
        self.state.calibration_mode = True
        self.state.dynamic_calibrated = False
        print("=" * 40)
        print("RECALIBRATION MODE")
        print("Point ArUco marker at BOTH cameras")
        print("=" * 40)

    def set_frame_callback(self, callback: Callable) -> None:
        """Set callback for frame processing (for local visualization)."""
        self._on_frame_callback = callback

    def set_point_callback(self, callback: Callable) -> None:
        """Set callback for point generation."""
        self._on_point_callback = callback


async def main(port: int = 8765, camera: int = 0) -> None:
    """Main entry point for standalone server."""
    server = AirPaintServer(port=port, camera_index=camera)

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

    parser = argparse.ArgumentParser(description="Air Paint WebSocket Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--camera", type=int, default=0, help="Mac webcam index")

    args = parser.parse_args()

    asyncio.run(main(port=args.port, camera=args.camera))
