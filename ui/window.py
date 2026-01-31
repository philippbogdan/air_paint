"""Main window composition with camera feeds and 3D viewport."""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for performance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple
from io import BytesIO

from config.settings import UI, DRAWING
from drawing.stroke import Stroke
from tracking.hands import HandLandmarks


class Viewport3D:
    """3D viewport using matplotlib for rendering strokes."""

    def __init__(
        self,
        size: Tuple[int, int] = UI.VIEW_3D_SIZE,
        auto_rotate_speed: float = UI.AUTO_ROTATE_SPEED
    ):
        """
        Initialize the 3D viewport.

        Args:
            size: Viewport size (width, height)
            auto_rotate_speed: Rotation speed in degrees per frame
        """
        self.size = size
        self.auto_rotate_speed = auto_rotate_speed
        self._rotation_angle = 30.0
        self._elevation = 20.0

        # Create figure with tight layout
        self.fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

    def render(
        self,
        strokes: List[Stroke],
        auto_rotate: bool = True,
        highlight_last: bool = True
    ) -> np.ndarray:
        """
        Render strokes to an image.

        Args:
            strokes: List of strokes to render
            auto_rotate: Whether to auto-rotate the view
            highlight_last: Whether to highlight the most recent stroke

        Returns:
            BGR image of the rendered viewport
        """
        self.ax.clear()

        # Set up axis
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')

        # Collect all points for axis limits
        all_points = []
        for stroke in strokes:
            if not stroke.is_empty:
                all_points.append(stroke.to_array())

        if all_points:
            points = np.vstack(all_points)

            # Set axis limits with some padding
            margin = 50  # mm
            x_range = [points[:, 0].min() - margin, points[:, 0].max() + margin]
            y_range = [points[:, 1].min() - margin, points[:, 1].max() + margin]
            z_range = [points[:, 2].min() - margin, points[:, 2].max() + margin]

            # Make axes equal scale
            max_range = max(
                x_range[1] - x_range[0],
                y_range[1] - y_range[0],
                z_range[1] - z_range[0]
            )
            mid_x = sum(x_range) / 2
            mid_y = sum(y_range) / 2
            mid_z = sum(z_range) / 2

            self.ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            self.ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            self.ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

        # Draw strokes
        for i, stroke in enumerate(strokes):
            if stroke.is_empty:
                continue

            points = stroke.to_array()
            is_last = (i == len(strokes) - 1) and highlight_last

            # Convert BGR to RGB for matplotlib
            color = tuple(c/255.0 for c in reversed(stroke.color))

            # Line properties
            linewidth = stroke.thickness * 1.5 if is_last else stroke.thickness
            alpha = 1.0 if is_last else 0.7

            self.ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                linewidth=linewidth,
                alpha=alpha
            )

            # Draw points
            self.ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=[color],
                s=10 if is_last else 5,
                alpha=alpha
            )

        # Update rotation
        if auto_rotate:
            self._rotation_angle += self.auto_rotate_speed
            if self._rotation_angle >= 360:
                self._rotation_angle -= 360

        self.ax.view_init(elev=self._elevation, azim=self._rotation_angle)

        # Render to image
        self.fig.canvas.draw()

        # Get image from figure
        buf = BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)

        # Decode PNG to numpy array
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        buf.close()

        # Resize to target size
        img = cv2.resize(img, self.size)

        return img

    def close(self) -> None:
        """Close the matplotlib figure."""
        plt.close(self.fig)


class MainWindow:
    """
    Main window compositing camera feeds and 3D viewport.

    Layout:
    ┌─────────────────┬─────────────────┐
    │  Camera A       │  Camera B       │
    │  + Skeleton     │  + Skeleton     │
    │  + 2D Strokes   │                 │
    ├─────────────────┴─────────────────┤
    │         3D Viewport               │
    │       (auto-rotating)             │
    └───────────────────────────────────┘
    """

    def __init__(
        self,
        window_name: str = UI.WINDOW_NAME,
        camera_size: Tuple[int, int] = (UI.CAMERA_PREVIEW_WIDTH, UI.CAMERA_PREVIEW_HEIGHT),
        viewport_size: Tuple[int, int] = UI.VIEW_3D_SIZE
    ):
        """
        Initialize the main window.

        Args:
            window_name: Window title
            camera_size: Size for each camera preview (width, height)
            viewport_size: Size for 3D viewport (width, height)
        """
        self.window_name = window_name
        self.camera_size = camera_size
        self.viewport_size = viewport_size

        self._viewport = Viewport3D(size=viewport_size)
        # Cache for 3D viewport (only re-render when needed)
        self._cached_viewport_img: Optional[np.ndarray] = None
        self._last_stroke_count = 0
        self._viewport_frame_counter = 0
        self._viewport_render_interval = 15  # Only render 3D every N frames

        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def draw_strokes_2d(
        self,
        image: np.ndarray,
        strokes: List[Stroke],
        project_func: callable
    ) -> np.ndarray:
        """
        Draw 2D projections of strokes on an image.

        Args:
            image: Image to draw on
            strokes: List of strokes
            project_func: Function to project 3D points to 2D (point_3d) -> (x, y)

        Returns:
            Image with strokes drawn
        """
        for stroke in strokes:
            if stroke.is_empty or len(stroke.points) < 2:
                continue

            # Project all points
            points_2d = []
            for point in stroke.points:
                try:
                    x, y = project_func(point.to_array())
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        points_2d.append((int(x), int(y)))
                except Exception:
                    continue

            # Draw lines between consecutive points
            for i in range(1, len(points_2d)):
                cv2.line(
                    image,
                    points_2d[i-1],
                    points_2d[i],
                    stroke.color,
                    int(stroke.thickness)
                )

            # Draw points
            for pt in points_2d:
                cv2.circle(image, pt, DRAWING.POINT_RADIUS, stroke.color, -1)

        return image

    def draw_status(
        self,
        image: np.ndarray,
        is_drawing: bool,
        num_strokes: int,
        fps: float = 0.0,
        marker_visible: bool = False,
        coordinate_frame: str = "camera_a",
        phase: str = "setup"
    ) -> np.ndarray:
        """
        Draw status overlay on image.

        Args:
            image: Image to draw on
            is_drawing: Whether currently drawing
            num_strokes: Number of strokes
            fps: Current FPS
            marker_visible: Whether ArUco marker is detected
            coordinate_frame: Current coordinate frame ("camera_a" or "world")
            phase: Current phase ("setup" or "drawing")

        Returns:
            Image with status overlay
        """
        h, w = image.shape[:2]

        # Phase-dependent display
        if phase == "setup":
            # Setup phase - show instructions
            status_text = "SHOW ARUCO + CLICK"
            status_color = (0, 255, 255)  # Yellow

            # Draw semi-transparent background
            overlay = image.copy()
            cv2.rectangle(overlay, (5, 5), (200, 75), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

            cv2.putText(
                image, status_text,
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2
            )

            # Marker status
            marker_text = "Marker: VISIBLE" if marker_visible else "Marker: NOT FOUND"
            marker_color = (0, 255, 0) if marker_visible else (0, 0, 255)
            cv2.putText(
                image, marker_text,
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, marker_color, 1
            )

            if fps > 0:
                cv2.putText(
                    image, f"FPS: {fps:.1f}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                )
        else:
            # Drawing phase
            status_text = "DRAWING" if is_drawing else "PAUSED"
            status_color = (0, 255, 0) if is_drawing else (0, 165, 255)

            # Draw semi-transparent background
            overlay = image.copy()
            cv2.rectangle(overlay, (5, 5), (150, 75), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

            # Draw text
            cv2.putText(
                image, status_text,
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2
            )
            cv2.putText(
                image, f"Strokes: {num_strokes}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            if fps > 0:
                cv2.putText(
                    image, f"FPS: {fps:.1f}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )

        # Drawing indicator circle (top right)
        indicator_pos = (w - 20, 20)
        if phase == "setup":
            # Blinking yellow during setup when marker visible
            indicator_color = (0, 255, 255) if marker_visible else (128, 128, 128)
        else:
            indicator_color = (0, 255, 0) if is_drawing else (0, 165, 255)
        cv2.circle(image, indicator_pos, 10, indicator_color, -1)

        return image

    def compose(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        strokes: List[Stroke],
        is_drawing: bool,
        hand_a: Optional[HandLandmarks] = None,
        hand_b: Optional[HandLandmarks] = None,
        project_func: Optional[callable] = None,
        fps: float = 0.0,
        marker_visible: bool = False,
        coordinate_frame: str = "camera_a",
        phase: str = "setup"
    ) -> np.ndarray:
        """
        Compose the full window display.

        Args:
            frame_a: Camera A frame
            frame_b: Camera B frame
            strokes: List of strokes to display
            is_drawing: Whether currently drawing
            hand_a: Detected hand in camera A
            hand_b: Detected hand in camera B
            project_func: Function to project 3D to camera A 2D
            fps: Current FPS
            marker_visible: Whether ArUco marker is detected
            coordinate_frame: Current coordinate frame ("camera_a" or "world")
            phase: Current phase ("setup" or "drawing")

        Returns:
            Composed display image
        """
        # Resize camera frames
        display_a = cv2.resize(frame_a.copy(), self.camera_size)
        display_b = cv2.resize(frame_b.copy(), self.camera_size)

        # Scale factors for resized display
        scale_x = self.camera_size[0] / frame_a.shape[1]
        scale_y = self.camera_size[1] / frame_a.shape[0]

        # Draw hand skeletons (simple circle for index finger tip - fast)
        if hand_a is not None:
            tip_x = int(hand_a.landmarks[8, 0] * scale_x)
            tip_y = int(hand_a.landmarks[8, 1] * scale_y)
            cv2.circle(display_a, (tip_x, tip_y), 10, (0, 255, 0), -1)
            cv2.circle(display_a, (tip_x, tip_y), 12, (0, 255, 0), 2)

        if hand_b is not None:
            tip_x = int(hand_b.landmarks[8, 0] * scale_x)
            tip_y = int(hand_b.landmarks[8, 1] * scale_y)
            cv2.circle(display_b, (tip_x, tip_y), 10, (0, 255, 0), -1)
            cv2.circle(display_b, (tip_x, tip_y), 12, (0, 255, 0), 2)

        # Draw 2D stroke projections on camera A
        if project_func is not None and strokes:
            def scaled_project(pt_3d):
                x, y = project_func(pt_3d)
                return x * scale_x, y * scale_y

            self.draw_strokes_2d(display_a, strokes, scaled_project)

        # Draw status overlay
        display_a = self.draw_status(
            display_a, is_drawing, len(strokes), fps,
            marker_visible=marker_visible,
            coordinate_frame=coordinate_frame,
            phase=phase
        )

        # Add labels
        cv2.putText(
            display_b, "Camera B",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        # Render 3D viewport (with caching for performance)
        self._viewport_frame_counter += 1
        current_stroke_count = sum(len(s.points) for s in strokes)
        strokes_changed = current_stroke_count != self._last_stroke_count
        should_render = (
            self._cached_viewport_img is None or
            strokes_changed or
            self._viewport_frame_counter >= self._viewport_render_interval
        )

        if should_render:
            viewport_img = self._viewport.render(strokes, auto_rotate=True)
            self._cached_viewport_img = viewport_img
            self._last_stroke_count = current_stroke_count
            self._viewport_frame_counter = 0
        else:
            viewport_img = self._cached_viewport_img

        # Ensure viewport width matches combined camera width
        combined_width = self.camera_size[0] * 2
        if viewport_img.shape[1] != combined_width:
            aspect = viewport_img.shape[0] / viewport_img.shape[1]
            new_height = int(combined_width * aspect)
            viewport_img = cv2.resize(viewport_img, (combined_width, new_height))

        # Combine cameras side by side
        cameras_row = np.hstack([display_a, display_b])

        # Stack cameras and viewport vertically
        # Ensure same width
        if cameras_row.shape[1] != viewport_img.shape[1]:
            viewport_img = cv2.resize(
                viewport_img,
                (cameras_row.shape[1], viewport_img.shape[0])
            )

        composed = np.vstack([cameras_row, viewport_img])

        return composed

    def show(self, image: np.ndarray) -> None:
        """
        Display an image in the window.

        Args:
            image: Image to display
        """
        cv2.imshow(self.window_name, image)

    def get_screenshot(self) -> Optional[np.ndarray]:
        """
        Get the current window contents.

        Returns:
            Screenshot image or None if window not visible
        """
        # This is a simplified version - in practice we'd store the last composed image
        return None

    def close(self) -> None:
        """Close the window and release resources."""
        self._viewport.close()
        cv2.destroyWindow(self.window_name)
