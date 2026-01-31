"""MediaPipe hands wrapper for hand tracking."""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path
import urllib.request
from config.settings import TRACKING, get_project_root


@dataclass
class HandLandmarks:
    """Container for detected hand landmarks."""
    landmarks: np.ndarray  # Shape: (21, 3) - all 21 landmarks with (x, y, z)
    handedness: str  # "Left" or "Right"
    confidence: float

    @property
    def index_finger_tip(self) -> Tuple[float, float]:
        """Get the index finger tip position (landmark 8) as (x, y) in pixel coords."""
        return (
            self.landmarks[TRACKING.INDEX_FINGER_TIP_LANDMARK, 0],
            self.landmarks[TRACKING.INDEX_FINGER_TIP_LANDMARK, 1]
        )

    @property
    def all_fingertips(self) -> np.ndarray:
        """Get all fingertip positions (landmarks 4, 8, 12, 16, 20)."""
        fingertip_indices = [4, 8, 12, 16, 20]
        return self.landmarks[fingertip_indices, :2]


# Model download URL
HAND_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def get_model_path() -> Path:
    """Get path to hand landmarker model, downloading if necessary."""
    model_dir = get_project_root() / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hand_landmarker.task"

    if not model_path.exists():
        print(f"Downloading hand landmarker model...")
        urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, model_path)
        print(f"Model saved to {model_path}")

    return model_path


class HandTracker:
    """
    Wrapper for MediaPipe Hands using the new Tasks API.

    Provides simplified interface for detecting and tracking hands.
    """

    # MediaPipe hand landmark connections for drawing
    CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]

    def __init__(
        self,
        max_num_hands: int = TRACKING.MAX_NUM_HANDS,
        min_detection_confidence: float = TRACKING.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = TRACKING.MIN_TRACKING_CONFIDENCE
    ):
        """
        Initialize the hand tracker.

        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        model_path = get_model_path()

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray, scale: float = None) -> List[HandLandmarks]:
        """
        Detect hands in an image.

        Args:
            image: BGR image
            scale: Optional downscale factor for faster detection (e.g., 0.5 = half size).
                   Coordinates are scaled back to original image size.
                   Defaults to TRACKING.DETECTION_SCALE.

        Returns:
            List of HandLandmarks for each detected hand
        """
        if scale is None:
            scale = TRACKING.DETECTION_SCALE

        h_orig, w_orig = image.shape[:2]

        # Downscale for faster detection if scale < 1
        if scale < 1.0:
            detect_image = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            detect_image = image

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(detect_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self.detector.detect(mp_image)

        hands_list = []

        if results.hand_landmarks:
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                # Convert normalized coordinates to ORIGINAL image pixel coordinates
                landmarks = np.array([
                    [lm.x * w_orig, lm.y * h_orig, lm.z * w_orig]  # z is scaled same as x
                    for lm in hand_landmarks
                ])

                # Get handedness
                handedness = "Right"
                confidence = 0.0
                if results.handedness and i < len(results.handedness):
                    handedness = results.handedness[i][0].category_name
                    confidence = results.handedness[i][0].score

                hands_list.append(HandLandmarks(
                    landmarks=landmarks,
                    handedness=handedness,
                    confidence=confidence
                ))

        return hands_list

    def get_index_finger_tip(
        self,
        image: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Get the index finger tip position from the first detected hand.

        Args:
            image: BGR image

        Returns:
            (x, y) position in pixels, or None if no hand detected
        """
        hands = self.detect(image)
        if not hands:
            return None
        return hands[0].index_finger_tip

    def draw_landmarks(
        self,
        image: np.ndarray,
        hand: HandLandmarks,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
        point_radius: int = 4,
        highlight_index: bool = True
    ) -> np.ndarray:
        """
        Draw hand landmarks on an image.

        Args:
            image: BGR image (will be modified in place)
            hand: HandLandmarks to draw
            color: Color for skeleton (BGR)
            thickness: Line thickness
            point_radius: Radius for landmark points
            highlight_index: Whether to highlight the index finger tip

        Returns:
            Image with landmarks drawn
        """
        landmarks = hand.landmarks[:, :2].astype(int)

        # Draw connections
        for start_idx, end_idx in self.CONNECTIONS:
            start = tuple(landmarks[start_idx])
            end = tuple(landmarks[end_idx])
            cv2.line(image, start, end, color, thickness)

        # Draw landmark points
        for i, (x, y) in enumerate(landmarks):
            # Highlight index finger tip
            if i == TRACKING.INDEX_FINGER_TIP_LANDMARK and highlight_index:
                cv2.circle(image, (x, y), point_radius + 2, (0, 255, 0), -1)
                cv2.circle(image, (x, y), point_radius + 4, (0, 255, 0), 2)
            else:
                cv2.circle(image, (x, y), point_radius, color, -1)

        return image

    def close(self) -> None:
        """Release resources."""
        self.detector.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
