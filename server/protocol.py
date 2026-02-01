"""WebSocket message protocol for real-time AR air painting.

Protocol Overview:
------------------
iPhone → Mac:
- frame: JPEG image from Meta glasses camera
- drawing_toggle: Bluetooth button was pressed

Mac → iPhone:
- point: New 3D point triangulated from stereo vision
- stroke_start: New stroke has begun
- stroke_end: Current stroke has ended
- world_anchor: ArUco marker pose for coordinate alignment
- status: Tracking status update
"""

import json
import base64
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union, Any
from enum import Enum


class MessageType(str, Enum):
    """WebSocket message types."""
    # iPhone → Mac
    FRAME = "frame"
    DRAWING_TOGGLE = "drawing_toggle"

    # Mac → iPhone
    POINT = "point"
    STROKE_START = "stroke_start"
    STROKE_END = "stroke_end"
    WORLD_ANCHOR = "world_anchor"
    STATUS = "status"

    # Bidirectional
    PING = "ping"
    PONG = "pong"
    ERROR = "error"


# ============================================================================
# iPhone → Mac Messages
# ============================================================================

@dataclass
class FrameMessage:
    """Frame from Meta glasses camera (sent as JPEG)."""
    type: str = field(default=MessageType.FRAME.value, init=False)
    data: str = ""  # Base64-encoded JPEG
    timestamp: float = field(default_factory=time.time)
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FrameMessage":
        return cls(
            data=d.get("data", ""),
            timestamp=d.get("timestamp", time.time()),
            width=d.get("width", 0),
            height=d.get("height", 0),
        )

    def decode_image(self) -> bytes:
        """Decode base64 data to raw JPEG bytes."""
        return base64.b64decode(self.data)

    @classmethod
    def from_jpeg_bytes(cls, jpeg_bytes: bytes, width: int = 0, height: int = 0) -> "FrameMessage":
        """Create message from raw JPEG bytes."""
        return cls(
            data=base64.b64encode(jpeg_bytes).decode('ascii'),
            timestamp=time.time(),
            width=width,
            height=height,
        )


@dataclass
class DrawingToggleMessage:
    """Toggle drawing on/off (triggered by Bluetooth button)."""
    type: str = field(default=MessageType.DRAWING_TOGGLE.value, init=False)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DrawingToggleMessage":
        return cls(timestamp=d.get("timestamp", time.time()))


# ============================================================================
# Mac → iPhone Messages
# ============================================================================

@dataclass
class PointMessage:
    """3D point triangulated from stereo vision.

    Coordinates are in world frame (ArUco marker origin).
    Units: meters (converted from mm for ARKit compatibility).
    """
    type: str = field(default=MessageType.POINT.value, init=False)
    x: float = 0.0  # meters
    y: float = 0.0  # meters
    z: float = 0.0  # meters
    stroke_id: int = 0
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0  # 0-1, based on reprojection error

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PointMessage":
        return cls(
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            z=d.get("z", 0.0),
            stroke_id=d.get("stroke_id", 0),
            timestamp=d.get("timestamp", time.time()),
            confidence=d.get("confidence", 1.0),
        )

    @classmethod
    def from_mm(cls, x_mm: float, y_mm: float, z_mm: float,
                stroke_id: int = 0, confidence: float = 1.0) -> "PointMessage":
        """Create from millimeter coordinates (converts to meters for ARKit)."""
        return cls(
            x=x_mm / 1000.0,
            y=y_mm / 1000.0,
            z=z_mm / 1000.0,
            stroke_id=stroke_id,
            confidence=confidence,
        )


@dataclass
class StrokeStartMessage:
    """New stroke has begun."""
    type: str = field(default=MessageType.STROKE_START.value, init=False)
    stroke_id: int = 0
    timestamp: float = field(default_factory=time.time)
    color: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])  # RGB, 0-1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrokeStartMessage":
        return cls(
            stroke_id=d.get("stroke_id", 0),
            timestamp=d.get("timestamp", time.time()),
            color=d.get("color", [1.0, 0.0, 0.0]),
        )


@dataclass
class StrokeEndMessage:
    """Current stroke has ended."""
    type: str = field(default=MessageType.STROKE_END.value, init=False)
    stroke_id: int = 0
    timestamp: float = field(default_factory=time.time)
    point_count: int = 0  # Total points in the completed stroke

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrokeEndMessage":
        return cls(
            stroke_id=d.get("stroke_id", 0),
            timestamp=d.get("timestamp", time.time()),
            point_count=d.get("point_count", 0),
        )


@dataclass
class WorldAnchorMessage:
    """ArUco marker pose for coordinate alignment.

    The iPhone uses this to align its ARKit coordinate system
    with the Mac's triangulated coordinates.

    marker_pose is a 4x4 transformation matrix (row-major, flattened to 16 floats)
    that transforms points from marker (world) frame to camera frame.
    """
    type: str = field(default=MessageType.WORLD_ANCHOR.value, init=False)
    marker_pose: List[float] = field(default_factory=lambda: [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ])  # 4x4 matrix, row-major
    marker_id: int = 0
    marker_size_m: float = 0.16  # meters
    timestamp: float = field(default_factory=time.time)
    visible: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorldAnchorMessage":
        return cls(
            marker_pose=d.get("marker_pose", [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
            marker_id=d.get("marker_id", 0),
            marker_size_m=d.get("marker_size_m", 0.16),
            timestamp=d.get("timestamp", time.time()),
            visible=d.get("visible", True),
        )

    @classmethod
    def from_matrix(cls, matrix_4x4, marker_id: int = 0,
                    marker_size_mm: float = 160.0, visible: bool = True) -> "WorldAnchorMessage":
        """Create from 4x4 numpy matrix."""
        import numpy as np
        flat = matrix_4x4.flatten().tolist()
        return cls(
            marker_pose=flat,
            marker_id=marker_id,
            marker_size_m=marker_size_mm / 1000.0,
            visible=visible,
        )


@dataclass
class StatusMessage:
    """Tracking status update."""
    type: str = field(default=MessageType.STATUS.value, init=False)
    tracking: bool = False  # Is hand being tracked?
    drawing: bool = False  # Is currently drawing?
    marker_visible: bool = False  # Is ArUco marker visible?
    fps: float = 0.0  # Processing FPS
    latency_ms: float = 0.0  # Estimated latency
    stroke_id: int = 0  # Current stroke ID
    total_strokes: int = 0  # Total stroke count
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StatusMessage":
        return cls(
            tracking=d.get("tracking", False),
            drawing=d.get("drawing", False),
            marker_visible=d.get("marker_visible", False),
            fps=d.get("fps", 0.0),
            latency_ms=d.get("latency_ms", 0.0),
            stroke_id=d.get("stroke_id", 0),
            total_strokes=d.get("total_strokes", 0),
            timestamp=d.get("timestamp", time.time()),
        )


# ============================================================================
# Utility Messages
# ============================================================================

@dataclass
class PingMessage:
    """Ping for latency measurement."""
    type: str = field(default=MessageType.PING.value, init=False)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PingMessage":
        return cls(timestamp=d.get("timestamp", time.time()))


@dataclass
class PongMessage:
    """Pong response."""
    type: str = field(default=MessageType.PONG.value, init=False)
    ping_timestamp: float = 0.0  # Original ping timestamp
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PongMessage":
        return cls(
            ping_timestamp=d.get("ping_timestamp", 0.0),
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class ErrorMessage:
    """Error message."""
    type: str = field(default=MessageType.ERROR.value, init=False)
    code: str = ""
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorMessage":
        return cls(
            code=d.get("code", ""),
            message=d.get("message", ""),
            timestamp=d.get("timestamp", time.time()),
        )


# ============================================================================
# Serialization / Parsing
# ============================================================================

# Type alias for all message types
Message = Union[
    FrameMessage, DrawingToggleMessage, PointMessage, StrokeStartMessage,
    StrokeEndMessage, WorldAnchorMessage, StatusMessage, PingMessage,
    PongMessage, ErrorMessage
]

# Message type to class mapping
MESSAGE_CLASSES = {
    MessageType.FRAME.value: FrameMessage,
    MessageType.DRAWING_TOGGLE.value: DrawingToggleMessage,
    MessageType.POINT.value: PointMessage,
    MessageType.STROKE_START.value: StrokeStartMessage,
    MessageType.STROKE_END.value: StrokeEndMessage,
    MessageType.WORLD_ANCHOR.value: WorldAnchorMessage,
    MessageType.STATUS.value: StatusMessage,
    MessageType.PING.value: PingMessage,
    MessageType.PONG.value: PongMessage,
    MessageType.ERROR.value: ErrorMessage,
}


def serialize_message(message: Message) -> str:
    """Serialize a message to JSON string."""
    return json.dumps(message.to_dict())


def parse_message(json_str: str) -> Optional[Message]:
    """Parse a JSON string into a message object.

    Args:
        json_str: JSON-encoded message string

    Returns:
        Parsed message object, or None if parsing fails
    """
    try:
        data = json.loads(json_str)
        msg_type = data.get("type")

        if msg_type not in MESSAGE_CLASSES:
            return None

        cls = MESSAGE_CLASSES[msg_type]
        return cls.from_dict(data)

    except (json.JSONDecodeError, KeyError, TypeError):
        return None
