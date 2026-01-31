"""Keyboard and shutter input handling with global HID capture."""

import cv2
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Optional
from pynput import keyboard


class InputAction(Enum):
    """Available input actions."""
    HID_CLICK = auto()  # Generic HID button click (Volume Up)
    TOGGLE_DRAWING = auto()
    UNDO = auto()
    CLEAR = auto()
    SAVE = auto()
    QUIT = auto()
    NONE = auto()


@dataclass
class InputEvent:
    """An input event."""
    action: InputAction
    key_code: int
    key_char: str


class HIDListener:
    """
    Global listener for HID events (Volume Up from Bluetooth shutter).

    Runs in background thread, captures Volume Up key globally.
    """

    def __init__(self, on_click: Callable[[], None]):
        """
        Initialize the HID listener.

        Args:
            on_click: Callback when HID button (Volume Up) is pressed
        """
        self._on_click = on_click
        self._listener: Optional[keyboard.Listener] = None
        self._running = False

    def start(self) -> None:
        """Start listening for HID events in background."""
        if self._running:
            return

        self._running = True
        self._listener = keyboard.Listener(on_press=self._on_key_press)
        self._listener.start()
        print("HID listener started (listening for Volume Up)")

    def stop(self) -> None:
        """Stop listening."""
        self._running = False
        if self._listener:
            self._listener.stop()
            self._listener = None

    def _on_key_press(self, key) -> None:
        """Handle key press events."""
        try:
            # Check for media keys (Volume Up)
            if hasattr(key, 'name'):
                # Some systems report as 'media_volume_up'
                if 'volume_up' in key.name.lower():
                    self._on_click()
            # Check for specific key codes
            elif hasattr(key, 'vk'):
                # Volume Up virtual key codes vary by system
                # macOS: 0x48 (72) or via media key
                if key.vk in [0x48, 0xAF, 0xE9]:
                    self._on_click()
        except Exception:
            pass


class InputHandler:
    """
    Handles keyboard input and Bluetooth HID shutter events.

    Two-phase operation:
    1. Setup phase: First HID click establishes world center (when ArUco visible)
    2. Drawing phase: HID clicks toggle drawing on/off

    Key mappings (keyboard fallback):
    - SPACE: Toggle drawing / establish world (same as HID)
    - Z: Undo last stroke
    - C: Clear all strokes
    - S: Save session
    - Q/ESC: Quit
    """

    # Default key mappings
    DEFAULT_MAPPINGS: Dict[int, InputAction] = {
        ord(' '): InputAction.HID_CLICK,  # Spacebar acts like HID
        ord('z'): InputAction.UNDO,
        ord('Z'): InputAction.UNDO,
        ord('c'): InputAction.CLEAR,
        ord('C'): InputAction.CLEAR,
        ord('s'): InputAction.SAVE,
        ord('S'): InputAction.SAVE,
        ord('q'): InputAction.QUIT,
        ord('Q'): InputAction.QUIT,
        27: InputAction.QUIT,  # ESC
    }

    def __init__(self):
        """Initialize the input handler."""
        self._mappings = dict(self.DEFAULT_MAPPINGS)
        self._callbacks: Dict[InputAction, Callable[[], None]] = {}

        # HID click state (thread-safe)
        self._hid_clicked = threading.Event()

        # Start global HID listener
        self._hid_listener = HIDListener(on_click=self._on_hid_click)
        self._hid_listener.start()

    def _on_hid_click(self) -> None:
        """Called when HID button (Volume Up) is pressed."""
        self._hid_clicked.set()

    def check_hid_click(self) -> bool:
        """
        Check if HID button was clicked since last check.

        Returns:
            True if clicked, False otherwise
        """
        if self._hid_clicked.is_set():
            self._hid_clicked.clear()
            return True
        return False

    def register_callback(
        self,
        action: InputAction,
        callback: Callable[[], None]
    ) -> None:
        """
        Register a callback for an input action.

        Args:
            action: The action to register
            callback: Function to call when action triggered
        """
        self._callbacks[action] = callback

    def add_key_mapping(self, key_code: int, action: InputAction) -> None:
        """
        Add or override a key mapping.

        Args:
            key_code: The key code
            action: The action to map to
        """
        self._mappings[key_code] = action

    def process_key(self, key_code: int) -> InputEvent:
        """
        Process a key press and return the corresponding event.

        Args:
            key_code: The pressed key code (from cv2.waitKey)

        Returns:
            InputEvent describing what happened
        """
        # Ignore -1 (no key pressed) or values > 255
        if key_code < 0 or key_code > 255:
            return InputEvent(
                action=InputAction.NONE,
                key_code=key_code,
                key_char=""
            )

        action = self._mappings.get(key_code, InputAction.NONE)

        # Try to get character representation
        try:
            key_char = chr(key_code) if 32 <= key_code <= 126 else ""
        except ValueError:
            key_char = ""

        event = InputEvent(
            action=action,
            key_code=key_code,
            key_char=key_char
        )

        # Execute callback if registered
        if action != InputAction.NONE and action in self._callbacks:
            self._callbacks[action]()

        return event

    def wait_key(self, delay_ms: int = 1) -> InputEvent:
        """
        Wait for a key press and process it.
        Also checks for HID clicks.

        Args:
            delay_ms: Milliseconds to wait (0 = wait forever)

        Returns:
            InputEvent describing what happened
        """
        # Check for HID click first
        if self.check_hid_click():
            event = InputEvent(
                action=InputAction.HID_CLICK,
                key_code=0,
                key_char=""
            )
            # Execute callback if registered
            if InputAction.HID_CLICK in self._callbacks:
                self._callbacks[InputAction.HID_CLICK]()
            return event

        # Then check keyboard
        key_code = cv2.waitKey(delay_ms) & 0xFF
        return self.process_key(key_code)

    def cleanup(self) -> None:
        """Stop the HID listener."""
        self._hid_listener.stop()


def get_key_name(key_code: int) -> str:
    """
    Get a human-readable name for a key code.

    Args:
        key_code: The key code

    Returns:
        Human-readable key name
    """
    special_keys = {
        27: "ESC",
        32: "SPACE",
        13: "ENTER",
        9: "TAB",
        8: "BACKSPACE",
        127: "DELETE",
    }

    if key_code in special_keys:
        return special_keys[key_code]

    if 32 <= key_code <= 126:
        return chr(key_code)

    return f"0x{key_code:02X}"
