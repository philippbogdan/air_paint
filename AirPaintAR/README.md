# AirPaintAR - Real-time AR Air Painting (Viewer Mode)

iOS app for real-time AR visualization of 3D air painting. iPhone acts as a pure AR viewer, receiving 3D points from Mac which uses dual local cameras for stereo hand tracking.

## Architecture

```
Camera A (Mac webcam) ──┐
                        ├→ Mac → Hand Detection → Triangulation → WebSocket → iPhone AR
Camera B (USB webcam) ──┘                                         (points)
```

1. **Mac**: Captures from two local cameras, runs MediaPipe hand tracking, triangulates 3D points
2. **iPhone**: Receives 3D points via WebSocket, renders strokes in AR

## Requirements

- iOS 17.0+
- iPhone with ARKit support (iPhone 6s or later)
- Mac running the air_paint server with dual cameras
- Two cameras connected to Mac (built-in webcam + USB webcam)

## Setup

### 1. Configure Signing

1. Open `AirPaintAR.xcodeproj` in Xcode
2. Select the AirPaintAR target
3. Under **Signing & Capabilities**, select your development team
4. Update the Bundle Identifier if needed

### 2. Build and Run

1. Connect your iPhone
2. Build and run (Cmd+R)
3. Grant camera permissions when prompted

## Usage

### Start the Mac Server

```bash
# List available cameras
python main.py --list-cameras

# Calibrate cameras (first time)
python main.py --calibrate --camera-a 0 --camera-b 1

# Start the local camera server
python main.py --server-local --camera-a 0 --camera-b 1 --port 8765
```

### Connect from iPhone

1. Open AirPaintAR on your iPhone
2. Tap the gear icon (Settings)
3. Enter the Mac's IP address (find via `ifconfig | grep "inet "`)
4. Enter port 8765 (default)
5. Tap "Connect"
6. Status should show green "Server" badge

### Drawing

1. Position yourself so both Mac cameras can see your hand
2. Press SPACE on Mac (or Bluetooth shutter button) to toggle drawing
3. Move your finger to draw in 3D
4. View your strokes in AR on iPhone!

## Project Structure

```
AirPaintAR/
├── AirPaintARApp.swift           # Entry point, AppState
├── Views/
│   ├── ContentView.swift         # Main UI + Settings
│   └── ARDrawingView.swift       # ARKit scene
├── ViewModels/
│   └── DrawingViewModel.swift    # WebSocket + state
├── Networking/
│   ├── Protocol.swift            # Message types
│   └── WebSocketClient.swift     # WebSocket client
└── Info.plist                    # Permissions
```

## Protocol

### Mac → iPhone
- `point`: 3D coordinate (x, y, z in meters)
- `stroke_start`: New stroke begun
- `stroke_end`: Stroke completed
- `world_anchor`: ArUco marker pose (4x4 matrix)
- `status`: FPS, tracking state, etc.
- `pong`: Ping response

### iPhone → Mac (optional)
- `drawing_toggle`: Toggle drawing state
- `ping`: Latency measurement

## Troubleshooting

### "Server not found" / Connection timeout
- Ensure Mac and iPhone are on same WiFi network
- Try using iPhone hotspot (connect Mac to it)
- Check Mac's IP address: `ifconfig | grep "inet "`
- Verify server is running: `python main.py --server-local`

### "No marker detected"
- Print an ArUco marker (ID 0, 160mm)
- Ensure good lighting
- Hold marker steady in view of camera A

### "Hand not tracked"
- Position hand clearly in view of BOTH cameras
- Ensure good lighting
- Try moving slower

### Low FPS
- Reduce camera resolution in config/settings.py
- Close other CPU-intensive apps

## Coordinate Systems

- **Mac**: Points in millimeters, ArUco marker as origin
- **iPhone**: Points in meters (ARKit convention)
- Conversion happens in `PointMessage` (divide by 1000)
