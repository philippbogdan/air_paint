# AirPaintAR - Real-time AR Air Painting

iOS app for real-time AR visualization of 3D air painting. Works with Meta glasses and a Mac running the air_paint server.

## Architecture

```
Meta Glasses (camera) → iPhone (hub) → Mac (processing) → iPhone (AR display)
```

1. **Meta Glasses**: Streams video via MWDATCamera SDK to iPhone
2. **iPhone**: Forwards frames to Mac via WebSocket, renders AR
3. **Mac**: Runs MediaPipe hand tracking, triangulates 3D points, sends back to iPhone

## Requirements

- iOS 17.0+
- iPhone with ARKit support (iPhone 6s or later)
- Mac running the air_paint server (`python main.py --server`)
- Meta Ray-Ban glasses or Meta Quest 3 for glasses streaming

## Setup

### 1. Add Meta Wearables SDK (Required)

1. Open `AirPaintAR.xcodeproj` in Xcode
2. Go to **File → Add Package Dependencies**
3. Enter the package URL:
   ```
   https://github.com/facebook/meta-wearables-dat-ios
   ```
4. Click **Add Package**
5. Select both `MWDATCore` and `MWDATCamera` products
6. Click **Add Package** to confirm

### 2. Configure the Project

1. In Xcode, go to **Project Settings → Build Settings**
2. Search for "xcconfig"
3. Set "Based on Configuration File" to point to `Config.xcconfig`

Or manually add to Build Settings:
- `CLIENT_TOKEN` = (your Meta client token)
- `META_APP_ID` = 901053085748763

### 3. Configure Signing

1. Select the AirPaintAR target
2. Under **Signing & Capabilities**, select your development team
3. Update the Bundle Identifier if needed

### 4. Build and Run

1. Connect your iPhone
2. Build and run (Cmd+R)
3. Grant camera, Bluetooth, and local network permissions

## Usage

### Initial Setup (First Time)

1. **Register with Meta**:
   - Open the app
   - Go to Settings (gear icon)
   - Tap **"Connect to Glasses"**
   - This opens the Meta mobile app for OAuth authentication
   - Complete the authorization flow
   - Return to AirPaintAR

2. **Pair your glasses**:
   - Ensure your Meta glasses are:
     - Charged
     - Paired with iPhone via Bluetooth (in Settings app)
     - Connected in the Meta app
   - The "Device" status should show "Connected"

3. **Connect to Mac server**:
   - Start the Mac server: `python main.py --server --camera 0`
   - Enter Mac's IP address in Settings
   - Tap "Connect"

### Drawing

1. Start the glasses stream: tap "Start Stream"
2. Position yourself so both cameras can see your hand
3. Use a Bluetooth shutter button to toggle drawing on/off
4. Your strokes appear in AR!

### Fallback Mode (Testing without Glasses)

Toggle **"Use iPhone Camera (Fallback)"** to test without glasses:

1. The iPhone's back camera acts as the "glasses" camera
2. Point it at your hand and the ArUco marker
3. Mac webcam provides the second stereo camera

## Project Structure

```
AirPaintAR/
├── AirPaintARApp.swift           # Entry point, SDK init
├── Views/
│   ├── ContentView.swift         # Main UI + Settings
│   └── ARDrawingView.swift       # ARKit scene
├── ViewModels/
│   ├── WearablesViewModel.swift  # SDK registration
│   ├── GlassesStreamViewModel.swift  # Video streaming
│   └── DrawingViewModel.swift    # WebSocket + state
├── Networking/
│   ├── Protocol.swift            # Message types
│   └── WebSocketClient.swift     # WebSocket client
├── Config.xcconfig               # Meta SDK credentials
└── Info.plist                    # Permissions + MWDAT config
```

## Meta SDK Flow

```
1. App Launch
   └── Wearables.configure()

2. User taps "Connect to Glasses"
   └── wearables.startRegistration()
   └── Opens Meta app for OAuth
   └── User authorizes
   └── Meta app redirects back (airpaintar://)
   └── Wearables.shared.handleUrl(url)
   └── registrationState → .registered

3. Device Connection
   └── deviceSelector.activeDeviceStream()
   └── hasActiveDevice = true when glasses connect

4. Streaming
   └── streamSession.start()
   └── videoFramePublisher.listen { frame in ... }
   └── Frames processed and sent to Mac
```

## Protocol

### iPhone → Mac
- `frame`: JPEG image from glasses (base64)
- `drawing_toggle`: Button pressed
- `ping`: Latency measurement

### Mac → iPhone
- `point`: 3D coordinate (x, y, z in meters)
- `stroke_start`: New stroke begun
- `stroke_end`: Stroke completed
- `world_anchor`: ArUco marker pose (4x4 matrix)
- `status`: FPS, tracking state, etc.
- `pong`: Ping response

## Troubleshooting

### "No such module 'MWDATCore'"
- Add the Meta Wearables SDK package (see Setup step 1)
- Ensure both `MWDATCore` and `MWDATCamera` are added

### Glasses not connecting
- Ensure glasses are paired in iPhone Bluetooth settings
- Check glasses are connected in the Meta app
- Registration must be completed first
- Try disconnecting and re-pairing

### "Server not found" / Connection timeout
- Ensure Mac and iPhone are on same WiFi network
- Try using iPhone hotspot (connect Mac to it)
- Check Mac's IP address: `ifconfig | grep "inet "`
- Verify server is running: `python main.py --server`

### "No marker detected"
- Print an ArUco marker (ID 0, 160mm)
- Ensure good lighting
- Hold marker steady in camera view

### "Hand not tracked"
- Position hand clearly in view of both cameras
- Ensure good lighting
- Try moving slower

## Coordinate Systems

- **Mac**: Points in millimeters, ArUco marker as origin
- **iPhone**: Points in meters (ARKit convention)
- Conversion happens in `PointMessage` (divide by 1000)
