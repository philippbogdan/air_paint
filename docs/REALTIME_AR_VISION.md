# Real-Time AR Air Painting with Meta Glasses

## Vision Statement

**Draw in mid-air while wearing Meta glasses. See the lines appear in AR on your iPhone in real-time, exactly where you drew them in physical space.**

The user wears Meta Ray-Ban glasses and moves their finger through the air. A stereo camera system (glasses + MacBook) triangulates the 3D position of the fingertip. The resulting 3D lines stream to an iPhone, which renders them in AR at the exact physical location where they were drawn.

---

## Table of Contents

1. [The Dream Experience](#the-dream-experience)
2. [System Overview](#system-overview)
3. [What We Have Today](#what-we-have-today)
4. [Hardware Constraints](#hardware-constraints)
5. [Software Constraints](#software-constraints)
6. [The Stereo Vision Problem](#the-stereo-vision-problem)
7. [The AR Rendering Problem](#the-ar-rendering-problem)
8. [The Networking Problem](#the-networking-problem)
9. [Latency Budget](#latency-budget)
10. [Coordinate System Challenges](#coordinate-system-challenges)
11. [Alternative Architectures](#alternative-architectures)
12. [Open Questions](#open-questions)
13. [Prior Art & References](#prior-art--references)

---

## The Dream Experience

### User Flow

1. User puts on Meta Ray-Ban Gen 2 glasses
2. User places iPhone on a surface or holds it
3. User places an ArUco marker in the scene (establishes world origin)
4. User opens the iPhone app, which connects to:
   - The glasses (via Meta DAT SDK)
   - The Mac (via local network)
5. User presses a button to start drawing
6. User moves their index finger through the air
7. **Lines appear in AR on the iPhone screen in real-time, floating in 3D space exactly where the finger moved**
8. User presses button to stop
9. The 3D drawing can be saved, exported, viewed from any angle, or 3D printed

### Key Experience Requirements

| Requirement | Target | Acceptable |
|-------------|--------|------------|
| Drawing latency (finger move → line appears) | <100ms | <200ms |
| Positional accuracy | ±5mm | ±10mm |
| AR registration (virtual vs real alignment) | ±10mm | ±20mm |
| Frame rate (visual smoothness) | 30 FPS | 15 FPS |
| Session duration | Unlimited | 10+ minutes |

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PHYSICAL WORLD                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐         │
│    │   USER      │         │   ArUco     │         │   DRAWING   │         │
│    │  (wearing   │         │   MARKER    │         │   SPACE     │         │
│    │  glasses)   │         │  (fixed)    │         │  (in air)   │         │
│    └──────┬──────┘         └─────────────┘         └─────────────┘         │
│           │                       ▲                       ▲                 │
│           │                       │                       │                 │
│    ┌──────▼──────┐         ┌──────┴───────────────────────┴──────┐         │
│    │ META GLASSES │         │              VIEWED BY              │         │
│    │  (Camera A)  │────────►│         BOTH CAMERAS               │         │
│    └─────────────┘         └─────────────────────────────────────┘         │
│                                          ▲                                  │
│    ┌─────────────┐                       │                                  │
│    │  MACBOOK    │───────────────────────┘                                  │
│    │  (Camera B) │                                                          │
│    └─────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                              ┌──────────────┐             │
│  │ META GLASSES │                              │   MACBOOK    │             │
│  │              │   MWDATCamera SDK            │              │             │
│  │  12MP camera │ ─────────────────────────►   │   Camera     │             │
│  │  @ 2-24 FPS  │   via iPhone app             │   @ 30 FPS   │             │
│  └──────────────┘                              └──────┬───────┘             │
│         │                                             │                     │
│         │                                             │                     │
│         ▼                                             ▼                     │
│  ┌──────────────┐                              ┌──────────────┐             │
│  │   iPHONE     │         WebSocket            │   PYTHON     │             │
│  │              │ ◄───────────────────────────►│   air_paint  │             │
│  │  Frames out  │         Local WiFi           │              │             │
│  │  Points in   │                              │  Triangulate │             │
│  └──────┬───────┘                              │  + Smooth    │             │
│         │                                      └──────────────┘             │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │   ARKit      │                                                           │
│  │              │                                                           │
│  │  Render 3D   │                                                           │
│  │  lines in AR │                                                           │
│  └──────────────┘                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Role | Key Capabilities |
|-----------|------|------------------|
| **Meta Glasses** | First-person camera | 12MP, video stream via DAT SDK |
| **iPhone** | Hub + AR display | Receives glasses stream, sends to Mac, renders AR |
| **MacBook** | Processing + second camera | Triangulation, hand tracking, coordinate math |
| **ArUco Marker** | World anchor | Establishes shared coordinate system |

---

## What We Have Today

### Existing air_paint (Mac Python App)

**Repository:** https://github.com/philippbogdan/air_paint

| Feature | Status | Notes |
|---------|--------|-------|
| Stereo camera calibration | ✅ Complete | Chessboard-based, <1px RMS error |
| MediaPipe hand tracking | ✅ Complete | Index fingertip (landmark 8) |
| 3D triangulation | ✅ Complete | cv2.triangulatePoints |
| Point smoothing (EMA) | ✅ Complete | Configurable alpha |
| Outlier rejection | ✅ Complete | Reprojection error threshold |
| ArUco world anchor | ✅ Complete | Locks world origin on first click |
| Drawing state machine | ✅ Complete | Toggle on/off, undo, clear |
| Live 3D preview | ✅ Complete | Matplotlib viewport |
| Export: JSON, STL, OBJ | ✅ Complete | All working |
| Export: USDZ | ✅ Complete | Via Pixar USD library |
| Background save with progress | ✅ Complete | Non-blocking UI |

**Current camera setup:** Two USB webcams (or MacBook + external)

### Existing myentropy (iOS App)

**Location:** `/Users/MOPOLLIKA/coding/myentropy/ios/`

| Feature | Status | Notes |
|---------|--------|-------|
| Meta glasses connection | ✅ Complete | MWDATCamera SDK |
| Video streaming from glasses | ✅ Complete | 2-24 FPS, raw codec |
| Frame extraction | ✅ Complete | `videoFrame.makeUIImage()` |
| iPhone camera capture | ✅ Complete | AVFoundation |
| Audio from glasses (HFP) | ✅ Complete | 8kHz Bluetooth |
| On-device ML (FastVLM) | ✅ Complete | Vision-language model |
| Google Calendar sync | ✅ Complete | OAuth + REST API |
| Network requests | ✅ Complete | URLSession |

**Key code for glasses streaming:**
```swift
// GlassesCaptureManager.swift
videoFrameListenerToken = streamSession.videoFramePublisher.listen { videoFrame in
    if let image = videoFrame.makeUIImage() {
        self.latestVideoFrame = image  // UIImage ready to use
    }
}
```

### Hardware Available

| Device | Model | Key Specs |
|--------|-------|-----------|
| **Meta Glasses** | Ray-Ban Meta Gen 2 | 12MP camera, no depth sensor |
| **iPhone** | (User's device) | A-series chip, ARKit capable |
| **MacBook** | (User's device) | Built-in webcam, USB for external |
| **ArUco Markers** | Printed | 160mm, DICT_6X6_250 |

---

## Hardware Constraints

### Meta Ray-Ban Gen 2 Glasses

**What they CAN do:**
- Stream video to iPhone app via MWDATCamera SDK
- 12MP photos, video up to 1080p
- Bluetooth audio (HFP profile)
- Multi-hour battery life

**What they CANNOT do:**
- Stream directly to Mac (must go through iPhone)
- Provide depth data (single camera, no stereo/ToF)
- Run custom code on the glasses themselves
- Access raw camera feed without Meta SDK

**SDK Constraints:**
- Requires Meta developer account and CLIENT_TOKEN
- Video streaming limited to specific frame rates (2, 6, 12, 24 FPS)
- SDK only available for iOS (no Android, no desktop)
- Must use official MWDATCamera framework

### iPhone

**What it CAN do:**
- Run ARKit for world tracking and rendering
- Connect to glasses via MWDATCamera SDK
- Stream data to Mac via WiFi (WebSocket, Bonjour, etc.)
- Render 3D content in real-time (SceneKit, RealityKit)
- Load and display USDZ files (QLPreviewController)

**What it CANNOT do:**
- Triangulate 3D points alone (single camera, needs stereo pair)
- Process video at >30 FPS
- Maintain sub-50ms latency to external services reliably

**ARKit Capabilities:**
- World tracking (knows where phone is in space)
- Plane detection (floor, walls, tables)
- Image/marker detection (including ArUco-style markers)
- Real-time rendering up to 60 FPS
- LiDAR on Pro models (additional depth data)

### MacBook

**What it CAN do:**
- Run Python with OpenCV, MediaPipe
- Process video at 30+ FPS
- Host WebSocket/HTTP server on local network
- Connect multiple USB cameras
- Heavy computation (triangulation, mesh generation)

**What it CANNOT do:**
- Run iOS frameworks (ARKit, MWDATCamera)
- Connect to Meta glasses directly
- Display AR content (no see-through display)

### ArUco Markers

**Purpose:** Establish a shared world coordinate system between all cameras

**Requirements:**
- Must be visible to BOTH cameras simultaneously (or each camera at some point)
- Must be rigidly fixed in the environment
- Size affects detection range (160mm works at 0.5-3m)

**Limitations:**
- Detection fails if marker is occluded, blurry, or too small in frame
- Pose estimation accuracy depends on marker size and camera calibration
- Only provides pose when marker is in view

---

## Software Constraints

### Meta Wearables DAT SDK (iOS)

```
Framework: MWDATCore, MWDATCamera
Version: 0.3.0
Platform: iOS only
```

**Streaming Configuration:**
```swift
StreamSessionConfig(
    videoCodec: .raw,           // or .compressed
    resolution: .high,          // 1080p-ish
    frameRate: 24               // 2, 6, 12, or 24
)
```

**Frame Access:**
```swift
streamSession.videoFramePublisher.listen { frame in
    let uiImage = frame.makeUIImage()  // UIImage
    let timestamp = frame.timestamp     // Capture time
}
```

**Constraints:**
- Must handle device connection/disconnection gracefully
- Stream can drop frames under load
- Bluetooth audio separate from video stream

### MediaPipe Hand Tracking

```
Framework: MediaPipe Tasks API
Model: hand_landmarker.task
Platform: Python (used on Mac)
```

**Performance:**
- ~30ms per frame on modern hardware
- 21 landmarks per hand
- Confidence scores for detection and tracking

**Constraints:**
- Works best with front-facing hand (palm visible)
- Struggles with extreme angles, occlusion, fast motion
- Index fingertip = landmark 8

### OpenCV Stereo Triangulation

**Method:** `cv2.triangulatePoints(P1, P2, points1, points2)`

**Requirements:**
- Calibrated projection matrices P1, P2
- Corresponding 2D points in both images
- Points must be undistorted or distortion accounted for

**Accuracy Factors:**
- Baseline distance (wider = better depth resolution)
- Camera calibration quality (RMS error < 1px ideal)
- Point correspondence accuracy (MediaPipe jitter)
- Angle between cameras (10-60° optimal, 180° still works)

### ARKit (iOS)

**Capabilities Needed:**
- World tracking (6DOF phone pose)
- Anchor placement (position virtual content)
- Real-time rendering (SceneKit or RealityKit)

**Constraints:**
- Requires good lighting and texture for tracking
- Can lose tracking if moved too fast
- Coordinate system origin at app start position

### USDZ Format

**What it is:** Apple's AR file format (compressed USD)

**Generation:** Pixar USD library (`pxr` Python package)

**Constraints:**
- File-based, not suitable for real-time streaming
- Regenerating and reloading takes 500ms+
- QLPreviewController not designed for frequent updates

---

## The Stereo Vision Problem

### Why Stereo?

A single camera cannot determine depth - the same 2D pixel could correspond to any point along a ray into the scene. Two cameras viewing the same point from different angles allow triangulation of the 3D position.

```
        Point in 3D space (P)
               *
              /|\
             / | \
            /  |  \
           /   |   \
          /    |    \
    Camera A   |   Camera B
        \      |      /
         \     |     /
          \    |    /
           \   |   /
            \  |  /
             \ | /
              \|/
         ArUco Marker
        (world origin)
```

### Traditional Fixed Stereo

**Setup:** Two cameras rigidly mounted, known fixed relationship
**Calibration:** One-time chessboard calibration gives P1, P2
**Triangulation:** Apply fixed matrices to corresponding points

**Pros:**
- Simple, well-understood
- Calibrate once, use forever
- Sub-millimeter accuracy possible

**Cons:**
- Cameras cannot move
- Limited workspace (both cameras must see the action)
- Requires careful physical setup

### Dynamic Stereo with ArUco

**Setup:** Cameras can move, but both see an ArUco marker
**Calibration:** Still need intrinsic calibration (K, distortion)
**Triangulation:** Compute P1, P2 dynamically from each camera's ArUco pose

**How it works:**
1. Camera A detects ArUco → computes pose (R_A, t_A) in world frame
2. Camera B detects ArUco → computes pose (R_B, t_B) in world frame
3. Construct P1 = K_A @ [R_A | t_A]
4. Construct P2 = K_B @ [R_B | t_B]
5. Triangulate using dynamic P1, P2

**Pros:**
- Cameras can move (wear glasses, handheld phone, ceiling mount)
- More flexible workspace
- Can add/remove cameras

**Cons:**
- Both cameras must see marker simultaneously (or use multiple markers)
- Pose estimation adds noise
- More complex implementation

### The Opposite-Facing Camera Challenge

**Typical stereo:** Cameras 10-60° apart, both facing the scene
**Our setup:** Glasses on face + MacBook on desk = potentially 90-180° apart

**Does it work?** YES, mathematically. Triangulation works regardless of camera angle.

**Challenges:**
- Hand looks different from front vs back (MediaPipe may struggle)
- ArUco marker must be visible to both (place it to the side)
- Larger baseline = better depth resolution but harder correspondence

```
Top-down view:

                    [ArUco on wall]
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         │                │                │
    [Glasses]────────[Hand]────────[MacBook]
    (Camera A)      (drawing)      (Camera B)
         │                │                │
         └────────────────┼────────────────┘
                          │
                   [Drawing space]
```

---

## The AR Rendering Problem

### The Goal

Render 3D lines on the iPhone screen such that they appear to float in physical space at the exact location where the user drew them.

### Why This Is Hard

1. **Coordinate System Alignment:**
   - Mac triangulates points in "ArUco world frame"
   - iPhone ARKit has its own "ARKit world frame"
   - These frames have different origins and orientations
   - Must transform points from one frame to the other

2. **Real-Time Updates:**
   - New points arrive every 33-100ms
   - Must add to scene without frame drops
   - Cannot regenerate entire mesh each frame

3. **AR Registration:**
   - Virtual content must align with physical world
   - If phone moves, virtual content must stay in place
   - Requires accurate world tracking

### Coordinate System Alignment Options

**Option A: Shared ArUco Marker**
- iPhone also detects the ArUco marker using ARKit
- Both Mac and iPhone express points relative to marker
- Points transfer directly (same coordinate system)

**Option B: Initial Calibration**
- At start, user aligns phone with a known pose
- Compute transform between ARKit frame and Mac world frame
- Apply transform to all incoming points

**Option C: Continuous Alignment**
- iPhone sends its ArUco pose to Mac
- Mac transforms points into iPhone's expected frame
- Handles phone movement during session

### Rendering Approaches

**Approach 1: Point Cloud**
- Each point is a small sphere
- Simple, fast to add new points
- May look sparse

**Approach 2: Line Segments**
- Connect consecutive points with cylinders
- Looks like actual drawing
- Slightly more complex geometry

**Approach 3: Tube Mesh**
- Full tube geometry like final USDZ
- Best visual quality
- Most expensive to update

**Recommended:** Start with line segments (SCNNode with geometry), optimize later

### AR Quick Look Limitations

AR Quick Look (QLPreviewController) is **NOT suitable** for real-time rendering:
- Loads a file once, displays it
- No API to update content
- Reloading causes visible flicker
- Designed for viewing, not live updates

**Instead, use:** SCNView with ARKit, or RealityKit with custom entities

---

## The Networking Problem

### Requirements

| Metric | Target |
|--------|--------|
| Latency (one-way) | <50ms |
| Throughput | ~1 MB/s (video frames) |
| Reliability | Ordered delivery, handle drops gracefully |
| Connection | Local WiFi, same network |

### Data Flows

**iPhone → Mac:**
- Video frames from glasses (JPEG compressed, ~50-200 KB each)
- Frame rate: 2-24 FPS
- Total: 0.1 - 5 MB/s depending on settings

**Mac → iPhone:**
- 3D points (12 bytes each: 3 floats)
- Point rate: ~30 points/second while drawing
- Total: <1 KB/s
- Plus: stroke start/end events, state sync

### Protocol Options

**WebSocket:**
- Bidirectional, low latency
- Easy to implement (Python `websockets`, iOS `URLSessionWebSocketTask`)
- Works over WiFi
- Recommended for this use case

**Bonjour/mDNS:**
- Zero-config service discovery
- iPhone finds Mac automatically on local network
- Use with WebSocket for connection

**UDP:**
- Lower latency than TCP
- No guaranteed delivery (must handle packet loss)
- Good for video frames (missing frame = skip it)

**HTTP:**
- Higher latency (connection overhead)
- Fine for occasional large transfers (final USDZ)
- Not ideal for real-time streaming

### Proposed Protocol

```
1. SERVICE DISCOVERY (Bonjour)
   Mac advertises: "_airpaint._tcp" on local network
   iPhone discovers Mac automatically

2. CONNECTION (WebSocket)
   iPhone connects to Mac's WebSocket server
   Bidirectional channel established

3. VIDEO STREAM (iPhone → Mac)
   Message type: "frame"
   Payload: JPEG data + timestamp
   Rate: 24 FPS

4. POINT STREAM (Mac → iPhone)
   Message type: "point"
   Payload: {x, y, z, strokeId, timestamp}
   Rate: As detected (~30/sec)

5. CONTROL MESSAGES (Bidirectional)
   "start_drawing", "stop_drawing"
   "undo", "clear"
   "save" → triggers USDZ generation
   "usdz_ready" → followed by file transfer

6. FILE TRANSFER (Mac → iPhone)
   Message type: "usdz"
   Payload: Binary USDZ data
   Triggered on save
```

---

## Latency Budget

### End-to-End Target: <200ms

| Stage | Target | Notes |
|-------|--------|-------|
| Glasses capture | 42ms | At 24 FPS |
| Glasses → iPhone (SDK) | 20ms | Internal SDK transfer |
| iPhone → Mac (WiFi) | 30ms | WebSocket, JPEG payload |
| Hand detection (MediaPipe) | 30ms | On Mac |
| Triangulation + smoothing | 5ms | Fast math |
| Mac → iPhone (WiFi) | 20ms | WebSocket, 12-byte point |
| ARKit rendering | 16ms | At 60 FPS |
| **Total** | **163ms** | Within budget |

### Latency Optimization Strategies

1. **Parallel Processing:**
   - Hand detection on Mac runs while next frame transfers
   - Pipelining hides latency

2. **Prediction:**
   - Extrapolate finger position based on velocity
   - Reduces perceived latency
   - Risk: overshoot on direction changes

3. **Frame Skipping:**
   - If behind, skip old frames
   - Always process most recent
   - Prevents latency accumulation

4. **Compression Tuning:**
   - Lower JPEG quality = smaller files = faster transfer
   - Trade visual quality for speed
   - Hand detection works fine with 50% JPEG quality

---

## Coordinate System Challenges

### The Three Coordinate Systems

**1. Glasses Camera Frame**
- Origin: Camera optical center
- Z: Forward (into scene)
- Y: Down (or up, check SDK)
- Changes as user moves head

**2. Mac/ArUco World Frame**
- Origin: ArUco marker center
- Z: Out of marker plane
- X, Y: Marker plane axes
- Fixed in physical space

**3. iPhone ARKit Frame**
- Origin: Where ARKit initialized
- Y: Up (gravity direction)
- Can drift over time

### Alignment Strategy

**Goal:** Points computed in Mac World Frame must appear at correct position in iPhone ARKit Frame.

**Method:**
1. Both Mac and iPhone detect the same ArUco marker
2. Mac computes point P in ArUco frame
3. iPhone knows ArUco pose in ARKit frame (T_arkit_aruco)
4. Transform: P_arkit = T_arkit_aruco @ P_aruco
5. Place SCNNode at P_arkit

**Challenge:** ArUco detection has noise. Averaging or filtering helps.

### Scale Consistency

All systems must agree on units:
- Mac triangulation: millimeters
- ARKit: meters
- USDZ: configurable (we use centimeters)

**Conversion:** Divide mm by 1000 to get meters for ARKit.

---

## Alternative Architectures

### Architecture A: Current (Mac-Centric)

```
Glasses → iPhone → Mac (process) → iPhone (display)
```

**Pros:** Leverage existing air_paint code, powerful Mac processing
**Cons:** Extra network hop, latency

### Architecture B: iPhone-Centric

```
Glasses → iPhone (process + display)
         ↓
      MacBook camera → iPhone
```

iPhone receives both camera streams and does all processing.

**Pros:** Single device does everything, lower latency
**Cons:** iPhone CPU limits, harder to implement stereo on iOS

### Architecture C: No Mac Needed

```
iPhone front camera + Glasses camera = stereo pair
All on iPhone, no Mac
```

**Pros:** Simplest setup, portable
**Cons:** iPhone front camera has limited FOV, may not see hands well

### Architecture D: Glasses + iPhone Cameras (No Mac)

```
Glasses (Camera A) + iPhone rear camera (Camera B)
iPhone holds both streams
iPhone does triangulation + AR
```

**Pros:** No Mac needed, self-contained
**Cons:** Must hold iPhone to point at drawing area, awkward UX

### Architecture E: Multiple ArUco Markers

```
Marker 1 visible to Camera A
Marker 2 visible to Camera B
Known spatial relationship between markers
```

**Pros:** Cameras don't need overlapping marker view
**Cons:** Requires marker calibration, more setup

---

## Open Questions

### Technical

1. **Can MediaPipe detect hands reliably from behind (back of hand)?**
   - Needed if glasses face opposite to MacBook
   - May need to test empirically

2. **What's the minimum frame rate needed for smooth drawing?**
   - 24 FPS from glasses might be enough
   - Lower rates (2-6 FPS) likely too choppy

3. **How to handle marker occlusion during drawing?**
   - User's hand may block marker from one camera
   - Options: multiple markers, continue with last known pose

4. **ARKit marker detection vs OpenCV ArUco detection - compatible?**
   - Both can detect ArUco markers
   - Need to verify pose conventions match

5. **What's the achievable accuracy with opposite-facing cameras?**
   - Simulation and testing needed
   - May need calibration refinement

### UX

6. **Where should the user look while drawing?**
   - At their hand? At the iPhone screen? At the marker?
   - Glasses POV may not match iPhone AR view

7. **How to indicate drawing start/stop without hands?**
   - Voice command? Bluetooth button? Gesture?
   - Current: spacebar or volume button

8. **What happens when tracking is lost?**
   - Pause drawing? Warn user? Attempt recovery?

### Product

9. **Is the setup complexity acceptable for users?**
   - ArUco marker, glasses, iPhone, Mac - lots of pieces
   - Could simplify with Architecture C or D

10. **What's the killer use case?**
    - Art creation? Spatial note-taking? Communication?
    - Affects feature priorities

---

## Prior Art & References

### Related Projects

- **Gravity Sketch:** VR 3D drawing (Quest headset)
- **Tilt Brush:** Google's VR painting app
- **AR Drawing apps:** Draw in AR using phone camera
- **Leap Motion:** Hand tracking for air gestures

### Key Differences

Our approach is unique in:
- Using consumer Meta glasses (not VR headset)
- Stereo from separate devices (not built-in stereo cameras)
- AR viewing on phone (not in headset)
- Dynamic stereo with moveable cameras

### Academic References

- Hartley & Zisserman, "Multiple View Geometry in Computer Vision"
- OpenCV stereo calibration and triangulation
- ARKit documentation (Apple)
- MediaPipe Hands (Google)

### SDK Documentation

- [Meta Wearables DAT SDK](https://wearables.developer.meta.com/docs/)
- [ARKit](https://developer.apple.com/documentation/arkit/)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV Stereo](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- [Pixar USD](https://graphics.pixar.com/usd/docs/index.html)

---

## Summary

### What We Want

**"Draw with glasses viewing the thing and on iPhone we see new lines appear in AR as we draw in the same position."**

### What We Have

- ✅ Meta glasses → iPhone video streaming (myentropy)
- ✅ Stereo triangulation with ArUco anchoring (air_paint)
- ✅ MediaPipe hand tracking (air_paint)
- ✅ Point smoothing and outlier rejection (air_paint)
- ✅ USDZ export (air_paint)
- ❌ iPhone ↔ Mac networking (to build)
- ❌ Real-time AR rendering on iPhone (to build)
- ❌ Coordinate system alignment (to solve)

### The Path Forward

1. **Build WebSocket bridge** between iPhone and Mac
2. **Add ARKit rendering** to iPhone app (not QLPreviewController)
3. **Solve coordinate alignment** (shared ArUco detection)
4. **Integrate glasses streaming** into the pipeline
5. **Optimize for latency** (<200ms end-to-end)
6. **Test and refine** the drawing experience

---

*Document created: January 2026*
*Project: air_paint*
*Status: Vision document - implementation pending*
