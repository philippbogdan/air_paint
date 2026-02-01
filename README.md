# Air Paint

Draw in the air with your finger. Two webcams track it in 3D. See it on your iPhone in AR.

---

## The Math

### Stereo Geometry

Two cameras see the same point from different positions. Given their relative pose, we can triangulate the 3D position.

**Camera intrinsic matrix** (per camera):
```
    [ fx   0  cx ]
K = [  0  fy  cy ]
    [  0   0   1 ]
```
- `fx, fy`: focal lengths in pixels
- `cx, cy`: principal point (image center)

**Distortion coefficients**:
```
D = [k1, k2, p1, p2, k3]
```
Radial (`k`) and tangential (`p`) lens distortion.

### Stereo Calibration

Camera B's pose relative to Camera A:
```
R = 3x3 rotation matrix
T = 3x1 translation vector (baseline)
```

**Projection matrices**:
```
P1 = K1 @ [I | 0]        # Camera A at origin
P2 = K2 @ [R | T]        # Camera B offset by R, T
```

### Triangulation

Given 2D point `(u, v)` in each camera, solve:

```
         [ u1 * P1[2,:] - P1[0,:] ]       [ 0 ]
         [ v1 * P1[2,:] - P1[1,:] ]       [ 0 ]
A * X =  [ u2 * P2[2,:] - P2[0,:] ] * X = [ 0 ]
         [ v2 * P2[2,:] - P2[1,:] ]       [ 0 ]
```

Solve via SVD. `X` is the 3D point in homogeneous coordinates:
```
X = [x, y, z, w]^T
point_3d = [x/w, y/w, z/w]
```

### Reprojection Error

Quality check—project 3D point back to 2D and measure pixel distance:
```
projected = P @ [X, Y, Z, 1]^T
u' = projected[0] / projected[2]
v' = projected[1] / projected[2]

error = sqrt((u - u')^2 + (v - v')^2)
```

### World Transform (ArUco)

ArUco marker detection gives rotation `rvec` and translation `tvec`.

Convert to 4x4 transform:
```
R, _ = cv2.Rodrigues(rvec)

        [ R   | tvec ]
T_CW =  [-----+------]    # Camera to World
        [ 0 0 0 |  1  ]

T_WC = inverse(T_CW)      # World to Camera
```

Transform points from camera frame to world frame:
```
point_world = T_CW @ [x, y, z, 1]^T
```

### Point Smoothing

Exponential moving average to reduce jitter:
```
smoothed = α * current + (1 - α) * previous
```
`α = 0.5` balances responsiveness and stability.

---

## Appendix: Quick Start

### Requirements

- macOS 13+
- Python 3.10+
- Two webcams (or Mac webcam + iPhone via Continuity Camera)
- iPhone with iOS 16+ (for AR viewing)

### Install

```bash
git clone https://github.com/philippbogdan/air_paint.git
cd air_paint
pip install -r requirements.txt
```

### Calibrate

Print a 9x6 chessboard. Run:
```bash
python main.py --calibrate
```
Hold chessboard visible to both cameras. Press SPACE to capture (15+ frames). Press C to calibrate. Press S to save.

### Run

```bash
python main.py --server-local
```

1. Show ArUco marker to camera
2. Press SPACE to lock origin
3. Remove marker
4. Press SPACE to start drawing
5. Wave your finger
6. Press S to export USDZ

### iPhone

Open the iOS app, enter your Mac's IP. Point phone at the origin axes (1m in front). Watch the drawing appear.

### Controls

| Key | Action |
|-----|--------|
| SPACE | Lock origin / Toggle drawing |
| S | Export USDZ |
| C | Recalibrate |
| Q | Quit |
