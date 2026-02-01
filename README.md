# Air Paint

Draw in the air with your finger. Two webcams track it in 3D. See it on your iPhone in AR.

---

## The Math

### Stereo Geometry

Two cameras observe the same point from different positions. Given their relative pose, we triangulate the 3D position.

**Camera Intrinsic Matrix** (per camera):

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

where $f_x, f_y$ are focal lengths in pixels and $(c_x, c_y)$ is the principal point.

**Distortion Coefficients**:

$$
D = \begin{bmatrix} k_1 & k_2 & p_1 & p_2 & k_3 \end{bmatrix}
$$

where $k_i$ are radial and $p_i$ are tangential distortion parameters.

### Stereo Calibration

Camera B's pose relative to Camera A is defined by:

$$
R \in SO(3), \quad T \in \mathbb{R}^3
$$

**Projection Matrices**:

$$
P_1 = K_1 \begin{bmatrix} I & \mathbf{0} \end{bmatrix}, \quad P_2 = K_2 \begin{bmatrix} R & T \end{bmatrix}
$$

Camera A is at the origin; Camera B is offset by rotation $R$ and translation $T$ (baseline).

### Triangulation

Given corresponding 2D points $\mathbf{x}_1 = (u_1, v_1)$ and $\mathbf{x}_2 = (u_2, v_2)$, we solve for $\mathbf{X} = (X, Y, Z, W)^T$ in homogeneous coordinates:

$$
A \mathbf{X} = \mathbf{0}
$$

where:

$$
A = \begin{bmatrix} u_1 P_1^{3T} - P_1^{1T} \\ v_1 P_1^{3T} - P_1^{2T} \\ u_2 P_2^{3T} - P_2^{1T} \\ v_2 P_2^{3T} - P_2^{2T} \end{bmatrix}
$$

Solved via SVD. The 3D point is recovered as:

$$
\mathbf{X}_{3D} = \begin{bmatrix} X/W \\ Y/W \\ Z/W \end{bmatrix}
$$

### Reprojection Error

Quality metric—project the 3D point back to 2D and measure deviation:

$$
\hat{\mathbf{x}} = \pi(P \cdot \mathbf{X}_h) = \begin{bmatrix} \hat{u} \\ \hat{v} \end{bmatrix}
$$

$$
\epsilon = \sqrt{(u - \hat{u})^2 + (v - \hat{v})^2}
$$

Points with $\epsilon > \tau$ are rejected as outliers.

### World Transform (ArUco)

ArUco marker detection yields rotation vector $\mathbf{r}$ and translation vector $\mathbf{t}$.

Convert to rotation matrix via Rodrigues' formula:

$$
R = I + \sin\theta \cdot [\mathbf{k}]_\times + (1 - \cos\theta) \cdot [\mathbf{k}]_\times^2
$$

where $\theta = \|\mathbf{r}\|$ and $\mathbf{k} = \mathbf{r}/\theta$.

**Camera-to-World Transform**:

$$
T_{CW} = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in SE(3)
$$

Transform points from camera frame to world frame:

$$
\mathbf{X}_W = T_{CW} \cdot \begin{bmatrix} \mathbf{X}_C \\ 1 \end{bmatrix}
$$

### Point Smoothing

Exponential moving average reduces jitter:

$$
\mathbf{X}_t^{smooth} = \alpha \cdot \mathbf{X}_t + (1 - \alpha) \cdot \mathbf{X}_{t-1}^{smooth}
$$

where $\alpha = 0.5$ balances responsiveness and stability.

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
