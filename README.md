# Air Paint

Draw in the air with your finger. Two webcams track it in 3D. See it on your iPhone in AR.

---

## The Math

### Pinhole Camera Model

We adopt the classical pinhole camera model. Each camera is characterized by its **intrinsic matrix**:

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 3}
$$

where $f_x, f_y$ are focal lengths in pixels and $(c_x, c_y)$ is the principal point.

Lens distortion is modeled using the **Brown-Conrady distortion model**:

$$
D = \begin{bmatrix} k_1 & k_2 & p_1 & p_2 & k_3 \end{bmatrix}
$$

where $k_i$ are radial distortion coefficients and $p_i$ are tangential (decentering) distortion coefficients.

### Epipolar Geometry

The geometric relationship between two cameras is governed by **epipolar geometry**. Camera B's pose relative to Camera A lies in the **Special Euclidean group**:

$$
(R, T) \in SE(3), \quad R \in SO(3), \quad T \in \mathbb{R}^3
$$

The **Essential Matrix** encodes this relationship:

$$
E = [T]_\times R
$$

where $[T]_\times$ is the skew-symmetric matrix form of $T$.

**Projection matrices** follow from the **perspective projection equation**:

$$
P_1 = K_1 \begin{bmatrix} I_{3 \times 3} & \mathbf{0} \end{bmatrix}, \quad P_2 = K_2 \begin{bmatrix} R & T \end{bmatrix}
$$

### Triangulation via Direct Linear Transform

Given corresponding points $\mathbf{x}_1 = (u_1, v_1)$ and $\mathbf{x}_2 = (u_2, v_2)$ satisfying the **epipolar constraint**, we recover the 3D point using the **Direct Linear Transform (DLT)** method.

We seek $\mathbf{X} \in \mathbb{P}^3$ (projective 3-space) such that:

$$
A \mathbf{X} = \mathbf{0}
$$

where $A$ is constructed from the **cross-product elimination** of the projection equations:

$$
A = \begin{bmatrix} u_1 \mathbf{p}_1^{3\top} - \mathbf{p}_1^{1\top} \\ v_1 \mathbf{p}_1^{3\top} - \mathbf{p}_1^{2\top} \\ u_2 \mathbf{p}_2^{3\top} - \mathbf{p}_2^{1\top} \\ v_2 \mathbf{p}_2^{3\top} - \mathbf{p}_2^{2\top} \end{bmatrix}
$$

The solution is the **null space** of $A$, computed via **Singular Value Decomposition (SVD)**. The 3D point in Euclidean coordinates:

$$
\mathbf{X}_{3D} = \begin{bmatrix} X/W & Y/W & Z/W \end{bmatrix}^\top
$$

### Reprojection Error & Outlier Rejection

We validate triangulation using the **reprojection error**—the Euclidean distance between observed and predicted image points:

$$
\epsilon = \| \mathbf{x} - \hat{\mathbf{x}} \|_2 = \sqrt{(u - \hat{u})^2 + (v - \hat{v})^2}
$$

Points exceeding threshold $\tau$ are rejected per **RANSAC-style** outlier filtering.

### Rigid Body Transform (ArUco Marker)

World frame registration uses **ArUco fiducial markers**. Detection yields the axis-angle representation $\mathbf{r} \in \mathfrak{so}(3)$.

Conversion to rotation matrix uses **Rodrigues' rotation formula**:

$$
R = I + \sin\theta \cdot [\mathbf{k}]_\times + (1 - \cos\theta) \cdot [\mathbf{k}]_\times^2
$$

where $\theta = \|\mathbf{r}\|$ is the rotation angle and $\mathbf{k} = \mathbf{r}/\theta$ is the unit axis.

The **homogeneous transformation matrix**:

$$
T_{CW} = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix} \in SE(3)
$$

Points transform from camera frame $\mathcal{F}_C$ to world frame $\mathcal{F}_W$:

$$
\mathbf{X}_W = T_{CW} \cdot \tilde{\mathbf{X}}_C
$$

where $\tilde{\mathbf{X}}_C$ denotes homogeneous coordinates.

### Temporal Filtering

Raw triangulated points exhibit jitter. We apply an **Exponential Moving Average (EMA)** filter:

$$
\mathbf{X}_t^{(s)} = \alpha \cdot \mathbf{X}_t + (1 - \alpha) \cdot \mathbf{X}_{t-1}^{(s)}
$$

with smoothing factor $\alpha = 0.5$, balancing latency and stability per the **bias-variance tradeoff**.

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
