# 3D Air Painting - Claude Code Instructions

## Project Overview

This is a macOS Python application for "3D air painting" using stereo vision from two cameras (Mac webcam + iPhone via USB-C). Users draw in 3D space by moving their index finger, with drawing toggled by spacebar or Bluetooth shutter button.

## Architecture

```
INPUT → PROCESSING → STATE → VISUALIZATION → EXPORT
(Cameras) (MediaPipe + Triangulation) (DrawingManager) (OpenCV + Matplotlib) (STL/USDZ)
```

## Directory Structure

- `main.py` - Application entry point
- `config/settings.py` - Global configuration constants
- `capture/` - Camera capture abstraction
- `calibration/` - Stereo calibration tools
- `tracking/` - Hand tracking and 3D triangulation
- `drawing/` - Stroke data structures and state management
- `ui/` - Window composition and input handling
- `export/` - Mesh generation (STL/OBJ/USDZ)

## Key Commands

```bash
# Run the application
python main.py

# Run with specific cameras
python main.py --camera-a 0 --camera-b 1

# List available cameras
python main.py --list-cameras

# Run calibration
python main.py --calibrate
# or
python -m calibration.calibrate

# Test camera availability
python -c "from capture.sync import list_available_cameras; print(list_available_cameras())"
```

## Calibration

The stereo calibration requires a chessboard pattern (9x6 inner corners by default).

1. Print a chessboard pattern
2. Run `python main.py --calibrate`
3. Hold the chessboard visible to both cameras
4. Press SPACE to capture (need 15+ frames)
5. Press C to calibrate
6. Press S to save

Calibration is saved to `data/calibration/stereo_calib.npz`.

## Key Files

| File | Purpose |
|------|---------|
| `calibration/stereo.py` | StereoCalibration dataclass with K, D, R, T, P matrices |
| `tracking/triangulate.py` | cv2.triangulatePoints wrapper |
| `drawing/manager.py` | Stroke state management with toggle logic |
| `ui/window.py` | Composite visualization with matplotlib 3D view |

## Technical Details

### Triangulation
- Uses cv2.triangulatePoints with calibrated projection matrices
- Points are undistorted before triangulation
- Results are in camera A's coordinate system (mm)

### Drawing Toggle
- Spacebar toggles drawing on/off
- When drawing: index finger tip position is triangulated and added to current stroke
- When paused: hand tracking continues but no points added

### Export Formats
- `strokes.json` - Raw point data
- `drawing.stl` - 3D mesh (tube geometry around strokes)
- `drawing.obj` - Standard 3D format
- `drawing.usdz` - AR format for iPhone viewing

## Dependencies

- numpy, opencv-python, mediapipe, matplotlib
- Optional: numpy-stl, trimesh (for mesh export)

## Common Issues

1. **iPhone not detected**: Ensure USB-C connection and Continuity Camera enabled (macOS 13+)
2. **Calibration drift**: Re-calibrate if cameras are moved
3. **Low FPS**: Reduce resolution in `config/settings.py`
4. **No 3D data**: Must run calibration first for triangulation

## Testing

```bash
# Test cameras
python main.py --list-cameras

# Test hand tracking (runs with whatever cameras available)
python main.py

# Verify calibration quality (should be < 1.0 RMS error)
python -c "from calibration.stereo import load_calibration; c = load_calibration(); print(f'RMS: {c.rms_error}' if c else 'No calibration')"
```
