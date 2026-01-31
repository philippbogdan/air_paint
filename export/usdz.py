"""USDZ conversion for AR viewing on iPhone."""

import subprocess
import shutil
from pathlib import Path
from typing import Optional


def convert_to_usdz(
    obj_path: Path,
    output_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Convert OBJ file to USDZ format for AR viewing.

    Uses Apple's usdzconvert tool (part of Reality Converter or Xcode).

    Args:
        obj_path: Path to input OBJ file
        output_path: Output USDZ path (default: same name with .usdz extension)

    Returns:
        Path to created USDZ file, or None if conversion failed
    """
    if output_path is None:
        output_path = obj_path.with_suffix('.usdz')

    # Try different conversion methods

    # Method 1: usdzconvert (from Reality Converter / Model I/O)
    if _try_usdzconvert(obj_path, output_path):
        return output_path

    # Method 2: xcrun usdz_converter (Xcode)
    if _try_xcrun_usdz(obj_path, output_path):
        return output_path

    # Method 3: trimesh + usd-core (if installed)
    if _try_trimesh_usdz(obj_path, output_path):
        return output_path

    print("USDZ conversion failed. Install Reality Converter or use trimesh with usd-core.")
    print("  - Reality Converter: https://developer.apple.com/augmented-reality/tools/")
    print("  - trimesh: pip install trimesh usd-core")

    return None


def _try_usdzconvert(obj_path: Path, output_path: Path) -> bool:
    """Try conversion with usdzconvert tool."""
    usdzconvert = shutil.which('usdzconvert')
    if usdzconvert is None:
        return False

    try:
        result = subprocess.run(
            [usdzconvert, str(obj_path), str(output_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"Converted to USDZ using usdzconvert: {output_path}")
            return True
        else:
            print(f"usdzconvert failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"usdzconvert error: {e}")
        return False


def _try_xcrun_usdz(obj_path: Path, output_path: Path) -> bool:
    """Try conversion with xcrun usdz_converter (Xcode)."""
    xcrun = shutil.which('xcrun')
    if xcrun is None:
        return False

    try:
        result = subprocess.run(
            [xcrun, 'usdz_converter', str(obj_path), str(output_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"Converted to USDZ using xcrun: {output_path}")
            return True
        else:
            # xcrun might not have usdz_converter available
            return False
    except Exception:
        return False


def _try_trimesh_usdz(obj_path: Path, output_path: Path) -> bool:
    """Try conversion using trimesh library."""
    try:
        import trimesh

        # Load the OBJ
        mesh = trimesh.load(str(obj_path))

        # Export to USDZ
        # Note: trimesh's USDZ export requires usd-core
        mesh.export(str(output_path), file_type='usdz')
        print(f"Converted to USDZ using trimesh: {output_path}")
        return True

    except ImportError:
        return False
    except Exception as e:
        print(f"trimesh USDZ export error: {e}")
        # Fall through to try direct USD export
        return _try_direct_usd(obj_path, output_path)


def _try_direct_usd(obj_path: Path, output_path: Path) -> bool:
    """Try conversion using Pixar USD library directly."""
    try:
        from pxr import Usd, UsdGeom, UsdUtils, Gf, Vt
        import numpy as np

        # Load OBJ manually
        vertices = []
        faces = []

        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()[1:4]
                    vertices.append([float(p) for p in parts])
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    # OBJ faces are 1-indexed, may have v/vt/vn format
                    face = []
                    for p in parts:
                        idx = int(p.split('/')[0]) - 1
                        face.append(idx)
                    faces.append(face)

        if not vertices or not faces:
            print("OBJ file is empty or invalid")
            return False

        vertices = np.array(vertices)

        # Create USD stage
        usda_path = output_path.with_suffix('.usda')
        stage = Usd.Stage.CreateNew(str(usda_path))

        # Set up axis and units for AR (Y-up, meters)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 0.001)  # mm to meters

        # Create mesh
        mesh_path = '/Drawing/Mesh'
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        # Set vertices (convert mm to meters for AR)
        points = [Gf.Vec3f(float(v[0])/1000, float(v[1])/1000, float(v[2])/1000) for v in vertices]
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))

        # Set face vertex counts and indices
        face_counts = [len(f) for f in faces]
        face_indices = []
        for f in faces:
            face_indices.extend(f)

        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))

        # Set subdivision scheme to none (we want hard edges)
        mesh.GetSubdivisionSchemeAttr().Set('none')

        # Add a simple green material
        material_path = '/Drawing/Material'
        from pxr import UsdShade, Sdf
        material = UsdShade.Material.Define(stage, material_path)

        shader = UsdShade.Shader.Define(stage, f'{material_path}/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.8, 0.3))  # Green
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.5)

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')

        # Bind material to mesh
        UsdShade.MaterialBindingAPI(mesh).Bind(material)

        # Set default prim
        stage.SetDefaultPrim(stage.GetPrimAtPath('/Drawing'))

        # Save USDA
        stage.GetRootLayer().Save()

        # Convert to USDZ
        UsdUtils.CreateNewUsdzPackage(str(usda_path), str(output_path))

        # Clean up intermediate file
        usda_path.unlink()

        print(f"Converted to USDZ using USD library: {output_path}")
        return True

    except ImportError as e:
        print(f"USD library not available: {e}")
        return False
    except Exception as e:
        print(f"USD export error: {e}")
        return False


def is_usdz_supported() -> bool:
    """Check if USDZ conversion is available on this system."""
    # Check for any available conversion tool
    if shutil.which('usdzconvert'):
        return True

    if shutil.which('xcrun'):
        # Check if usdz_converter is available
        try:
            result = subprocess.run(
                ['xcrun', '--find', 'usdz_converter'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    # Check for trimesh with USD support
    try:
        import trimesh
        # Try to check if USD export is available
        # This is a heuristic - actual support depends on usd-core
        return True
    except ImportError:
        pass

    return False
