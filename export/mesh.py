"""STL/OBJ tube mesh generation from strokes."""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from drawing.stroke import Stroke
from config.settings import EXPORT


def generate_tube_vertices(
    points: np.ndarray,
    radius: float = EXPORT.TUBE_RADIUS_MM,
    segments: int = EXPORT.TUBE_SEGMENTS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate tube mesh vertices and faces from a polyline.

    Args:
        points: (N, 3) array of 3D points
        radius: Tube radius in mm
        segments: Number of segments around the tube circumference

    Returns:
        (vertices, faces) tuple where:
        - vertices is (M, 3) array of vertex positions
        - faces is (F, 3) array of triangle indices
    """
    if len(points) < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)

    vertices = []
    faces = []

    n_points = len(points)

    # Generate circles at each point
    for i in range(n_points):
        # Calculate tangent direction
        if i == 0:
            tangent = points[1] - points[0]
        elif i == n_points - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[i+1] - points[i-1]

        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

        # Find perpendicular vectors
        if abs(tangent[0]) < 0.9:
            perp1 = np.cross(tangent, [1, 0, 0])
        else:
            perp1 = np.cross(tangent, [0, 1, 0])
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
        perp2 = np.cross(tangent, perp1)

        # Generate circle vertices
        for j in range(segments):
            angle = 2 * np.pi * j / segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(points[i] + offset)

    vertices = np.array(vertices)

    # Generate faces (triangles connecting adjacent circles)
    for i in range(n_points - 1):
        for j in range(segments):
            # Indices of the quad corners
            v0 = i * segments + j
            v1 = i * segments + (j + 1) % segments
            v2 = (i + 1) * segments + j
            v3 = (i + 1) * segments + (j + 1) % segments

            # Two triangles per quad
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # Cap the ends
    # Start cap
    center_start = len(vertices)
    vertices = np.vstack([vertices, points[0].reshape(1, 3)])
    for j in range(segments):
        v0 = j
        v1 = (j + 1) % segments
        faces.append([center_start, v1, v0])

    # End cap
    center_end = len(vertices)
    vertices = np.vstack([vertices, points[-1].reshape(1, 3)])
    base = (n_points - 1) * segments
    for j in range(segments):
        v0 = base + j
        v1 = base + (j + 1) % segments
        faces.append([center_end, v0, v1])

    return vertices, np.array(faces, dtype=int)


def generate_tube_mesh(
    strokes: List[Stroke],
    radius: float = EXPORT.TUBE_RADIUS_MM,
    segments: int = EXPORT.TUBE_SEGMENTS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate combined tube mesh from multiple strokes.

    Args:
        strokes: List of strokes to convert
        radius: Tube radius in mm
        segments: Number of segments around tubes

    Returns:
        (vertices, faces) for the combined mesh
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for stroke in strokes:
        if stroke.is_empty or stroke.num_points < 2:
            continue

        points = stroke.to_array()
        vertices, faces = generate_tube_vertices(points, radius, segments)

        if len(vertices) > 0:
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(vertices)

    if not all_vertices:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)

    return np.vstack(all_vertices), np.vstack(all_faces)


def save_stl(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Path,
    binary: bool = True
) -> None:
    """
    Save mesh as STL file.

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) triangle indices
        path: Output file path
        binary: Whether to save as binary STL (smaller file)
    """
    try:
        from stl import mesh as stl_mesh

        # Create the mesh
        mesh = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))

        for i, face in enumerate(faces):
            for j in range(3):
                mesh.vectors[i][j] = vertices[face[j]]

        # Save
        if binary:
            mesh.save(str(path))
        else:
            mesh.save(str(path), mode=stl_mesh.Mode.ASCII)

        print(f"Saved STL to {path}")

    except ImportError:
        print("numpy-stl not installed. Install with: pip install numpy-stl")
        # Fallback: save as ASCII STL manually
        _save_stl_ascii(vertices, faces, path)


def _save_stl_ascii(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Path
) -> None:
    """Fallback ASCII STL writer."""
    with open(path, 'w') as f:
        f.write("solid drawing\n")
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            # Calculate normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = np.array([0, 0, 1])

            f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
            f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
            f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid drawing\n")

    print(f"Saved ASCII STL to {path}")


def save_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Path
) -> None:
    """
    Save mesh as OBJ file.

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) triangle indices (0-indexed)
        path: Output file path
    """
    with open(path, 'w') as f:
        f.write("# 3D Air Painting Export\n")
        f.write(f"# {len(vertices)} vertices, {len(faces)} faces\n\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Write faces (OBJ uses 1-indexed)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"Saved OBJ to {path}")
