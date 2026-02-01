#!/usr/bin/env python3
"""
Generate a printable ArUco marker for world coordinate frame.

Usage:
    python -m calibration.generate_aruco

This creates an ArUco marker with:
- Dictionary: 6x6 (250 markers)
- Marker ID: 0 (default)
- Size: 80mm x 80mm (default)
- White border for detection reliability
"""

import cv2
import numpy as np
from pathlib import Path

# Import settings or use defaults
try:
    from config.settings import ARUCO
    MARKER_ID = ARUCO.MARKER_ID
    MARKER_SIZE_MM = ARUCO.MARKER_SIZE_MM
    DICTIONARY_ID = ARUCO.DICTIONARY_ID
except ImportError:
    MARKER_ID = 0
    MARKER_SIZE_MM = 80.0
    DICTIONARY_ID = 10  # cv2.aruco.DICT_6X6_250


def generate_aruco_marker(
    marker_id: int = MARKER_ID,
    marker_size_mm: float = MARKER_SIZE_MM,
    border_mm: float = 10.0,
    dpi: int = 300,
    dictionary_id: int = DICTIONARY_ID
) -> np.ndarray:
    """
    Generate an ArUco marker image.

    Args:
        marker_id: Marker ID (0-249 for 6x6_250 dictionary)
        marker_size_mm: Size of the marker (not including border) in mm
        border_mm: White border around the marker in mm
        dpi: Output resolution in dots per inch
        dictionary_id: ArUco dictionary ID

    Returns:
        numpy array of the marker image (grayscale)
    """
    # Get the ArUco dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    # Calculate dimensions in pixels
    mm_to_inch = 1 / 25.4
    marker_size_px = int(marker_size_mm * mm_to_inch * dpi)
    border_px = int(border_mm * mm_to_inch * dpi)

    # Generate the marker
    # The marker itself needs to be a specific size for OpenCV
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)

    # Add white border
    total_size = marker_size_px + 2 * border_px
    bordered_img = np.ones((total_size, total_size), dtype=np.uint8) * 255

    # Place marker in center
    bordered_img[border_px:border_px + marker_size_px,
                 border_px:border_px + marker_size_px] = marker_img

    return bordered_img


def main():
    print("Generating ArUco marker...")
    print(f"  Dictionary: 6x6 (250 markers)")
    print(f"  Marker ID: {MARKER_ID}")
    print(f"  Marker size: {MARKER_SIZE_MM} mm")

    border_mm = 10.0
    total_size_mm = MARKER_SIZE_MM + 2 * border_mm

    image = generate_aruco_marker()

    print(f"\nMarker dimensions: {MARKER_SIZE_MM} x {MARKER_SIZE_MM} mm")
    print(f"Total with border: {total_size_mm} x {total_size_mm} mm")
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels @ 300 DPI")

    # Save the image
    output_dir = Path(__file__).parent.parent
    output_path = output_dir / f"aruco_marker_{MARKER_ID}_{int(MARKER_SIZE_MM)}mm.png"
    cv2.imwrite(str(output_path), image)
    print(f"\nSaved to: {output_path}")

    # Also save as JPG
    jpg_path = output_dir / f"aruco_marker_{MARKER_ID}_{int(MARKER_SIZE_MM)}mm.jpg"
    cv2.imwrite(str(jpg_path), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"Saved to: {jpg_path}")

    # Save a version for iOS assets (named aruco_marker_0.png)
    ios_assets_path = output_dir / "AirPaintAR" / "AirPaintAR" / "Assets.xcassets"
    if ios_assets_path.exists():
        # Save directly in the assets folder for easy import
        ios_marker_path = ios_assets_path / f"aruco_marker_{MARKER_ID}.imageset"
        ios_marker_path.mkdir(exist_ok=True)

        # Save the image
        cv2.imwrite(str(ios_marker_path / f"aruco_marker_{MARKER_ID}.png"), image)

        # Create Contents.json for the asset
        contents_json = '''{
  "images" : [
    {
      "filename" : "aruco_marker_0.png",
      "idiom" : "universal",
      "scale" : "1x"
    }
  ],
  "info" : {
    "author" : "xcode",
    "version" : 1
  }
}'''
        (ios_marker_path / "Contents.json").write_text(contents_json)
        print(f"Saved iOS asset to: {ios_marker_path}")
    else:
        # Just save in the output directory
        ios_path = output_dir / f"aruco_marker_{MARKER_ID}.png"
        cv2.imwrite(str(ios_path), image)
        print(f"Saved iOS marker to: {ios_path}")
        print(f"  (Add this to AirPaintAR Assets.xcassets as 'aruco_marker_0')")

    # Print instructions
    print("\n" + "=" * 60)
    print("PRINTING INSTRUCTIONS")
    print("=" * 60)
    print(f"""
Paper size needed: A4 (210 x 297 mm) or US Letter (216 x 279 mm)

The ArUco marker is:
  - {MARKER_SIZE_MM} mm x {MARKER_SIZE_MM} mm (marker itself)
  - {total_size_mm} mm x {total_size_mm} mm (with white border)
  - ID: {MARKER_ID}
  - Dictionary: 6x6_250

IMPORTANT - Print settings:
  1. Open the image in Preview (Mac) or any image viewer
  2. Print at "Actual Size" or "100%" scale
  3. Do NOT select "Fit to Page" or "Scale to Fit"
  4. Use a flat, rigid surface (glue to cardboard or foam board)

To verify correct size:
  - Measure the black marker area (not including white border)
  - It should be exactly {MARKER_SIZE_MM} mm ({MARKER_SIZE_MM/25.4:.2f} inches)

PLACEMENT:
  - Place the marker on a flat surface visible to Camera A
  - Keep it stationary during drawing session
  - The marker defines the "world" origin:
    - Center of marker = origin (0, 0, 0)
    - X-axis = right along marker
    - Y-axis = up along marker
    - Z-axis = out of marker (towards camera)
""")


if __name__ == "__main__":
    main()
