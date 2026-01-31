#!/usr/bin/env python3
"""
Generate a printable chessboard pattern for camera calibration.

Usage:
    python -m calibration.generate_chessboard

This creates a chessboard with 9x6 inner corners (10x7 squares),
with 25mm squares - suitable for A4 or Letter paper.
"""

import numpy as np
from pathlib import Path

# Import settings or use defaults
try:
    from config.settings import CALIBRATION
    INNER_CORNERS = CALIBRATION.CHESSBOARD_SIZE  # (9, 6)
    SQUARE_SIZE_MM = CALIBRATION.SQUARE_SIZE_MM  # 25.0
except ImportError:
    INNER_CORNERS = (9, 6)
    SQUARE_SIZE_MM = 25.0


def generate_chessboard_image(
    inner_corners: tuple = INNER_CORNERS,
    square_size_mm: float = SQUARE_SIZE_MM,
    dpi: int = 300,
    margin_mm: float = 15.0
) -> np.ndarray:
    """
    Generate a chessboard pattern image.

    Args:
        inner_corners: Number of inner corners (cols, rows)
        square_size_mm: Size of each square in mm
        dpi: Output resolution in dots per inch
        margin_mm: White margin around the board in mm

    Returns:
        numpy array of the chessboard image (grayscale)
    """
    # Number of squares is inner_corners + 1
    num_cols = inner_corners[0] + 1  # 10 squares wide
    num_rows = inner_corners[1] + 1  # 7 squares tall

    # Calculate dimensions in pixels
    mm_to_inch = 1 / 25.4
    square_size_px = int(square_size_mm * mm_to_inch * dpi)
    margin_px = int(margin_mm * mm_to_inch * dpi)

    # Total image size
    board_width_px = num_cols * square_size_px
    board_height_px = num_rows * square_size_px
    img_width = board_width_px + 2 * margin_px
    img_height = board_height_px + 2 * margin_px

    # Create white image
    image = np.ones((img_height, img_width), dtype=np.uint8) * 255

    # Draw black squares
    for row in range(num_rows):
        for col in range(num_cols):
            # Checkerboard pattern: black if (row + col) is even
            if (row + col) % 2 == 0:
                x1 = margin_px + col * square_size_px
                y1 = margin_px + row * square_size_px
                x2 = x1 + square_size_px
                y2 = y1 + square_size_px
                image[y1:y2, x1:x2] = 0

    return image


def main():
    import cv2

    # Generate the chessboard
    print("Generating chessboard pattern...")
    print(f"  Inner corners: {INNER_CORNERS[0]} x {INNER_CORNERS[1]}")
    print(f"  Squares: {INNER_CORNERS[0]+1} x {INNER_CORNERS[1]+1}")
    print(f"  Square size: {SQUARE_SIZE_MM} mm")

    image = generate_chessboard_image()

    # Calculate actual dimensions
    num_cols = INNER_CORNERS[0] + 1
    num_rows = INNER_CORNERS[1] + 1
    board_width_mm = num_cols * SQUARE_SIZE_MM
    board_height_mm = num_rows * SQUARE_SIZE_MM
    margin_mm = 15.0
    total_width_mm = board_width_mm + 2 * margin_mm
    total_height_mm = board_height_mm + 2 * margin_mm

    print(f"\nBoard dimensions: {board_width_mm} x {board_height_mm} mm")
    print(f"Total with margins: {total_width_mm} x {total_height_mm} mm")
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels @ 300 DPI")

    # Save the image
    output_dir = Path(__file__).parent.parent
    output_path = output_dir / "chessboard_9x6_25mm.png"
    cv2.imwrite(str(output_path), image)
    print(f"\nSaved to: {output_path}")

    # Also save as JPG
    jpg_path = output_dir / "chessboard_9x6_25mm.jpg"
    cv2.imwrite(str(jpg_path), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"Saved to: {jpg_path}")

    # Print instructions
    print("\n" + "=" * 60)
    print("PRINTING INSTRUCTIONS")
    print("=" * 60)
    print(f"""
Paper size needed: A4 (210 x 297 mm) or US Letter (216 x 279 mm)

The chessboard is:
  - {board_width_mm} mm wide x {board_height_mm} mm tall
  - {num_cols} x {num_rows} squares
  - {INNER_CORNERS[0]} x {INNER_CORNERS[1]} inner corners (what OpenCV detects)

IMPORTANT - Print settings:
  1. Open the image in Preview (Mac) or any image viewer
  2. Print at "Actual Size" or "100%" scale
  3. Do NOT select "Fit to Page" or "Scale to Fit"
  4. Use a flat, rigid surface (glue to cardboard or foam board)

To verify correct size:
  - Measure any square with a ruler
  - It should be exactly {SQUARE_SIZE_MM} mm ({SQUARE_SIZE_MM/25.4:.2f} inches)

If squares are wrong size:
  - Your printer scaled the image
  - Try "Custom Scale: 100%" in print dialog
  - Or adjust scale until squares measure {SQUARE_SIZE_MM} mm
""")


if __name__ == "__main__":
    main()
