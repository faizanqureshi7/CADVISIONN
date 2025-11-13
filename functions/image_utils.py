import cv2
import os
from pathlib import Path
from .detector import detect_objects

# list of extensions OpenCV knows how to write (common ones)
# we only use these for simple detection of whether user provided a file path
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def _ensure_output_filepath(input_path: str, output_path: str) -> str:
    """
    Ensure output_path is a filepath with an image extension.
    If output_path is a directory (exists or ends with os.sep) or has no image extension,
    create a filename based on the input filename and return a new path.

    Returns a string that is guaranteed to be a file path ending with '.png'
    unless the user explicitly provided another valid image extension.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)

    # If user explicitly passed a directory or the path looks like a directory or no suffix:
    looks_like_dir = (
        output_path.endswith(os.sep)
        or (out_p.exists() and out_p.is_dir())
    )
    has_image_ext = out_p.suffix.lower() in IMAGE_EXTS

    if looks_like_dir or not has_image_ext:
        # create directory if needed
        out_dir = output_path if looks_like_dir else str(out_p)
        # if output_path looks like a file path with no extension (e.g. "out/dir/file")
        # we want to treat the parent folder as directory.
        if not looks_like_dir and out_p.parent != Path('.'):
            out_dir = str(out_p.parent)
        # ensure directory exists
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # create filename from input file stem
        suffix = out_p.suffix.lower() if has_image_ext else '.png'
        filename = f"{in_p.stem}_detected{suffix}"
        return str(Path(out_dir) / filename)

    # otherwise user provided a filename with a valid image extension - ensure dir exists
    out_p.parent.mkdir(parents=True, exist_ok=True)
    return str(out_p)


def process_image(input_path, output_path, **detect_params):
    """
    Read an image, run detect_objects, save annotated image, and print detected boxes.
    Accepts:
      - output_path as a filename: writes there
      - output_path as a directory: writes <input_stem>_detected.png inside that directory
      - output_path missing an extension: will create <parent>/<input_stem>_detected.png

    Keeps the original behaviour and console output.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f"Error: could not read image '{input_path}'")

    out_img, objects, meta = detect_objects(img, **detect_params)

    print(f"Detected {len(objects)} object{'s' if len(objects) != 1 else ''}.")
    for i, o in enumerate(objects, start=1):
        x, y, w_box, h_box = o['bbox']
        print(f"  {i}) bbox=({x},{y},{w_box},{h_box}), points={o['points_count']}")

    # ensure we have a proper filepath for writing
    final_output_path = _ensure_output_filepath(input_path, output_path)

    # write image
    success = cv2.imwrite(final_output_path, out_img)
    if not success:
        raise RuntimeError(f"cv2.imwrite failed for path: {final_output_path}")

    print(f"Saved output image to: {final_output_path}")

    return out_img, objects, meta