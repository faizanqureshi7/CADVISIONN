import cv2
import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
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
    try:
        in_p = Path(input_path)
        out_p = Path(output_path)

        # If user explicitly passed a directory or the path looks like a directory or no suffix:
        looks_like_dir = (
            output_path.endswith(os.sep)
            or (out_p.exists() and out_p.is_dir())
        )
        has_image_ext = out_p.suffix.lower() in IMAGE_EXTS

        if looks_like_dir or not has_image_ext:
            try:
                # create directory if needed
                out_dir = output_path if looks_like_dir else str(out_p)
                # if output_path looks like a file path with no extension (e.g. "out/dir/file")
                # we want to treat the parent folder as directory.
                if not looks_like_dir and out_p.parent != Path('.'):
                    out_dir = str(out_p.parent)
                # ensure directory exists
                Path(out_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Failed to create output directory: {e}")

            try:
                # create filename from input file stem
                suffix = out_p.suffix.lower() if has_image_ext else '.png'
                filename = f"{in_p.stem}_detected{suffix}"
                return str(Path(out_dir) / filename)
            except Exception as e:
                raise RuntimeError(f"Failed to generate output filename: {e}")

        # otherwise user provided a filename with a valid image extension - ensure dir exists
        try:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            return str(out_p)
        except Exception as e:
            raise RuntimeError(f"Failed to create parent directory for output file: {e}")
            
    except Exception as e:
        print(f"Error in _ensure_output_filepath: {e}")
        raise


def process_image(input_path: str, 
                  output_path: str, 
                  **detect_params: Any) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Read an image, run detect_objects, save annotated image, and print detected boxes.
    Accepts:
      - output_path as a filename: writes there
      - output_path as a directory: writes <input_stem>_detected.png inside that directory
      - output_path missing an extension: will create <parent>/<input_stem>_detected.png

    Keeps the original behaviour and console output.
    
    Args:
        input_path: Path to input image file
        output_path: Path to output image file or directory
        **detect_params: Additional parameters to pass to detect_objects
        
    Returns:
        Tuple of (output_image, objects_list, metadata_dict)
    """
    try:
        # Read input image
        try:
            img = cv2.imread(input_path)
            if img is None:
                raise RuntimeError(f"Error: could not read image '{input_path}'")
        except Exception as e:
            raise RuntimeError(f"Failed to read input image '{input_path}': {e}")

        # Run object detection
        try:
            out_img, objects, meta = detect_objects(img, **detect_params)
        except Exception as e:
            raise RuntimeError(f"Object detection failed: {e}")

        # Print detection results
        try:
            print(f"Detected {len(objects)} object{'s' if len(objects) != 1 else ''}.")
            for i, o in enumerate(objects, start=1):
                try:
                    x, y, w_box, h_box = o['bbox']
                    print(f"  {i}) bbox=({x},{y},{w_box},{h_box}), points={o['points_count']}")
                except Exception as e:
                    print(f"Warning: Failed to print object {i} details: {e}")
                    continue
        except Exception as e:
            print(f"Warning: Failed to print detection summary: {e}")

        # Ensure we have a proper filepath for writing
        try:
            final_output_path = _ensure_output_filepath(input_path, output_path)
        except Exception as e:
            raise RuntimeError(f"Failed to determine output path: {e}")

        # Write image
        try:
            success = cv2.imwrite(final_output_path, out_img)
            if not success:
                raise RuntimeError(f"cv2.imwrite failed for path: {final_output_path}")
            print(f"Saved output image to: {final_output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save output image: {e}")

        return out_img, objects, meta
        
    except Exception as e:
        print(f"Critical error in process_image: {e}")
        raise