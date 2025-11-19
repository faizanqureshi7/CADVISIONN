from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from pdf2image import convert_from_path

from .detector import detect_objects
from .clip_compare import compare_and_annotate

DEFAULT_DETECT_PARAMS: Dict[str, Any] = {
    "bin_threshold": 200,
    "edge_threshold": 50,
    "min_area": 10000,
    "max_area_ratio": 0.75,
}

DEFAULT_CLIP_PARAMS: Dict[str, Any] = {
    "clip_threshold": 0.20,
}


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if necessary."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {path}: {e}")


def _pdf_to_image_first_page(pdf_path: Path, output_dir: Path, dpi: int = 200) -> Path:
    """
    Converts the first page of a PDF into a PNG within `output_dir` and returns its path.

    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory to save converted image
        dpi: DPI resolution for conversion

    Returns:
        Path to converted PNG image
    """
    try:
        try:
            images = convert_from_path(str(pdf_path), dpi=dpi)
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF '{pdf_path}': {e}")

        if not images:
            raise RuntimeError(f"No pages found in PDF: {pdf_path}")

        try:
            temp_path = output_dir / f"{pdf_path.stem}_page1.png"
            images[0].save(temp_path, "PNG")
        except Exception as e:
            raise RuntimeError(f"Failed to save converted image to '{temp_path}': {e}")

        return temp_path

    except Exception as e:
        print(f"Error in _pdf_to_image_first_page: {e}")
        raise


def prepare_input(path: Path, output_dir: Path, dpi: int = 200) -> Path:
    """
    If path is a PDF, convert its first page to PNG (within `output_dir`) and return the new path.
    Otherwise return the original path.

    Args:
        path: Input file path (PDF or image)
        output_dir: Directory for temporary conversions
        dpi: DPI resolution for PDF conversion

    Returns:
        Path to image file (original or converted)
    """
    try:
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        if path.suffix.lower() == ".pdf":
            try:
                _ensure_dir(output_dir)
                return _pdf_to_image_first_page(path, output_dir, dpi=dpi)
            except Exception as e:
                raise RuntimeError(f"Failed to prepare PDF input '{path}': {e}")

        return path

    except Exception as e:
        print(f"Error in prepare_input: {e}")
        raise


def run_detection(img_path: Path, save_path: Path, detect_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run `detect_objects` on an image file and save the annotated output.

    Args:
        img_path: Path to input image
        save_path: Path to save annotated output
        detect_params: Parameters for object detection

    Returns:
        List of detected objects
    """
    try:
        # Read image
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read image '{img_path}': {e}")

        # Run detection
        try:
            out_img, objects, _meta = detect_objects(img, **detect_params)
        except Exception as e:
            raise RuntimeError(f"Object detection failed on '{img_path}': {e}")

        # Save output
        try:
            _ensure_dir(save_path.parent)
            success = cv2.imwrite(str(save_path), out_img)
            if not success:
                raise RuntimeError(f"cv2.imwrite failed for '{save_path}'")
        except Exception as e:
            raise RuntimeError(f"Failed to save detected image to '{save_path}': {e}")

        return objects

    except Exception as e:
        print(f"Error in run_detection: {e}")
        raise


def _default_output_paths(
    base_output_dir: Path,
    input1: Path,
    input2: Path,
) -> Dict[str, Path]:
    """
    Generate default output paths for all pipeline outputs.

    Args:
        base_output_dir: Base directory for outputs
        input1: First input file path
        input2: Second input file path

    Returns:
        Dictionary mapping output names to paths
    """
    try:
        base1 = input1.stem
        base2 = input2.stem

        return {
            "detected_1": base_output_dir / f"{base1}_detected.png",
            "detected_2": base_output_dir / f"{base2}_detected.png",
            "matched_1": base_output_dir / f"{base1}_matched.png",
            "matched_2": base_output_dir / f"{base2}_matched.png",
            "highlighted_1": base_output_dir / f"{base1}_highlighted.png",
        }
    except Exception as e:
        print(f"Error generating default output paths: {e}")
        raise


def process_documents(
    input1_path: str,
    input2_path: str,
    output_dir: str,
    detect_params: Optional[Dict] = None,
    clip_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Main pipeline function for document comparison.

    Args:
        input1_path: Path to first input document (PDF or image)
        input2_path: Path to second input document (PDF or image)
        output_dir: Directory to save all outputs
        detect_params: Optional parameters for object detection
        clip_params: Optional parameters for CLIP comparison

    Returns:
        {
            "matches": dict,
            "outputs": {
                "detected_1": str,
                "detected_2": str,
                "matched_1": str,
                "matched_2": str,
                "highlighted_1": str,
            },
            "prepared_inputs": {
                "img1": str,
                "img2": str,
            }
        }
    """
    try:
        # Create output directory
        try:
            output_dir_path = _ensure_dir(Path(output_dir))
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory: {e}")

        # Merge parameters with defaults
        try:
            detect_params = {**DEFAULT_DETECT_PARAMS, **(detect_params or {})}
            clip_params = {**DEFAULT_CLIP_PARAMS, **(clip_params or {})}
        except Exception as e:
            raise RuntimeError(f"Failed to merge parameters: {e}")

        # Prepare input paths
        try:
            input1 = Path(input1_path)
            input2 = Path(input2_path)
        except Exception as e:
            raise RuntimeError(f"Invalid input paths: {e}")

        # Prepare inputs (convert PDFs if necessary)
        try:
            prepared1 = prepare_input(input1, output_dir_path, dpi=200)
            prepared2 = prepare_input(input2, output_dir_path, dpi=200)
        except Exception as e:
            raise RuntimeError(f"Failed to prepare inputs: {e}")

        # Resolve output paths
        try:
            resolved_output_paths = _default_output_paths(output_dir_path, input1, input2)
        except Exception as e:
            raise RuntimeError(f"Failed to resolve output paths: {e}")

        try:
            detected1_path = resolved_output_paths["detected_1"]
            detected2_path = resolved_output_paths["detected_2"]
            matched1_path = resolved_output_paths["matched_1"]
            matched2_path = resolved_output_paths["matched_2"]
            highlighted1_path = resolved_output_paths["highlighted_1"]
        except KeyError as e:
            raise RuntimeError(f"Missing required output path: {e}")

        # Run detection on both images
        try:
            objects1 = run_detection(Path(prepared1), detected1_path, detect_params)
        except Exception as e:
            raise RuntimeError(f"Detection failed on first input: {e}")

        try:
            objects2 = run_detection(Path(prepared2), detected2_path, detect_params)
        except Exception as e:
            raise RuntimeError(f"Detection failed on second input: {e}")

        # CLIP comparison and matching
        try:
            print("\n🔍 Running CLIP-based object matching...")
            
            # Unpack the tuple returned by compare_and_annotate
            matches_result = compare_and_annotate(
                img1_path=prepared1,
                img2_path=prepared2,
                boxes1=objects1,
                boxes2=objects2,
                output1_matched_path=matched1_path,
                output2_matched_path=matched2_path,
                output1_highlighted_path=highlighted1_path,
                **clip_params
            )
            
            # Handle the return value (it might be a tuple now)
            if isinstance(matches_result, tuple):
                matches_dict, img1_for_metrics, img2_for_metrics = matches_result
            else:
                # Fallback if it returns just dict
                matches_dict = matches_result
                img1_for_metrics = None
                img2_for_metrics = None
                
            print(f"✅ CLIP matching complete. Found {len(matches_dict)} matched groups.")
            
        except Exception as e:
            print(f"❌ Error during CLIP matching: {e}")
            raise RuntimeError(f"CLIP comparison failed: {e}")

        # Prepare outputs
        outputs = {
            "matched_1": matched1_path,
            "matched_2": matched2_path,
            "highlighted_1": highlighted1_path,
        }

        prepared_inputs = {
            "img1": prepared1,
            "img2": prepared2,
        }

        # Return comprehensive result
        return {
            "outputs": outputs,
            "prepared_inputs": prepared_inputs,
            "matches": matches_dict,      # Now correctly a dict, not tuple
            "objects1": objects1,
            "objects2": objects2,
            "img1_for_metrics": img1_for_metrics,  # For metrics calculation
            "img2_for_metrics": img2_for_metrics,  # For metrics calculation
        }

    except Exception as e:
        print(f"Critical error in process_documents: {e}")
        raise