"""
Processing module for CAD comparison pipeline.
Contains all processing logic and helper functions.
"""
import base64
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from backend.functions.pipeline import process_documents, DEFAULT_DETECT_PARAMS, DEFAULT_CLIP_PARAMS
from backend.functions.preprocess import load_image_any_format, optimize_image_for_processing
from backend.functions.summary import create_summary_visualization, generate_summary_with_gemini


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def encode_image(path: Path) -> str:
    """
    Encode image to base64 string.
    
    Args:
        path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    try:
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
            
        with path.open("rb") as f:
            data = f.read()
            if not data:
                raise ValueError(f"Empty image file: {path}")
            return base64.b64encode(data).decode("utf-8")
            
    except Exception as e:
        print(f"Error in encode_image: {e}")
        raise


def serialize_matches(matches: Dict[int, Any]) -> Dict[str, Any]:
    """
    Serialize matches dictionary for JSON response.
    
    Args:
        matches: Dictionary mapping indices to match lists
        
    Returns:
        Serialized matches dictionary
    """
    try:
        serialized = {}
        for idx, match_list in matches.items():
            try:
                serialized[str(idx)] = [
                    {"img2_index": int(img2_idx), "similarity": float(similarity)}
                    for img2_idx, similarity in match_list
                ]
            except Exception as e:
                print(f"Warning: Failed to serialize match {idx}: {e}")
                serialized[str(idx)] = []
                continue
                
        return serialized
        
    except Exception as e:
        print(f"Error in serialize_matches: {e}")
        raise


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"


# ============================================================================
# PROCESSING PIPELINE FUNCTIONS
# ============================================================================

def save_uploaded_files(file1, file2, temp_path: Path) -> Tuple[Path, Path, float]:
    """
    Save uploaded files to temporary directory.
    
    Args:
        file1: First uploaded file
        file2: Second uploaded file
        temp_path: Temporary directory path
        
    Returns:
        Tuple of (input1_path, input2_path, elapsed_time)
    """
    t_start = time.time()
    
    try:
        input1_path = temp_path / f"input1_{file1.filename}"
        input2_path = temp_path / f"input2_{file2.filename}"
    except Exception as e:
        raise RuntimeError(f"Failed to create input file paths: {e}")

    try:
        with input1_path.open("wb") as f:
            file1.file.seek(0)
            shutil.copyfileobj(file1.file, f)
        print(f"✅ Saved: {input1_path.name}")
    except Exception as e:
        raise RuntimeError(f"Failed to save file 1 '{file1.filename}': {e}")

    try:
        with input2_path.open("wb") as f:
            file2.file.seek(0)
            shutil.copyfileobj(file2.file, f)
        print(f"✅ Saved: {input2_path.name}")
    except Exception as e:
        raise RuntimeError(f"Failed to save file 2 '{file2.filename}': {e}")
    
    elapsed = time.time() - t_start
    return input1_path, input2_path, elapsed


def load_and_validate_images(input1_path: Path, input2_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load images from paths and validate them.
    
    Args:
        input1_path: Path to first image
        input2_path: Path to second image
        
    Returns:
        Tuple of (img1, img2, elapsed_time)
    """
    t_start = time.time()
    
    try:
        img1 = load_image_any_format(str(input1_path), all_pages=False, page_num=0)
    except Exception as e:
        raise RuntimeError(f"Failed to load image 1: {e}")
        
    try:
        img2 = load_image_any_format(str(input2_path), all_pages=False, page_num=0)
    except Exception as e:
        raise RuntimeError(f"Failed to load image 2: {e}")

    # Handle case where images might be lists
    try:
        if isinstance(img1, list):
            print("⚠️ Image 1 returned as list, using first page")
            if len(img1) == 0:
                raise ValueError("Image 1 list is empty")
            img1 = img1[0]
        if isinstance(img2, list):
            print("⚠️ Image 2 returned as list, using first page")
            if len(img2) == 0:
                raise ValueError("Image 2 list is empty")
            img2 = img2[0]
    except Exception as e:
        raise RuntimeError(f"Failed to extract images from lists: {e}")

    # Validate images
    try:
        if img1 is None or img1.size == 0:
            raise ValueError("Image 1 is invalid or empty")
        if img2 is None or img2.size == 0:
            raise ValueError("Image 2 is invalid or empty")
    except Exception as e:
        raise RuntimeError(f"Image validation failed: {e}")
    
    elapsed = time.time() - t_start
    return img1, img2, elapsed


def optimize_and_save_images(img1: np.ndarray, img2: np.ndarray, 
                             temp_path: Path, max_size: int = 1500, 
                             quality: int = 90) -> Tuple[np.ndarray, np.ndarray, Path, Path, float]:
    """
    Optimize images and save to temporary directory.
    
    Args:
        img1: First image
        img2: Second image
        temp_path: Temporary directory path
        max_size: Maximum image dimension
        quality: JPEG quality (0-100)
        
    Returns:
        Tuple of (img1_optimized, img2_optimized, optimized1_path, optimized2_path, elapsed_time)
    """
    t_start = time.time()
    
    print("⚙️ Optimizing Image 1...")
    try:
        img1_optimized, scale1 = optimize_image_for_processing(img1, max_size=max_size, quality=quality)
    except Exception as e:
        raise RuntimeError(f"Failed to optimize image 1: {e}")
    
    print("⚙️ Optimizing Image 2...")
    try:
        img2_optimized, scale2 = optimize_image_for_processing(img2, max_size=max_size, quality=quality)
    except Exception as e:
        raise RuntimeError(f"Failed to optimize image 2: {e}")

    # Save optimized images
    try:
        optimized1_path = temp_path / "optimized1.jpg"
        optimized2_path = temp_path / "optimized2.jpg"
        
        success1 = cv2.imwrite(str(optimized1_path), img1_optimized)
        success2 = cv2.imwrite(str(optimized2_path), img2_optimized)
        
        if not success1 or not success2:
            raise RuntimeError("Failed to write optimized images")
            
        print(f"💾 Saved optimized images to temp directory")
    except Exception as e:
        raise RuntimeError(f"Failed to save optimized images: {e}")
    
    elapsed = time.time() - t_start
    return img1_optimized, img2_optimized, optimized1_path, optimized2_path, elapsed


def create_output_directory(temp_path: Path) -> Path:
    """
    Create output directory in temporary path.
    
    Args:
        temp_path: Temporary directory path
        
    Returns:
        Path to output directory
    """
    try:
        output_dir = temp_path / "outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory: {e}")


def run_comparison_pipeline(optimized1_path: Path, optimized2_path: Path, 
                            output_dir: Path) -> Tuple[Dict[str, Any], float]:
    """
    Run the comparison pipeline on optimized images.
    
    Args:
        optimized1_path: Path to optimized image 1
        optimized2_path: Path to optimized image 2
        output_dir: Output directory for results
        
    Returns:
        Tuple of (result_dict, elapsed_time)
    """
    t_start = time.time()
    
    try:
        result = process_documents(
            input1_path=str(optimized1_path),
            input2_path=str(optimized2_path),
            output_dir=str(output_dir),
            detect_params=DEFAULT_DETECT_PARAMS,
            clip_params=DEFAULT_CLIP_PARAMS,
        )
    except Exception as e:
        raise RuntimeError(f"Document comparison pipeline failed: {e}")

    print("✅ Pipeline completed successfully")
    try:
        match_count = len(result.get('matches', {}))
        print(f"🎯 Found {match_count} matched groups")
    except Exception as e:
        print(f"Warning: Failed to count matches: {e}")
    
    elapsed = time.time() - t_start
    return result, elapsed


def validate_pipeline_results(result: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate and extract pipeline results.
    
    Args:
        result: Pipeline result dictionary
        
    Returns:
        Tuple of (outputs_dict, prepared_inputs_dict)
    """
    try:
        outputs = result.get("outputs", {})
        prepared_inputs = result.get("prepared_inputs", {})
        
        if not outputs or not prepared_inputs:
            raise ValueError("Missing outputs or prepared_inputs in pipeline result")
            
        return outputs, prepared_inputs
    except Exception as e:
        raise RuntimeError(f"Failed to extract pipeline results: {e}")


def generate_ai_summary(img1_optimized: np.ndarray, img2_optimized: np.ndarray, 
                       highlighted_path: Path) -> Tuple[str, float]:
    """
    Generate AI summary from comparison results.
    
    Args:
        img1_optimized: Optimized original image
        img2_optimized: Optimized revised image
        highlighted_path: Path to highlighted differences image
        
    Returns:
        Tuple of (ai_summary_text, elapsed_time)
    """
    t_start = time.time()
    
    try:
        print("   📂 Loading highlighted image from pipeline...")
        try:
            highlighted = cv2.imread(str(highlighted_path))
            if highlighted is None:
                raise ValueError(f"Could not read highlighted image: {highlighted_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load highlighted image: {e}")
        
        print("   📊 Creating visualization for AI analysis...")
        try:
            composite = create_summary_visualization(
                img1_optimized,
                img2_optimized,
                highlighted
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create visualization: {e}")
        
        print("   💬 Generating summary text...")
        try:
            ai_summary = generate_summary_with_gemini(composite)
        except Exception as e:
            print(f"Warning: AI summary generation failed: {e}")
            ai_summary = f"⚠️ AI summary unavailable: {e}"
        
        print(f"   ✅ Summary generated ({len(ai_summary)} characters)")
        
    except Exception as e:
        print(f"Warning: Summary generation failed: {e}")
        ai_summary = f"❌ Summary generation failed: {e}"
    
    elapsed = time.time() - t_start
    return ai_summary, elapsed


def create_response_payload(job_id: str, result: Dict[str, Any], 
                           outputs: Dict[str, Any], prepared_inputs: Dict[str, Any],
                           optimized1_path: Path, optimized2_path: Path,
                           ai_summary: str) -> Tuple[Dict[str, Any], float]:
    """
    Create final response payload with encoded images.
    
    Args:
        job_id: Unique job identifier
        result: Pipeline result dictionary
        outputs: Output files dictionary
        prepared_inputs: Prepared input files dictionary
        optimized1_path: Path to optimized image 1
        optimized2_path: Path to optimized image 2
        ai_summary: AI-generated summary text
        
    Returns:
        Tuple of (response_payload_dict, elapsed_time)
    """
    t_start = time.time()
    
    try:
        response_payload = {
            "job_id": job_id,
            "matches": serialize_matches(result.get("matches", {})),
            "images": {
                "highlighted_1": encode_image(Path(outputs["highlighted_1"])),
                "matched_1": encode_image(Path(outputs["matched_1"])),
                "matched_2": encode_image(Path(outputs["matched_2"])),
                "input_1": encode_image(Path(prepared_inputs["img1"])),
                "input_2": encode_image(Path(prepared_inputs["img2"])),
                "optimized_input_1": encode_image(optimized1_path),
                "optimized_input_2": encode_image(optimized2_path),
            },
            "ai_summary": ai_summary
        }
        
    except KeyError as e:
        raise RuntimeError(f"Missing required output file: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create response payload: {e}")
    
    elapsed = time.time() - t_start
    return response_payload, elapsed


def print_performance_summary(timers: Dict[str, float], total_time: float, 
                             response_payload: Dict[str, Any], ai_summary: Optional[str]):
    """
    Print performance summary to console.
    
    Args:
        timers: Dictionary of timed operations
        total_time: Total execution time
        response_payload: Final response payload
        ai_summary: AI summary text (optional)
    """
    print("\n" + "="*80)
    print("⏱️  PERFORMANCE SUMMARY")
    print("="*80)
    print(f"📥 File Upload:          {format_time(timers.get('file_upload', 0))}")
    print(f"🔄 Image Loading:        {format_time(timers.get('image_loading', 0))}")
    print(f"⚙️  Optimization:         {format_time(timers.get('optimization', 0))}")
    print(f"🔍 Pipeline (Detection): {format_time(timers.get('pipeline', 0))}")
    print(f"🤖 AI Summary:           {format_time(timers.get('ai_summary', 0))}")
    print(f"📤 Image Encoding:       {format_time(timers.get('encoding', 0))}")
    print("-" * 80)
    print(f"⏱️  TOTAL INFERENCE TIME: {format_time(total_time)}")
    print("="*80)
    
    try:
        print(f"\n✅ Response prepared successfully")
        print(f"📦 Response payload size: {len(str(response_payload))} bytes")
        if ai_summary:
            print(f"📝 AI summary: {len(ai_summary)} characters")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Warning: Failed to print response summary: {e}")


# ============================================================================
# MAIN WRAPPER FUNCTION
# ============================================================================

def process_cad_comparison(file1, file2, job_id: str, temp_path: Path) -> Dict[str, Any]:
    """
    Main wrapper function that runs the entire CAD comparison pipeline.
    
    Args:
        file1: First uploaded file
        file2: Second uploaded file
        job_id: Unique job identifier
        temp_path: Temporary directory path
        
    Returns:
        Dictionary containing comparison results and metadata
    """
    start_time = time.time()
    timers = {}
    
    try:
        # Step 1: Save uploaded files
        print("\n📥 Saving uploaded files...")
        input1_path, input2_path, timers['file_upload'] = save_uploaded_files(
            file1, file2, temp_path
        )

        # Step 2: Load and validate images
        print("\n🔄 Loading and preprocessing images...")
        img1, img2, timers['image_loading'] = load_and_validate_images(
            input1_path, input2_path
        )

        # Step 3: Optimize images
        (img1_optimized, img2_optimized, 
         optimized1_path, optimized2_path, 
         timers['optimization']) = optimize_and_save_images(img1, img2, temp_path)

        # Step 4: Create output directory
        output_dir = create_output_directory(temp_path)

        # Step 5: Run comparison pipeline
        print("\n🔍 Starting document comparison pipeline...")
        result, timers['pipeline'] = run_comparison_pipeline(
            optimized1_path, optimized2_path, output_dir
        )

        # Step 6: Validate pipeline results
        outputs, prepared_inputs = validate_pipeline_results(result)

        # Step 7: Generate AI summary
        print("\n🤖 Generating AI summary...")
        ai_summary, timers['ai_summary'] = generate_ai_summary(
            img1_optimized, img2_optimized, Path(outputs["highlighted_1"])
        )

        # Step 8: Create response payload
        print("\n📤 Encoding output images...")
        response_payload, timers['encoding'] = create_response_payload(
            job_id, result, outputs, prepared_inputs,
            optimized1_path, optimized2_path, ai_summary
        )

        # Print performance summary
        total_time = time.time() - start_time
        print_performance_summary(timers, total_time, response_payload, ai_summary)
        
        return response_payload
        
    except Exception as e:
        print(f"\n❌ Error in process_cad_comparison: {e}")
        raise