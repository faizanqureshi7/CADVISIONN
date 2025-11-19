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
from backend.functions.summary import create_summary_visualization, generate_summary_with_gpt5
from backend.functions.metrics import calculate_comprehensive_metrics


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
        # Ensure matches is a dictionary
        if not isinstance(matches, dict):
            print(f"Warning: matches is type {type(matches)}, expected dict. Converting...")
            if matches is None:
                return {}
            # If it's somehow a tuple or list, return empty
            return {}
        
        serialized = {}
        for idx, match_list in matches.items():
            try:
                # Handle different formats of match_list
                if isinstance(match_list, list):
                    serialized[str(idx)] = [
                        {"img2_index": int(img2_idx), "similarity": float(similarity)}
                        for img2_idx, similarity in match_list
                    ]
                else:
                    print(f"Warning: match_list for idx {idx} is not a list: {type(match_list)}")
                    serialized[str(idx)] = []
                    
            except Exception as e:
                print(f"Warning: Failed to serialize match {idx}: {e}")
                serialized[str(idx)] = []
                continue
                
        return serialized
        
    except Exception as e:
        print(f"Error in serialize_matches: {e}")
        print(f"Matches type: {type(matches)}, value: {matches}")
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


def resize_to_same_height(images: list, target_height: Optional[int] = None) -> list:
    """
    Resize all images to the same height while maintaining aspect ratio.
    
    Args:
        images: List of images (numpy arrays)
        target_height: Target height in pixels (if None, use max height)
        
    Returns:
        List of resized images
    """
    if not images:
        return []
    
    # Find target height
    if target_height is None:
        target_height = max(img.shape[0] for img in images)
    
    resized_images = []
    for img in images:
        h, w = img.shape[:2]
        
        if h == target_height:
            resized_images.append(img)
        else:
            # Calculate new width to maintain aspect ratio
            aspect_ratio = w / h
            new_width = int(target_height * aspect_ratio)
            
            # Resize image
            resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
            resized_images.append(resized)
    
    return resized_images


def add_label_to_image(img: np.ndarray, label: str, 
                       font_scale: float = 1.0, 
                       thickness: int = 2,
                       bg_color: Tuple[int, int, int] = (50, 50, 50),
                       text_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Add a label banner at the top of an image.
    
    Args:
        img: Input image
        label: Label text
        font_scale: Font size scale
        thickness: Text thickness
        bg_color: Background color (BGR)
        text_color: Text color (BGR)
        
    Returns:
        Image with label banner
    """
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Create banner
    banner_height = text_height + baseline + 20  # 20px padding
    banner = np.full((banner_height, img.shape[1], 3), bg_color, dtype=np.uint8)
    
    # Calculate text position (centered)
    text_x = (img.shape[1] - text_width) // 2
    text_y = (banner_height + text_height) // 2
    
    # Put text on banner
    cv2.putText(banner, label, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # Stack banner on top of image
    result = np.vstack([banner, img])
    
    return result


def add_ssim_overlay(img: np.ndarray, 
                     ssim_score: float,
                     position: str = 'top-right',
                     font_scale: float = 1.2,
                     thickness: int = 3) -> np.ndarray:
    """
    Add SSIM score overlay to an image. Shows "NO CHANGES" in red if SSIM > 0.95.
    
    Args:
        img: Input image
        ssim_score: SSIM score (0-1)
        position: Position of text ('top-right', 'top-left', 'bottom-right', 'bottom-left', 'center')
        font_scale: Font size scale
        thickness: Text thickness
        
    Returns:
        Image with SSIM overlay
    """
    try:
        # Create a copy to avoid modifying original
        result = img.copy()
        
        # Ensure image is BGR
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Determine text and color based on SSIM
        if ssim_score > 0.95:
            text = "NO CHANGES"
            text_color = (0, 0, 255)  # Red in BGR
            bg_color = (240, 240, 240)  # Light gray background
            ssim_text = f"SSIM: {ssim_score:.4f}"
        else:
            text = f"SSIM: {ssim_score:.4f}"
            ssim_text = None
            # Color based on SSIM score (green to yellow to red)
            if ssim_score > 0.8:
                text_color = (0, 200, 0)  # Green
            elif ssim_score > 0.6:
                text_color = (0, 200, 200)  # Yellow
            else:
                text_color = (0, 100, 255)  # Orange
            bg_color = (50, 50, 50)  # Dark gray background
        
        # Get text sizes
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        if ssim_text:
            (ssim_width, ssim_height), ssim_baseline = cv2.getTextSize(
                ssim_text, font, font_scale * 0.6, thickness - 1
            )
            total_height = text_height + ssim_height + baseline + ssim_baseline + 30
            max_width = max(text_width, ssim_width)
        else:
            total_height = text_height + baseline + 20
            max_width = text_width
        
        # Calculate position
        h, w = result.shape[:2]
        padding = 20
        
        if position == 'top-right':
            x = w - max_width - padding - 20
            y = padding
        elif position == 'top-left':
            x = padding
            y = padding
        elif position == 'bottom-right':
            x = w - max_width - padding - 20
            y = h - total_height - padding
        elif position == 'bottom-left':
            x = padding
            y = h - total_height - padding
        else:  # center
            x = (w - max_width) // 2
            y = (h - total_height) // 2
        
        # Draw semi-transparent background rectangle
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y - 10),
            (x + max_width + 20, y + total_height + 10),
            bg_color,
            -1
        )
        # Blend with original
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        # Draw border
        cv2.rectangle(
            result,
            (x - 10, y - 10),
            (x + max_width + 20, y + total_height + 10),
            (200, 200, 200),
            2
        )
        
        # Draw main text
        text_x = x + (max_width - text_width) // 2
        text_y = y + text_height + 5
        cv2.putText(result, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Draw SSIM score below if showing "NO CHANGES"
        if ssim_text:
            ssim_x = x + (max_width - ssim_width) // 2
            ssim_y = text_y + ssim_height + 15
            cv2.putText(result, ssim_text, (ssim_x, ssim_y), font, 
                       font_scale * 0.6, (100, 100, 100), thickness - 1, cv2.LINE_AA)
        
        return result
        
    except Exception as e:
        print(f"   ⚠️  Warning: Failed to add SSIM overlay: {e}")
        return img


def add_clip_similarity_overlay(img: np.ndarray, 
                                clip_similarity: float,
                                position: str = 'top-right',
                                font_scale: float = 1.2,
                                thickness: int = 3) -> np.ndarray:
    """
    Add CLIP similarity score overlay to an image. Shows "NO CHANGES" in red if CLIP > 0.99.
    
    Args:
        img: Input image
        clip_similarity: CLIP similarity score (0-1)
        position: Position of text ('top-right', 'top-left', 'bottom-right', 'bottom-left', 'center')
        font_scale: Font size scale
        thickness: Text thickness
        
    Returns:
        Image with CLIP similarity overlay
    """
    try:
        # Create a copy to avoid modifying original
        result = img.copy()
        
        # Ensure image is BGR
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Determine text and color based on CLIP similarity
        if clip_similarity > 0.99:
            text = "NO CHANGES"
            text_color = (0, 0, 255)  # Red in BGR
            bg_color = (240, 240, 240)  # Light gray background
            clip_text = f"CLIP: {clip_similarity:.4f}"
        else:
            text = f"CLIP: {clip_similarity:.4f}"
            clip_text = None
            # Color based on CLIP score (green to yellow to red)
            if clip_similarity > 0.95:
                text_color = (0, 200, 0)  # Green
            elif clip_similarity > 0.90:
                text_color = (0, 200, 200)  # Yellow
            else:
                text_color = (0, 100, 255)  # Orange
            bg_color = (50, 50, 50)  # Dark gray background
        
        # Get text sizes
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        if clip_text:
            (clip_width, clip_height), clip_baseline = cv2.getTextSize(
                clip_text, font, font_scale * 0.6, thickness - 1
            )
            total_height = text_height + clip_height + baseline + clip_baseline + 30
            max_width = max(text_width, clip_width)
        else:
            total_height = text_height + baseline + 20
            max_width = text_width
        
        # Calculate position
        h, w = result.shape[:2]
        padding = 20
        
        if position == 'top-right':
            x = w - max_width - padding - 20
            y = padding
        elif position == 'top-left':
            x = padding
            y = padding
        elif position == 'bottom-right':
            x = w - max_width - padding - 20
            y = h - total_height - padding
        elif position == 'bottom-left':
            x = padding
            y = h - total_height - padding
        else:  # center
            x = (w - max_width) // 2
            y = (h - total_height) // 2
        
        # Draw semi-transparent background rectangle
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y - 10),
            (x + max_width + 20, y + total_height + 10),
            bg_color,
            -1
        )
        # Blend with original
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        # Draw border
        cv2.rectangle(
            result,
            (x - 10, y - 10),
            (x + max_width + 20, y + total_height + 10),
            (200, 200, 200),
            2
        )
        
        # Draw main text
        text_x = x + (max_width - text_width) // 2
        text_y = y + text_height + 5
        cv2.putText(result, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Draw CLIP score below if showing "NO CHANGES"
        if clip_text:
            clip_x = x + (max_width - clip_width) // 2
            clip_y = text_y + clip_height + 15
            cv2.putText(result, clip_text, (clip_x, clip_y), font, 
                       font_scale * 0.6, (100, 100, 100), thickness - 1, cv2.LINE_AA)
        
        return result
        
    except Exception as e:
        print(f"   ⚠️  Warning: Failed to add CLIP overlay: {e}")
        return img


def add_object_ssim_overlay(img: np.ndarray, 
                            avg_object_ssim: float,
                            position: str = 'top-right',
                            font_scale: float = 1.2,
                            thickness: int = 3) -> np.ndarray:
    """
    Add Average Object SSIM score overlay to an image. Shows "NO CHANGES" in red if Avg SSIM > 0.95.
    
    Args:
        img: Input image
        avg_object_ssim: Average Object SSIM score (0-1)
        position: Position of text ('top-right', 'top-left', 'bottom-right', 'bottom-left', 'center')
        font_scale: Font size scale
        thickness: Text thickness
        
    Returns:
        Image with Average Object SSIM overlay
    """
    try:
        # Create a copy to avoid modifying original
        result = img.copy()
        
        # Ensure image is BGR
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Determine text and color based on Average Object SSIM
        if avg_object_ssim > 0.95:
            text = "NO CHANGES"
            text_color = (0, 0, 255)  # Red in BGR
            bg_color = (240, 240, 240)  # Light gray background
            ssim_text = f"Avg Object SSIM: {avg_object_ssim:.4f}"
        else:
            text = f"Avg Obj SSIM: {avg_object_ssim:.4f}"
            ssim_text = None
            # Color based on SSIM score (green to yellow to red)
            if avg_object_ssim > 0.85:
                text_color = (0, 200, 0)  # Green
            elif avg_object_ssim > 0.70:
                text_color = (0, 200, 200)  # Yellow
            else:
                text_color = (0, 100, 255)  # Orange/Red
            bg_color = (50, 50, 50)  # Dark gray background
        
        # Get text sizes
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        if ssim_text:
            (ssim_width, ssim_height), ssim_baseline = cv2.getTextSize(
                ssim_text, font, font_scale * 0.6, thickness - 1
            )
            total_height = text_height + ssim_height + baseline + ssim_baseline + 30
            max_width = max(text_width, ssim_width)
        else:
            total_height = text_height + baseline + 20
            max_width = text_width
        
        # Calculate position
        h, w = result.shape[:2]
        padding = 20
        
        if position == 'top-right':
            x = w - max_width - padding - 20
            y = padding
        elif position == 'top-left':
            x = padding
            y = padding
        elif position == 'bottom-right':
            x = w - max_width - padding - 20
            y = h - total_height - padding
        elif position == 'bottom-left':
            x = padding
            y = h - total_height - padding
        else:  # center
            x = (w - max_width) // 2
            y = (h - total_height) // 2
        
        # Draw semi-transparent background rectangle
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y - 10),
            (x + max_width + 20, y + total_height + 10),
            bg_color,
            -1
        )
        # Blend with original
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        # Draw border
        cv2.rectangle(
            result,
            (x - 10, y - 10),
            (x + max_width + 20, y + total_height + 10),
            (200, 200, 200),
            2
        )
        
        # Draw main text
        text_x = x + (max_width - text_width) // 2
        text_y = y + text_height + 5
        cv2.putText(result, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Draw SSIM score below if showing "NO CHANGES"
        if ssim_text:
            ssim_x = x + (max_width - ssim_width) // 2
            ssim_y = text_y + ssim_height + 15
            cv2.putText(result, ssim_text, (ssim_x, ssim_y), font, 
                       font_scale * 0.6, (100, 100, 100), thickness - 1, cv2.LINE_AA)
        
        return result
        
    except Exception as e:
        print(f"   ⚠️  Warning: Failed to add Object SSIM overlay: {e}")
        return img


def create_stitched_comparison(img1: np.ndarray, 
                               img2_aligned: np.ndarray, 
                               highlighted: np.ndarray,
                               avg_object_ssim: float,
                               labels: Tuple[str, str, str] = ("Original", "Revised (Aligned)", "Highlighted Changes"),
                               spacing: int = 10) -> np.ndarray:
    """
    Create a horizontally stitched comparison image with labels and Average Object SSIM score.
    
    Args:
        img1: Original image
        img2_aligned: Aligned revised image
        highlighted: Highlighted differences image
        avg_object_ssim: Average Object SSIM score (0-1)
        labels: Tuple of labels for each image
        spacing: Spacing between images in pixels
        
    Returns:
        Stitched comparison image
    """
    try:
        print(f"   🖼️  Creating stitched comparison image...")
        print(f"   📊 Average Object SSIM: {avg_object_ssim:.4f}")
        
        # Ensure all images are BGR
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2_aligned.shape) == 2:
            img2_aligned = cv2.cvtColor(img2_aligned, cv2.COLOR_GRAY2BGR)
        if len(highlighted.shape) == 2:
            highlighted = cv2.cvtColor(highlighted, cv2.COLOR_GRAY2BGR)
        
        # Add Average Object SSIM overlay to highlighted image
        highlighted_with_ssim = add_object_ssim_overlay(
            highlighted, 
            avg_object_ssim, 
            position='top-right',
            font_scale=1.2,
            thickness=3
        )
        
        # Resize all images to same height
        images = [img1, img2_aligned, highlighted_with_ssim]
        resized_images = resize_to_same_height(images)
        
        # Add labels
        labeled_images = []
        for img, label in zip(resized_images, labels):
            labeled_img = add_label_to_image(img, label, font_scale=0.8, thickness=2)
            labeled_images.append(labeled_img)
        
        # Create spacing columns
        height = labeled_images[0].shape[0]
        spacer = np.full((height, spacing, 3), 255, dtype=np.uint8)  # White spacing
        
        # Horizontally concatenate with spacing
        stitched = np.hstack([
            labeled_images[0],
            spacer,
            labeled_images[1],
            spacer,
            labeled_images[2]
        ])
        
        print(f"   ✅ Stitched image created: {stitched.shape[1]}x{stitched.shape[0]} px")
        if avg_object_ssim > 0.95:
            print(f"   🔴 NO CHANGES detected (Avg Object SSIM > 0.95)")
        
        return stitched
        
    except Exception as e:
        print(f"   ❌ Error creating stitched image: {e}")
        raise


def save_stitched_output(img1: np.ndarray,
                         img2_aligned: np.ndarray,
                         highlighted: np.ndarray,
                         avg_object_ssim: float,
                         output_dir: Path,
                         job_id: str) -> Tuple[Path, float]:
    """
    Create and save stitched comparison image with Average Object SSIM score.
    
    Args:
        img1: Original image
        img2_aligned: Aligned revised image
        highlighted: Highlighted differences image
        avg_object_ssim: Average Object SSIM score (0-1)
        output_dir: Output directory
        job_id: Job identifier
        
    Returns:
        Tuple of (stitched_image_path, elapsed_time)
    """
    t_start = time.time()
    
    try:
        print("\n🎨 Creating stitched comparison output...")
        
        # Create stitched image with Average Object SSIM
        stitched = create_stitched_comparison(
            img1, img2_aligned, highlighted, avg_object_ssim
        )
        
        # Create output path
        output_path = output_dir / f"comparison_stitched_{job_id}.jpg"
        
        # Save image
        success = cv2.imwrite(str(output_path), stitched, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if not success:
            raise RuntimeError(f"Failed to write stitched image to {output_path}")
        
        print(f"   ✅ Saved stitched comparison: {output_path.name}")
        print(f"   📏 Size: {stitched.shape[1]}x{stitched.shape[0]} px")
        
        elapsed = time.time() - t_start
        return output_path, elapsed
        
    except Exception as e:
        print(f"   ❌ Error saving stitched output: {e}")
        raise


def create_persistent_output_folder() -> Path:
    """
    Create a persistent output folder in the project directory.
    
    Returns:
        Path to output folder
    """
    try:
        # Create outputs folder in project root
        project_root = Path(__file__).resolve().parent.parent.parent
        output_folder = project_root / "comparison_outputs"
        output_folder.mkdir(exist_ok=True, parents=True)
        
        print(f"📁 Output folder: {output_folder}")
        
        return output_folder
        
    except Exception as e:
        print(f"Warning: Failed to create persistent output folder: {e}")
        # Fallback to temp directory
        return Path(tempfile.gettempdir()) / "cad_comparison_outputs"


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
    Generate AI summary from comparison results using GPT-5.
    
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
        
        print("   💬 Generating summary with GPT-5...")
        try:
            # Call GPT-5 with two separate images (exact same logic)
            ai_summary = generate_summary_with_gpt5(img1_optimized, img2_optimized)
            
        except Exception as e:
            print(f"Warning: GPT-5 summary generation failed: {e}")
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
                           ai_summary: str,
                           stitched_path: Optional[Path] = None) -> Tuple[Dict[str, Any], float]:
    """
    Create final response payload with encoded images.
    """
    t_start = time.time()
    
    try:
        # Get matches and ensure it's a dict
        matches = result.get("matches", {})
        
        # Debug print
        print(f"   🔍 Matches type: {type(matches)}")
        
        # Ensure matches is a dictionary
        if not isinstance(matches, dict):
            print(f"   ⚠️  Warning: matches is {type(matches)}, using empty dict")
            matches = {}
        
        response_payload = {
            "job_id": job_id,
            "matches": serialize_matches(matches),
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
        
        # Add stitched image if available
        if stitched_path and stitched_path.exists():
            response_payload["images"]["stitched_comparison"] = encode_image(stitched_path)
            response_payload["stitched_output_path"] = str(stitched_path)
        
    except KeyError as e:
        raise RuntimeError(f"Missing required output file: {e}")
    except Exception as e:
        print(f"   ❌ Error in create_response_payload: {e}")
        import traceback
        print(traceback.format_exc())
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
    print(f"🎨 Stitched Output:      {format_time(timers.get('stitched_output', 0))}")
    print(f"📤 Image Encoding:       {format_time(timers.get('encoding', 0))}")
    print("-" * 80)
    print(f"⏱️  TOTAL INFERENCE TIME: {format_time(total_time)}")
    print("="*80)
    
    try:
        print(f"\n✅ Response prepared successfully")
        print(f"📦 Response payload size: {len(str(response_payload))} bytes")
        if ai_summary:
            print(f"📝 AI summary: {len(ai_summary)} characters")
        if "stitched_output_path" in response_payload:
            print(f"🎨 Stitched output: {response_payload['stitched_output_path']}")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Warning: Failed to print response summary: {e}")


# ============================================================================
# MAIN WRAPPER FUNCTION
# ============================================================================

def process_cad_comparison(file1, file2, job_id: str, temp_path: Path) -> Dict[str, Any]:
    """
    Main wrapper function that runs the entire CAD comparison pipeline.
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

        # Debug: Check what's in result
        print(f"\n🔍 DEBUG - Pipeline result keys: {result.keys()}")
        print(f"   Matches type: {type(result.get('matches'))}")
        if isinstance(result.get('matches'), dict):
            print(f"   Matches count: {len(result.get('matches', {}))}")
        
        # Step 7: Generate AI summary
        print("\n🤖 Generating AI summary...")
        ai_summary, timers['ai_summary'] = generate_ai_summary(
            img1_optimized, img2_optimized, Path(outputs["highlighted_1"])
        )

        # Step 8: Calculate metrics FIRST (to get Average Object SSIM score)
        print("\n📊 Calculating evaluation metrics...")
        t_start = time.time()
        
        avg_object_ssim = 0.0  # Default value
        
        try:
            highlighted_img = cv2.imread(str(outputs["highlighted_1"]))
            
            # Get match statistics and boxes from pipeline result
            matches_dict = result.get("matches", {})
            
            # Ensure it's a dict
            if not isinstance(matches_dict, dict):
                print(f"   ⚠️  Warning: matches_dict is {type(matches_dict)}, using empty dict")
                matches_dict = {}
            
            matches_count = len(matches_dict)
            total_features = result.get("total_features", matches_count * 2)
            inliers_count = result.get("inliers_count", None)
            
            # Get detected boxes from result
            boxes1 = result.get("objects1", [])
            boxes2 = result.get("objects2", [])
            
            # Use img2_optimized as aligned version (or get from result if available)
            img2_aligned = result.get("img2_for_metrics", img2_optimized);
            
            comprehensive_metrics = calculate_comprehensive_metrics(
                img1_optimized,
                img2_optimized,
                img2_aligned,
                highlighted_img,
                timers,
                ai_summary,
                matches_count,
                total_features,
                inliers_count,
                matches_dict=matches_dict,
                boxes1=boxes1,
                boxes2=boxes2
            )
            
            # Extract Average Object SSIM score for stitched image
            if "object_level_similarity" in comprehensive_metrics:
                obj_stats = comprehensive_metrics["object_level_similarity"].get("aggregate_statistics", {})
                avg_object_ssim = obj_stats.get("average_object_ssim", 0.0)
                print(f"   📊 Extracted Average Object SSIM: {avg_object_ssim:.4f}")
            
            timers['metrics_calculation'] = time.time() - t_start;
            
            print(f"   ✅ Metrics calculated successfully")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to calculate metrics: {e}")
            import traceback
            traceback.print_exc()
            comprehensive_metrics = {}

        # Step 9: Create stitched output (NOW with Average Object SSIM)
        try:
            # Load the highlighted image
            highlighted_img = cv2.imread(str(outputs["highlighted_1"]))
            
            # Get img2_aligned from result or use optimized version
            img2_aligned = result.get("img2_for_metrics", img2_optimized)
            
            # Create persistent output folder
            persistent_output_folder = create_persistent_output_folder()
            
            # Save stitched output WITH AVERAGE OBJECT SSIM SCORE
            stitched_path, timers['stitched_output'] = save_stitched_output(
                img1_optimized,
                img2_aligned,
                highlighted_img,
                avg_object_ssim,  # NEW: Pass Average Object SSIM instead of CLIP
                persistent_output_folder,
                job_id
            )
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to create stitched output: {e}")
            stitched_path = None
            timers['stitched_output'] = 0.0

        # Step 10: Create response payload
        print("\n📤 Encoding output images...")
        response_payload, timers['encoding'] = create_response_payload(
            job_id, result, outputs, prepared_inputs,
            optimized1_path, optimized2_path, ai_summary,
            stitched_path=stitched_path
        )

        # Add metrics to response
        response_payload["metrics"] = comprehensive_metrics

        # Print performance summary
        total_time = time.time() - start_time
        print_performance_summary(timers, total_time, response_payload, ai_summary)
        
        # Print key metrics (including object-level)
        if "metrics" in response_payload and response_payload["metrics"]:
            print("\n📊 KEY EVALUATION METRICS:")
            print("="*80)
            metrics = response_payload["metrics"]
            
            # Print object-level metrics FIRST
            if "object_level_similarity" in metrics:
                obj_stats = metrics["object_level_similarity"].get("aggregate_statistics", {})
                if "average_object_ssim" in obj_stats:
                    print(f"🎯 OBJECT-LEVEL METRICS:")
                    print(f"   Avg Object SSIM:      {obj_stats['average_object_ssim']:.4f}")
                    print(f"   Min Object SSIM:      {obj_stats['min_object_ssim']:.4f}")
                    print(f"   Max Object SSIM:      {obj_stats['max_object_ssim']:.4f}")
                    print(f"   Avg CLIP Similarity:  {obj_stats['average_clip_similarity']:.4f}")
                    print(f"   Matched Pairs:        {obj_stats['total_matched_pairs']}")
                    
                    # Highlight if NO CHANGES based on Average Object SSIM
                    if obj_stats['average_object_ssim'] > 0.95:
                        print(f"🔴 STATUS:               NO CHANGES DETECTED (Avg Object SSIM > 0.95)")
            
            if "similarity" in metrics:
                print(f"\n🔍 Image-level Metrics:")
                print(f"   SSIM:                 {metrics['similarity']['ssim']:.4f}")
                print(f"   PSNR:                 {metrics['similarity']['psnr_db']:.2f} dB")
            
            if "change_detection" in metrics:
                print(f"\n🎯 CHANGE DETECTION:")
                print(f"   Total Changes:        {metrics['change_detection']['change_percentage']:.2f}%")
                print(f"   ➕ Additions:         {metrics['change_detection']['additions']['percentage']:.2f}%")
                print(f"   ➖ Deletions:         {metrics['change_detection']['deletions']['percentage']:.2f}%")
            
            print("="*80)
        
        return response_payload
        
    except Exception as e:
        print(f"\n❌ Error in process_cad_comparison: {e}")
        raise