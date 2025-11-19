"""
Metrics module for CAD comparison evaluation.
Contains functions to calculate various quality and performance metrics.
"""
import time
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import label


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def calculate_alignment_metrics(img1: np.ndarray, img2_aligned: np.ndarray, 
                                matches_count: int, total_features: int,
                                inliers_count: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate alignment quality metrics.
    
    Args:
        img1: Original image
        img2_aligned: Aligned image
        matches_count: Number of good matches
        total_features: Total features detected
        inliers_count: Number of RANSAC inliers (optional)
        
    Returns:
        Dictionary of alignment metrics
    """
    try:
        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY) if len(img2_aligned.shape) == 3 else img2_aligned
        
        # Ensure same size
        h, w = gray1.shape
        gray2_resized = cv2.resize(gray2, (w, h))
        
        # Calculate RMSE
        mse = np.mean((gray1.astype(float) - gray2_resized.astype(float)) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate feature match ratio
        match_ratio = matches_count / total_features if total_features > 0 else 0.0
        
        # Calculate inlier ratio
        inlier_ratio = inliers_count / matches_count if inliers_count and matches_count > 0 else None
        
        metrics = {
            "rmse": float(rmse),
            "mse": float(mse),
            "feature_match_ratio": float(match_ratio),
            "good_matches": int(matches_count),
            "total_features": int(total_features)
        }
        
        if inlier_ratio is not None:
            metrics["inlier_ratio"] = float(inlier_ratio)
            metrics["inliers_count"] = int(inliers_count)
        
        # At the end, convert numpy types
        return convert_numpy_types(metrics)
        
    except Exception as e:
        print(f"Error calculating alignment metrics: {e}")
        return {
            "rmse": 0.0,
            "mse": 0.0,
            "feature_match_ratio": 0.0,
            "good_matches": 0,
            "total_features": 0
        }


def calculate_image_similarity_metrics(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """
    Calculate image similarity metrics (SSIM, PSNR, MSE).
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        Dictionary of similarity metrics
    """
    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Ensure same size
        h, w = gray1.shape
        gray2_resized = cv2.resize(gray2, (w, h))
        
        # Calculate SSIM
        ssim_score, _ = ssim(gray1, gray2_resized, full=True)
        
        # Calculate MSE
        mse_value = np.mean((gray1.astype(float) - gray2_resized.astype(float)) ** 2)
        
        # Calculate PSNR
        if mse_value == 0:
            psnr_value = 100.0  # Perfect match
        else:
            max_pixel = 255.0
            psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
        
        # At the end, convert numpy types
        return convert_numpy_types({
            "ssim": float(ssim_score),
            "psnr_db": float(psnr_value),
            "mse": float(mse_value),
            "similarity_percentage": float(ssim_score * 100)
        })
        
    except Exception as e:
        print(f"Error calculating similarity metrics: {e}")
        return {
            "ssim": 0.0,
            "psnr_db": 0.0,
            "mse": 0.0,
            "similarity_percentage": 0.0
        }


def calculate_change_detection_metrics(highlighted: np.ndarray) -> Dict[str, Any]:
    """
    Calculate change detection metrics from highlighted image.
    
    Args:
        highlighted: Highlighted changes image (BGR)
        
    Returns:
        Dictionary of change detection metrics
    """
    try:
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(highlighted, cv2.COLOR_BGR2HSV)
        
        # Define color masks
        red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))
        
        # Calculate pixel counts
        total_pixels = highlighted.shape[0] * highlighted.shape[1]
        
        deletions_pixels = int(np.sum(red_mask > 0))
        additions_pixels = int(np.sum(green_mask > 0))
        modifications_pixels = int(np.sum(yellow_mask > 0))
        total_changed_pixels = deletions_pixels + additions_pixels + modifications_pixels
        
        # Calculate percentages
        deletions_percent = (deletions_pixels / total_pixels) * 100
        additions_percent = (additions_pixels / total_pixels) * 100
        modifications_percent = (modifications_pixels / total_pixels) * 100
        total_change_percent = (total_changed_pixels / total_pixels) * 100
        
        result = {
            "total_pixels": total_pixels,
            "changed_pixels": total_changed_pixels,
            "change_percentage": float(total_change_percent),
            "deletions": {
                "pixels": deletions_pixels,
                "percentage": float(deletions_percent)
            },
            "additions": {
                "pixels": additions_pixels,
                "percentage": float(additions_percent)
            },
            "modifications": {
                "pixels": modifications_pixels,
                "percentage": float(modifications_percent)
            },
            "unchanged_percentage": float(100 - total_change_percent)
        }
        
        # At the end, convert numpy types
        return convert_numpy_types(result)
        
    except Exception as e:
        print(f"Error calculating change detection metrics: {e}")
        return {
            "total_pixels": 0,
            "changed_pixels": 0,
            "change_percentage": 0.0,
            "deletions": {"pixels": 0, "percentage": 0.0},
            "additions": {"pixels": 0, "percentage": 0.0},
            "modifications": {"pixels": 0, "percentage": 0.0},
            "unchanged_percentage": 100.0
        }


def calculate_performance_metrics(timers: Dict[str, float], 
                                  img1_size: Tuple[int, int],
                                  img2_size: Tuple[int, int]) -> Dict[str, Any]:
    """
    Calculate performance metrics.
    
    Args:
        timers: Dictionary of timed operations
        img1_size: Size of first image (height, width)
        img2_size: Size of second image (height, width)
        
    Returns:
        Dictionary of performance metrics
    """
    try:
        total_time = sum(timers.values())
        total_pixels = (img1_size[0] * img1_size[1]) + (img2_size[0] * img2_size[1])
        
        return {
            "total_time_seconds": float(total_time),
            "file_upload_seconds": float(timers.get('file_upload', 0)),
            "image_loading_seconds": float(timers.get('image_loading', 0)),
            "optimization_seconds": float(timers.get('optimization', 0)),
            "pipeline_seconds": float(timers.get('pipeline', 0)),
            "ai_summary_seconds": float(timers.get('ai_summary', 0)),
            "encoding_seconds": float(timers.get('encoding', 0)),
            "throughput_pixels_per_second": float(total_pixels / total_time) if total_time > 0 else 0.0,
            "input_image_sizes": {
                "image1": {"height": img1_size[0], "width": img1_size[1]},
                "image2": {"height": img2_size[0], "width": img2_size[1]}
            }
        }
        
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
        return {
            "total_time_seconds": 0.0,
            "throughput_pixels_per_second": 0.0
        }


def calculate_summary_metrics(ai_summary: str, generation_time: float) -> Dict[str, Any]:
    """
    Calculate AI summary quality metrics.
    
    Args:
        ai_summary: Generated AI summary text
        generation_time: Time taken to generate summary
        
    Returns:
        Dictionary of summary metrics
    """
    try:
        word_count = len(ai_summary.split())
        char_count = len(ai_summary)
        line_count = len(ai_summary.split('\n'))
        
        # Count sections (## headers)
        section_count = ai_summary.count('##')
        
        # Count subsections (### headers)
        subsection_count = ai_summary.count('###')
        
        return {
            "character_count": char_count,
            "word_count": word_count,
            "line_count": line_count,
            "section_count": section_count,
            "subsection_count": subsection_count,
            "generation_time_seconds": float(generation_time),
            "words_per_second": float(word_count / generation_time) if generation_time > 0 else 0.0
        }
        
    except Exception as e:
        print(f"Error calculating summary metrics: {e}")
        return {
            "character_count": 0,
            "word_count": 0,
            "line_count": 0,
            "generation_time_seconds": 0.0
        }


def calculate_object_level_similarity(img1: np.ndarray, 
                                      img2: np.ndarray,
                                      bbox1: Tuple[int, int, int, int],
                                      bbox2: Tuple[int, int, int, int]) -> Dict[str, float]:
    """
    Calculate SSIM and other similarity metrics between two cropped object regions.
    
    Args:
        img1: Full image 1
        img2: Full image 2 (should be aligned)
        bbox1: Bounding box in image 1 (x, y, w, h)
        bbox2: Bounding box in image 2 (x, y, w, h)
        
    Returns:
        Dictionary containing object-level similarity metrics
    """
    try:
        # Extract bounding box coordinates
        x1, y1, w1, h1 = int(bbox1[0]), int(bbox1[1]), int(bbox1[2]), int(bbox1[3])
        x2, y2, w2, h2 = int(bbox2[0]), int(bbox2[1]), int(bbox2[2]), int(bbox2[3])
        
        # Crop regions from both images
        crop1 = img1[y1:y1+h1, x1:x1+w1].copy()
        crop2 = img2[y2:y2+h2, x2:x2+w2].copy()
        
        # Ensure crops are valid
        if crop1.size == 0 or crop2.size == 0:
            return {
                "object_ssim": 0.0,
                "object_psnr_db": 0.0,
                "object_mse": 0.0,
                "error": "Empty crop region"
            }
        
        # Resize to same dimensions (use larger dimensions)
        max_h = max(h1, h2)
        max_w = max(w1, w2)
        
        crop1_resized = cv2.resize(crop1, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
        crop2_resized = cv2.resize(crop2, (max_w, max_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(crop1_resized, cv2.COLOR_BGR2GRAY) if len(crop1_resized.shape) == 3 else crop1_resized
        gray2 = cv2.cvtColor(crop2_resized, cv2.COLOR_BGR2GRAY) if len(crop2_resized.shape) == 3 else crop2_resized
        
        # Calculate SSIM
        ssim_score, _ = ssim(gray1, gray2, full=True)
        
        # Calculate MSE
        mse_value = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        
        # Calculate PSNR
        if mse_value == 0:
            psnr_value = 100.0
        else:
            max_pixel = 255.0
            psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
        
        return {
            "object_ssim": float(ssim_score),
            "object_psnr_db": float(psnr_value),
            "object_mse": float(mse_value),
            "object_similarity_percentage": float(ssim_score * 100),
            "crop_size": (int(max_w), int(max_h))
        }
        
    except Exception as e:
        print(f"Error calculating object-level similarity: {e}")
        return {
            "object_ssim": 0.0,
            "object_psnr_db": 0.0,
            "object_mse": 0.0,
            "error": str(e)
        }


def calculate_matched_objects_metrics(img1: np.ndarray,
                                      img2_aligned: np.ndarray,
                                      matches_dict: Dict[int, List[Tuple[int, float]]],
                                      boxes1: List[Dict],
                                      boxes2: List[Dict]) -> Dict[str, Any]:
    """
    Calculate object-level similarity metrics for all matched object pairs.
    
    Args:
        img1: Original image 1
        img2_aligned: Aligned image 2
        matches_dict: Dictionary of matches {img1_idx: [(img2_idx, clip_sim), ...]}
        boxes1: List of detected boxes in image 1
        boxes2: List of detected boxes in image 2
        
    Returns:
        Dictionary containing per-object metrics and aggregated statistics
    """
    try:
        object_metrics = []
        
        print(f"\n   🎯 Calculating SSIM for {len(matches_dict)} matched groups...")
        
        for idx1, matches_list in matches_dict.items():
            try:
                bbox1 = tuple(boxes1[idx1]['bbox'])
                
                # For many-to-one matches, calculate metrics for each img2 object
                for idx2, clip_similarity in matches_list:
                    try:
                        bbox2 = tuple(boxes2[idx2]['bbox'])
                        
                        # Calculate object-level similarity
                        obj_sim = calculate_object_level_similarity(
                            img1, img2_aligned, bbox1, bbox2
                        )
                        
                        # Add match information and convert all to native Python types
                        obj_sim.update({
                            "img1_index": int(idx1),
                            "img2_index": int(idx2),
                            "clip_similarity": float(clip_similarity),
                            "bbox1": (int(bbox1[0]), int(bbox1[1]), int(bbox1[2]), int(bbox1[3])),
                            "bbox2": (int(bbox2[0]), int(bbox2[1]), int(bbox2[2]), int(bbox2[3]))
                        })
                        
                        object_metrics.append(obj_sim)
                        
                        # Print individual SSIM
                        print(f"      Match img1[{idx1}] <-> img2[{idx2}]: SSIM={obj_sim['object_ssim']:.4f}, "
                              f"CLIP={clip_similarity:.4f}")
                        
                    except Exception as e:
                        print(f"   ⚠️  Failed to calculate metrics for match img1[{idx1}] <-> img2[{idx2}]: {e}")
                        continue
                        
            except Exception as e:
                print(f"   ⚠️  Failed to process matches for img1[{idx1}]: {e}")
                continue
        
        # Calculate aggregate statistics
        if object_metrics:
            ssim_scores = [m['object_ssim'] for m in object_metrics if 'object_ssim' in m]
            psnr_scores = [m['object_psnr_db'] for m in object_metrics if 'object_psnr_db' in m]
            clip_scores = [m['clip_similarity'] for m in object_metrics if 'clip_similarity' in m]
            
            aggregate_stats = {
                "average_object_ssim": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
                "min_object_ssim": float(np.min(ssim_scores)) if ssim_scores else 0.0,
                "max_object_ssim": float(np.max(ssim_scores)) if ssim_scores else 0.0,
                "std_object_ssim": float(np.std(ssim_scores)) if ssim_scores else 0.0,
                
                "average_object_psnr": float(np.mean(psnr_scores)) if psnr_scores else 0.0,
                "min_object_psnr": float(np.min(psnr_scores)) if psnr_scores else 0.0,
                "max_object_psnr": float(np.max(psnr_scores)) if psnr_scores else 0.0,
                
                "average_clip_similarity": float(np.mean(clip_scores)) if clip_scores else 0.0,
                "min_clip_similarity": float(np.min(clip_scores)) if clip_scores else 0.0,
                "max_clip_similarity": float(np.max(clip_scores)) if clip_scores else 0.0,
                
                "total_matched_pairs": int(len(object_metrics)),
                "total_matched_img1_objects": int(len(matches_dict)),
                "total_matched_img2_objects": int(sum(len(matches) for matches in matches_dict.values()))
            }
            
            print(f"\n   📊 Aggregate Statistics:")
            print(f"      Average SSIM: {aggregate_stats['average_object_ssim']:.4f}")
            print(f"      Min SSIM: {aggregate_stats['min_object_ssim']:.4f}")
            print(f"      Max SSIM: {aggregate_stats['max_object_ssim']:.4f}")
            print(f"      Total Pairs: {aggregate_stats['total_matched_pairs']}")
            
        else:
            aggregate_stats = {
                "average_object_ssim": 0.0,
                "total_matched_pairs": 0,
                "error": "No matched objects to calculate metrics"
            }
        
        # Convert all numpy types before returning
        result = {
            "per_object_metrics": convert_numpy_types(object_metrics),
            "aggregate_statistics": convert_numpy_types(aggregate_stats)
        }
        
        return result
        
    except Exception as e:
        print(f"   ❌ Error calculating matched objects metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            "per_object_metrics": [],
            "aggregate_statistics": {
                "error": str(e),
                "total_matched_pairs": 0
            }
        }


def calculate_comprehensive_metrics(img1: np.ndarray, 
                                    img2: np.ndarray,
                                    img2_aligned: np.ndarray,
                                    highlighted: np.ndarray,
                                    timers: Dict[str, float],
                                    ai_summary: str,
                                    matches_count: int = 0,
                                    total_features: int = 0,
                                    inliers_count: Optional[int] = None,
                                    matches_dict: Optional[Dict[int, List[Tuple[int, float]]]] = None,
                                    boxes1: Optional[List[Dict]] = None,
                                    boxes2: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Calculate all metrics for comprehensive evaluation including object-level SSIM.
    
    Args:
        img1: Original image
        img2: Revised image
        img2_aligned: Aligned revised image
        highlighted: Highlighted changes image
        timers: Dictionary of timed operations
        ai_summary: AI-generated summary
        matches_count: Number of feature matches
        total_features: Total features detected
        inliers_count: Number of RANSAC inliers
        matches_dict: Dictionary of CLIP matches {img1_idx: [(img2_idx, sim), ...]}
        boxes1: List of detected boxes in image 1
        boxes2: List of detected boxes in image 2
        
    Returns:
        Comprehensive metrics dictionary
    """
    try:
        metrics = {
            "alignment": calculate_alignment_metrics(
                img1, img2_aligned, matches_count, total_features, inliers_count
            ),
            "similarity": calculate_image_similarity_metrics(img1, img2),
            "change_detection": calculate_change_detection_metrics(highlighted),
            "performance": calculate_performance_metrics(
                timers, img1.shape[:2], img2.shape[:2]
            ),
            "summary": calculate_summary_metrics(
                ai_summary, timers.get('ai_summary', 0)
            )
        }
        
        # Add object-level metrics if matches are available
        if matches_dict is not None and boxes1 is not None and boxes2 is not None:
            print("   📊 Calculating object-level SSIM metrics...")
            object_level_metrics = calculate_matched_objects_metrics(
                img1, img2_aligned, matches_dict, boxes1, boxes2
            )
            metrics["object_level_similarity"] = object_level_metrics
            
            # Print summary
            agg = object_level_metrics.get("aggregate_statistics", {})
            if "average_object_ssim" in agg:
                print(f"   ✅ Object-level metrics: Avg SSIM={agg['average_object_ssim']:.4f}, "
                      f"Matched pairs={agg['total_matched_pairs']}")
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating comprehensive metrics: {e}")
        return {}