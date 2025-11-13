# functions/highlight.py
import cv2
import numpy as np

def highlight_color_differences(img1, img2_aligned, diff_threshold=25, min_area=15, white_threshold=240):
    """
    Compare two images using color difference (Lab space).
    Shows additions (green), deletions (red), overlaps (yellow).
    
    FIXED: Ignores white background to prevent false positives.

    Inputs:
      img1, img2_aligned : BGR numpy arrays (same size)
      diff_threshold: threshold for detecting differences
      min_area: minimum area for connected components
      white_threshold: grayscale value above which pixels are considered white (default 240)
    Returns:
      highlighted (BGR numpy array),
      mask_added (uint8 binary),
      mask_removed (uint8 binary),
      overlap (uint8 binary)
    """
    # Convert to grayscale to detect white regions
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
    
    # Create masks for white/background regions (anything above white_threshold is considered background)
    white_mask1 = (gray1 > white_threshold).astype(np.uint8) * 255
    white_mask2 = (gray2 > white_threshold).astype(np.uint8) * 255
    
    # Convert to LAB color space for better color difference detection
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2LAB)

    # Compute differences
    diff_added = cv2.subtract(lab1, lab2)    # New in img1 (vs img2)
    diff_removed = cv2.subtract(lab2, lab1)  # Missing in img1 (vs img2)

    # Compute intensity of differences
    diff_added_intensity = np.linalg.norm(diff_added.astype(np.float32), axis=2)
    diff_removed_intensity = np.linalg.norm(diff_removed.astype(np.float32), axis=2)

    # Normalize to 0-255
    diff_added_intensity = cv2.normalize(diff_added_intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    diff_removed_intensity = cv2.normalize(diff_removed_intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold to create binary masks
    _, mask_added = cv2.threshold(diff_added_intensity, diff_threshold, 255, cv2.THRESH_BINARY)
    _, mask_removed = cv2.threshold(diff_removed_intensity, diff_threshold, 255, cv2.THRESH_BINARY)

    # CRITICAL FIX: Remove white regions from the masks
    # If img2 is white (background), don't count it as "added content"
    # If img1 is white (background), don't count it as "removed content"
    mask_added = cv2.bitwise_and(mask_added, cv2.bitwise_not(white_mask2))
    mask_removed = cv2.bitwise_and(mask_removed, cv2.bitwise_not(white_mask1))

    # Filter small noise areas
    def filter_small(mask, min_area):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        filtered = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == i] = 255
        return filtered

    mask_added = filter_small(mask_added, min_area)
    mask_removed = filter_small(mask_removed, min_area)

    # Apply median blur and erosion for cleaner results
    mask_added = cv2.medianBlur(mask_added, 3)
    mask_removed = cv2.medianBlur(mask_removed, 3)

    mask_added = cv2.erode(mask_added, np.ones((2, 2), np.uint8), iterations=1)
    mask_removed = cv2.erode(mask_removed, np.ones((2, 2), np.uint8), iterations=1)

    # Compute overlap (areas that show both addition and removal - legitimate changes)
    overlap = cv2.bitwise_and(mask_added, mask_removed)

    # Create overlay with color coding
    overlay = img2_aligned.copy()
    overlay[mask_added > 0] = [0, 255, 0]      # Green → Additions
    overlay[mask_removed > 0] = [0, 0, 255]    # Red → Deletions
    overlay[overlap > 0] = [0, 255, 255]       # Yellow → Replacements/Overlaps

    # Blend with original image
    highlighted = cv2.addWeighted(img2_aligned, 0.6, overlay, 0.4, 0)

    return highlighted, mask_added, mask_removed, overlap


def align_cad_images(
    img1,
    img2,
    max_features=5000,
    good_match_percent=0.15,
    feature_type="ORB",
    visualize=False
):
    """
    Align img2 to img1 using feature matching.

    Inputs:
      img1, img2: BGR numpy arrays.
    Returns:
      aligned_img2 (BGR numpy array), h (2x3 affine matrix)
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
    gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)

    feature_type = feature_type.upper()
    if feature_type == "ORB":
        detector = cv2.ORB_create(max_features)
        norm_type = cv2.NORM_HAMMING
    elif feature_type == "AKAZE":
        detector = cv2.AKAZE_create()
        norm_type = cv2.NORM_HAMMING
    elif feature_type == "BRISK":
        detector = cv2.BRISK_create()
        norm_type = cv2.NORM_HAMMING
    elif feature_type == "SIFT":
        detector = cv2.SIFT_create(max_features)
        norm_type = cv2.NORM_L2
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("No descriptors found in one or both images.")

    matcher = cv2.BFMatcher(norm_type, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    if len(matches) == 0:
        raise ValueError("No matches found between images.")

    matches = sorted(matches, key=lambda x: x.distance)
    num_good_matches = max(10, int(len(matches) * good_match_percent))
    matches = matches[:num_good_matches]

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC)

    if h is None:
        raise ValueError("Affine transform estimation failed.")

    height, width = img1.shape[:2]
    aligned_img2 = cv2.warpAffine(img2, h, (width, height),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

    return aligned_img2, h