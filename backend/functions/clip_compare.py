# functions/clip_compare.py
import os
import math
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw
import torch
from datetime import datetime
import cv2

from backend.functions.highlight import align_cad_images, highlight_color_differences


# Try open_clip then fallback to clip
try:
    import open_clip
    _CLIP_BACKEND = "open_clip"
except Exception:
    try:
        import clip  # from openai/CLIP (pip install git+https://github.com/openai/CLIP)
        _CLIP_BACKEND = "clip"
    except Exception:
        raise ImportError(
            "No CLIP backend available. Install 'open_clip_torch' (recommended) or the 'clip' repo.\n"
            "For example: pip install open_clip_torch --extra-index-url https://download.pytorch.org/whl/cu117"
        )


def _load_clip_model(device: torch.device = None):
    """Load a CLIP image encoder (and preprocess). Returns model, preprocess, embed_dim"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _CLIP_BACKEND == "open_clip":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model.to(device).eval()
        embed_dim = model.text_projection.shape[1] if hasattr(model, 'text_projection') else model.proj.shape[1]
        return model, preprocess, embed_dim, device
    else:
        import clip as _clip
        model, preprocess = _clip.load("ViT-B/32", device=device)
        model.to(device).eval()
        embed_dim = model.visual.output_dim if hasattr(model.visual, 'output_dim') else 512
        return model, preprocess, embed_dim, device


def _crop_image_for_bbox(pil_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = bbox
    # ensure box inside image
    img_w, img_h = pil_img.size
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))
    if x2 <= x1 or y2 <= y1:
        # return a minimal 1x1 image if box invalid (shouldn't happen)
        return pil_img.crop((0, 0, 1, 1))
    return pil_img.crop((x1, y1, x2, y2))


def _get_image_embeddings(model, preprocess, pil_img: Image.Image, bboxes: List[Tuple[int, int, int, int]],
                          device: torch.device, batch_size:int=32) -> np.ndarray:
    """
    Given a PIL image and list of bboxes, return NxD normalized numpy array of embeddings.
    """
    crops = [ _crop_image_for_bbox(pil_img, bb) for bb in bboxes ]
    # Preprocess and batch
    tensors = []
    for c in crops:
        t = preprocess(c).unsqueeze(0)  # 1,C,H,W
        tensors.append(t)
    if len(tensors) == 0:
        return np.zeros((0, model.visual.output_dim if hasattr(model, 'visual') else 512), dtype=np.float32)

    tensors = torch.cat(tensors, dim=0).to(device)

    # forward in batches
    with torch.no_grad():
        embeds = []
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i+batch_size]
            if _CLIP_BACKEND == "open_clip":
                emb = model.encode_image(batch)
            else:
                emb = model.encode_image(batch)
            emb = emb.float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeds.append(emb.cpu().numpy())
    emb_np = np.vstack(embeds)
    return emb_np


def match_boxes_by_clip_many_to_one(emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.28) -> Dict[int, List[Tuple[int, float]]]:
    """
    Compute pairwise cosine similarities and return many-to-one matches.
    Returns dict: {index_in_img1: [(index_in_img2, similarity), ...]}
    
    Multiple objects in img2 can match the same object in img1 if they are similar enough.
    """
    if emb1.shape[0] == 0 or emb2.shape[0] == 0:
        return {}

    sims = np.dot(emb1, emb2.T)  # N1 x N2 cosine similarity matrix

    matches_dict = {}  # img1_idx -> list of (img2_idx, sim)
    matched_in_img2 = set()

    # For each object in img2, find its best match in img1
    for j in range(sims.shape[1]):  # iterate over img2
        best_i = int(np.argmax(sims[:, j]))
        best_sim = float(sims[best_i, j])
        
        if best_sim >= threshold:
            if best_i not in matches_dict:
                matches_dict[best_i] = []
            matches_dict[best_i].append((j, best_sim))
            matched_in_img2.add(j)

    # Handle last resort: if exactly 1 unmatched in each image, match them
    unmatched_img1 = [i for i in range(emb1.shape[0]) if i not in matches_dict]
    unmatched_img2 = [j for j in range(emb2.shape[0]) if j not in matched_in_img2]
    
    if len(unmatched_img1) == 1 and len(unmatched_img2) == 1:
        i = unmatched_img1[0]
        j = unmatched_img2[0]
        sim = float(sims[i, j])
        matches_dict[i] = [(j, sim)]
        print(f"Last resort matching: img1[{i}] <-> img2[{j}] with similarity {sim:.3f}")

    return matches_dict


def compute_combined_bbox(bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """
    Given a list of bboxes (x, y, w, h), compute the combined bounding box that covers all.
    Returns (x, y, w, h) of the combined box.
    """
    if not bboxes:
        return (0, 0, 0, 0)
    
    min_x = min(bb[0] for bb in bboxes)
    min_y = min(bb[1] for bb in bboxes)
    max_x = max(bb[0] + bb[2] for bb in bboxes)
    max_y = max(bb[1] + bb[3] for bb in bboxes)
    
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def compare_and_annotate(img1_path: str, img2_path: str,
                         boxes1: List[Dict], boxes2: List[Dict],
                         output1_matched_path: str, output2_matched_path: str,
                         output1_highlighted_path: str,
                         clip_threshold: float = 0.28,
                         min_area: int = 10000,
                         device: torch.device = None,
                         batch_size:int=32,
                         # new tuning params for alignment/highlight
                         align_feature_type: str = "ORB",
                         align_good_match_percent: float = 0.15,
                         highlight_diff_threshold: int = 25,
                         highlight_min_area: int = 15,
                         line_width: int = 4):
    """
    Main entry with many-to-one matching support.
    
    For each matched group (1 img1 object -> N img2 objects):
    1. Compute combined bbox for all img2 objects in the group
    2. Find the larger dimensions between img1 bbox and combined img2 bbox
    3. Align and compare
    4. Generate THREE output images:
       - output1_matched_path: Image 1 with color-coded matched boxes
       - output2_matched_path: Image 2 with color-coded matched boxes
       - output1_highlighted_path: Image 1 with highlighted differences
    
    Returns dict of matches {img1_idx: [(img2_idx, sim), ...]}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Filter boxes by min_area before processing
    def area_of_box(bbox):
        return bbox[2] * bbox[3]  # w * h
    
    filtered_boxes1 = [box for box in boxes1 if area_of_box(box['bbox']) >= min_area]
    filtered_boxes2 = [box for box in boxes2 if area_of_box(box['bbox']) >= min_area]
    
    print(f"Filtered boxes: Image1 {len(boxes1)} -> {len(filtered_boxes1)}, Image2 {len(boxes2)} -> {len(filtered_boxes2)}")
    
    boxes1 = filtered_boxes1
    boxes2 = filtered_boxes2

    # Load CLIP
    model, preprocess, embed_dim, device = _load_clip_model(device)

    # Prepare PIL images and bbox lists
    pil1 = Image.open(img1_path).convert("RGB")
    pil2 = Image.open(img2_path).convert("RGB")

    bboxes1 = [tuple(o['bbox']) for o in boxes1]
    bboxes2 = [tuple(o['bbox']) for o in boxes2]

    emb1 = _get_image_embeddings(model, preprocess, pil1, bboxes1, device, batch_size=batch_size)
    emb2 = _get_image_embeddings(model, preprocess, pil2, bboxes2, device, batch_size=batch_size)

    # Match with many-to-one logic
    matches_dict = match_boxes_by_clip_many_to_one(emb1, emb2, threshold=clip_threshold)

    print(f"\nFound {len(matches_dict)} matched groups:")
    for i, matches_list in matches_dict.items():
        img2_indices = [j for j, sim in matches_list]
        print(f"  Image1[{i}] <-> Image2{img2_indices}")

    # Generate colors for each matched group
    def _color_for_idx(i):
        import colorsys
        h = (i * 37) % 360
        r, g, b = colorsys.hsv_to_rgb(h/360.0, 0.8, 0.95)
        return (int(255*r), int(255*g), int(255*b))

    colors_for_groups = {}
    for group_id, i in enumerate(matches_dict.keys()):
        colors_for_groups[i] = _color_for_idx(group_id)

    # Load images for processing
    full1 = cv2.imread(img1_path)
    full2 = cv2.imread(img2_path)
    if full1 is None or full2 is None:
        raise RuntimeError("Could not read input images for diff generation.")

    # Create result images - separate for matched and highlighted
    result_img1_matched = full1.copy()
    result_img2_matched = full2.copy()
    result_img1_highlighted = full1.copy()
    
    img1_h, img1_w = full1.shape[:2]
    img2_h, img2_w = full2.shape[:2]

    # Process each matched group
    for i, matches_list in matches_dict.items():
        # Get img1 bbox
        bx1 = bboxes1[i]
        x1, y1, w1, h1 = [int(v) for v in bx1]

        # Get all img2 bboxes for this group
        img2_bboxes_in_group = [bboxes2[j] for j, sim in matches_list]
        
        # Compute combined bbox for img2 objects
        combined_bx2 = compute_combined_bbox(img2_bboxes_in_group)
        x2, y2, w2, h2 = [int(v) for v in combined_bx2]

        # Determine LARGER dimensions
        max_w = max(w1, w2)
        max_h = max(h1, h2)

        # Helper to get corners
        def corners(b):
            x, y, w, h = b
            return {
                'tl': np.array([x, y]),
                'tr': np.array([x + w, y]),
                'bl': np.array([x, y + h]),
                'br': np.array([x + w, y + h]),
            }

        # Determine which bbox is larger
        area1 = w1 * h1
        area2 = w2 * h2
        larger_bb = bx1 if area1 >= area2 else combined_bx2
        smaller_bb = combined_bx2 if area1 >= area2 else bx1

        large_c = corners(larger_bb)
        small_c = corners(smaller_bb)

        # Compute distances between corresponding corners
        dists = {
            'tl': np.linalg.norm(large_c['tl'] - small_c['tl']),
            'tr': np.linalg.norm(large_c['tr'] - small_c['tr']),
            'bl': np.linalg.norm(large_c['bl'] - small_c['bl']),
            'br': np.linalg.norm(large_c['br'] - small_c['br']),
        }
        matched_corner = min(dists, key=dists.get)

        # Compute new coordinates based on matched corner
        def compute_new_xy_from_anchor(x, y, w, h, anchor):
            if anchor == 'tl':
                new_x = int(x)
                new_y = int(y)
            elif anchor == 'tr':
                new_x = int(max(0, (x + w) - max_w))
                new_y = int(y)
            elif anchor == 'bl':
                new_x = int(x)
                new_y = int(max(0, (y + h) - max_h))
            else:  # 'br'
                new_x = int(max(0, (x + w) - max_w))
                new_y = int(max(0, (y + h) - max_h))
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            return new_x, new_y

        new_x1, new_y1 = compute_new_xy_from_anchor(x1, y1, w1, h1, matched_corner)
        new_x2, new_y2 = compute_new_xy_from_anchor(x2, y2, w2, h2, matched_corner)

        # Clamp to image bounds
        final_w1 = min(max_w, img1_w - new_x1)
        final_h1 = min(max_h, img1_h - new_y1)
        final_w2 = min(max_w, img2_w - new_x2)
        final_h2 = min(max_h, img2_h - new_y2)

        final_w = int(min(final_w1, final_w2))
        final_h = int(min(final_h1, final_h2))

        if final_w <= 0 or final_h <= 0:
            continue

        # Crop from both images
        crop1 = full1[new_y1:new_y1+final_h, new_x1:new_x1+final_w].copy()
        crop2 = full2[new_y2:new_y2+final_h, new_x2:new_x2+final_w].copy()

        # Resize if necessary
        if crop1.shape[:2] != (final_h, final_w):
            crop1 = cv2.resize(crop1, (final_w, final_h), interpolation=cv2.INTER_CUBIC)
        if crop2.shape[:2] != (final_h, final_w):
            crop2 = cv2.resize(crop2, (final_w, final_h), interpolation=cv2.INTER_CUBIC)

        if crop1.size == 0 or crop2.size == 0:
            continue

        # Align crop2 to crop1
        try:
            aligned_crop2, H = align_cad_images(
                crop1, crop2,
                max_features=5000,
                good_match_percent=align_good_match_percent,
                feature_type=align_feature_type,
                visualize=False
            )
        except Exception as e:
            print(f"Warning: alignment failed for group img1[{i}]: {e}")
            continue

        # Run highlight algorithm
        highlighted_crop, mask_added, mask_removed, overlap = highlight_color_differences(
            crop1, aligned_crop2,
            diff_threshold=highlight_diff_threshold,
            min_area=highlight_min_area
        )

        # Paste highlighted crop back at img1's location using NEW coordinates
        result_img1_highlighted[new_y1:new_y1+final_h, new_x1:new_x1+final_w] = highlighted_crop

    # Draw colored bounding boxes on matched images
    # Image 1: draw boxes with matched colors
    matched_in_img1 = set(matches_dict.keys())
    for idx, bb in enumerate(bboxes1):
        x, y, w, h = bb
        
        if idx in matched_in_img1:
            col = colors_for_groups[idx]
        else:
            col = (200, 200, 200)  # gray for unmatched
        
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        for lw in range(line_width):
            cv2.rectangle(result_img1_matched, 
                         (x1-lw, y1-lw), 
                         (x2+lw, y2+lw), 
                         col, 
                         1)

    # Image 2: draw boxes with matched colors (and combined boxes)
    matched_in_img2 = set()
    for i, matches_list in matches_dict.items():
        for j, sim in matches_list:
            matched_in_img2.add(j)
        
        # Draw combined bbox for this group in img2
        img2_bboxes_in_group = [bboxes2[j] for j, sim in matches_list]
        combined_bx2 = compute_combined_bbox(img2_bboxes_in_group)
        x, y, w, h = [int(v) for v in combined_bx2]
        
        col = colors_for_groups[i]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        for lw in range(line_width):
            cv2.rectangle(result_img2_matched, 
                         (x1-lw, y1-lw), 
                         (x2+lw, y2+lw), 
                         col, 
                         1)

    # Draw unmatched boxes in img2
    for idx, bb in enumerate(bboxes2):
        if idx not in matched_in_img2:
            x, y, w, h = bb
            col = (200, 200, 200)
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            for lw in range(line_width):
                cv2.rectangle(result_img2_matched, 
                             (x1-lw, y1-lw), 
                             (x2+lw, y2+lw), 
                             col, 
                             1)

    # Save all three outputs
    Path(output1_matched_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output2_matched_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output1_highlighted_path).parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(output1_matched_path, result_img1_matched)
    cv2.imwrite(output2_matched_path, result_img2_matched)
    cv2.imwrite(output1_highlighted_path, result_img1_highlighted)
    
    print(f"\nSaved Image 1 with matched boxes: {output1_matched_path}")
    print(f"Saved Image 2 with matched boxes: {output2_matched_path}")
    print(f"Saved Image 1 with highlighted differences: {output1_highlighted_path}")

    return matches_dict