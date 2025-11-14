import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

def detect_objects(image_bgr: np.ndarray,
                   bin_threshold: int = 200,
                   edge_threshold: int = 50,
                   min_area: int = 1000,
                   max_area_ratio: float = 0.8,
                   merge_gap: int = 0) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (output_image_bgr_with_boxes, objects_list, meta)
    objects_list is a list of dicts: {'bbox': (x,y,w,h), 'points_count': n}

    NOTE: Logic is unchanged from the original script except for the
    final post-processing step that removes nested boxes and merges overlaps
    or nearby boxes (within merge_gap pixels).
    """
    try:
        # Make a copy to draw on later
        out_img = image_bgr.copy()
        h, w = image_bgr.shape[:2]
        img_area = h * w

        # 1) grayscale
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise RuntimeError(f"Failed to convert image to grayscale: {e}")

        # 2) binary threshold (same as original)
        try:
            _, binary = cv2.threshold(gray, bin_threshold, 255, cv2.THRESH_BINARY)
        except Exception as e:
            raise RuntimeError(f"Failed to apply binary threshold: {e}")

        # 3) simple Sobel-like edge detection on the binary image
        try:
            kx = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
            ky = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)

            bin_f = binary.astype(np.float32)
            gx = cv2.filter2D(bin_f, -1, kx)
            gy = cv2.filter2D(bin_f, -1, ky)
            edges = np.sqrt(gx * gx + gy * gy)
            edges = np.clip((edges / edges.max()) * 255, 0, 255).astype(np.uint8)
        except Exception as e:
            raise RuntimeError(f"Failed to detect edges: {e}")

        # 4) connected components by flood-fill on edge pixels above edge_threshold
        visited = np.zeros_like(edges, dtype=np.uint8)
        objects = []

        def flood_fill(start_y: int, start_x: int) -> Optional[Dict[str, Any]]:
            """Iterative stack flood-fill returning bbox and count of pixels"""
            try:
                stack = [(start_y, start_x)]
                min_x, max_x = w, 0
                min_y, max_y = h, 0
                count = 0

                while stack:
                    try:
                        y, x = stack.pop()
                        if x < 0 or x >= w or y < 0 or y >= h:
                            continue
                        idx_val = edges[y, x]
                        if visited[y, x] or idx_val <= edge_threshold:
                            continue

                        visited[y, x] = 1
                        count += 1
                        if x < min_x: min_x = x
                        if x > max_x: max_x = x
                        if y < min_y: min_y = y
                        if y > max_y: max_y = y

                        # push 8 neighbors
                        for ny in (y-1, y, y+1):
                            for nx in (x-1, x, x+1):
                                if ny == y and nx == x:
                                    continue
                                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                                    # we'll check threshold when popping to avoid duplicate checks
                                    stack.append((ny, nx))
                    except Exception as e:
                        print(f"Warning: Error processing pixel ({x}, {y}) in flood fill: {e}")
                        continue

                if count == 0:
                    return None

                # bbox width/height (ensure >=1)
                width = max_x - min_x + 1
                height = max_y - min_y + 1

                return {
                    'bbox': (min_x, min_y, width, height),
                    'points_count': count,
                }
            except Exception as e:
                print(f"Warning: Flood fill failed at ({start_x}, {start_y}): {e}")
                return None

        # iterate through edge pixels and flood-fill
        try:
            ys, xs = np.where(edges > edge_threshold)
            for y, x in zip(ys, xs):
                try:
                    if visited[y, x]:
                        continue
                    obj = flood_fill(y, x)
                    if obj is None:
                        continue
                    x0, y0, ww, hh = obj['bbox']
                    area = ww * hh
                    # filter small noise and very large (likely border) areas
                    if area > min_area and area < (img_area * max_area_ratio):
                        objects.append(obj)
                except Exception as e:
                    print(f"Warning: Failed to process edge pixel ({x}, {y}): {e}")
                    continue
        except Exception as e:
            raise RuntimeError(f"Failed during flood fill iteration: {e}")

        # ---------------------------
        # NEW: post-processing filters
        # 1) remove components that are completely inside another component
        # 2) merge overlapping or nearby components into their union bbox (iterative)
        # ---------------------------

        def rect_union(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
            # a and b are (x, y, w, h) -> return union bbox (x,y,w,h)
            try:
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                x1 = min(ax, bx)
                y1 = min(ay, by)
                x2 = max(ax + aw, bx + bw)
                y2 = max(ay + ah, by + bh)
                return (x1, y1, x2 - x1, y2 - y1)
            except Exception as e:
                print(f"Warning: Failed to compute rect union: {e}")
                return a

        def rect_intersect_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
            try:
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                ix1 = max(ax, bx)
                iy1 = max(ay, by)
                ix2 = min(ax + aw, bx + bw)
                iy2 = min(ay + ah, by + bh)
                if ix2 <= ix1 or iy2 <= iy1:
                    return 0
                return (ix2 - ix1) * (iy2 - iy1)
            except Exception as e:
                print(f"Warning: Failed to compute rect intersection: {e}")
                return 0

        def rect_gap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
            # compute the gap (in pixels) between two axis-aligned rectangles.
            # if they overlap or touch, gap is 0. otherwise gap is positive.
            try:
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                a_x2 = ax + aw
                b_x2 = bx + bw
                a_y2 = ay + ah
                b_y2 = by + bh

                # horizontal gap
                if a_x2 < bx:
                    dx = bx - a_x2
                elif b_x2 < ax:
                    dx = ax - b_x2
                else:
                    dx = 0

                # vertical gap
                if a_y2 < by:
                    dy = by - a_y2
                elif b_y2 < ay:
                    dy = ay - b_y2
                else:
                    dy = 0

                # use the larger axis gap as the measure (you may also use sqrt(dx^2+dy^2))
                return max(dx, dy)
            except Exception as e:
                print(f"Warning: Failed to compute rect gap: {e}")
                return 0

        # Convert objects -> list of bboxes + points_count
        try:
            boxes = [{'bbox': tuple(o['bbox']), 'points_count': o['points_count']} for o in objects]
        except Exception as e:
            print(f"Warning: Failed to convert objects to boxes: {e}")
            boxes = []

        # If no boxes, skip
        if not boxes:
            objects = []
        else:
            try:
                # Union-Find (Disjoint Set) for clustering boxes
                n = len(boxes)
                parent = list(range(n))

                def find(x: int) -> int:
                    try:
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    except Exception as e:
                        print(f"Warning: Find operation failed for {x}: {e}")
                        return x

                def union(a: int, b: int) -> None:
                    try:
                        ra = find(a)
                        rb = find(b)
                        if ra != rb:
                            parent[rb] = ra
                    except Exception as e:
                        print(f"Warning: Union operation failed for {a} and {b}: {e}")

                # build edges between boxes that overlap or are within merge_gap
                for i in range(n):
                    for j in range(i + 1, n):
                        try:
                            a = boxes[i]['bbox']
                            b = boxes[j]['bbox']
                            if rect_intersect_area(a, b) > 0 or rect_gap(a, b) <= merge_gap:
                                union(i, j)
                        except Exception as e:
                            print(f"Warning: Failed to process box pair ({i}, {j}): {e}")
                            continue

                # collect clusters
                clusters = {}
                for i in range(n):
                    try:
                        r = find(i)
                        clusters.setdefault(r, []).append(i)
                    except Exception as e:
                        print(f"Warning: Failed to cluster box {i}: {e}")
                        continue

                # compute union bbox and summed points_count for each cluster
                merged_boxes = []
                for indices in clusters.values():
                    try:
                        bx = boxes[indices[0]]['bbox']
                        pts = 0
                        ux1, uy1 = bx[0], bx[1]
                        ux2, uy2 = bx[0] + bx[2], bx[1] + bx[3]
                        for idx in indices:
                            try:
                                b = boxes[idx]['bbox']
                                pts += boxes[idx]['points_count']
                                ux1 = min(ux1, b[0])
                                uy1 = min(uy1, b[1])
                                ux2 = max(ux2, b[0] + b[2])
                                uy2 = max(uy2, b[1] + b[3])
                            except Exception as e:
                                print(f"Warning: Failed to process box {idx} in cluster: {e}")
                                continue
                        merged_bbox = (ux1, uy1, ux2 - ux1, uy2 - uy1)
                        merged_boxes.append({'bbox': merged_bbox, 'points_count': pts})
                    except Exception as e:
                        print(f"Warning: Failed to merge cluster: {e}")
                        continue

                # final objects is merged_boxes; still apply area filter safety (in case union produced huge boxes)
                final_boxes = []
                for mb in merged_boxes:
                    try:
                        x0, y0, ww, hh = mb['bbox']
                        area = ww * hh
                        if area > min_area and area < (img_area * max_area_ratio):
                            final_boxes.append(mb)
                    except Exception as e:
                        print(f"Warning: Failed to filter merged box: {e}")
                        continue

                objects = final_boxes
            except Exception as e:
                print(f"Warning: Failed during box merging process: {e}")
                # Keep original objects if merging fails
                pass

        # 5) draw bounding boxes and labels on the output image
        for idx, obj in enumerate(objects):
            try:
                x0, y0, ww, hh = obj['bbox']
                # rectangle
                cv2.rectangle(out_img, (x0, y0), (x0 + ww - 1, y0 + hh - 1), (0, 255, 0), 2)
                # label
                label = str(idx + 1)
                # choose a font scale that adapts to image size
                font_scale = max(0.5, min(2.0, w / 800.0))
                cv2.putText(out_img, label, (x0 + 5, y0 + int(20 * font_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Warning: Failed to draw box {idx}: {e}")
                continue

        meta = {
            'original_shape': (h, w),
            'binary_threshold': bin_threshold,
            'edge_threshold': edge_threshold,
            'min_area': min_area,
            'max_area_ratio': max_area_ratio,
            'merge_gap': merge_gap
        }

        return out_img, objects, meta
        
    except Exception as e:
        print(f"Critical error in detect_objects: {e}")
        raise