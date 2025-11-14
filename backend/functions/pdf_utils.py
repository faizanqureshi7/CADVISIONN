import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any
from pdf2image import convert_from_path
from .detector import detect_objects
import cv2

def process_pdf(pdf_path: str, 
                output_dir: str, 
                **detect_params: Any) -> int:
    """Process a PDF file, detecting objects on each page.

    Saves pages into `output_dir`. If a target filename already exists, appends
    a timestamp suffix to the filename so previous files are not overwritten.
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory to save output images
        **detect_params: Additional parameters to pass to detect_objects
        
    Returns:
        Total number of objects detected across all pages
    """
    try:
        # Create output directory
        try:
            out_base = str(output_dir)
            os.makedirs(out_base, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory '{output_dir}': {e}")

        # Convert PDF to images
        print(f"Converting PDF pages to images...")
        try:
            images = convert_from_path(pdf_path, dpi=200)
        except Exception as e:
            raise RuntimeError(f"Error converting PDF '{pdf_path}': {e}")

        if not images:
            raise ValueError(f"No pages found in PDF: {pdf_path}")

        print(f"Processing {len(images)} page(s)...")

        total_objects = 0
        for page_num, pil_image in enumerate(images, start=1):
            try:
                # Convert PIL image to OpenCV format
                try:
                    img_rgb = np.array(pil_image)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Warning: Failed to convert page {page_num} to BGR: {e}")
                    continue

                # Run object detection
                try:
                    out_img, objects, meta = detect_objects(img_bgr, **detect_params)
                except Exception as e:
                    print(f"Warning: Object detection failed on page {page_num}: {e}")
                    continue

                # Generate output filename
                try:
                    base_name = f"page_{page_num:03d}_detected.png"
                    output_path = os.path.join(out_base, base_name)

                    # if file exists, append timestamp (keeps all results in same folder)
                    if os.path.exists(output_path):
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        name, ext = os.path.splitext(base_name)
                        output_path = os.path.join(out_base, f"{name}_{ts}{ext}")
                except Exception as e:
                    print(f"Warning: Failed to generate output path for page {page_num}: {e}")
                    continue

                # Save output image
                try:
                    success = cv2.imwrite(output_path, out_img)
                    if not success:
                        raise RuntimeError(f"cv2.imwrite failed for: {output_path}")
                except Exception as e:
                    print(f"Warning: Failed to save output for page {page_num}: {e}")
                    continue

                # Print detection results
                try:
                    print(f"  Page {page_num}: Detected {len(objects)} object{'s' if len(objects) != 1 else ''}")
                    for i, o in enumerate(objects, start=1):
                        try:
                            x, y, w_box, h_box = o['bbox']
                            print(f"    {i}) bbox=({x},{y},{w_box},{h_box}), points={o['points_count']}")
                        except Exception as e:
                            print(f"Warning: Failed to print object {i} details on page {page_num}: {e}")
                            continue
                except Exception as e:
                    print(f"Warning: Failed to print detection summary for page {page_num}: {e}")

                total_objects += len(objects)

            except Exception as e:
                print(f"Warning: Failed to process page {page_num}: {e}")
                continue

        # Print summary
        try:
            print(f"\nTotal objects detected across all pages: {total_objects}")
            print(f"Saved output images to: {out_base}")
        except Exception as e:
            print(f"Warning: Failed to print final summary: {e}")

        return total_objects
        
    except Exception as e:
        print(f"Critical error in process_pdf: {e}", file=sys.stderr)
        raise