import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from pdf2image import convert_from_path
from .detector import detect_objects
import cv2

def process_pdf(pdf_path, output_dir, **detect_params):
    """Process a PDF file, detecting objects on each page.

    Saves pages into `output_dir`. If a target filename already exists, appends
    a timestamp suffix to the filename so previous files are not overwritten.
    """
    out_base = str(output_dir)
    os.makedirs(out_base, exist_ok=True)

    print(f"Converting PDF pages to images...")
    try:
        images = convert_from_path(pdf_path, dpi=200)
    except Exception as e:
        print(f"Error converting PDF: {e}", file=sys.stderr)
        raise

    print(f"Processing {len(images)} page(s)...")

    total_objects = 0
    for page_num, pil_image in enumerate(images, start=1):
        img_rgb = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        out_img, objects, meta = detect_objects(img_bgr, **detect_params)

        # base filename
        base_name = f"page_{page_num:03d}_detected.png"
        output_path = os.path.join(out_base, base_name)

        # if file exists, append timestamp (keeps all results in same folder)
        if os.path.exists(output_path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(out_base, f"{name}_{ts}{ext}")

        cv2.imwrite(output_path, out_img)

        print(f"  Page {page_num}: Detected {len(objects)} object{'s' if len(objects) != 1 else ''}")
        for i, o in enumerate(objects, start=1):
            x, y, w_box, h_box = o['bbox']
            print(f"    {i}) bbox=({x},{y},{w_box},{h_box}), points={o['points_count']}")

        total_objects += len(objects)

    print(f"\nTotal objects detected across all pages: {total_objects}")
    print(f"Saved output images to: {out_base}")

    return total_objects