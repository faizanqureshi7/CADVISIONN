import cv2
import numpy as np
from pdf2image import convert_from_path
import os


def load_image_any_format(file_path, dpi=300, all_pages=True, page_num=0):
    """
    Loads an image or PDF (multi-page supported) and returns one or more images.

    Args:
        file_path (str): Path to image or PDF file.
        dpi (int): Resolution for rendering PDF pages.
        all_pages (bool): If True, load all pages; if False, load only one (page_num).
        page_num (int): For PDFs, page index to load (0-based, used if all_pages=False).

    Returns:
        - np.ndarray for single-page image files
        - list[np.ndarray] for PDFs (one per page)
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"❌ Could not read image file: {file_path}")
        print(f"✅ Loaded image: {os.path.basename(file_path)}")
        return img

    elif ext == ".pdf":
        try:
            pages = convert_from_path(file_path, dpi=dpi)
            if len(pages) == 0:
                raise ValueError("❌ No pages found in PDF.")
            
            if all_pages:
                imgs = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]
                print(f"📄 Loaded {len(imgs)} pages from PDF: {os.path.basename(file_path)}")
                return imgs
            else:
                page = pages[page_num]
                img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                print(f"📄 Loaded page {page_num} from PDF: {os.path.basename(file_path)}")
                return img
        except Exception as e:
            raise ValueError(f"❌ Error reading PDF file: {e}")
    else:
        raise ValueError(f"⚠️ Unsupported file type: {ext}")


def optimize_image_for_processing(img, max_size=1500, quality=90):
    """
    Downscales and optimizes the image for faster processing.
    - Keeps aspect ratio.
    - Ensures largest dimension <= max_size.
    - Converts 16-bit to 8-bit if needed.
    
    Returns:
        tuple: (optimized_image, scale_factor)
    """
    h, w = img.shape[:2]
    print(f"🔍 Original image size: {w}x{h}")

    # Convert 16-bit images to 8-bit
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=(255.0 / np.max(img)))
        print("🔄 Converted 16-bit image to 8-bit.")

    # Resize if needed
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"📉 Image resized from ({w}, {h}) → ({new_w}, {new_h}) (scale={scale:.2f})")
    else:
        print("✅ Image within size limits, no resize needed.")

    # JPEG-like compression (optional; mainly for memory, not disk)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed = cv2.imencode(".jpg", img, encode_param)
    img = cv2.imdecode(compressed, cv2.IMREAD_COLOR)

    print(f"⚙️ Optimization complete. Final size: {img.shape[1]}x{img.shape[0]} | Quality: {quality}%")
    return img, scale