import cv2
import numpy as np
from pdf2image import convert_from_path
import os
from typing import Union, List, Tuple


def load_image_any_format(file_path: str, 
                          dpi: int = 300, 
                          all_pages: bool = True, 
                          page_num: int = 0) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Loads an image or PDF (multi-page supported) and returns one or more images.

    Args:
        file_path: Path to image or PDF file.
        dpi: Resolution for rendering PDF pages.
        all_pages: If True, load all pages; if False, load only one (page_num).
        page_num: For PDFs, page index to load (0-based, used if all_pages=False).

    Returns:
        - np.ndarray for single-page image files
        - list[np.ndarray] for PDFs (one per page)
    """
    try:
        # Validate file path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            ext = os.path.splitext(file_path)[-1].lower()
        except Exception as e:
            raise RuntimeError(f"Failed to extract file extension: {e}")

        if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            try:
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError(f"❌ Could not read image file: {file_path}")
                print(f"✅ Loaded image: {os.path.basename(file_path)}")
                return img
            except Exception as e:
                raise RuntimeError(f"Failed to load image file '{file_path}': {e}")

        elif ext == ".pdf":
            try:
                pages = convert_from_path(file_path, dpi=dpi)
            except Exception as e:
                raise RuntimeError(f"Failed to convert PDF '{file_path}': {e}")
                
            if len(pages) == 0:
                raise ValueError("❌ No pages found in PDF.")
            
            if all_pages:
                try:
                    imgs = []
                    for i, p in enumerate(pages):
                        try:
                            img = cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
                            imgs.append(img)
                        except Exception as e:
                            print(f"Warning: Failed to convert page {i}: {e}")
                            continue
                            
                    if len(imgs) == 0:
                        raise RuntimeError("Failed to convert any PDF pages")
                        
                    print(f"📄 Loaded {len(imgs)} pages from PDF: {os.path.basename(file_path)}")
                    return imgs
                except Exception as e:
                    raise RuntimeError(f"Failed to process PDF pages: {e}")
            else:
                try:
                    if page_num >= len(pages):
                        raise ValueError(f"Page number {page_num} out of range (PDF has {len(pages)} pages)")
                        
                    page = pages[page_num]
                    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                    print(f"📄 Loaded page {page_num} from PDF: {os.path.basename(file_path)}")
                    return img
                except Exception as e:
                    raise RuntimeError(f"Failed to load page {page_num} from PDF: {e}")
        else:
            raise ValueError(f"⚠️ Unsupported file type: {ext}")
            
    except Exception as e:
        print(f"Critical error in load_image_any_format: {e}")
        raise


def optimize_image_for_processing(img: np.ndarray, 
                                  max_size: int = 1500, 
                                  quality: int = 90) -> Tuple[np.ndarray, float]:
    """
    Downscales and optimizes the image for faster processing.
    - Keeps aspect ratio.
    - Ensures largest dimension <= max_size.
    - Converts 16-bit to 8-bit if needed.
    
    Args:
        img: Input image as numpy array
        max_size: Maximum dimension size
        quality: JPEG quality for compression (0-100)
        
    Returns:
        tuple: (optimized_image, scale_factor)
    """
    try:
        # Validate input
        if img is None or img.size == 0:
            raise ValueError("Invalid input image")
            
        try:
            h, w = img.shape[:2]
            print(f"🔍 Original image size: {w}x{h}")
        except Exception as e:
            raise RuntimeError(f"Failed to get image dimensions: {e}")

        # Convert 16-bit images to 8-bit
        if img.dtype != np.uint8:
            try:
                max_val = np.max(img)
                if max_val == 0:
                    max_val = 1  # Prevent division by zero
                img = cv2.convertScaleAbs(img, alpha=(255.0 / max_val))
                print("🔄 Converted 16-bit image to 8-bit.")
            except Exception as e:
                raise RuntimeError(f"Failed to convert image bit depth: {e}")

        # Resize if needed
        scale = 1.0
        if max(h, w) > max_size:
            try:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Ensure dimensions are at least 1
                new_w = max(1, new_w)
                new_h = max(1, new_h)
                
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"📉 Image resized from ({w}, {h}) → ({new_w}, {new_h}) (scale={scale:.2f})")
            except Exception as e:
                raise RuntimeError(f"Failed to resize image: {e}")
        else:
            print("✅ Image within size limits, no resize needed.")

        # JPEG-like compression (optional; mainly for memory, not disk)
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, compressed = cv2.imencode(".jpg", img, encode_param)
            if compressed is None or len(compressed) == 0:
                raise RuntimeError("JPEG encoding produced empty result")
                
            img = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("JPEG decoding failed")
        except Exception as e:
            print(f"Warning: JPEG compression failed, using original: {e}")
            # Continue with uncompressed image

        try:
            print(f"⚙️ Optimization complete. Final size: {img.shape[1]}x{img.shape[0]} | Quality: {quality}%")
        except Exception as e:
            print(f"Warning: Failed to print optimization summary: {e}")
            
        return img, scale
        
    except Exception as e:
        print(f"Critical error in optimize_image_for_processing: {e}")
        raise