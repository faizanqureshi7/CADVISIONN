import base64
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

from backend.functions.pipeline import process_documents, DEFAULT_DETECT_PARAMS, DEFAULT_CLIP_PARAMS
from backend.functions.preprocess import load_image_any_format, optimize_image_for_processing
from backend.functions.summary import create_summary_visualization, generate_summary_with_gemini


BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="CAD Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _encode_image(path: Path) -> str:
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
            
        try:
            with path.open("rb") as f:
                data = f.read()
                if not data:
                    raise ValueError(f"Empty image file: {path}")
                return base64.b64encode(data).decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to read and encode image '{path}': {e}")
            
    except Exception as e:
        print(f"Error in _encode_image: {e}")
        raise


def _serialize_matches(matches: Dict[int, Any]) -> Dict[str, Any]:
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
        print(f"Error in _serialize_matches: {e}")
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    try:
        print("🏥 Health check requested")
        return {"status": "ok"}
    except Exception as e:
        print(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.post("/compare")
async def compare_documents(
    file1: UploadFile = File(...), 
    file2: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Compare two CAD documents and return matched regions with highlighted differences and AI summary.
    
    Args:
        file1: First uploaded file (PDF or image)
        file2: Second uploaded file (PDF or image)
        
    Returns:
        Dictionary containing job ID, matches, base64-encoded images, and AI summary text
    """
    print("\n" + "="*80)
    print("🚀 New comparison request received")
    print("="*80)
    
    try:
        # Validate filenames
        if file1.filename is None or file2.filename is None:
            print("❌ Error: Missing filenames")
            raise HTTPException(status_code=400, detail="Both files must include filenames.")

        print(f"📁 File 1: {file1.filename}")
        print(f"📁 File 2: {file2.filename}")

        try:
            job_id = uuid4().hex
            print(f"🆔 Job ID: {job_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate job ID: {e}")

        # Use system temp directory - automatically cleaned up after request
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # Save uploaded files temporarily
                print("\n📥 Saving uploaded files...")
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

                # Load and preprocess images
                print("\n🔄 Loading and preprocessing images...")
                try:
                    img1 = load_image_any_format(str(input1_path), all_pages=False, page_num=0)
                except Exception as e:
                    raise RuntimeError(f"Failed to load image 1: {e}")
                    
                try:
                    img2 = load_image_any_format(str(input2_path), all_pages=False, page_num=0)
                except Exception as e:
                    raise RuntimeError(f"Failed to load image 2: {e}")

                # Handle case where images might be lists (safety check)
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

                # Optimize images
                print("⚙️ Optimizing Image 1...")
                try:
                    img1_optimized, scale1 = optimize_image_for_processing(img1, max_size=1500, quality=90)
                except Exception as e:
                    raise RuntimeError(f"Failed to optimize image 1: {e}")
                
                print("⚙️ Optimizing Image 2...")
                try:
                    img2_optimized, scale2 = optimize_image_for_processing(img2, max_size=1500, quality=90)
                except Exception as e:
                    raise RuntimeError(f"Failed to optimize image 2: {e}")

                # Save optimized images to temp directory
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

                # Create output directory in temp
                try:
                    output_dir = temp_path / "outputs"
                    output_dir.mkdir(exist_ok=True)
                except Exception as e:
                    raise RuntimeError(f"Failed to create output directory: {e}")

                # Process documents
                print("\n🔍 Starting document comparison pipeline...")
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

                # Get output paths
                try:
                    outputs = result.get("outputs", {})
                    prepared_inputs = result.get("prepared_inputs", {})
                    
                    if not outputs or not prepared_inputs:
                        raise ValueError("Missing outputs or prepared_inputs in pipeline result")
                except Exception as e:
                    raise RuntimeError(f"Failed to extract pipeline results: {e}")

                # Generate AI summary using highlighted image from pipeline
                print("\n🤖 Generating AI summary...")
                ai_summary: Optional[str] = None
                
                try:
                    # Load the highlighted image created by the pipeline
                    print("   📂 Loading highlighted image from pipeline...")
                    try:
                        highlighted_path = Path(outputs["highlighted_1"])
                        highlighted = cv2.imread(str(highlighted_path))
                        if highlighted is None:
                            raise ValueError(f"Could not read highlighted image: {highlighted_path}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to load highlighted image: {e}")
                    
                    # Create composite visualization for Gemini using optimized images
                    print("   📊 Creating visualization for AI analysis...")
                    try:
                        composite = create_summary_visualization(
                            img1_optimized,  # Use optimized image 1
                            img2_optimized,  # Use optimized image 2
                            highlighted      # Use highlighted image from pipeline
                        )
                    except Exception as e:
                        raise RuntimeError(f"Failed to create visualization: {e}")
                    
                    # Generate summary text only
                    print("   💬 Generating summary text...")
                    try:
                        ai_summary = generate_summary_with_gemini(composite)
                    except Exception as e:
                        print(f"Warning: AI summary generation failed: {e}")
                        ai_summary = f"⚠️ AI summary unavailable: {e}"
                    
                    print(f"   ✅ Summary generated ({len(ai_summary)} characters)")
                    # print(f" AI summary is {ai_summary}")
                    
                except Exception as e:
                    print(f"Warning: Summary generation failed: {e}")
                    ai_summary = f"❌ Summary generation failed: {e}"

                # Read output images and encode to base64
                print("\n📤 Encoding output images...")
                
                try:
                    response_payload = {
                        "job_id": job_id,
                        "matches": _serialize_matches(result.get("matches", {})),
                        "images": {
                            "highlighted_1": _encode_image(Path(outputs["highlighted_1"])),
                            "matched_1": _encode_image(Path(outputs["matched_1"])),
                            "matched_2": _encode_image(Path(outputs["matched_2"])),
                            "input_1": _encode_image(Path(prepared_inputs["img1"])),
                            "input_2": _encode_image(Path(prepared_inputs["img2"])),
                            "optimized_input_1": _encode_image(optimized1_path),  # Add optimized image 1
                            "optimized_input_2": _encode_image(optimized2_path),  # Add optimized image 2
                        },
                    }
                    
                    # Add AI summary if available
                    if ai_summary:
                        response_payload["ai_summary"] = ai_summary
                    
                except KeyError as e:
                    raise RuntimeError(f"Missing required output file: {e}")
                except Exception as e:
                    raise RuntimeError(f"Failed to create response payload: {e}")

                try:
                    print("✅ Response prepared successfully")
                    print(f"📦 Response payload size: {len(str(response_payload))} bytes")
                    if ai_summary:
                        print(f"📝 AI summary: {len(ai_summary)} characters")
                    print("="*80 + "\n")
                except Exception as e:
                    print(f"Warning: Failed to print response summary: {e}")
                
                return response_payload
                
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except Exception as exc:
                print(f"\n❌ ERROR: {str(exc)}")
                import traceback
                print(traceback.format_exc())
                print("="*80 + "\n")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
                
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as exc:
        print(f"\n❌ CRITICAL ERROR: {str(exc)}")
        import traceback
        print(traceback.format_exc())
        print("="*80 + "\n")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(exc)}") from exc