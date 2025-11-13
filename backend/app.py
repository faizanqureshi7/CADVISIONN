import base64
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2

from backend.functions.pipeline import process_documents, DEFAULT_DETECT_PARAMS, DEFAULT_CLIP_PARAMS
from backend.functions.preprocess import load_image_any_format, optimize_image_for_processing


BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="CAD Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _encode_image(path: Path) -> str:
    """Encode image to base64 string."""
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _serialize_matches(matches: Dict[int, Any]) -> Dict[str, Any]:
    """Serialize matches dictionary for JSON response."""
    return {
        str(idx): [
            {"img2_index": img2_idx, "similarity": float(similarity)}
            for img2_idx, similarity in match_list
        ]
        for idx, match_list in matches.items()
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    print("🏥 Health check requested")
    return {"status": "ok"}


@app.post("/compare")
async def compare_documents(
    file1: UploadFile = File(...), 
    file2: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Compare two CAD documents and return matched regions with highlighted differences.
    """
    print("\n" + "="*80)
    print("🚀 New comparison request received")
    print("="*80)
    
    if file1.filename is None or file2.filename is None:
        print("❌ Error: Missing filenames")
        raise HTTPException(status_code=400, detail="Both files must include filenames.")

    print(f"📁 File 1: {file1.filename}")
    print(f"📁 File 2: {file2.filename}")

    job_id = uuid4().hex
    print(f"🆔 Job ID: {job_id}")

    # Use system temp directory - automatically cleaned up after request
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Save uploaded files temporarily
            print("\n📥 Saving uploaded files...")
            input1_path = temp_path / f"input1_{file1.filename}"
            input2_path = temp_path / f"input2_{file2.filename}"

            with input1_path.open("wb") as f:
                file1.file.seek(0)
                shutil.copyfileobj(file1.file, f)
            print(f"✅ Saved: {input1_path.name}")

            with input2_path.open("wb") as f:
                file2.file.seek(0)
                shutil.copyfileobj(file2.file, f)
            print(f"✅ Saved: {input2_path.name}")

            # Load and preprocess images
            print("\n🔄 Loading and preprocessing images...")
            img1 = load_image_any_format(str(input1_path), all_pages=False, page_num=0)
            img2 = load_image_any_format(str(input2_path), all_pages=False, page_num=0)

            # Handle case where images might be lists (safety check)
            if isinstance(img1, list):
                print("⚠️ Image 1 returned as list, using first page")
                img1 = img1[0]
            if isinstance(img2, list):
                print("⚠️ Image 2 returned as list, using first page")
                img2 = img2[0]

            print("⚙️ Optimizing Image 1...")
            img1_optimized, scale1 = optimize_image_for_processing(img1, max_size=1500, quality=90)
            
            print("⚙️ Optimizing Image 2...")
            img2_optimized, scale2 = optimize_image_for_processing(img2, max_size=1500, quality=90)

            # Save optimized images to temp directory
            optimized1_path = temp_path / "optimized1.jpg"
            optimized2_path = temp_path / "optimized2.jpg"
            
            cv2.imwrite(str(optimized1_path), img1_optimized)
            cv2.imwrite(str(optimized2_path), img2_optimized)
            print(f"💾 Saved optimized images to temp directory")

            # Create output directory in temp
            output_dir = temp_path / "outputs"
            output_dir.mkdir(exist_ok=True)

            # Process documents
            print("\n🔍 Starting document comparison pipeline...")
            result = process_documents(
                input1_path=str(optimized1_path),
                input2_path=str(optimized2_path),
                output_dir=str(output_dir),
                detect_params=DEFAULT_DETECT_PARAMS,
                clip_params=DEFAULT_CLIP_PARAMS,
            )

            print("✅ Pipeline completed successfully")
            print(f"🎯 Found {len(result['matches'])} matched groups")

            # Read output images and encode to base64
            print("\n📤 Encoding output images...")
            outputs = result["outputs"]
            prepared_inputs = result["prepared_inputs"]
            
            response_payload = {
                "job_id": job_id,
                "matches": _serialize_matches(result["matches"]),
                "images": {
                    "highlighted_1": _encode_image(Path(outputs["highlighted_1"])),
                    "matched_1": _encode_image(Path(outputs["matched_1"])),
                    "matched_2": _encode_image(Path(outputs["matched_2"])),
                    "input_1": _encode_image(Path(prepared_inputs["img1"])),
                    "input_2": _encode_image(Path(prepared_inputs["img2"])),
                },
            }

            print("✅ Response prepared successfully")
            print(f"📦 Response payload size: {len(str(response_payload))} bytes")
            print("="*80 + "\n")
            
            return response_payload
            
        except Exception as exc:
            print(f"\n❌ ERROR: {str(exc)}")
            import traceback
            print(traceback.format_exc())
            print("="*80 + "\n")
            raise HTTPException(status_code=500, detail=str(exc)) from exc