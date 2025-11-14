import tempfile
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.functions.processing import process_cad_comparison


BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="CAD Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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

        job_id = uuid4().hex
        print(f"🆔 Job ID: {job_id}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # Run entire processing pipeline
                response_payload = process_cad_comparison(file1, file2, job_id, temp_path)
                return response_payload
                
            except Exception as exc:
                print(f"\n❌ ERROR: {str(exc)}")
                import traceback
                print(traceback.format_exc())
                print("="*80 + "\n")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
                
    except HTTPException:
        raise
    except Exception as exc:
        print(f"\n❌ CRITICAL ERROR: {str(exc)}")
        import traceback
        print(traceback.format_exc())
        print("="*80 + "\n")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(exc)}") from exc