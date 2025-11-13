import base64
import shutil
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from functions.pipeline import process_documents, DEFAULT_DETECT_PARAMS, DEFAULT_CLIP_PARAMS


BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_JOBS_DIR = BASE_DIR / "temp_jobs"
OUTPUT_ROOT = BASE_DIR / "output_api"

app = FastAPI(title="CAD Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload_to_path(upload: UploadFile, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as out_file:
        upload.file.seek(0)
        shutil.copyfileobj(upload.file, out_file)
    upload.file.seek(0)


def _encode_image(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _serialize_matches(matches: Dict[int, Any]) -> Dict[str, Any]:
    return {
        str(idx): [
            {"img2_index": img2_idx, "similarity": float(similarity)}
            for img2_idx, similarity in match_list
        ]
        for idx, match_list in matches.items()
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/compare")
async def compare_documents(file1: UploadFile = File(...), file2: UploadFile = File(...)) -> Dict[str, Any]:
    if file1.filename is None or file2.filename is None:
        raise HTTPException(status_code=400, detail="Both files must include filenames.")

    job_id = uuid4().hex
    job_dir = TEMP_JOBS_DIR / job_id
    job_inputs_dir = job_dir / "inputs"
    job_outputs_dir = OUTPUT_ROOT / job_id

    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        job_inputs_dir.mkdir(parents=True, exist_ok=True)

        input1_path = job_inputs_dir / file1.filename
        input2_path = job_inputs_dir / file2.filename

        _save_upload_to_path(file1, input1_path)
        _save_upload_to_path(file2, input2_path)

        result = process_documents(
            input1_path=str(input1_path),
            input2_path=str(input2_path),
            output_dir=str(job_outputs_dir),
            detect_params=DEFAULT_DETECT_PARAMS,
            clip_params=DEFAULT_CLIP_PARAMS,
        )

        outputs = result["outputs"]
        response_payload = {
            "job_id": job_id,
            "matches": _serialize_matches(result["matches"]),
            "images": {
                "highlighted_1": _encode_image(Path(outputs["highlighted_1"])),
                "matched_1": _encode_image(Path(outputs["matched_1"])),
                "matched_2": _encode_image(Path(outputs["matched_2"])),
            },
        }

        return response_payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc