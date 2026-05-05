import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.agent import AgenticOCRAgent
from app.utils.file_manager import FileManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/agent/convert")
async def agent_convert_to_word(
    images: List[UploadFile] = File(...),
    user_goal: str = Form("Digitize these handwritten notes faithfully into a structured Word document."),
    session_id: Optional[str] = Form(None),
):
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")

    file_manager = FileManager()
    uploads_dir = file_manager.get_uploads_dir()
    outputs_dir = file_manager.get_outputs_dir()
    saved_paths: List[str] = []

    try:
        for index, image in enumerate(images):
            suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                prefix=f"agent_{index}_",
                dir=uploads_dir,
            )
            content = await image.read()
            temp_file.write(content)
            temp_file.close()
            saved_paths.append(temp_file.name)
            await image.seek(0)

        agent = AgenticOCRAgent(base_dir=file_manager.base_dir)
        state = await agent.run(
            image_paths=saved_paths,
            uploads_dir=uploads_dir,
            outputs_dir=outputs_dir,
            user_goal=user_goal,
            session_id=session_id,
        )

        if not state.final_document_path or not Path(state.final_document_path).exists():
            raise HTTPException(status_code=500, detail="Agent did not generate a Word document")

        response_headers = {
            "X-Agent-Session-Id": state.session_id,
            "X-Agent-Review-Required": str(state.human_review_required).lower(),
            "X-Agent-Confidence-Report": json.dumps(state.confidence_report),
        }
        return FileResponse(
            state.final_document_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=Path(state.final_document_path).name,
            headers=response_headers,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Agentic conversion failed")
        raise HTTPException(status_code=500, detail=f"Agentic conversion failed: {str(exc)}")


@router.post("/agent/analyze")
async def agent_analyze_only(
    images: List[UploadFile] = File(...),
    user_goal: str = Form("Analyze these handwritten notes for OCR conversion."),
    session_id: Optional[str] = Form(None),
):
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")

    file_manager = FileManager()
    uploads_dir = file_manager.get_uploads_dir()
    outputs_dir = file_manager.get_outputs_dir()
    saved_paths: List[str] = []

    try:
        for index, image in enumerate(images):
            suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
                prefix=f"agent_analysis_{index}_",
                dir=uploads_dir,
            )
            content = await image.read()
            temp_file.write(content)
            temp_file.close()
            saved_paths.append(temp_file.name)
            await image.seek(0)

        agent = AgenticOCRAgent(base_dir=file_manager.base_dir)
        state = await agent.run(
            image_paths=saved_paths,
            uploads_dir=uploads_dir,
            outputs_dir=outputs_dir,
            user_goal=user_goal,
            session_id=session_id,
        )
        payload = _json_safe(asdict(state))
        return JSONResponse(content=payload)
    except Exception as exc:
        logger.exception("Agentic analysis failed")
        raise HTTPException(status_code=500, detail=f"Agentic analysis failed: {str(exc)}")


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
