"""
ConvI — Conversation Router

Main API endpoint for submitting audio or text conversations for analysis.
"""

import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from typing import Optional
from loguru import logger

from app.config import Settings, get_settings
from app.schemas import ConversationAnalyticsResponse, InputType

router = APIRouter(prefix="/api/v1", tags=["Conversation Intelligence"])


@router.post(
    "/analyze",
    response_model=ConversationAnalyticsResponse,
    summary="Analyze a conversation (audio or text)",
    description=(
        "Submit either an **audio file** (WAV/MP3) or a **text transcript**. "
        "The system will run the appropriate pipeline and return full structured analytics."
    ),
    status_code=status.HTTP_200_OK,
)
async def analyze_conversation(
    audio_file: Optional[UploadFile] = File(
        default=None,
        description="Audio recording (WAV preferred). Multi-speaker supported.",
    ),
    text_transcript: Optional[str] = Form(
        default=None,
        description="Raw text transcript (used when no audio file is provided).",
    ),
    domain: str = Form(default="financial_banking"),
    session_id: Optional[str] = Form(default=None),
    settings: Settings = Depends(get_settings),
):
    # ── Input validation ──────────────────────────────────────────────────
    if audio_file is None and not text_transcript:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either 'audio_file' or 'text_transcript'.",
        )

    session = session_id or str(uuid.uuid4())
    input_type = InputType.audio if audio_file else InputType.text

    logger.info(f"[{session}] Received {input_type.value} request | domain={domain}")

    # ── TODO: wire up pipeline modules ───────────────────────────────────
    # Step 1: Route to audio OR text pipeline
    # Step 2: Conversation Normalizer
    # Step 3: RAG Engine
    # Step 4: LLM Engine
    # Step 5: Analytics Engine
    # Step 6: Storage Layer
    # Step 7: Return ConversationAnalyticsResponse

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Pipeline modules not yet implemented. Structure ready for integration.",
    )
