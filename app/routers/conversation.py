"""
ConvI — Conversation Router

API endpoints for audio and text conversation analysis.
"""

import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from typing import Optional
from loguru import logger

from app.config import Settings, get_settings
from app.schemas import (
    PipelineResponse,
    PipelineSegment,
    InputType,
)
from app.speech_pipeline.pipeline import (
    run_speech_pipeline_async,
    save_upload_to_temp,
    delete_temp_file,
)
from app.text_pipeline.text_parser import parse_transcript, text_turns_to_speech_segments

router = APIRouter(prefix="/api/v1", tags=["Conversation Intelligence"])


def _segments_to_response(
    segments: list,
    session_id: str,
    input_type: InputType,
    domain: str,
    audio_language: Optional[str] = None,
) -> PipelineResponse:
    """Convert pipeline segments to PipelineResponse."""
    pipeline_segments = [
        PipelineSegment(
            speaker_id=s.speaker_id,
            start_time=s.start_time,
            end_time=s.end_time,
            original_text=s.original_text,
            language=s.language,
            emotion=s.emotion,
            emotion_confidence=s.emotion_confidence or 0.0,
            transcription_confidence=s.transcription_confidence,
        )
        for s in segments
    ]
    return PipelineResponse(
        session_id=session_id,
        input_type=input_type,
        domain=domain,
        segments=pipeline_segments,
        audio_language=audio_language or (segments[0].audio_language if segments else None),
        total_segments=len(segments),
        unique_speakers=len({s.speaker_id for s in segments}),
    )


@router.post(
    "/analyze/audio",
    response_model=PipelineResponse,
    summary="Analyze audio conversation",
    description=(
        "Upload an **audio file** (WAV preferred). Runs the full speech pipeline: "
        "speaker diarization, ASR, and emotion detection. Returns segmented transcript."
    ),
    status_code=status.HTTP_200_OK,
)
async def analyze_audio(
    audio_file: UploadFile = File(
        ...,
        description="Audio recording (WAV preferred). Multi-speaker supported.",
    ),
    domain: str = Form(default="financial_banking"),
    session_id: Optional[str] = Form(default=None),
    skip_emotion: bool = Form(
        default=False,
        description="Skip emotion detection for faster processing.",
    ),
    settings: Settings = Depends(get_settings),
):
    session = session_id or str(uuid.uuid4())
    logger.info(f"[{session}] Audio upload: {audio_file.filename}")

    tmp_path: Optional[Path] = None
    try:
        tmp_path = await save_upload_to_temp(audio_file)
        segments = await run_speech_pipeline_async(
            audio_path=tmp_path,
            forced_language=None,
            skip_emotion=skip_emotion,
        )

        if not segments:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No speech detected in audio. Ensure the file contains clear speech.",
            )

        return _segments_to_response(
            segments=segments,
            session_id=session,
            input_type=InputType.audio,
            domain=domain,
            audio_language=segments[0].audio_language if segments else None,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {e}",
        )
    finally:
        if tmp_path:
            delete_temp_file(tmp_path)


@router.post(
    "/analyze/text",
    response_model=PipelineResponse,
    summary="Analyze text transcript",
    description=(
        "Submit a **text transcript**. Parses Agent/Customer turns and returns "
        "structured segments. Use format: 'Agent: ...' / 'Customer: ...' or plain lines."
    ),
    status_code=status.HTTP_200_OK,
)
async def analyze_text(
    text_transcript: str = Form(
        ...,
        description="Raw text transcript. Use 'Agent: ...' and 'Customer: ...' for speaker labels.",
    ),
    domain: str = Form(default="financial_banking"),
    session_id: Optional[str] = Form(default=None),
    settings: Settings = Depends(get_settings),
):
    session = session_id or str(uuid.uuid4())
    logger.info(f"[{session}] Text transcript ({len(text_transcript)} chars)")

    turns = parse_transcript(text_transcript)
    segments = text_turns_to_speech_segments(turns)

    if not segments:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No turns parsed from transcript. Use 'Agent: ...' and 'Customer: ...' format.",
        )

    return _segments_to_response(
        segments=segments,
        session_id=session,
        input_type=InputType.text,
        domain=domain,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ANALYTICS ENDPOINT — RAG + LLM + Storage
# Accepts audio OR text, runs the complete intelligence pipeline and returns
# the full ConversationAnalyticsResponse (summary, intent, compliance, risk, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/analyze",
    response_model="ConversationAnalyticsResponse",
    summary="Full conversation intelligence (audio or text)",
    description=(
        "Submit either an **audio file** (WAV/MP3) or a **text transcript**. "
        "Runs the complete pipeline: speech/text processing → conversation "
        "normalization → RAG retrieval → local LLM reasoning → analytics. "
        "Returns full structured banking intelligence JSON."
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
    from app.schemas import ConversationAnalyticsResponse

    # ── 0. Validate input ─────────────────────────────────────────────────
    if audio_file is None and not text_transcript:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either 'audio_file' or 'text_transcript'.",
        )

    session    = session_id or str(uuid.uuid4())
    input_type = InputType.audio if audio_file else InputType.text
    logger.info(f"[{session}] ▶ FULL ANALYZE {input_type.value.upper()} | domain={domain}")

    # ── Lazy imports ──────────────────────────────────────────────────────
    from app.conversation_normalizer import (
        normalize_from_speech,
        normalize_from_text,
        turns_to_dialogue_string,
    )
    from app.rag_engine import retriever as rag_retriever
    from app.llm_engine import run_llm_analysis
    from app import storage

    # ── 1a. AUDIO PATH ────────────────────────────────────────────────────
    turns = []
    if input_type == InputType.audio:
        tmp_path: Optional[Path] = None
        try:
            tmp_path = await save_upload_to_temp(audio_file)
            logger.info(f"[{session}] Audio saved → {tmp_path}")
            speech_segments = await run_speech_pipeline_async(
                audio_path=tmp_path,
                forced_language=None,
            )
            turns = normalize_from_speech(speech_segments)
        except Exception as e:
            logger.error(f"[{session}] Speech pipeline failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Speech pipeline error: {str(e)}",
            )
        finally:
            if tmp_path:
                delete_temp_file(tmp_path)

    # ── 1b. TEXT PATH ─────────────────────────────────────────────────────
    else:
        # Use the text_pipeline parser if available, fall back to normalizer
        try:
            raw_turns   = parse_transcript(text_transcript)
            segments    = text_turns_to_speech_segments(raw_turns)
            turns       = normalize_from_speech(segments)
        except Exception:
            turns = normalize_from_text(text_transcript, language="en")

    if not turns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No conversation content could be extracted from the input.",
        )
    logger.info(f"[{session}] {len(turns)} turns normalized.")

    # ── 2. RAG Retrieval ──────────────────────────────────────────────────
    dialogue_str = turns_to_dialogue_string(turns)
    rag_query    = " ".join(t.normalized_text_en for t in turns)[:1000]
    try:
        if not rag_retriever.is_ready:
            rag_retriever.load()
        rag_result = rag_retriever.retrieve(rag_query)
        logger.info(f"[{session}] RAG: {len(rag_result['rag_context_chunks'])} chunks.")
    except Exception as e:
        logger.warning(f"[{session}] RAG failed (non-fatal): {e}")
        rag_result = {"rag_context_chunks": [], "policy_references": []}

    # ── 3. LLM Analysis ───────────────────────────────────────────────────
    try:
        analysis = run_llm_analysis(
            turns=turns,
            rag_result=rag_result,
            domain=domain,
            dialogue_str=dialogue_str,
        )
    except RuntimeError as e:
        logger.error(f"[{session}] LLM error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    # ── 4. Storage (non-fatal) ────────────────────────────────────────────
    storage.save_session(
        session_id=session,
        domain=domain,
        input_type=input_type.value,
        risk_score=analysis["risk_score"],
        escalation_level=analysis["escalation_level"].value,
        call_outcome=analysis["basic_conversational_analysis"].call_outcome,
    )
    storage.save_turns(session, turns)
    storage.save_analytics(session, analysis)
    storage.log_event(session, "analyze_complete", f"risk={analysis['risk_score']}")

    logger.info(
        f"[{session}] ✅ Done | risk={analysis['risk_score']} "
        f"| escalation={analysis['escalation_level'].value}"
    )

    # ── 5. Response ───────────────────────────────────────────────────────
    return ConversationAnalyticsResponse(
        session_id=session,
        input_type=input_type,
        domain=domain,
        conversation_timeline=turns,
        basic_conversational_analysis=analysis["basic_conversational_analysis"],
        rag_based_analysis=analysis["rag_based_analysis"],
        timeline_analysis=analysis["timeline_analysis"],
        agent_performance_analysis=analysis["agent_performance_analysis"],
        confidence_scores=analysis["confidence_scores"],
        risk_score=analysis["risk_score"],
        escalation_level=analysis["escalation_level"],
    )
