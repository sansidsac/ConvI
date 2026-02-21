"""
ConvI — Full Pipeline Integration Test
=======================================
Step 1 : Take PipelineResponse (from /analyze/audio)
Step 2 : Conversation Normalizer  →  List[ConversationTurn]
Step 3 : RAG Retrieval            →  top-5 context chunks
Step 4 : LLM Engine (Ollama)      →  full analytics

Usage:
    venv\Scripts\python test_full_pipeline.py
    venv\Scripts\python test_full_pipeline.py --skip-llm   (to skip Ollama)
"""

from __future__ import annotations
import sys
import json
import pprint
sys.path.insert(0, ".")          # make sure app/ is importable

SKIP_LLM = "--skip-llm" in sys.argv

# ── Reproduce the /analyze/audio output ──────────────────────────────────────
# This is the actual response returned by the live server test.
PIPELINE_RESPONSE = {
    "session_id": "4f24a554-bcd1-4edb-888c-d0f681c16e40",
    "input_type": "audio",
    "domain": "financial_banking",
    "audio_language": "en",
    "total_segments": 11,
    "unique_speakers": 2,
    "segments": [
        {"speaker_id": "SPEAKER_00", "start_time": 0.031,  "end_time": 2.039,  "original_text": "Good morning. This is XYZ Bank.",                                                                               "language": "en", "emotion": "neutral", "emotion_confidence": 1.0,    "transcription_confidence": 0.7175},
        {"speaker_id": "SPEAKER_01", "start_time": 2.039,  "end_time": 3.035,  "original_text": "How may I help you?",                                                                                          "language": "en", "emotion": "neutral", "emotion_confidence": 1.0,    "transcription_confidence": 0.9808},
        {"speaker_id": "SPEAKER_01", "start_time": 3.727,  "end_time": 6.46,   "original_text": "accidentally transferred 25,000 to the wrong account number.",                                                  "language": "en", "emotion": "neutral", "emotion_confidence": 1.0,    "transcription_confidence": 0.9095},
        {"speaker_id": "SPEAKER_00", "start_time": 7.068,  "end_time": 9.97,   "original_text": "I am sorry to hear that. When was the transaction made?",                                                      "language": "en", "emotion": "neutral", "emotion_confidence": 1.0,    "transcription_confidence": 0.9469},
        {"speaker_id": "SPEAKER_00", "start_time": 10.122, "end_time": 18.56,  "original_text": "Around 28 minutes ago through UPI. I will need to verify your identity. Please confirm your name and last four digits of your mobile number.", "language": "en", "emotion": "neutral", "emotion_confidence": 0.9999, "transcription_confidence": 0.9153},
        {"speaker_id": "SPEAKER_00", "start_time": 18.796, "end_time": 30.659, "original_text": "Kiran 4582. Thank you. I can see the transaction was successful. We can raise a wrong credit recovery request. Will I get my money back? I am really worried.", "language": "en", "emotion": "neutral", "emotion_confidence": 0.9999, "transcription_confidence": 0.9214},
        {"speaker_id": "SPEAKER_00", "start_time": 32.245, "end_time": 38.422, "original_text": "per the RBI guidelines, we will contact the recipient bank and update you within 7 to 10 working days. Okay, please register",                          "language": "en", "emotion": "neutral", "emotion_confidence": 1.0,    "transcription_confidence": 0.9087},
        {"speaker_id": "SPEAKER_01", "start_time": 37.747, "end_time": 41.071, "original_text": "please register the complaint. I need the money back. Really.",                                                 "language": "en", "emotion": "neutral", "emotion_confidence": 1.0,    "transcription_confidence": 0.8922},
        {"speaker_id": "SPEAKER_00", "start_time": 41.628, "end_time": 41.898, "original_text": "Yeah,",                                                                                                         "language": "en", "emotion": "neutral", "emotion_confidence": 0.0,    "transcription_confidence": 0.3513},
        {"speaker_id": "SPEAKER_01", "start_time": 42.117, "end_time": 42.337, "original_text": "surely.",                                                                                                        "language": "en", "emotion": "neutral", "emotion_confidence": 0.0,    "transcription_confidence": 0.8773},
        {"speaker_id": "SPEAKER_01", "start_time": 42.522, "end_time": 43.855, "original_text": "Thank you.",                                                                                                     "language": "en", "emotion": "neutral", "emotion_confidence": 1.0,    "transcription_confidence": 0.9585},
    ],
}

SEP = "=" * 70

def step(n, title):
    print(f"\n{SEP}\n  STEP {n}: {title}\n{SEP}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Convert PipelineResponse segments → SpeechSegment objects
# ─────────────────────────────────────────────────────────────────────────────
step(1, "PipelineResponse → SpeechSegment objects")

from app.speech_pipeline.schemas import SpeechSegment

speech_segments = [
    SpeechSegment(
        speaker_id              = s["speaker_id"],
        start_time              = s["start_time"],
        end_time                = s["end_time"],
        original_text           = s["original_text"],
        language                = s["language"],
        emotion                 = s.get("emotion"),
        emotion_confidence      = s.get("emotion_confidence", 0.0),
        transcription_confidence= s.get("transcription_confidence", 1.0),
        audio_language          = PIPELINE_RESPONSE["audio_language"],
    )
    for s in PIPELINE_RESPONSE["segments"]
]

print(f"  ✅ {len(speech_segments)} SpeechSegments reconstructed.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Conversation Normalizer
# ─────────────────────────────────────────────────────────────────────────────
step(2, "Conversation Normalizer")

from app.conversation_normalizer import normalize_from_speech, turns_to_dialogue_string

turns = normalize_from_speech(speech_segments)

print(f"\n  ✅ {len(turns)} ConversationTurns generated")
print(f"\n  {'Speaker':<12} {'Role':<10} {'Emotion':<10} {'Text'}")
print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*40}")
for t in turns:
    print(f"  {t.speaker_id:<12} {t.role.value:<10} {str(t.emotion):<10} {t.normalized_text_en[:60]}")

dialogue_string = turns_to_dialogue_string(turns)
print(f"\n  ── Dialogue String ──\n")
print(dialogue_string)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — RAG Retrieval (top 5 chunks)
# ─────────────────────────────────────────────────────────────────────────────
step(3, "RAG Retrieval — top 5 chunks")

from app.rag_engine import retriever as rag

rag_result = rag.retrieve(dialogue_string, top_k=5)

rag_chunks = rag_result.get("rag_context_chunks", [])
policy_refs = rag_result.get("policy_references", [])

print(f"\n  ✅ Retrieved {len(rag_chunks)} chunks")
print(f"\n  Policy references:")
for ref in policy_refs:
    print(f"    • {ref['source']} — page {ref['page']} [{ref['doc_type']}]  score={ref['score']:.4f}")

print(f"\n  ── Top 5 RAG Chunks ──")
for i, chunk in enumerate(rag_chunks, 1):
    print(f"\n  [{i}] {chunk[:300]}{'...' if len(chunk) > 300 else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — LLM Analysis (Ollama / Qwen2.5:7b)
# ─────────────────────────────────────────────────────────────────────────────
step(4, "LLM Analysis via Ollama (Qwen2.5:7b)")

if SKIP_LLM:
    print("  ⏭  --skip-llm flag set. Skipping Ollama call.")
    print("     Run without --skip-llm to get full analysis.")
    sys.exit(0)

from app.llm_engine import run_llm_analysis

print("  Calling Ollama... (may take 30-120s on CPU)")
try:
    result = run_llm_analysis(
        turns       = turns,
        rag_result  = rag_result,
        domain      = PIPELINE_RESPONSE["domain"],
        dialogue_str= dialogue_string,
    )

    print(f"\n  ✅ LLM analysis complete!\n")

    # ── Print structured output ───────────────────────────────────────────
    basic  = result["basic_conversational_analysis"]
    rag_an = result["rag_based_analysis"]
    tl     = result["timeline_analysis"]
    agent  = result["agent_performance_analysis"]
    conf   = result["confidence_scores"]

    print(f"\n{'─'*70}")
    print(f"  BASIC CONVERSATIONAL ANALYSIS")
    print(f"{'─'*70}")
    print(f"  Summary      : {basic.conversation_summary}")
    print(f"  Intent       : {basic.customer_intention}")
    print(f"  Topics       : {basic.key_topics}")
    print(f"  Emotion Tone : {basic.overall_emotional_tone}")
    print(f"  Outcome      : {basic.call_outcome}")
    print(f"  Language     : {basic.language_detected}")

    print(f"\n{'─'*70}")
    print(f"  RAG-BASED ANALYSIS")
    print(f"{'─'*70}")
    print(f"  Compliance Flags  : {rag_an.compliance_flags or 'none'}")
    print(f"  Fraud Indicators  : {rag_an.fraud_indicators or 'none'}")
    print(f"  Policy References : {rag_an.policy_references[:3]}")

    print(f"\n{'─'*70}")
    print(f"  RISK & ESCALATION")
    print(f"{'─'*70}")
    print(f"  Risk Score       : {result['risk_score']}")
    print(f"  Escalation Level : {result['escalation_level'].value}")

    print(f"\n{'─'*70}")
    print(f"  AGENT PERFORMANCE")
    print(f"{'─'*70}")
    print(f"  Score            : {agent.performance_score}/10")
    print(f"  De-escalation    : {agent.de_escalation_detected}")
    print(f"  Tone Shift       : {agent.tone_shift_detected}")
    print(f"  Metrics          : {agent.interaction_metrics}")

    print(f"\n{'─'*70}")
    print(f"  CONFIDENCE SCORES")
    print(f"{'─'*70}")
    print(f"  Transcription    : {conf.transcription}")
    print(f"  Language Detect  : {conf.language_detection}")
    print(f"  Emotion          : {conf.emotion_detection}")
    print(f"  LLM Reasoning    : {conf.llm_reasoning}")
    print(f"  RAG Retrieval    : {conf.rag_retrieval}")

    print(f"\n{'─'*70}")
    print(f"  EMOTION TIMELINE (first 5 points)")
    print(f"{'─'*70}")
    for pt in tl.emotion_timeline[:5]:
        print(f"  {pt.speaker_id} @ {pt.timestamp:.1f}s  emotion={pt.emotion}  sentiment={pt.sentiment_score}  risk={pt.risk_score}")

except RuntimeError as e:
    if "ollama" in str(e).lower() or "connect" in str(e).lower():
        print(f"\n  ⚠  Ollama not reachable: {e}")
        print("     Start Ollama with:  ollama serve")
        print("     Pull model with:    ollama pull qwen2.5:7b")
    else:
        raise
