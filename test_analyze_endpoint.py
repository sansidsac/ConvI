"""
ConvI — Full /analyze Endpoint Test
=====================================
Tests the complete HTTP pipeline:
    POST /api/v1/analyze   (audio file)
    → PipelineResponse → Normalizer → RAG → LLM → ConversationAnalyticsResponse

Usage:
    venv\Scripts\python test_analyze_endpoint.py
    venv\Scripts\python test_analyze_endpoint.py --text   (text mode)
"""

from __future__ import annotations
import sys
import json

import httpx

BASE_URL   = "http://localhost:8000/api/v1"
AUDIO_FILE = "inputs/Trial 1 convi.wav"
TIMEOUT    = 600   # seconds — Whisper + Ollama both take time

TEXT_TRANSCRIPT = """
Agent: Good morning. This is XYZ Bank. How may I help you?
Customer: I accidentally transferred 25,000 to the wrong account number.
Agent: I am sorry to hear that. When was the transaction made?
Customer: Around 28 minutes ago through UPI.
Agent: I will need to verify your identity. Please confirm your name and last four digits of your mobile number.
Customer: Kiran 4582.
Agent: Thank you. I can see the transaction was successful. We can raise a wrong credit recovery request.
Customer: Will I get my money back? I am really worried.
Agent: As per the RBI guidelines, we will contact the recipient bank and update you within 7 to 10 working days.
Customer: Please register the complaint. I need the money back. Really.
Agent: Of course, the complaint has been registered. You will receive a reference number via SMS.
Customer: Thank you.
""".strip()

SEP = "=" * 70

def print_section(title: str, data):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"  {k:<30}: {v}")
    elif isinstance(data, list):
        for item in data:
            print(f"  • {item}")
    else:
        print(f"  {data}")


use_text = "--text" in sys.argv

print(f"\n{SEP}")
print(f"  ConvI — Full /analyze Endpoint Test")
print(f"  Mode: {'TEXT' if use_text else 'AUDIO'}")
print(f"{SEP}\n")

# ── POST to /analyze ──────────────────────────────────────────────────────────
if use_text:
    print(f"  POSTing text transcript to {BASE_URL}/analyze ...")
    resp = httpx.post(
        f"{BASE_URL}/analyze",
        data={
            "text_transcript": TEXT_TRANSCRIPT,
            "domain":          "financial_banking",
        },
        timeout=TIMEOUT,
    )
else:
    print(f"  POSTing {AUDIO_FILE} to {BASE_URL}/analyze ...")
    print(f"  ⏱  This may take 3-5 minutes (Whisper + Ollama on CPU) ...\n")
    with open(AUDIO_FILE, "rb") as f:
        resp = httpx.post(
            f"{BASE_URL}/analyze",
            files={"audio_file": (AUDIO_FILE, f, "audio/wav")},
            data={"domain": "financial_banking"},
            timeout=TIMEOUT,
        )

print(f"\n{SEP}")
print(f"  HTTP Status : {resp.status_code}")
print(f"{SEP}")

if resp.status_code != 200:
    print(f"\n  ❌ Error response:\n{resp.text}")
    sys.exit(1)

data = resp.json()

# ── Session info ──────────────────────────────────────────────────────────────
print(f"\n  session_id     : {data['session_id']}")
print(f"  input_type     : {data['input_type']}")
print(f"  domain         : {data['domain']}")
print(f"  risk_score     : {data['risk_score']}")
print(f"  escalation     : {data['escalation_level']}")

# ── Conversation timeline ─────────────────────────────────────────────────────
print_section("CONVERSATION TIMELINE", None)
for t in data.get("conversation_timeline", []):
    ts = f"@{t.get('start_time', 0):.1f}s" if t.get("start_time") is not None else ""
    print(f"  [{t['role'].upper():<8}] {ts:<8} {t['speaker_id']}: {t['normalized_text_en'][:80]}")

# ── Basic analysis ────────────────────────────────────────────────────────────
basic = data.get("basic_conversational_analysis", {})
print_section("BASIC CONVERSATIONAL ANALYSIS", {
    "Summary"      : basic.get("conversation_summary", "-"),
    "Intent"       : basic.get("customer_intention", "-"),
    "Topics"       : ", ".join(basic.get("key_topics", [])),
    "Emotion Tone" : basic.get("overall_emotional_tone", "-"),
    "Outcome"      : basic.get("call_outcome", "-"),
    "Language"     : basic.get("language_detected", "-"),
})

# ── RAG analysis ──────────────────────────────────────────────────────────────
rag = data.get("rag_based_analysis", {})
print_section("RAG-BASED ANALYSIS", None)
print(f"  Compliance Flags  : {rag.get('compliance_flags') or ['none']}")
print(f"  Fraud Indicators  : {rag.get('fraud_indicators') or ['none']}")
print(f"  Policy References :")
for ref in rag.get("policy_references", []):
    print(f"    • {ref}")

# ── Agent performance ─────────────────────────────────────────────────────────
agent = data.get("agent_performance_analysis", {})
print_section("AGENT PERFORMANCE", {
    "Score"         : f"{agent.get('performance_score', 0)}/10",
    "De-escalation" : agent.get("de_escalation_detected", False),
    "Tone Shift"    : agent.get("tone_shift_detected", False),
})

# ── Confidence scores ─────────────────────────────────────────────────────────
conf = data.get("confidence_scores", {})
print_section("CONFIDENCE SCORES", {
    "Transcription"    : conf.get("transcription"),
    "Language Detect"  : conf.get("language_detection"),
    "Emotion"          : conf.get("emotion_detection"),
    "LLM Reasoning"    : conf.get("llm_reasoning"),
    "RAG Retrieval"    : conf.get("rag_retrieval"),
})

# ── Emotion timeline ──────────────────────────────────────────────────────────
print_section("EMOTION TIMELINE (first 6 points)", None)
tl = data.get("timeline_analysis", {}).get("emotion_timeline", [])
for pt in tl[:6]:
    ts = pt.get("timestamp")
    print(f"  {pt['speaker_id']:<12} @{ts:.1f}s  emotion={pt['emotion']}  sentiment={pt['sentiment_score']}  risk={pt['risk_score']}")

print(f"\n{SEP}")
print(f"  ✅ Full pipeline complete!")
print(f"{SEP}\n")
