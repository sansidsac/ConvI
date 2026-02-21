"""
ConvI — LLM Engine
===================
Local LLM reasoning via Ollama (Qwen2.5:7b at http://localhost:11434).

Flow:
  1. Build a structured prompt from the conversation timeline + RAG context
  2. POST to Ollama /api/chat
  3. Parse JSON response into typed output objects
  4. Compute analytics (risk scoring, timeline, agent performance)

Ollama endpoint: http://localhost:11434
Model          : qwen2.5:7b
"""

from __future__ import annotations

import json
import re
import httpx
from typing import List, Optional
from loguru import logger

from app.schemas import (
    ConversationTurn,
    BasicConversationalAnalysis,
    RAGBasedAnalysis,
    TimelineAnalysis,
    TimelinePoint,
    AgentPerformanceAnalysis,
    ConfidenceScores,
    EscalationLevel,
    Role,
)

# ── Ollama config ─────────────────────────────────────────────────────────────
import os as _os
OLLAMA_URL   = _os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = _os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
TIMEOUT      = 180.0  # seconds — larger prompts need more time
OLLAMA_CTX   = 8192   # context window tokens (default 2048 is too small for RAG prompts)

# ── Sentiment keywords ────────────────────────────────────────────────────────
_NEGATIVE_WORDS = {
    "fraud", "scam", "stolen", "unauthorized", "block", "complaint",
    "angry", "furious", "upset", "wrong", "error", "problem", "issue",
    "cancel", "refund", "dispute", "lost", "missing", "hack", "breach"
}
_POSITIVE_WORDS = {
    "thank", "resolved", "great", "excellent", "satisfied", "perfect",
    "appreciated", "sorted", "happy", "helped", "good", "wonderful"
}


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_analysis_prompt(
    dialogue: str,
    rag_chunks: List[str],
    domain: str,
) -> str:
    rag_section = "\n\n".join(rag_chunks[:5]) if rag_chunks else "No policy context retrieved."

    return f"""You are an expert financial banking conversation analyst.

DOMAIN: {domain}

RELEVANT POLICY CONTEXT (from internal knowledge base):
---
{rag_section}
---

CONVERSATION TRANSCRIPT:
---
{dialogue}
---

Analyze the above banking support conversation and respond with a VALID JSON object (no markdown, no extra text) in EXACTLY this structure:

{{
  "conversation_summary": "<2-3 sentence summary>",
  "customer_intention": "<primary reason customer called>",
  "key_topics": ["<topic1>", "<topic2>", "<topic3>"],
  "overall_emotional_tone": "<positive|neutral|negative|mixed>",
  "call_outcome": "<resolved|escalated|pending|unresolved>",
  "language_detected": "<en|ml|hi|other>",
  "compliance_flags": ["<flag1 if any>"],
  "fraud_indicators": ["<indicator1 if any>"],
  "policy_violations": ["<violation1 if any>"]
}}

Rules:
- compliance_flags: list issues that may violate banking regulations
- fraud_indicators: list any signals of fraud or suspicious behavior
- If nothing detected, use empty lists []
- Respond ONLY with the JSON object, nothing else."""


def _build_role_classification_prompt(dialogue: str) -> str:
    return f"""You are analyzing a banking support call.

CONVERSATION:
---
{dialogue}
---

For each speaker turn, determine if they are the "agent" (bank support staff) or "customer" (banking client).
Respond with ONLY a JSON array:
[{{"speaker_id": "SPEAKER_00", "role": "agent"}}, {{"speaker_id": "SPEAKER_01", "role": "customer"}}]"""


# ── Ollama caller ─────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, temperature: float = 0.1) -> str:
    """
    Call Ollama /api/chat with the given prompt.
    Returns the assistant's reply as a raw string.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
            "num_ctx":     OLLAMA_CTX,   # raise context window beyond default 2048
        },
    }
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. "
            "Is `ollama serve` running?"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")


def _parse_json_response(raw: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    # Find first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


# ── Sentiment scoring helper ──────────────────────────────────────────────────

def _score_sentiment(text: str) -> float:
    """Simple lexicon-based sentiment score [-1.0, 1.0]."""
    words = set(text.lower().split())
    neg = len(words & _NEGATIVE_WORDS)
    pos = len(words & _POSITIVE_WORDS)
    total = neg + pos
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def _score_risk(text: str, emotion: Optional[str]) -> float:
    """Heuristic risk score [0,10]."""
    score = 0.0
    words = set(text.lower().split())
    risk_hits = len(words & _NEGATIVE_WORDS)
    score += min(risk_hits * 1.5, 6.0)
    if emotion in {"angry", "fear", "disgust"}:
        score += 2.0
    return round(min(score, 10.0), 2)


# ── Timeline builder ──────────────────────────────────────────────────────────

def _build_timelines(turns: List[ConversationTurn]) -> TimelineAnalysis:
    emotion_tl:    List[TimelinePoint] = []
    sentiment_tl:  List[TimelinePoint] = []
    risk_tl:       List[TimelinePoint] = []

    for t in turns:
        sent  = _score_sentiment(t.normalized_text_en)
        risk  = _score_risk(t.normalized_text_en, t.emotion)
        pt = TimelinePoint(
            speaker_id      = t.speaker_id,
            timestamp       = t.start_time,
            emotion         = t.emotion,
            sentiment_score = sent,
            risk_score      = risk,
        )
        emotion_tl.append(pt)
        sentiment_tl.append(TimelinePoint(
            speaker_id=t.speaker_id, timestamp=t.start_time,
            emotion=None, sentiment_score=sent, risk_score=None,
        ))
        risk_tl.append(TimelinePoint(
            speaker_id=t.speaker_id, timestamp=t.start_time,
            emotion=None, sentiment_score=None, risk_score=risk,
        ))

    return TimelineAnalysis(
        emotion_timeline=emotion_tl,
        sentiment_timeline=sentiment_tl,
        risk_timeline=risk_tl,
    )


# ── Agent performance ─────────────────────────────────────────────────────────

def _score_agent(turns: List[ConversationTurn]) -> AgentPerformanceAnalysis:
    agent_turns    = [t for t in turns if t.role == Role.agent]
    customer_turns = [t for t in turns if t.role == Role.customer]

    if not agent_turns:
        return AgentPerformanceAnalysis(
            performance_score=5.0,
            de_escalation_detected=False,
            tone_shift_detected=False,
            interaction_metrics={},
        )

    # Check emotional trajectory of customer turns
    customer_emotions = [t.emotion for t in customer_turns if t.emotion]
    negative_emotions = {"angry", "fear", "disgust", "sad"}

    de_escalation = False
    if len(customer_emotions) >= 2:
        early_neg = customer_emotions[0] in negative_emotions
        late_neg  = customer_emotions[-1] in negative_emotions
        de_escalation = early_neg and not late_neg

    # Sentiment trajectory of agent turns
    agent_sentiments = [_score_sentiment(t.normalized_text_en) for t in agent_turns]
    tone_shift = (
        len(agent_sentiments) >= 2
        and agent_sentiments[-1] > agent_sentiments[0]
    )

    # Performance score heuristic
    perf = 5.0
    if de_escalation:
        perf += 2.5
    if tone_shift:
        perf += 1.5
    avg_agent_sentiment = sum(agent_sentiments) / len(agent_sentiments) if agent_sentiments else 0
    perf += avg_agent_sentiment * 1.0
    perf = round(min(max(perf, 0.0), 10.0), 2)

    return AgentPerformanceAnalysis(
        performance_score=perf,
        de_escalation_detected=de_escalation,
        tone_shift_detected=tone_shift,
        interaction_metrics={
            "total_turns":         len(turns),
            "agent_turns":         len(agent_turns),
            "customer_turns":      len(customer_turns),
            "avg_agent_sentiment": round(avg_agent_sentiment, 3),
            "customer_emotions":   customer_emotions,
        },
    )


# ── Risk & escalation ─────────────────────────────────────────────────────────

def _compute_overall_risk(
    turns: List[ConversationTurn],
    compliance_flags: List[str],
    fraud_indicators: List[str],
) -> tuple[float, EscalationLevel]:
    base_risk = sum(_score_risk(t.normalized_text_en, t.emotion) for t in turns)
    avg_risk  = base_risk / len(turns) if turns else 0.0
    avg_risk  += len(compliance_flags) * 0.5
    avg_risk  += len(fraud_indicators) * 1.5
    avg_risk   = round(min(avg_risk, 10.0), 2)

    if avg_risk >= 7.5:
        level = EscalationLevel.critical
    elif avg_risk >= 5.0:
        level = EscalationLevel.high
    elif avg_risk >= 2.5:
        level = EscalationLevel.medium
    else:
        level = EscalationLevel.low

    return avg_risk, level


# ── Main entry point ──────────────────────────────────────────────────────────

def run_llm_analysis(
    turns: List[ConversationTurn],
    rag_result: dict,
    domain: str = "financial_banking",
    dialogue_str: Optional[str] = None,
) -> dict:
    """
    Full LLM + analytics pipeline.

    Parameters
    ----------
    turns        : List[ConversationTurn] from the normalizer
    rag_result   : dict from RAGRetriever.retrieve()
    domain       : business domain string
    dialogue_str : optional pre-rendered dialogue (avoids re-rendering)

    Returns
    -------
    dict with keys:
        basic_conversational_analysis : BasicConversationalAnalysis
        rag_based_analysis            : RAGBasedAnalysis
        timeline_analysis             : TimelineAnalysis
        agent_performance_analysis    : AgentPerformanceAnalysis
        confidence_scores             : ConfidenceScores
        risk_score                    : float
        escalation_level              : EscalationLevel
    """
    from app.conversation_normalizer import turns_to_dialogue_string

    if not dialogue_str:
        dialogue_str = turns_to_dialogue_string(turns)

    rag_chunks     = rag_result.get("rag_context_chunks", [])
    policy_refs    = rag_result.get("policy_references", [])

    # ── 1. LLM call ───────────────────────────────────────────────────────
    logger.info(f"[LLM] Calling Ollama ({OLLAMA_MODEL}) for analysis...")
    prompt = _build_analysis_prompt(dialogue_str, rag_chunks, domain)

    try:
        raw = _call_ollama(prompt)
        llm_json = _parse_json_response(raw)
        llm_ok = True
        logger.info("[LLM] Response parsed successfully.")
    except Exception as e:
        logger.error(f"[LLM] Failed: {e} — using fallback defaults.")
        llm_json = {}
        llm_ok = False

    # ── 2. Build BasicConversationalAnalysis ──────────────────────────────
    basic = BasicConversationalAnalysis(
        conversation_summary  = llm_json.get("conversation_summary", "Analysis unavailable."),
        customer_intention    = llm_json.get("customer_intention", "Unknown"),
        key_topics            = llm_json.get("key_topics", []),
        overall_emotional_tone= llm_json.get("overall_emotional_tone", "neutral"),
        call_outcome          = llm_json.get("call_outcome", "pending"),
        language_detected     = llm_json.get("language_detected", "en"),
    )

    # ── 3. Build RAGBasedAnalysis ─────────────────────────────────────────
    policy_references_str = [
        f"{r['source']} (p{r['page']}) [{r['doc_type']}] score={r['score']}"
        for r in policy_refs
    ]
    rag_analysis = RAGBasedAnalysis(
        compliance_flags  = llm_json.get("compliance_flags", []),
        fraud_indicators  = llm_json.get("fraud_indicators", []),
        policy_references = policy_references_str,
        rag_context_chunks= rag_chunks[:3],   # top 3 chunks in response
    )

    # ── 4. Timeline analysis ──────────────────────────────────────────────
    timeline = _build_timelines(turns)

    # ── 5. Agent performance ──────────────────────────────────────────────
    agent_perf = _score_agent(turns)

    # ── 6. Risk & escalation ──────────────────────────────────────────────
    risk_score, escalation = _compute_overall_risk(
        turns,
        rag_analysis.compliance_flags,
        rag_analysis.fraud_indicators,
    )

    # ── 7. Confidence scores ──────────────────────────────────────────────
    avg_transcription_conf = None
    if turns and turns[0].start_time is not None:
        # audio path — transcription confidence from speech pipeline
        confs = [
            getattr(t, "transcription_confidence", None) for t in turns
            if getattr(t, "transcription_confidence", None) is not None
        ]
        avg_transcription_conf = round(sum(confs) / len(confs), 3) if confs else None

    confidence = ConfidenceScores(
        transcription    = avg_transcription_conf,
        language_detection= 0.95,
        emotion_detection = 0.80 if any(t.emotion for t in turns) else None,
        llm_reasoning    = 0.90 if llm_ok else 0.0,
        rag_retrieval    = 0.85 if rag_chunks else 0.0,
    )

    logger.info(
        f"[LLM] Done | risk={risk_score} | escalation={escalation.value} | "
        f"outcome={basic.call_outcome}"
    )

    return {
        "basic_conversational_analysis": basic,
        "rag_based_analysis":            rag_analysis,
        "timeline_analysis":             timeline,
        "agent_performance_analysis":    agent_perf,
        "confidence_scores":             confidence,
        "risk_score":                    risk_score,
        "escalation_level":              escalation,
    }
