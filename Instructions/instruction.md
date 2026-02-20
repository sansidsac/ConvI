================================================================================
TRANSIGHT HACKATHON — FINANCIAL BANKING MULTIMODAL CONVERSATION INTELLIGENCE BACKEND
MASTER DEVELOPER + AGENTIC AI IMPLEMENTATION PROMPT
===================================================

## PROJECT IDEA (DETAILED OVERVIEW)

We are building a LOCAL-FIRST, BACKEND-ONLY, ENTERPRISE-GRADE multimodal conversation
intelligence system focused on the FINANCIAL BANKING domain.

The system must analyze multilingual customer support conversations coming either as:

1. AUDIO FILE (multi-speaker call recording)
   OR
2. TEXT TRANSCRIPT

The backend processes conversations end-to-end and produces structured enterprise-ready
JSON analytics including:

- Conversation summary
- Customer intention
- Key topics
- Emotional tone
- Call outcome classification
- Compliance / fraud detection (RAG-driven)
- Risk & escalation scoring
- Timeline-based emotion and sentiment evolution
- Agent performance evaluation (de-escalation detection)

The system MUST be:

- API-first (FastAPI)
- Fully local (no external APIs)
- Modular
- Enterprise-scalable
- Cleanly structured for agentic AI coding tools (Copilot, Cursor, Antigravity)

This system aligns with the Transight Hackathon Problem Statement:

Design a backend API system capable of multimodal conversation intelligence
that extracts structured insights and applies configurable business/domain rules.

================================================================================
TECHNOLOGY STACK (FINALIZED)
============================

## CORE BACKEND

- FastAPI
- Pydantic
- Python

## SPEECH PROCESSING

- pyannote.audio → speaker diarization
- faster-whisper → speech recognition + audio language detection
- speechbrain wav2vec2 → emotion detection from audio

## TEXT & NLP

- fastText → language detection
- spaCy → NLP preprocessing & entity tagging

## LOCAL LLM

- Qwen2.5 Instruct (local runtime)

## RAG STACK

- bge-m3 embeddings
- FAISS vector database

## STORAGE

- PostgreSQL → session memory, analytics, audit data

================================================================================
HIGH LEVEL ARCHITECTURE (CONDITIONAL FLOW)
==========================================

## INPUT

[FastAPI API Gateway]
Request:

- audio_file (optional)
- text_transcript (optional)
- domain="financial_banking"

CONDITION:
IF audio_file EXISTS → AUDIO PIPELINE
ELSE IF text_transcript EXISTS → TEXT PIPELINE

================================================================================
INPUT ROUTER (CONDITIONAL MODULE)
=================================

```
                       ┌────────────────────────┐
                       │     FastAPI Router     │
                       └───────────┬────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
                │                                     │
```

============================== AUDIO PATH ====================================

[Speaker Diarization]
Tech: pyannote.audio
Input: audio.wav
Output: speaker_segments

```
    ↓
```

[ASR + Language Detection]
Tech: faster-whisper
Output:

- original_text
- language_audio
- timestamps

  ```
   ↓
  ```

[Audio Emotion Detection]
Tech: speechbrain wav2vec2
Output:

- audio_emotion per segment

  ```
   ↓
   ▼
  ```

============================== TEXT PATH =====================================

[Text Input Handler]
Tech: FastAPI/Pydantic

```
    ↓
```

[Language Detection]
Tech: fastText
Output:

- language_text

  ```
   ↓
  ```

[NLP Preprocessing]
Tech: spaCy
Output:

- entities
- cleaned_text

  ```
   ↓
   ▼
  ```

====================== MERGE POINT — SHARED PIPELINE =========================

[Conversation Normalizer + Role Classifier]
Tech:

- Python Service
- Local LLM (Qwen2.5)

Creates Unified Conversation Timeline Object:

{
speaker_id,
role: agent | customer,
original_text,
normalized_text_en,
language,
emotion,
start_time,
end_time
}

================================================================================
FINANCIAL BANKING DOMAIN INTELLIGENCE (RAG)
===========================================

[RAG Retrieval Engine]
Tech:

- Embeddings: bge-m3
- Vector DB: FAISS

Domain Knowledge Sources:

- Banking privacy policies
- Fraud & refund rules
- Compliance guidelines
- Finance SOP documents
- FAQ datasets

Output:

- rag_context_chunks
- policy_references

================================================================================
LOCAL LLM REASONING ENGINE
==========================

Tech: Qwen2.5 Instruct

Generates:

- conversation summary
- customer intention
- key topics
- overall emotional tone
- call outcome classification
- compliance & fraud reasoning

================================================================================
ANALYTICS ENGINE
================

## Risk & Escalation Analyzer

Outputs:

- risk_score
- escalation_level

## Timeline Analyzer

Outputs:

- emotion_timeline
- sentiment_timeline
- risk_timeline

## Agent Performance Analyzer

Logic:

- detects emotional improvement
- evaluates tone shift
  Outputs:
- performance_score
- interaction_metrics

================================================================================
MEMORY & STORAGE
================

PostgreSQL:

- sessions
- conversation turns
- analytics results
- audit logs

FAISS Vector Store:

- domain document embeddings

================================================================================
FINAL OUTPUT STRUCTURE
======================

Structured JSON Response MUST include:

{
basic_conversational_analysis,
rag_based_analysis,
timeline_analysis,
agent_performance_analysis,
confidence_scores
}

================================================================================
MODULE INPUT / OUTPUT CONTRACTS (FOR AGENTIC AI CODING)
=======================================================

1. FastAPI Gateway
   Input:
   {
   "audio_file": optional,
   "text_transcript": optional,
   "domain": "financial_banking"
   }
   Output:
   {
   "input_type": "audio | text",
   "payload": ...
   }

2. Speaker Diarization (Audio Only)
   Input:
   { "audio_path": "file.wav" }
   Output:
   { "segments": [ {speaker_id, start_time, end_time} ] }

3. ASR + Language Detection
   Input:
   { "audio_path": "...", "segments": [...] }
   Output:
   { "text", "language_audio", "timestamps" }

4. Emotion Detection
   Input:
   { "audio_path": "...", "segments": [...] }
   Output:
   { "audio_emotion", "confidence" }

5. Text NLP Pipeline
   Input:
   { "text": "..." }
   Output:
   { "language_text", "entities", "cleaned_text" }

6. Conversation Normalizer
   Input:
   { transcriptions + emotions + roles }
   Output:
   Unified Timeline Object

7. RAG Retrieval
   Input:
   { normalized_text_en, domain }
   Output:
   { rag_context_chunks, policy_references }

8. Local LLM Engine
   Input:
   { conversation, rag_context }
   Output:
   { summary, intent, topics, sentiment, outcome, compliance_flags }

9. Analytics Engine
   Input:
   { llm_output, timeline_data }
   Output:
   { risk_score, timelines, agent_performance }

10. Storage Layer
    Input:
    { session_id, analytics }
    Output:
    { status: stored }

================================================================================
END-TO-END FLOW SUMMARY
=======================

INPUT (Audio OR Text)
│
├── IF AUDIO → Diarization → ASR → Emotion Detection
│
└── IF TEXT → NLP + Language Detection
│
▼
Conversation Normalizer (Unified Object)
│
▼
Banking RAG Knowledge Engine
│
▼
Local LLM Reasoning Engine
│
▼
Analytics Engine
│
▼
PostgreSQL + JSON Response

================================================================================
DEVELOPER EXECUTION INSTRUCTIONS
================================

Each developer may independently own modules:

DEV 1 → FastAPI + Input Router + Output Formatter
DEV 2 → Speech Pipeline (pyannote + whisper + speechbrain)
DEV 3 → RAG Engine + Local LLM Integration
DEV 4 → Analytics Engine + Memory DB

All modules must respect IO contracts above.
System must remain fully local and modular.

================================================================================
END MASTER PROMPT
=================
