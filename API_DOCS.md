# ConvI API Documentation

> **Base URL:** `http://localhost:8000`  
> **OpenAPI UI:** http://localhost:8000/docs  
> **ReDoc:** http://localhost:8000/redoc  
> **Content-Type:** All `POST` endpoints accept `multipart/form-data`

---

## Table of Contents

1. [Health Endpoints](#1-health-endpoints)  
2. [POST /api/v1/analyze — Full Pipeline](#2-post-apiv1analyze--full-pipeline)  
3. [POST /api/v1/analyze/audio — Speech Pipeline](#3-post-apiv1analyzeaudio--speech-pipeline)  
4. [POST /api/v1/analyze/text — Text Pipeline](#4-post-apiv1analyzetext--text-pipeline)  
5. [Response Schemas Reference](#5-response-schemas-reference)  
6. [Error Responses](#6-error-responses)  

---

## 1. Health Endpoints

### `GET /`

Returns service identity and version.

**Request**
```http
GET / HTTP/1.1
Host: localhost:8000
```

**Response `200 OK`**
```json
{
  "service": "ConvI — Conversation Intelligence API",
  "version": "0.1.0",
  "status": "ok",
  "domain": "financial_banking"
}
```

---

### `GET /health`

Liveness probe — confirms the server is running.

**Request**
```http
GET /health HTTP/1.1
Host: localhost:8000
```

**Response `200 OK`**
```json
{
  "status": "healthy"
}
```

**curl**
```bash
curl http://localhost:8000/health
```

---

## 2. `POST /api/v1/analyze` — Full Pipeline

The **primary endpoint**. Accepts either an audio file or a text transcript and runs the complete intelligence pipeline:

> Speech/Text Processing → Conversation Normalization → RAG Retrieval → LLM Analysis → PostgreSQL Storage

Returns the full `ConversationAnalyticsResponse`.

### Request Fields (`multipart/form-data`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_file` | File (WAV/MP3) | One of these two | Audio recording — WAV 16-bit PCM recommended |
| `text_transcript` | string | is required | Raw text transcript |
| `domain` | string | No | Business domain context (default: `financial_banking`) |
| `session_id` | string | No | Custom session ID for linking multiple requests |

> **Note:** Submit exactly one of `audio_file` or `text_transcript`. Submitting both causes `audio_file` to take precedence.

---

### Sample Request A — Audio File

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "audio_file=@/path/to/call_recording.wav" \
  -F "domain=financial_banking" \
  -F "session_id=sess_20260221_001"
```

---

### Sample Request B — Text Transcript

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "text_transcript=Agent: Thank you for calling ConvI Bank. How can I help you today?
Customer: I want to check why my KYC update was rejected. I submitted all documents.
Agent: I understand your frustration. Let me pull up your account details.
Customer: This is the third time I am calling! Nobody is fixing this issue!
Agent: I sincerely apologize for the inconvenience. I am escalating this to our KYC team right now." \
  -F "domain=financial_banking"
```

---

### Response `200 OK` — `ConversationAnalyticsResponse`

```json
{
  "session_id": "sess_20260221_001",
  "input_type": "text",
  "domain": "financial_banking",

  "conversation_timeline": [
    {
      "speaker_id": "SPEAKER_00",
      "role": "agent",
      "original_text": "Thank you for calling ConvI Bank. How can I help you today?",
      "normalized_text_en": "Thank you for calling ConvI Bank. How can I help you today?",
      "language": "en",
      "emotion": "neutral",
      "start_time": null,
      "end_time": null
    },
    {
      "speaker_id": "SPEAKER_01",
      "role": "customer",
      "original_text": "I want to check why my KYC update was rejected. I submitted all documents.",
      "normalized_text_en": "I want to check why my KYC update was rejected. I submitted all documents.",
      "language": "en",
      "emotion": "neutral",
      "start_time": null,
      "end_time": null
    },
    {
      "speaker_id": "SPEAKER_01",
      "role": "customer",
      "original_text": "This is the third time I am calling! Nobody is fixing this issue!",
      "normalized_text_en": "This is the third time I am calling! Nobody is fixing this issue!",
      "language": "en",
      "emotion": "angry",
      "start_time": null,
      "end_time": null
    }
  ],

  "basic_conversational_analysis": {
    "conversation_summary": "Customer contacted bank for the third time regarding a rejected KYC document update. Agent acknowledged the issue and escalated to the KYC team.",
    "customer_intention": "Resolve KYC document rejection and complete account update",
    "key_topics": ["KYC update", "document rejection", "account access", "escalation"],
    "overall_emotional_tone": "frustrated",
    "call_outcome": "Escalated to KYC team for resolution",
    "language_detected": "en"
  },

  "rag_based_analysis": {
    "compliance_flags": [
      "KYC document rejection without written communication may violate RBI KYC Direction Section 16",
      "Customer not informed of rejection reason — potential breach of grievance redressal norms"
    ],
    "fraud_indicators": [],
    "policy_references": [
      "Master Direction - Know Your Customer (KYC) Direction, 2016 — Section 16: Customer Due Diligence",
      "RBI Corporate Communication Policy 2025 — Section 4: Customer Grievance Communication"
    ],
    "rag_context_chunks": [
      "Banks shall communicate the reasons for rejection of KYC documents to the customer in writing within 7 working days...",
      "Customers have the right to know the specific deficiencies in their submitted documents..."
    ]
  },

  "timeline_analysis": {
    "emotion_timeline": [
      { "speaker_id": "SPEAKER_01", "timestamp": null, "emotion": "neutral", "sentiment_score": 0.1, "risk_score": 1.5 },
      { "speaker_id": "SPEAKER_01", "timestamp": null, "emotion": "angry",   "sentiment_score": -0.7, "risk_score": 6.2 }
    ],
    "sentiment_timeline": [
      { "speaker_id": "SPEAKER_01", "timestamp": null, "emotion": null, "sentiment_score": 0.1,  "risk_score": null },
      { "speaker_id": "SPEAKER_01", "timestamp": null, "emotion": null, "sentiment_score": -0.7, "risk_score": null }
    ],
    "risk_timeline": [
      { "speaker_id": "SPEAKER_01", "timestamp": null, "emotion": null, "sentiment_score": null, "risk_score": 1.5 },
      { "speaker_id": "SPEAKER_01", "timestamp": null, "emotion": null, "sentiment_score": null, "risk_score": 6.2 }
    ]
  },

  "agent_performance_analysis": {
    "performance_score": 7.2,
    "de_escalation_detected": true,
    "tone_shift_detected": false,
    "interaction_metrics": {
      "avg_response_time_s": null,
      "empathy_statements": 1,
      "apology_count": 1,
      "escalation_offered": true,
      "resolution_offered": true
    }
  },

  "confidence_scores": {
    "transcription": null,
    "language_detection": 0.99,
    "emotion_detection": 0.72,
    "llm_reasoning": 0.85,
    "rag_retrieval": 0.78
  },

  "risk_score": 5.8,
  "escalation_level": "medium"
}
```

---

### Sample Request C — Audio File with skip emotion flag (Speech-only endpoint)

See [Section 3](#3-post-apiv1analyzeaudio--speech-pipeline) for the lighter audio endpoint.

---

## 3. `POST /api/v1/analyze/audio` — Speech Pipeline

Runs **only** the speech pipeline (diarization + ASR + emotion). Does not invoke RAG or LLM. Returns a segmented transcript.

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_file` | File (WAV/MP3) | Yes | Audio recording |
| `domain` | string | No | Default: `financial_banking` |
| `session_id` | string | No | Custom session ID |
| `skip_emotion` | bool | No | `true` to skip emotion detection (faster) |

### Sample Request

```bash
curl -X POST http://localhost:8000/api/v1/analyze/audio \
  -F "audio_file=@/path/to/call.wav" \
  -F "skip_emotion=false"
```

### Response `200 OK` — `PipelineResponse`

```json
{
  "session_id": "7f3a2b1c-4d5e-6789-abcd-ef0123456789",
  "input_type": "audio",
  "domain": "financial_banking",
  "audio_language": "en",
  "total_segments": 6,
  "unique_speakers": 2,
  "segments": [
    {
      "speaker_id": "SPEAKER_00",
      "start_time": 0.51,
      "end_time": 4.82,
      "original_text": "Thank you for calling ConvI Bank. How can I help you?",
      "language": "en",
      "emotion": "neutral",
      "emotion_confidence": 0.81,
      "transcription_confidence": 0.95
    },
    {
      "speaker_id": "SPEAKER_01",
      "start_time": 5.20,
      "end_time": 12.64,
      "original_text": "I want to report an unauthorized transaction on my account.",
      "language": "en",
      "emotion": "neutral",
      "emotion_confidence": 0.74,
      "transcription_confidence": 0.92
    },
    {
      "speaker_id": "SPEAKER_01",
      "start_time": 22.10,
      "end_time": 30.45,
      "original_text": "Why is this taking so long? This is completely unacceptable!",
      "language": "en",
      "emotion": "angry",
      "emotion_confidence": 0.88,
      "transcription_confidence": 0.89
    }
  ]
}
```

> **Emotion override note:** If the `emotion` field is `angry` and the audio energy was unusually high, the label may have been set by the RMS energy override layer rather than the wav2vec2 model alone.

---

## 4. `POST /api/v1/analyze/text` — Text Pipeline

Parses a raw text transcript into structured turns. Does not invoke RAG or LLM.

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text_transcript` | string | Yes | Raw conversation text. Use `Agent:` / `Customer:` speaker labels for best results |
| `domain` | string | No | Default: `financial_banking` |
| `session_id` | string | No | Custom session ID |

### Supported Transcript Formats

**Labelled format (recommended):**
```
Agent: Good morning, how can I assist you?
Customer: I need to update my registered mobile number.
Agent: I can help with that. Please provide your account number.
```

**Plain format (fallback — all turns assigned unknown speaker):**
```
Good morning, how can I assist you?
I need to update my registered mobile number.
```

### Sample Request

```bash
curl -X POST http://localhost:8000/api/v1/analyze/text \
  -F "text_transcript=Agent: Good morning, how can I assist you?
Customer: I need to update my registered mobile number. The OTP is not coming.
Agent: I understand. Let me verify your account first.
Customer: I have been waiting for 20 minutes. Please hurry.
Agent: I apologize for the wait. Your number has been updated successfully."
```

### Response `200 OK` — `PipelineResponse`

```json
{
  "session_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
  "input_type": "text",
  "domain": "financial_banking",
  "audio_language": null,
  "total_segments": 5,
  "unique_speakers": 2,
  "segments": [
    {
      "speaker_id": "SPEAKER_00",
      "start_time": 0.0,
      "end_time": 1.0,
      "original_text": "Good morning, how can I assist you?",
      "language": "en",
      "emotion": null,
      "emotion_confidence": 0.0,
      "transcription_confidence": 1.0
    },
    {
      "speaker_id": "SPEAKER_01",
      "start_time": 1.0,
      "end_time": 2.0,
      "original_text": "I need to update my registered mobile number. The OTP is not coming.",
      "language": "en",
      "emotion": null,
      "emotion_confidence": 0.0,
      "transcription_confidence": 1.0
    },
    {
      "speaker_id": "SPEAKER_01",
      "start_time": 3.0,
      "end_time": 4.0,
      "original_text": "I have been waiting for 20 minutes. Please hurry.",
      "language": "en",
      "emotion": null,
      "emotion_confidence": 0.0,
      "transcription_confidence": 1.0
    }
  ]
}
```

> **Note:** Text-path segments have no audio-based emotion detection. Emotion fields will be `null`/`0.0`. Use `/api/v1/analyze` for full emotion + LLM analysis from text.

---

## 5. Response Schemas Reference

### `ConversationAnalyticsResponse`

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | UUID session identifier |
| `input_type` | `"audio"` \| `"text"` | Input modality |
| `domain` | string | Business domain used for RAG context |
| `conversation_timeline` | `ConversationTurn[]` | Ordered list of conversation turns |
| `basic_conversational_analysis` | `BasicConversationalAnalysis` | LLM-generated summary block |
| `rag_based_analysis` | `RAGBasedAnalysis` | Compliance, fraud, and policy retrieval |
| `timeline_analysis` | `TimelineAnalysis` | Per-turn emotion / sentiment / risk timelines |
| `agent_performance_analysis` | `AgentPerformanceAnalysis` | Agent scoring and interaction metrics |
| `confidence_scores` | `ConfidenceScores` | Per-stage pipeline confidence values |
| `risk_score` | float (0–10) | Overall call risk score |
| `escalation_level` | `"low"` \| `"medium"` \| `"high"` \| `"critical"` | Escalation classification |

---

### `ConversationTurn`

| Field | Type | Description |
|-------|------|-------------|
| `speaker_id` | string | e.g. `"SPEAKER_00"`, `"SPEAKER_01"` |
| `role` | `"agent"` \| `"customer"` \| `"unknown"` | Identified speaker role |
| `original_text` | string | Transcribed or parsed text in source language |
| `normalized_text_en` | string | English text (translated if source was non-English) |
| `language` | string | ISO-639-1 language code e.g. `"en"`, `"ml"` |
| `emotion` | string \| null | `"neutral"`, `"happy"`, `"angry"`, `"sad"` |
| `start_time` | float \| null | Segment start time in seconds (audio only) |
| `end_time` | float \| null | Segment end time in seconds (audio only) |

---

### `BasicConversationalAnalysis`

| Field | Type | Description |
|-------|------|-------------|
| `conversation_summary` | string | LLM-generated 2–4 sentence summary |
| `customer_intention` | string | Primary customer goal or reason for contact |
| `key_topics` | string[] | List of main topics discussed |
| `overall_emotional_tone` | string | Dominant emotional tone across the conversation |
| `call_outcome` | string | Resolution status: resolved / escalated / unresolved |
| `language_detected` | string | Primary language of the conversation |

---

### `RAGBasedAnalysis`

| Field | Type | Description |
|-------|------|-------------|
| `compliance_flags` | string[] | Potential regulatory / policy violations identified |
| `fraud_indicators` | string[] | Phrases or patterns suggesting fraudulent activity |
| `policy_references` | string[] | Source document titles from the FAISS index |
| `rag_context_chunks` | string[] | Raw retrieved policy text chunks |

---

### `TimelineAnalysis`

Contains three parallel timelines, each as a list of `TimelinePoint`:

| Field | Description |
|-------|-------------|
| `emotion_timeline` | Per-turn emotion label and score |
| `sentiment_timeline` | Per-turn sentiment score (−1 negative → +1 positive) |
| `risk_timeline` | Per-turn risk score (0–10) |

**`TimelinePoint`**

| Field | Type | Description |
|-------|------|-------------|
| `speaker_id` | string | Speaker identifier |
| `timestamp` | float \| null | Turn start time in seconds |
| `emotion` | string \| null | Emotion label |
| `sentiment_score` | float \| null | Sentiment polarity |
| `risk_score` | float \| null | Turn-level risk |

---

### `AgentPerformanceAnalysis`

| Field | Type | Description |
|-------|------|-------------|
| `performance_score` | float (0–10) | Overall agent performance rating |
| `de_escalation_detected` | bool | Whether agent successfully reduced customer tension |
| `tone_shift_detected` | bool | Whether a significant agent tone change was detected |
| `interaction_metrics` | dict | Flexible key/value metrics (empathy statements, apologies, etc.) |

---

### `ConfidenceScores`

| Field | Type | Description |
|-------|------|-------------|
| `transcription` | float \| null | Average Whisper word-level confidence |
| `language_detection` | float \| null | Language ID confidence |
| `emotion_detection` | float \| null | Emotion classification confidence |
| `llm_reasoning` | float \| null | LLM output quality estimate |
| `rag_retrieval` | float \| null | Top-K RAG retrieval relevance score |

---

### `PipelineResponse` (interim — audio and text endpoints)

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | UUID session identifier |
| `input_type` | `"audio"` \| `"text"` | Input modality |
| `domain` | string | Business domain |
| `segments` | `PipelineSegment[]` | Ordered speaker turn segments |
| `audio_language` | string \| null | Detected audio language (`"en"`, `"ml"`) |
| `total_segments` | int | Total number of turns |
| `unique_speakers` | int | Number of distinct speakers detected |

---

### `PipelineSegment`

| Field | Type | Description |
|-------|------|-------------|
| `speaker_id` | string | Speaker label |
| `start_time` | float | Turn start (seconds) |
| `end_time` | float | Turn end (seconds) |
| `original_text` | string | Transcribed or parsed text |
| `language` | string | ISO-639-1 language code |
| `emotion` | string \| null | Detected emotion label |
| `emotion_confidence` | float | Confidence for emotion label (0–1) |
| `transcription_confidence` | float | ASR word probability (0–1) |

---

## 6. Error Responses

All errors follow the standard FastAPI JSON error format:

```json
{
  "detail": "<human-readable error message>"
}
```

| HTTP Status | Cause | Example Detail |
|-------------|-------|----------------|
| `400 Bad Request` | Audio file not found or unreadable | `"[EmotionDetector] File not found: /tmp/upload.wav"` |
| `422 Unprocessable Entity` | Neither `audio_file` nor `text_transcript` provided | `"Provide either 'audio_file' or 'text_transcript'."` |
| `422 Unprocessable Entity` | No speech detected in audio | `"No speech detected in audio. Ensure the file contains clear speech."` |
| `422 Unprocessable Entity` | No turns parsed from transcript | `"No turns parsed from transcript. Use 'Agent: ...' and 'Customer: ...' format."` |
| `500 Internal Server Error` | Speech pipeline crash | `"Pipeline error: Diarization failed — check pyannote token"` |
| `503 Service Unavailable` | Ollama LLM not reachable | `"LLM error: Cannot connect to Ollama at http://localhost:11434"` |

---

## Sample curl Commands — Quick Reference

```bash
# Health check
curl http://localhost:8000/health

# Full analysis — text transcript
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "text_transcript=Agent: Hello, how can I help?
Customer: My transaction of 50000 rupees was unauthorized. Reverse it now!" \
  -F "domain=financial_banking"

# Full analysis — audio file
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "audio_file=@call.wav" \
  -F "session_id=demo-001"

# Speech pipeline only (no LLM/RAG)
curl -X POST http://localhost:8000/api/v1/analyze/audio \
  -F "audio_file=@call.wav" \
  -F "skip_emotion=false"

# Text pipeline only (no LLM/RAG/emotion)
curl -X POST http://localhost:8000/api/v1/analyze/text \
  -F "text_transcript=Agent: Good morning.
Customer: I need a loan statement."
```

---

## Python Client Example

```python
import httplib2
import requests

# Full analysis via text
resp = requests.post(
    "http://localhost:8000/api/v1/analyze",
    data={
        "text_transcript": (
            "Agent: Good morning, ConvI Bank. How may I help you?\n"
            "Customer: I want to know why my fixed deposit was auto-renewed "
            "without my consent.\n"
            "Agent: I apologize for the inconvenience. Let me check your account.\n"
            "Customer: This is fraud! I never agreed to auto-renewal!"
        ),
        "domain": "financial_banking",
        "session_id": "py-client-001",
    },
)

data = resp.json()
print(f"Risk score: {data['risk_score']}")
print(f"Escalation: {data['escalation_level']}")
print(f"Summary: {data['basic_conversational_analysis']['conversation_summary']}")
print(f"Compliance flags: {data['rag_based_analysis']['compliance_flags']}")
```
