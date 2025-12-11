# SER Service Full Infrastructure Migration Plan

## Overview

This plan maps out the complete infrastructure structure for migrating the SER (Speech Emotion Recognition) service from `Well-Bot_SER/` to `Well-Bot_cloud/ser/`. The structure follows existing patterns (`fusion/`, `intervention/`) with a flat directory layout and shared utilities in `utils/`.

## Target Directory Structure

```
Well-Bot_cloud/
├── ser/                                    # SER service (flat structure)
│   ├── __init__.py                         # Module exports
│   ├── config.py                           # SER-specific configuration
│   ├── models.py                           # SER-specific Pydantic models
│   ├── api.py                              # FastAPI routes + CRUD operations
│   │
│   ├── audio_preprocessing.py              # Audio preprocessing utilities
│   ├── emotion_recognition.py              # Emotion recognition ML model
│   ├── transcription.py                    # ASR/transcription ML model
│   ├── sentiment_analysis.py               # Sentiment analysis ML model
│   ├── processing_pipeline.py              # Orchestrates all ML services
│   ├── queue_manager.py                    # Queue management & background worker
│   ├── session_manager.py                  # Session management & in-memory storage
│   └── aggregator.py                       # Aggregation & logging
│
├── utils/
│   └── database.py                         # Add SER database functions here
│
└── main.py                                 # Include SER router
```

## Component Mapping & Purpose

### 1. Configuration (`ser/config.py`)

**Source**: `Well-Bot_SER/app/core/config.py`

**Purpose**: SER-specific configuration settings

- `AGGREGATION_WINDOW_SECONDS` (default: 300)
- `SESSION_GAP_THRESHOLD_SECONDS` (default: 60)
- `AGGREGATION_LOG_DIR` (default: "data/aggregation_logs")
- `FUSION_SERVICE_URL` (optional)

**Note**: Remove Supabase config - use `utils.database.get_supabase_config()` instead

---

### 2. Models (`ser/models.py`)

**Source**: `Well-Bot_SER/app/models/speech.py`

**Purpose**: SER-specific Pydantic models

- `VoiceEmotionCreate` - For creating voice emotion records in database
- `ChunkResult` - Individual inference result for a single audio chunk
- `AggregatedResult` - Aggregated result for a time window

**Note**: Do NOT include fusion models (`ModelSignal`, `ModelPredictResponse`, etc.) - these exist in `fusion/models.py` and will be imported when needed

---

### 3. API Routes (`ser/api.py`)

**Source**:

- `Well-Bot_SER/app/api/speech.py`
- `Well-Bot_SER/app/api/dashboard.py`
- `Well-Bot_SER/app/crud/speech.py` (CRUD operations integrated here)

**Purpose**: FastAPI routes and database operations

- `POST /ser/analyze-speech` - Enqueue audio chunk for processing
- `POST /ser/predict` - Get predictions for fusion service (returns ModelSignal format)
- `GET /ser/dashboard` - Dashboard HTML page
- `GET /ser/api/dashboard/status` - Dashboard status JSON endpoint
- `insert_voice_emotion()` - Database insert function (calls `utils.database`)

**Note**: CRUD operations call `utils.database.insert_voice_emotion()` directly

---

### 4. ML Processing Services

#### 4.1 Audio Preprocessing (`ser/audio_preprocessing.py`)

**Source**: `Well-Bot_SER/app/services/speech_AudioPreprocessing.py`

**Purpose**: Audio preprocessing utilities

- `preprocess_audio()` - Resample, normalize, remove silence
- `validate_audio()` - Validate audio file format
- `get_audio_info()` - Get audio metadata

**Dependencies**: `librosa`, `soundfile`, `numpy`

---

#### 4.2 Emotion Recognition (`ser/emotion_recognition.py`)

**Source**: `Well-Bot_SER/app/services/speech_EmotionRecognition.py`

**Purpose**: Speech emotion recognition using ML model

- `predict_emotion()` - Predict emotion from audio file
- `_load_emotion_model()` - Lazy-load emotion model (wav2vec2-base-superb-er)

**Dependencies**: `transformers`, `torch`, `torchaudio`

**Model**: `superb/wav2vec2-base-superb-er` (CPU-only)

---

#### 4.3 Transcription (`ser/transcription.py`)

**Source**: `Well-Bot_SER/app/services/speech_transcription.py`

**Purpose**: Automatic Speech Recognition (ASR)

- `transcribe_audio()` - Transcribe audio to text
- `_load_transcription_model()` - Lazy-load Whisper model

**Dependencies**: `transformers`, `torch`, `torchaudio`

**Model**: `openai/whisper-small` (CPU-only)

---

#### 4.4 Sentiment Analysis (`ser/sentiment_analysis.py`)

**Source**: `Well-Bot_SER/app/services/speech_SentimentAnalysis.py`

**Purpose**: Sentiment analysis on transcribed text

- `analyze_sentiment()` - Analyze sentiment of text
- `_load_sentiment_pipeline()` - Lazy-load sentiment pipeline

**Dependencies**: `transformers`

**Model**: `cardiffnlp/twitter-xlm-roberta-base-sentiment`

---

#### 4.5 Processing Pipeline (`ser/processing_pipeline.py`)

**Source**: `Well-Bot_SER/app/services/speech_ProcessingPipeline.py`

**Purpose**: Orchestrates all ML services in sequence

- `analyze_full()` - Main pipeline function that:

  1. Validates audio
  2. Preprocesses audio
  3. Runs emotion recognition
  4. Runs transcription
  5. Detects language
  6. Runs sentiment analysis
  7. Returns combined results

**Dependencies**: All ML services above, `langdetect`

---

### 5. Queue & Session Management

#### 5.1 Queue Manager (`ser/queue_manager.py`)

**Source**: `Well-Bot_SER/app/services/speech_QueueManager.py`

**Purpose**: Asynchronous processing queue with background worker thread

- `QueueManager` (singleton) - Manages processing queue
- `enqueue_chunk()` - Add audio chunk to queue
- `start_worker()` - Start background worker thread
- `stop_worker()` - Stop background worker thread
- `get_queue_size()` - Get current queue size
- `get_queue_items()` - Get queue items for dashboard
- `get_processing_item()` - Get currently processing item
- `get_recent_results()` - Get recent processing results

**Dependencies**: `speech_ProcessingPipeline`, `speech_SessionManager`

---

#### 5.2 Session Manager (`ser/session_manager.py`)

**Source**: `Well-Bot_SER/app/services/speech_SessionManager.py`

**Purpose**: In-memory session management and chunk result storage

- `SessionManager` (singleton) - Manages user sessions
- `add_result()` - Add chunk result to appropriate session
- `get_results_in_window()` - Get results within time window (for fusion)
- `get_all_sessions()` - Get all sessions for a user
- `_detect_or_create_session()` - Auto-detect sessions based on time gaps

**Dependencies**: `ser.models.ChunkResult`, `ser.config`

---

#### 5.3 Aggregator (`ser/aggregator.py`)

**Source**: `Well-Bot_SER/app/services/speech_Aggregator.py`

**Purpose**: Periodic aggregation of chunk results over time windows

- `Aggregator` (singleton) - Aggregates results periodically
- `start_periodic_aggregation()` - Start periodic timer thread
- `stop_periodic_aggregation()` - Stop periodic timer
- `aggregate_window()` - Aggregate results for a time window
- Writes aggregated results to JSONL log files

**Dependencies**: `speech_SessionManager`, `ser.models.AggregatedResult`, `ser.config`

---

### 6. Database Functions (`utils/database.py`)

**Source**: `Well-Bot_SER/app/crud/speech.py`

**Purpose**: Add SER-specific database operations to shared database module

**New Function**:

- `insert_voice_emotion(data: VoiceEmotionCreate) -> Optional[Dict]`
  - Inserts voice emotion record into `voice_emotion` table
  - Uses `get_supabase_client()` from `utils.database`
  - Uses `get_current_time_utc8()` for timestamp

**Note**: This follows the pattern of other database functions already in `utils/database.py`

---

### 7. Module Initialization (`ser/__init__.py`)

**Purpose**: Export main components for easy imports

**Exports**:

- `config` - Configuration settings
- `models` - Pydantic models
- `api` - FastAPI router

---

### 8. Main Application Integration (`main.py`)

**Source**: `Well-Bot_SER/app/main.py` (startup/shutdown events)

**Purpose**: Integrate SER router and lifecycle management

**Changes**:

- Import SER router: `from ser import api as ser_api`
- Include router: `app.include_router(ser_api.router, prefix="/ser")`
- Add startup events: Start `QueueManager` and `Aggregator`
- Add shutdown events: Stop `QueueManager` and `Aggregator`

---

## File Naming Conventions

**Old → New**:

- `speech_AudioPreprocessing.py` → `audio_preprocessing.py`
- `speech_EmotionRecognition.py` → `emotion_recognition.py`
- `speech_transcription.py` → `transcription.py`
- `speech_SentimentAnalysis.py` → `sentiment_analysis.py`
- `speech_ProcessingPipeline.py` → `processing_pipeline.py`
- `speech_QueueManager.py` → `queue_manager.py`
- `speech_SessionManager.py` → `session_manager.py`
- `speech_Aggregator.py` → `aggregator.py`

**Rationale**: Remove `speech_` prefix, use snake_case consistently

---

## Dependencies (`requirements.txt`)

**Add to `Well-Bot_cloud/requirements.txt`**:

- `transformers` - Hugging Face transformers (for ML models)
- `torch` - PyTorch (CPU-only)
- `torchaudio` - Audio processing (CPU-only)
- `librosa` - Audio analysis
- `soundfile` - Audio file I/O
- `langdetect` - Language detection

**Note**: Use CPU-only PyTorch to avoid GPU dependencies

---

## Import Path Updates

**Old → New**:

- `app.core.config` → `ser.config`
- `app.core.db` → `utils.database` (for Supabase client)
- `app.models.speech` → `ser.models`
- `app.crud.speech` → `utils.database` (for insert_voice_emotion)
- `app.services.speech_*` → `ser.*` (e.g., `ser.audio_preprocessing`)

---

## Key Design Principles

1. **Flat Structure**: No nested subdirectories (follows `fusion/` and `intervention/` patterns)
2. **Shared Utilities**: Database functions in `utils/database.py`, other utilities in `utils/`
3. **No Separate CRUD**: Database operations integrated into `ser/api.py` calling `utils.database`
4. **SER-Specific Models Only**: Fusion models remain in `fusion/models.py`
5. **Singleton Pattern**: QueueManager, SessionManager, Aggregator remain singletons
6. **Lazy Loading**: ML models loaded on first use to reduce startup time
7. **Thread Safety**: Queue and session management use locks for thread safety

---

## Testing Checklist

- [ ] Import `ser` module successfully
- [ ] Load `ser.config.settings` and verify environment variables
- [ ] Create instances of all Pydantic models
- [ ] Call `utils.database.insert_voice_emotion()` with test data
- [ ] Import all service modules without errors
- [ ] Verify QueueManager singleton pattern
- [ ] Verify SessionManager singleton pattern
- [ ] Verify Aggregator singleton pattern
- [ ] Test ML model lazy loading
- [ ] Test API endpoints (after implementation)
- [ ] Test startup/shutdown events in main.py