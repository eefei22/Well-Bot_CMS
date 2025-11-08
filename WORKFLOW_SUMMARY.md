# Well-Bot CMS - Complete Workflow Summary

## 📋 Project Overview

**Well-Bot CMS (Content Management System)** is a FastAPI-based service that processes user conversation messages from Supabase and generates two types of AI-powered summaries:
1. **Persona Facts** - Stable characteristics (communication style, interests, personality traits, values, concerns)
2. **Daily Life Context** - Experiential stories (routines, relationships, work life, daily activities)

Both summaries are extracted using DeepSeek's reasoning model and stored in the database for use by the Well-Bot conversational AI system.

---

## 🏗️ Architecture & Workflow

### High-Level Flow

```
User Request (user_id)
    ↓
FastAPI Endpoint (/api/context/process)
    ↓
┌─────────────────────────────────────┐
│  Step 1: Facts Extraction           │
│  - Load messages from Supabase       │
│  - Filter & normalize                │
│  - Extract persona facts via LLM    │
│  - Save to facts field               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Step 2: Context Extraction          │
│  - Load messages from Supabase       │
│  - Filter & normalize                │
│  - Extract daily life context       │
│  - Save to persona_summary field     │
└─────────────────────────────────────┘
    ↓
Return Response (both summaries)
```

### Detailed Processing Pipeline

1. **Message Loading** (`database.load_user_messages()`)
   - Queries `wb_conversation` table for all user conversations
   - Queries `wb_message` table for user messages (role="user" only)
   - Groups messages by conversation
   - Returns structured conversation data

2. **Message Preprocessing** (both extractors)
   - **Filtering**: Removes messages with < 4 words
   - **Normalization**: Lowercase, strip whitespace, remove extra spaces
   - **Formatting**: Converts to bullet-point list for LLM prompt

3. **LLM Processing** (DeepSeek Reasoning Model)
   - **Facts Extractor**: Categorizes persona characteristics
   - **Context Processor**: Extracts daily life stories and experiences
   - Both use separate, specialized prompts
   - Processing time: 1-3 minutes per extraction (2-6 minutes total)

4. **Database Storage** (`database.write_users_context_bundle()`)
   - Upserts to `users_context_bundle` table
   - Supports partial updates (can update one or both fields)
   - Both fields saved to same row (same user_id)

---

## ✅ Functional Requirements

### Core Features

1. **User Message Retrieval**
   - Load all user messages from Supabase database
   - Filter by user_id and role="user"
   - Group by conversation with metadata

2. **Message Preprocessing**
   - Filter short messages (< 4 words)
   - Normalize text (lowercase, whitespace cleanup)
   - Format for LLM consumption

3. **Persona Facts Extraction**
   - Extract communication style and patterns
   - Identify interests and preferences
   - Determine personality traits
   - Capture values and concerns
   - Note behavioral patterns
   - Output: Structured text summary

4. **Daily Life Context Extraction**
   - Extract daily routines and activities
   - Capture stories and experiences
   - Identify people and relationships
   - Extract work life context
   - Note life events and significant moments
   - Output: Structured text summary

5. **Data Persistence**
   - Save facts to `facts` field
   - Save context to `persona_summary` field
   - Support partial updates
   - Maintain version timestamps

6. **RESTful API**
   - Process user context via POST endpoint
   - Return both summaries in response
   - Handle errors gracefully
   - Provide health check endpoint

---

## 📦 Dependencies

### Python Packages

```python
# Web Framework & Server
fastapi==0.109.0              # REST API framework
uvicorn[standard]==0.27.0     # ASGI server with auto-reload

# Database
supabase>=2.4.0               # Supabase Python client
httpx[http2]>=0.26,<0.28       # HTTP client (pinned for compatibility)

# Data Validation
pydantic==2.5.3                # Data validation and settings
pydantic-settings==2.1.0       # Settings management

# Environment & Configuration
python-dotenv==1.0.0           # .env file support

# Testing
requests==2.31.0               # HTTP client for testing
```

### External Services

1. **Supabase** (PostgreSQL Database)
   - Tables: `wb_conversation`, `wb_message`, `users_context_bundle`
   - Authentication: Service role key for admin operations

2. **DeepSeek API**
   - Model: `deepseek-reasoner` (reasoning model)
   - Endpoint: `https://api.deepseek.com/v1/chat/completions`
   - Timeout: 680 seconds (11+ minutes) per request

### Environment Variables Required

```bash
# Supabase Configuration
SUPABASE_URL=<your-supabase-url>
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>

# DeepSeek API
DEEPSEEK_API_KEY=<your-deepseek-api-key>

# Development (optional)
DEV_USER_ID=<test-user-uuid>
```

---

## 🔌 API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. **GET /** - Root Endpoint
- **Purpose**: Health check / API status
- **Response**:
  ```json
  {
    "message": "Well-Bot CMS API is running"
  }
  ```

#### 2. **GET /health** - Health Check
- **Purpose**: Service health monitoring
- **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```

#### 3. **POST /api/context/process** - Process User Context
- **Purpose**: Extract and save both persona facts and daily life context
- **Request Body**:
  ```json
  {
    "user_id": "8517c97f-66ef-4955-86ed-531013d33d3e"
  }
  ```
- **Response** (Success):
  ```json
  {
    "status": "success",
    "user_id": "8517c97f-66ef-4955-86ed-531013d33d3e",
    "facts": "• Communication style: [extracted facts]\n• Interests: [interests]\n...",
    "persona_summary": "• Daily routines: [routines]\n• Stories: [experiences]\n..."
  }
  ```
- **Response** (Error):
  ```json
  {
    "detail": "Error message here"
  }
  ```
- **Status Codes**:
  - `200`: Success
  - `400`: Bad Request (missing user_id, no messages found, etc.)
  - `500`: Internal Server Error (LLM failure, database error, etc.)
- **Processing Time**: 2-6 minutes (both LLM extractions)
- **Error Handling**:
  - If facts extraction fails, context extraction still proceeds
  - If context extraction fails, request returns error
  - Both results saved independently to database

### Interactive API Documentation

FastAPI automatically generates interactive docs:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🗄️ Database Schema

### Table: `users_context_bundle`

```sql
CREATE TABLE public.users_context_bundle (
  user_id uuid NOT NULL PRIMARY KEY,
  version_ts timestamp with time zone NOT NULL DEFAULT now(),
  persona_summary text,           -- Daily life context (stories, routines, relationships)
  last_session_summary text,       -- Reserved for future use
  facts text                       -- Persona characteristics (traits, interests, values)
);
```

**Field Usage**:
- `user_id`: Primary key, identifies the user
- `version_ts`: Auto-updated timestamp on each write
- `persona_summary`: Stores daily life context extracted by `context_processor.py`
- `facts`: Stores persona facts extracted by `facts_extractor.py`
- `last_session_summary`: Reserved for future session-based summaries

### Related Tables (Read-Only)

**`wb_conversation`**:
- `id`: Conversation UUID
- `user_id`: User UUID
- `started_at`: Conversation start timestamp

**`wb_message`**:
- `id`: Message UUID
- `conversation_id`: Conversation UUID (FK)
- `role`: Message role ("user" or "assistant")
- `text`: Message content
- `created_at`: Message timestamp

---

## 📁 File Structure

```
Well-Bot_CMS/
├── main.py                    # FastAPI application & endpoints
├── context_processor.py       # Daily life context extraction
├── facts_extractor.py         # Persona facts extraction
├── database.py                # Supabase connection & operations
├── llm.py                     # DeepSeek API client
├── schemas.py                 # Pydantic request/response models
├── requirements.txt           # Python dependencies
├── test_api.py               # API testing script
├── test_context_processor.py # Direct function testing
├── .env                      # Environment variables (not in repo)
└── archive/                  # Legacy code
    ├── db_connect.py
    └── db_read.py
```

### Module Responsibilities

| File | Purpose |
|------|---------|
| `main.py` | API server, endpoint routing, request handling |
| `context_processor.py` | Extracts daily life stories, routines, relationships |
| `facts_extractor.py` | Extracts persona characteristics and traits |
| `database.py` | Supabase connection, message loading, context saving |
| `llm.py` | DeepSeek API client (streaming & non-streaming) |
| `schemas.py` | Request/response data models |

---

## 🔄 Data Flow

### Request Flow

```
1. Client sends POST /api/context/process
   {
     "user_id": "uuid-here"
   }
   ↓
2. main.py receives request
   ↓
3. facts_extractor.extract_user_facts(user_id)
   ├─→ database.load_user_messages(user_id)
   │   ├─→ Query wb_conversation table
   │   └─→ Query wb_message table (role="user")
   ├─→ Filter messages (< 4 words)
   ├─→ Normalize messages
   ├─→ Call DeepSeek API (persona facts prompt)
   └─→ database.write_users_context_bundle(user_id, facts=...)
       └─→ Upsert to users_context_bundle table
   ↓
4. context_processor.process_user_context(user_id)
   ├─→ database.load_user_messages(user_id) [same as above]
   ├─→ Filter messages (< 4 words)
   ├─→ Normalize messages
   ├─→ Call DeepSeek API (daily life context prompt)
   └─→ database.write_users_context_bundle(user_id, persona_summary=...)
       └─→ Upsert to users_context_bundle table
   ↓
5. Return response
   {
     "status": "success",
     "user_id": "uuid-here",
     "facts": "...",
     "persona_summary": "..."
   }
```

### Database Operations

**Read Operations**:
- `load_user_messages()`: Reads from `wb_conversation` and `wb_message`

**Write Operations**:
- `write_users_context_bundle()`: Upserts to `users_context_bundle`
  - First call: Saves `facts` field
  - Second call: Saves `persona_summary` field
  - Both updates same row (same `user_id`)

---

## 🚀 Running the Application

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` file:
```bash
SUPABASE_URL=your-supabase-url
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
DEEPSEEK_API_KEY=your-deepseek-api-key
DEV_USER_ID=8517c97f-66ef-4955-86ed-531013d33d3e
```

### 3. Start Server

```bash
python main.py
```

Server runs on `http://localhost:8000` with auto-reload enabled.

### 4. Test the API

**Option A: Using test script**
```bash
python test_api.py
```

**Option B: Using curl**
```bash
curl -X POST "http://localhost:8000/api/context/process" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "8517c97f-66ef-4955-86ed-531013d33d3e"}'
```

**Option C: Using FastAPI docs**
- Open `http://localhost:8000/docs`
- Click on `POST /api/context/process`
- Click "Try it out"
- Enter user_id and execute

---

## ⚙️ Configuration & Settings

### Logging

- **Level**: INFO (default)
- **Format**: `%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s`
- **Location**: Console output
- **Modules**: All modules use conditional logging (prevents conflicts with uvicorn)

### LLM Configuration

- **Model**: `deepseek-reasoner`
- **Base URL**: `https://api.deepseek.com`
- **Timeout**: 680 seconds (11+ minutes)
- **Streaming**: Supported but not used in current implementation

### Message Processing

- **Min Words Filter**: 4 words (messages with < 4 words are discarded)
- **Normalization**: Lowercase, strip whitespace, remove extra spaces
- **Message Role**: Only processes messages with `role="user"`

---
## User differentation

The endpoint requires the user UUID in the request body to identify which user to process.
### Current Implementation

**Request Schema** (`schemas.py`):
```python
class ProcessContextRequest(BaseModel):
    user_id: str  # Required field
```

**Request Example**:
```json
{
  "user_id": "8517c97f-66ef-4955-86ed-531013d33d3e"
}
```

### How User Differentiation Works

1. Request body: The `user_id` is passed in the JSON body.
2. Message loading: `database.load_user_messages(user_id)` queries:
   - `wb_conversation` table filtered by `user_id`
   - `wb_message` table for messages in those conversations
3. Processing: Both extractors use this `user_id` to process that user's messages.
4. Storage: Results are saved to `users_context_bundle` with `user_id` as the primary key.

### Why This Design?

- Each request targets a specific user.
- The `user_id` is required (Pydantic validation).
- The database uses `user_id` as the primary key for storage.

So yes, every call to `/api/context/process` must include the user UUID in the request body to identify which user's messages to process.