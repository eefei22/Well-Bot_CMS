
Context needed to build an external service that integrates with the smalltalk module:

## **1. Message Data Structure & Format**

**Message Format (LLM):**
```python
messages: List[Dict[str, str]] = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]
```

**Database Message Schema (`wb_message`):**
- `id`: uuid (primary key)
- `conversation_id`: uuid (FK to `wb_conversation`)
- `role`: text ('user' | 'assistant')
- `text`: text (message content)
- `created_at`: timestamp
- `tokens`: integer (optional)
- `metadata`: jsonb (optional, currently `{}`)

**Database Conversation Schema (`wb_conversation`):**
- `id`: uuid (primary key)
- `user_id`: uuid (FK to `auth.users`)
- `started_at`: timestamp
- `ended_at`: timestamp (nullable)
- `reason_ended`: text (nullable)

## **2. Conversation Lifecycle & Timing**

**Lifecycle Events:**
1. Conversation Start: `start_conversation()` → returns `conversation_id`
2. User Message: Saved immediately after STT capture
3. LLM Processing: `self.llm.stream_chat(self.messages, temperature=0.6)`
4. Assistant Message: Saved after LLM response completes
5. Turn Completion: `complete_turn(user_text, assistant_text)`
6. Conversation End: `end_conversation(conversation_id)` sets `ended_at`

**Turn Flow:**
```
User speaks → STT → user_text
  ↓
Save to DB (role='user', intent='small_talk')
  ↓
Append to self.messages
  ↓
LLM.stream_chat(self.messages) → assistant_text
  ↓
Append to self.messages
  ↓
Save to DB (role='assistant')
  ↓
complete_turn() → increments turn_count
```

## **3. System Prompt & Configuration**

**System Prompt Location:**
- Default: `"You are a friendly, concise wellness assistant. Keep responses short unless asked."`
- Configurable: `backend/config/{lang}.json` → `smalltalk.system_prompt`
- Can be overridden: `SmallTalkSession(system_prompt=...)`
- Can be injected: `add_system_message(content)` method exists

**Current System Prompt (from `en.json`):**
```json
"You are Well-Bot, a friendly emotional wellness assistant having a small talk with your user. Keep your responses casual, short and concise. Do not use emojis. Here is what the user said:"
```

## **4. Database Access Patterns**

**Key Functions (from `backend/src/supabase/database.py`):**
```python
# Start conversation
conversation_id = start_conversation(user_id=DEV_USER_ID, title="Small Talk")

# Add message
message_id = add_message(
    conversation_id=conversation_id,
    role="user" | "assistant",
    content="...",
    intent="small_talk",  # optional
    lang="en-US",  # optional
    tokens=None,  # optional
    metadata={}  # optional
)

# End conversation
end_conversation(conversation_id)

# List messages
messages = list_messages(conversation_id, limit=100)

# List conversations
conversations = list_conversations(limit=20, user_id=DEV_USER_ID)
```

**User ID Resolution:**
- Currently: `get_current_user_id()` → reads `DEV_USER_ID` env var
- Default: `"8517c97f-66ef-4955-86ed-531013d33d3e"`
- Future: Will extract from JWT token

## **5. LLM Integration Details**

**LLM Client:**
- Provider: DeepSeek (REST API)
- Endpoint: `https://api.deepseek.com/v1/chat/completions`
- Model: `deepseek-chat` (default)
- Streaming: Yes (`stream_chat()` yields text chunks)
- Temperature: `0.6` (hardcoded in `_stream_llm_and_tts()`)

**LLM Call Pattern:**
```python
for text_chunk in self.llm.stream_chat(self.messages, temperature=0.6):
    # Process chunks
full_text = "".join(text_chunks)
self.messages.append({"role": "assistant", "content": full_text})
```

## **6. Integration Hooks & Extension Points**

**Current Extension Points:**
1. System Prompt Injection: `add_system_message(content)` (line 147-150 in `smalltalk.py`)
2. Seed System Prompt: `start(seed_system_prompt=...)` (line 152)
3. Custom Start Prompt: `start(custom_start_prompt=...)` (line 152)
4. Metadata Field: `add_message(..., metadata={...})` - currently unused but available

**No Current Hooks For:**
- Pre-LLM message processing
- Post-LLM response modification
- Turn-level callbacks
- Conversation-level analytics

## **7. Message Context & State**

**In-Memory State (`SmallTalkSession`):**
- `self.messages`: Full conversation history (system + user + assistant)
- `self.conversation_id`: Current conversation UUID
- `self._active`: Boolean session state

**Session State (`ConversationSession`):**
- `_turn_count`: Current turn number
- `max_turns`: Default 20 (configurable)
- `language_code`: e.g., "en-US"

## **8. Configuration Structure**

**Language Config (`backend/config/en.json`):**
```json
{
  "smalltalk": {
    "system_prompt": "...",
    "termination_phrases": [...],
    "prompts": {
      "start": "...",
      "nudge": "...",
      "timeout": "...",
      "end": "..."
    }
  }
}
```

**Global Config (`backend/config/global.json`):**
- Language codes (STT/TTS)
- Audio settings
- Smalltalk settings (max_turns, timeouts, etc.)


## **10. Critical Constraints**

1. No real-time streaming hooks — messages are saved after completion
2. Synchronous execution — activities run in main thread
3. Single user per instance — `DEV_USER_ID` is instance-level
4. Metadata field available — can store persona_summary or other data
5. No built-in retry mechanism — external service failures won't block conversation
6. Language-aware — messages have `lang` field, configs are language-specific

## **11. Example Integration Points**

**After Turn Completion:**
```python
# In smalltalk.py, after line 430
if not self.session_manager.complete_turn(user_text, assistant_text):
    break

# NEW: Call external service
if external_service_url:
    call_persona_updater(
        conversation_id=self.session_manager.conversation_id,
        user_id=self.user_id,
        messages=self.llm_pipeline.messages,
        turn_count=self.session_manager.get_turn_count()
    )
```

**At Conversation End:**
```python
# In smalltalk.py, after line 252
self.session_manager.end_conversation()

# NEW: Final persona summary update
if external_service_url:
    final_persona_update(conversation_id, user_id)
```
