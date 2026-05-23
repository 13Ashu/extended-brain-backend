# extended-brain-backend — CLAUDE.md

## Strategic Context

> **The native iOS app is the only active client.** Telegram and WhatsApp integrations exist in this codebase but are **legacy and slated for retirement**. Do not build new features that depend on them. When reading or modifying any flow, focus on the iOS API path. The messaging webhook handlers (`/webhook/telegram`, `/webhook/whatsapp`) and their associated files (`telegram.py`, `whatsapp.py`, `messaging_factory.py`, `messaging_interface.py`) are candidates for removal.

---

## What This Service Does

FastAPI backend powering the Extended Minds iOS app and web dashboard. It:
- Accepts message captures and runs them through an AI pipeline (intent detection, categorization, embedding)
- Stores all knowledge in Neon PostgreSQL with pgvector for semantic search
- Handles authentication (phone + OTP + password → JWT)
- Delivers reminders via APNs (primary) and Telegram (legacy, to be retired)
- Manages Pro accounts, collaborative groups, and live WebSocket feeds
- Runs a background scheduler for reminders, briefings, and nudges

**Runtime:** Python 3.11+, FastAPI, SQLAlchemy (async), uvicorn  
**Deployed on:** Railway  
**Database:** Neon PostgreSQL (`asyncpg` driver, `pgvector` extension)  
**Cache:** Upstash Redis (optional, REST API)

---

## Folder Structure

```
extended-brain-backend/
├── main.py                   ← ALL routes + scheduler + startup (3600+ lines)
├── database.py               ← SQLAlchemy ORM models + async engine setup
├── models.py                 ← Re-exports from database.py (thin wrapper)
├── config.py                 ← Config class reading all env vars
├── cerebras_client.py        ← Unified LLM client: Gemini / Cerebras / OpenRouter
├── requirements.txt          ← Python dependencies
├── Procfile                  ← Railway/Heroku start command
├── railway.json              ← Railway build + deploy config
├── README.md                 ← User-facing setup guide
├── SETUP_INSTRUCTIONS.md     ← Developer setup steps
│
├── services/                 ← 22 focused service modules
│   ├── auth_service.py       ← OTP, JWT creation/verify, password hashing
│   ├── message_processor.py  ← Fast intent bucket detection (signal-based)
│   ├── intent_service.py     ← Deep multi-action parse via LLM
│   ├── search_service.py     ← Semantic search: embed → pgvector → LLM rank
│   ├── embedding_service.py  ← Gemini embedding generation (1536 dims)
│   ├── redis_cache.py        ← Upstash Redis wrapper (async, graceful)
│   ├── list_service.py       ← Named list management (shopping, packing, etc.)
│   ├── group_service.py      ← Pro account + collaborative group logic
│   ├── reminder_service.py   ← Reminder scheduling + Telegram/APNs delivery
│   ├── briefing_service.py   ← Morning briefing + carry-forward todos
│   ├── nudge_service.py      ← Idle task nudges + 48hr follow-up checks
│   ├── category_manager.py   ← Custom category CRUD + LLM suggestions
│   ├── document_processor.py ← PDF/DOCX text extraction
│   ├── vision_service.py     ← Image analysis via Gemini multimodal
│   ├── project_service.py    ← Project grouping + completion tracking
│   ├── subtask_service.py    ← Task breakdown into subtasks
│   ├── recurrence_service.py ← Recurring reminders (daily/weekly/monthly)
│   ├── coupon_service.py     ← Coupon validation and redemption
│   └── context_service.py    ← Multi-turn conversation context storage
│
├── messaging_interface.py    ← Abstract base class for messaging clients
├── messaging_factory.py      ← Factory: returns WhatsApp or Telegram client
├── whatsapp.py               ← WhatsApp Business API client
├── telegram.py               ← Telegram Bot API client
│
└── (migration/test scripts)  ← migrate_*.py, backfill_*.py, test_*.py
```

---

## Key Files Mapped to Concerns

| Concern | File(s) |
|---------|---------|
| All HTTP routes | `main.py` |
| Background scheduler | `main.py` (`_master_scheduler` function) |
| ORM models + DB schema | `database.py` |
| Database connection / engine | `database.py` (`create_async_engine`, `get_db`) |
| Model re-exports | `models.py` |
| All env vars / config | `config.py` (`Config` class) |
| LLM calls (Gemini/Cerebras) | `cerebras_client.py` |
| JWT issuance + verification | `services/auth_service.py` |
| Password hashing | `services/auth_service.py` (`hash_password`, `verify_password`) |
| Current user dependency | `services/auth_service.py` (`get_current_user`) |
| AI intent parsing | `services/intent_service.py` |
| Quick bucket detection | `services/message_processor.py` |
| Semantic search | `services/search_service.py` |
| Embeddings | `services/embedding_service.py` |
| Redis caching | `services/redis_cache.py` |
| Pro accounts / groups | `services/group_service.py` |
| Reminder scheduling | `services/reminder_service.py` |
| APNs push delivery | `services/reminder_service.py` |
| Telegram delivery | `telegram.py` |
| WhatsApp delivery | `whatsapp.py` |
| Platform selection | `config.py` + `messaging_factory.py` |

---

## API Route Inventory

### Core
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info & feature list |
| GET | `/health` | Health check |

### Authentication
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/send-otp` | Send OTP to phone via Telegram |
| POST | `/api/auth/verify-otp` | Verify 6-digit OTP (max 5 attempts, 10 min TTL) |
| POST | `/api/auth/login` | Phone + password → JWT |
| POST | `/api/auth/forgot-password` | Reset password |
| DELETE | `/api/auth/delete-account` | Delete account (auth required) |
| POST | `/api/auth/link-telegram` | Associate Telegram chat_id with account |

### Users
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/users/me` | Current user profile |
| POST | `/api/users/register` | Create account (OTP must be verified if ENABLE_OTP=true) |
| POST | `/api/users/device-token` | Register APNs device token |

### Messages
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/messages/capture` | Save message — runs full AI pipeline |
| GET | `/api/messages/recent` | Recent messages (`?limit=&group_id=&after=`) |
| GET | `/api/messages/assigned` | Tasks @assigned to current user (their "Assigned to Me" feed) |
| GET | `/api/messages/assigned-to-others` | Group tasks the current user assigned to others (their delegation dashboard) |
| PATCH | `/api/messages/{id}/assignments/{idx}/complete` | Assignee marks their slot done; notifies assigner via APNs + WS broadcast |
| GET | `/api/messages/detail/{id}` | Single message detail |
| PATCH | `/api/messages/{id}/done` | Mark task done/undone `{"done": bool}` |
| PATCH | `/api/messages/{id}/items/{idx}/complete` | Check off a list item |
| DELETE | `/api/messages/{id}` | Delete message |

### Search & Upload
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/search` | Semantic search (`?fast=true` skips LLM rerank) |
| POST | `/api/upload` | Upload image (multipart) → returns URL |
| GET | `/api/images/{id}` | Download stored image |

### Categories
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/categories/manage` | Create / edit / delete / merge categories |

### Bootstrap & Analytics
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/bootstrap` | Initial app load (`?limit=&group_id=`) — recent + members + unread |
| GET | `/api/analytics` | User usage statistics |

### Groups
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/groups` | List user's groups |
| POST | `/api/groups` | Create group (Pro required) |
| GET | `/api/groups/{id}/messages` | Group message feed |
| GET | `/api/groups/{id}/members` | Group members |
| POST | `/api/groups/{id}/members` | Add member by user_id |
| POST | `/api/groups/{id}/invite` | Invite member by phone number |
| DELETE | `/api/groups/{id}/leave` | Leave group |
| DELETE | `/api/groups/{id}` | Delete group |
| POST | `/api/groups/{id}/seen` | Mark all group messages as read |
| GET | `/api/groups/unread` | Unread count per group `{group_id: count}` |
| WS | `/ws/group/{id}` | WebSocket: live incoming group messages |

### Pro Account
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/pro/status` | Pro subscription info + members list |
| POST | `/api/pro/invite` | Invite member by phone |
| GET | `/api/pro/my-invites` | Pending invites for current user |
| POST | `/api/pro/accept-invite` | Accept a team invite |
| DELETE | `/api/pro/members/{phone}` | Remove Pro team member |
| POST | `/api/pro/validate-coupon` | Check if coupon code is valid |
| POST | `/api/pro/redeem-coupon` | Apply coupon to account |

### Admin (requires `X-Admin-Secret` header)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/stats` | System-wide statistics |
| POST | `/api/admin/coupons` | Create coupon code |
| GET | `/api/admin/coupons` | List all coupons |
| DELETE | `/api/admin/coupons/{id}` | Deactivate coupon |
| POST | `/api/admin/grant-pro` | Manually grant Pro access |

### Webhooks
| Method | Path | Description |
|--------|------|-------------|
| GET | `/webhook/whatsapp` | WhatsApp webhook verification |
| POST | `/webhook/whatsapp` | Incoming WhatsApp message |
| POST | `/webhook/telegram` | Incoming Telegram update |
| GET | `/api/webhook/info` | Webhook registration status |

---

## Database Schema Summary

All models in `database.py`. Uses SQLAlchemy 2.0 async with `asyncpg`.

### Core Tables

**`users`**
- `id` (PK), `phone_number` (unique), `email`, `name`, `age`, `occupation`
- `password_hash` (SHA256+salt: `salt$hash`)
- `timezone`, `briefing_time`
- `is_pro` (bool), `telegram_chat_id` (nullable), `active_group_id` (nullable FK→groups)
- `created_at`, `last_login`

**`messages`**
- `id` (PK), `user_id` (FK), `group_id` (FK, nullable), `category_id` (FK, nullable)
- `content` (TEXT), `message_type` (enum: text/image/audio/pdf/link)
- `media_url` (nullable), `summary` (nullable)
- `tags` (JSONB): `{buckets, primary_bucket, priority, due_date, event_time, done, reminded_at, original_dump, split_from, assignments, group_reminder, ...}`
  - `assignments`: `[{user_id, name, phone, done, done_at}]` — per-assignee completion slots for group @mention tasks
  - `group_reminder`: `true` — set when a group message has a due time but no specific @mention (fires APNs to all members)
  - `original_dump`: original capture text (for retrieval when LLM paraphrases single tasks)
  - `split_from`: original multi-task dump text (for retrieval of split tasks)
- `entities` (JSONB): `{people, locations, dates, numbers}`
- `embedding` (pgvector, 1536 dims)
- `created_at`

**`categories`**
- `id`, `user_id`, `name`, `description`, `color`, `icon`

**`reminders`**
- `id`, `user_id`, `message_id` (nullable), `content`, `task`
- `remind_at` (datetime), `timezone`, `telegram_chat_id`
- `is_sent`, `is_cancelled`, `snooze_count`, `sent_at`
- `recurrence` (JSON): `{type: daily|weekly|monthly, interval: N}`

### Auth Tables

**`otp_verifications`** — `phone_number`, `otp_code`, `is_verified`, `attempts` (max 5), `expires_at`

**`device_tokens`** — `user_id`, `token` (APNs), `platform` ("ios")

### Pro & Groups Tables

**`pro_accounts`** — `owner_id` (FK→users, unique), `plan_type`, `max_members`, `expires_at`

**`pro_account_members`** — `account_id`, `user_id` (nullable), `phone_number`, `invite_token`, `status` (pending/active), `invited_at`, `joined_at`

**`groups`** — `id`, `account_id` (FK→pro_accounts), `name`, `description`, `emoji`, `created_by`, `created_at`

**`group_members`** — `group_id`, `user_id`, `role`, `joined_at`

**`group_last_seen`** — `(user_id, group_id)` unique, `last_seen_at` — drives unread counts

### Other Tables

**`coupon_codes`** — `code`, `discount_type` (free/percent/fixed), `discount_value`, `duration_days`, `max_uses`, `expires_at`, `is_active`

**`coupon_redemptions`** — `coupon_id`, `user_id`, `redeemed_at`

**`stored_images`** — `user_id`, `data` (LargeBinary), `mime_type` — fallback for images without CDN URL

---

## Environment Variables

```bash
# Required
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
SECRET_KEY=<long-random-string>           # JWT signing — MUST override default
GEMINI_API_KEY=AIzaS...                    # Primary LLM + embeddings

# Messaging (one platform active at a time)
MESSAGING_PLATFORM=telegram                # "telegram" | "whatsapp"
TELEGRAM_BOT_TOKEN=123456:ABCDef...
TELEGRAM_WEBHOOK_URL=https://<railway-domain>/webhook/telegram
# OR for WhatsApp:
WHATSAPP_ACCESS_TOKEN=...
WHATSAPP_PHONE_NUMBER_ID=...
WHATSAPP_VERIFY_TOKEN=...

# Optional but recommended
CEREBRAS_API_KEY=csk-...                   # Fallback LLM
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...

# iOS Push Notifications
APNS_KEY_ID=XXXXX
APNS_TEAM_ID=ABCDE
APNS_AUTH_KEY=-----BEGIN PRIVATE KEY-----...
APNS_BUNDLE_ID=com.extendedminds.app
APNS_PRODUCTION=true                       # false = sandbox

# Admin
ADMIN_SECRET=<password>                    # Gating for /api/admin/* endpoints

# Feature flags
ENABLE_OTP=true                            # Default false — enables phone verification
DEBUG=false

# Set automatically by Railway
PORT=8000
RAILWAY_PUBLIC_DOMAIN=...
```

---

## How to Run Locally

```bash
cd extended-brain-backend

# 1. Create virtualenv
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file with required vars (see above)
cp .env.example .env   # if it exists, otherwise create manually

# 4. Start server
uvicorn main:app --reload --port 8000

# 5. Visit http://localhost:8000 for API info
#    http://localhost:8000/docs for Swagger UI
```

**Database init:** On startup, `init_db()` runs `CREATE TABLE IF NOT EXISTS` migrations + safe column additions.

**Telegram webhook (local dev):** Use `ngrok http 8000` to expose locally, then POST to `/api/webhook/info` or set `TELEGRAM_WEBHOOK_URL` manually.

---

## Deployment Notes (Railway)

- **Build:** Nixpacks auto-detects Python; runs `pip install -r requirements.txt`
- **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Restart:** `ON_FAILURE`, max 10 retries (`railway.json`)
- **Neon DB:** SSL required; pool_size=5 (Neon free tier limit is hardcoded in `database.py`)
- **Telegram webhook:** Set `TELEGRAM_WEBHOOK_URL=https://<app>.up.railway.app/webhook/telegram` — app auto-registers on startup
- **Single process:** No workers configuration; uvicorn runs one async process (no multiprocessing)

---

## Scheduler Architecture

Runs as an `asyncio` background task at startup (`_master_scheduler`):

```
Every 60 seconds:
  → reminder_service.run_scheduler_tick()    fire due reminders
  → recurrence_service.run()                 process recurring tasks

Every 30 minutes (every 30th tick):
  → nudge_service.run_idle_nudges()          poke stale todos
  → nudge_service.run_followup_checks()      48hr follow-ups

Daily at user's briefing_time:
  → briefing_service.run()                   morning briefing + carry-forward
```

---

## AI Pipeline

**Primary:** Gemini 2.0 Flash Lite (`gemini-2.0-flash-lite`)  
**Fallback:** Gemini 2.0 Flash → Cerebras (`qwen-3-235b-a22b-instruct-2507`) → Llama 3.1 8B

**LLM use cases:**
1. Multi-action intent parse (`intent_service`) — one LLM call determines all actions for a message
2. Search result re-ranking (`search_service`)
3. Morning briefing generation (`briefing_service`)
4. Category suggestions (`category_manager`)

**Embeddings:** Gemini text-embedding via REST, 1536 dims, stored in pgvector column.

**Retry:** 3 attempts, exponential backoff on 429/500/503.

---

## Things to Always Do

- Use `async with get_db() as db:` (or the FastAPI `Depends(get_db)` pattern) — never create a session manually.
- Run through `cerebras_client` for all LLM calls — it handles retries, caching, and fallback.
- Invalidate the relevant Redis cache keys when mutating messages or groups.
- Respect the 7 bucket names exactly: `"Remember"`, `"To-Do"`, `"Ideas"`, `"Track"`, `"Events"`, `"List"`, `"Random"`.
- New routes go in `main.py` (that's the current convention, even if undesirable).

## Things to Never Do

- Never import `SECRET_KEY` inline — always use `Config.SECRET_KEY` (or the import from config).
- Never create a synchronous SQLAlchemy session — the engine is async-only.
- Never call `db.execute()` without `await`.
- Never log full JWT tokens or password hashes.
- Never hardcode the Neon DB URL — always use `DATABASE_URL` from env.
- **Never build new features that route through Telegram or WhatsApp** — these channels are being retired. All new flows must be reachable from the iOS app via the REST API.

---

## ⚠️ Noted Issues

- `main.py` is 3600+ lines with all routes, scheduler, WebSocket manager, and startup code in one file — extremely hard to navigate and untestable.
- No test suite exists for routes or services.
- `SECRET_KEY` has a default value hardcoded in `main.py` — this is a critical security risk in production.
- Password hashing uses SHA256 with a random salt (`salt$hash`) — not bcrypt or argon2. Adequate for low-risk but should be migrated.
- `ENABLE_OTP=false` default means phone ownership is never verified in dev environments.
- `stored_images` table stores raw binary (LargeBinary) in PostgreSQL — this will not scale; should use a CDN/object store.
- Two share extensions in the iOS repo (`ShareExtension/` and `extendedMindShareExtension/`) — backend doesn't differentiate, but iOS project maintenance is split.
- `❓ Unclear`: The `try_gemini.py`, `test_cerebras_again.py`, and similar root-level scripts appear to be one-off experiments — no documentation on whether they are safe to delete.
