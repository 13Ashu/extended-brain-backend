# extended-brain-backend — CLAUDE.md

## Strategic Context

> **The native iOS app is the only active client.** Telegram and WhatsApp integrations exist in this codebase but are **legacy and slated for retirement**. Do not build new features that depend on them. When reading or modifying any flow, focus on the iOS API path. The messaging webhook handlers (`/webhook/telegram`, `/webhook/whatsapp`) and their associated files (`telegram.py`, `whatsapp.py`, `messaging_factory.py`, `messaging_interface.py`) are candidates for removal.

---

## What This Service Does

FastAPI backend powering the Extended Minds iOS app and web dashboard. It:
- Accepts message captures and classifies them via a two-path AI pipeline:
  - **Fast path**: on-server ONNX classifier (~10ms, no network) for bucket classification
  - **Slow path**: Gemini 2.5 Flash Lite LLM, paid (~500ms) when classifier is absent or low-confidence
- Stores all knowledge in Neon PostgreSQL with pgvector for semantic search
- Handles authentication (phone + OTP + password → JWT)
- Delivers reminders via APNs (primary) and Telegram (legacy, to be retired)
- Manages Pro accounts, collaborative groups, and live WebSocket feeds
- Runs a background scheduler for reminders, briefings, and nudges
- Stores user bucket-correction annotations to a `label_annotations` table for model retraining

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
├── cerebras_client.py        ← Unified LLM client: Gemini 2.5 Flash Lite only (Cerebras removed)
├── requirements.txt          ← Python dependencies
├── Procfile                  ← Railway/Heroku start command
├── railway.json              ← Railway build + deploy config
├── README.md                 ← User-facing setup guide
├── SETUP_INSTRUCTIONS.md     ← Developer setup steps
│
├── models/                   ← ML model weights (not fully in git)
│   └── intent_classifier/    ← ONNX intent classifier (loaded at startup)
│       ├── backbone.onnx     ← fine-tuned all-MiniLM-L6-v2 v4 (86 MB, .gitignored — deployed via GitHub release)
│       ├── head_weights.npz  ← logistic regression weights (in git)
│       ├── input_prefix.txt  ← empty for MiniLM-L6 (contains "query: " for E5 models)
│       └── tokenizer_*.json  ← tokenizer files (in git)
│
├── services/                 ← 23 focused service modules
│   ├── classifier_service.py ← ★ ONNX intent classifier (fast path, ~10ms, no network)
│   ├── intent_service.py     ← ★ Two-path parse: ONNX fast → Gemini slow
│   ├── message_processor.py  ← Rule-based bucket pre-filter + time/entity extraction
│   ├── auth_service.py       ← OTP, JWT creation/verify, password hashing
│   ├── search_service.py     ← Two-tier search: keyword (fast) + embed+expand (slow, parallel)
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
│   ├── payment_service.py    ← Razorpay web payments (web dashboard only — not iOS)
│   ├── iap_service.py        ← Apple IAP: JWS verification, App Store Server API, webhook handler
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
| LLM calls (Gemini 2.5 Flash Lite only) | `cerebras_client.py` |
| JWT issuance + verification | `services/auth_service.py` |
| Password hashing | `services/auth_service.py` (`hash_password`, `verify_password`) |
| Current user dependency | `services/auth_service.py` (`get_current_user`) |
| **Intent classifier (ONNX fast path)** | **`services/classifier_service.py`** |
| **Intent parse orchestrator** | **`services/intent_service.py`** — routes to ONNX or Gemini |
| Rule-based time/entity extraction | `services/message_processor.py` |
| Semantic search + fuzzy text ranking | `services/search_service.py` |
| Embeddings | `services/embedding_service.py` |
| Redis caching | `services/redis_cache.py` |
| Pro accounts / groups | `services/group_service.py` |
| Reminder scheduling | `services/reminder_service.py` |
| APNs push delivery | `services/reminder_service.py` |
| **Annotation storage (retraining data)** | **`database.py` → `LabelAnnotation`** |
| **Write annotation on bucket move** | **`main.py` → `PATCH /api/messages/{id}/bucket`** |
| **Export annotations for retraining** | **`main.py` → `GET /api/annotations/export`** |
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
| PATCH | `/api/messages/{id}/assign` | Retroactively assign a group To-Do to a member; body: `{user_id, name, phone}`; updates `assigned_to_user_id` + `tags.assignments`, mirrors To-Do in assignee feed, sends APNs; cannot self-assign; 409 if already assigned to same member |
| GET | `/api/messages/detail/{id}` | Single message detail |
| PATCH | `/api/messages/{id}/done` | Mark task done/undone `{"done": bool}` |
| PATCH | `/api/messages/{id}/content` | Edit message text `{"content": str}` — updates `content` + `summary`; busts bootstrap cache |
| PATCH | `/api/messages/{id}/items/{idx}/complete` | Check off a list item |
| DELETE | `/api/messages/{id}` | Delete message |

### Search & Upload
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/search` | Two-tier search: `fast=true` = keyword-only (~5ms); `fast=false` = embed + optional LLM expand in parallel |
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
| GET | `/api/groups` | List user's groups (includes `member_count`, `max_members`, `invite_token`) |
| POST | `/api/groups` | Create group (Pro required) — mints `invite_token`, returns it |
| GET | `/api/groups/{id}/messages` | Group message feed |
| GET | `/api/groups/{id}/members` | Group members |
| POST | `/api/groups/{id}/members` | **Owner/admin only** — add existing user by `user_id` (per-group cap; 409 if full). Invitee needs no Pro |
| POST | `/api/groups/{id}/invite` | **Owner/admin only** — add existing user by phone. `user_exists=false` → tell client to share the link |
| GET | `/api/groups/{id}/invite-link` | **Owner/admin only** — shareable join link `https://www.extendedmindsai.com/join/{token}` |
| POST | `/api/groups/join/{token}` | Join via invite link — **no Pro required**, any signed-in user (per-group cap) |
| DELETE | `/api/groups/{id}/leave` | Leave group |
| DELETE | `/api/groups/{id}` | Delete group |
| POST | `/api/groups/{id}/seen` | Mark all group messages as read |
| GET | `/api/groups/unread` | Unread count per group `{group_id: count}` |
| WS | `/ws/group/{id}` | WebSocket: live incoming group messages |

### Pro Account
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/pro/status` | Pro subscription info (still returns a `members` array for back-compat) |
| POST | `/api/pro/validate-coupon` | Check if coupon code is valid |
| POST | `/api/pro/redeem-coupon` | Apply coupon to account |
| ~~POST~~ | ~~`/api/pro/invite`~~ | ⚠️ **Defunct** — account-roster invite. Left for back-compat; not used. Add members at the group level instead |
| ~~GET~~ | ~~`/api/pro/my-invites`~~ | ⚠️ **Defunct** roster path |
| ~~POST~~ | ~~`/api/pro/accept-invite`~~ | ⚠️ **Defunct** roster path |
| ~~DELETE~~ | ~~`/api/pro/members/{phone}`~~ | ⚠️ **Defunct** roster path |

### Payments — Razorpay (web dashboard only)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/payments/create-order` | Create Razorpay order for web checkout |
| POST | `/api/payments/verify` | Verify Razorpay signature + activate Pro |

### Payments — Apple IAP (iOS app)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/iap/verify` | Verify StoreKit 2 transaction with Apple + activate Pro (auth required) |

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
| POST | `/webhook/razorpay` | Razorpay payment events (web only) |
| POST | `/webhook/apple` | App Store Server Notifications V2 — renewal, expiry, refund |

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
- `content` (TEXT), `message_type` (enum: text/image/audio/document/video) — `"link"` is **not** a valid value
- `media_url` (nullable), `summary` (nullable)
- `tags` (JSONB): `{buckets, primary_bucket, priority, due_date, event_time, done, reminded_at, original_dump, split_from, assignments, group_reminder, expense_amount, expense_category, expense_payer_id, expense_payer_name, ...}`
  - `assignments`: `[{user_id, name, phone, done, done_at}]` — per-assignee completion slots for group @mention tasks
  - `group_reminder`: `true` — set when a group message has a due time but no specific @mention (fires APNs to all members)
  - `original_dump`: original capture text (for retrieval when LLM paraphrases single tasks)
  - `split_from`: original multi-task dump text (for retrieval of split tasks)
  - `expense_amount`: `float` — set by the Spent chip; forces bucket = Track
  - `expense_category`: `string` — one of the 12 bank-style categories (Food & Dining, Transport, etc.)
  - `expense_payer_id`: `int` — user ID of who paid; defaults to capturing user when not supplied
  - `expense_payer_name`: `string` — display name of payer; used in group expense summary and payer chip
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

**Membership model (WhatsApp-style, since 2026-06-13):** groups are self-contained. **Creating** a group requires Pro. Only the **group owner (admin/creator — the Pro user)** adds members — by phone, or by sharing the group's invite link. Everyone else just participates; **joining via a shared link is free** (no Pro). There is **no account-wide roster** and **no global member cap** — the cap is **per-group** (`groups.max_members`, default 10). The parent `ProAccount` is now only a billing/ownership anchor.

**`pro_accounts`** — `owner_id` (FK→users, unique), `plan_type`, `max_members` (**dead — cap is now per-group**), `expires_at`

**`pro_account_members`** — `account_id`, `user_id` (nullable), `phone_number`, `invite_token`, `status` (pending/active), `invited_at`, `joined_at`. ⚠️ **Defunct** — the old account roster. Table kept (account-deletion logic still references it) but nothing gates group access on it anymore.

**`groups`** — `id`, `account_id` (FK→pro_accounts), `name`, `description`, `emoji`, `invite_token` (unique — the shareable join link), `max_members` (default 10), `created_by`, `created_at`

**`group_members`** — `group_id`, `user_id`, `role`, `joined_at`

**`group_last_seen`** — `(user_id, group_id)` unique, `last_seen_at` — drives unread counts

### Other Tables

**`coupon_codes`** — `code`, `discount_type` (free/percent/fixed), `discount_value`, `duration_days`, `max_uses`, `expires_at`, `is_active`

**`coupon_redemptions`** — `coupon_id`, `user_id`, `redeemed_at`

**`payment_orders`** — `razorpay_order_id` (unique), `razorpay_payment_id`, `user_id`, `plan` (monthly/annual), `amount` (paise), `status` (created/paid/failed) — Razorpay web payments only

**`iap_transactions`** — `transaction_id` (unique), `original_transaction_id` (indexed), `user_id`, `product_id`, `environment` (Production/Sandbox), `expires_at` — links Apple transaction IDs to backend users for webhook resolution

**`stored_images`** — `user_id`, `data` (LargeBinary), `mime_type` — fallback for images without CDN URL

### Annotation Table (Classifier Flywheel)

**`label_annotations`** — `id`, `user_id` (FK→users), `message_id` (FK→messages, nullable), `text` (TEXT), `label` (e.g. `"To-Do"`), `source` (default `"user_correction"`), `created_at`

Written every time a user moves a message to a different bucket via `PATCH /api/messages/{id}/bucket`. Exported via `GET /api/annotations/export` and consumed by `retrain.py` in the POC repo to improve the ONNX classifier over time. Annotations always win over base training data on exact-text matches.

---

## Environment Variables

```bash
# Required
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
SECRET_KEY=<long-random-string>           # JWT signing — MUST override default
GEMINI_API_KEY=AIzaS...                    # LLM (2.5 Flash Lite paid) + embeddings — REQUIRED

# Messaging (one platform active at a time)
MESSAGING_PLATFORM=telegram                # "telegram" | "whatsapp"
TELEGRAM_BOT_TOKEN=123456:ABCDef...
TELEGRAM_WEBHOOK_URL=https://<railway-domain>/webhook/telegram
# OR for WhatsApp:
WHATSAPP_ACCESS_TOKEN=...
WHATSAPP_PHONE_NUMBER_ID=...
WHATSAPP_VERIFY_TOKEN=...

# Optional but recommended
CEREBRAS_API_KEY=csk-...                   # ⚠️ Not used — Cerebras removed; key can be omitted
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...

# iOS Push Notifications
APNS_KEY_ID=XXXXX
APNS_TEAM_ID=ABCDE
APNS_AUTH_KEY=-----BEGIN PRIVATE KEY-----...
APNS_BUNDLE_ID=com.extendedminds.app
APNS_PRODUCTION=true                       # false = sandbox

# Razorpay (web dashboard payments only — NOT used in iOS app)
RAZORPAY_KEY_ID=rzp_live_xxx
RAZORPAY_KEY_SECRET=xxx
RAZORPAY_WEBHOOK_SECRET=xxx

# Apple IAP — App Store Server API (iOS subscriptions)
# Keys from App Store Connect → Users and Access → Integrations → In-App Purchase
APPLE_ISSUER_ID=<UUID from App Store Connect → Keys → Issuer ID>
APPLE_KEY_ID=<Key ID of the In-App Purchase .p8 key>
APPLE_PRIVATE_KEY=<contents of .p8 file, \n-escaped for Railway>

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

### Intent Classification — Two-Path Architecture

Every captured message goes through `intent_service.parse()`:

```
Incoming message text
       │
       ▼
classifier_service.classify(text)        ← ONNX, ~10ms, no network
       │
  confidence ≥ 0.50?
  ├── YES → _build_from_bucket(text, bucket)   ← regex for time/date/reminder
  │             return result + _classifier_confidence
  │
  └── NO  → _llm_parse(text)                  ← Gemini Flash Lite, ~500ms
                return full LLM result
```

**Fast path** (`services/classifier_service.py`):
- Loads `models/intent_classifier/backbone.onnx` + `head_weights.npz` + `input_prefix.txt` at startup via `lifespan()`
- Tokenizes with `AutoTokenizer` (all-MiniLM-L6-v2 v4), runs ONNX inference, applies sklearn LogisticRegression head
- **v4 accuracy:** 89.7% overall · To-Do recall 94.1% · Events recall 95.5% (722 training examples, GPU contrastive fine-tuning)
- **Deploy:** backbone.onnx published as GitHub release asset; Railway pulls via `ONNX_MODEL_URL` env var
- Returns `(bucket: str, confidence: float)`; threshold `CONF_THRESHOLD = 0.50`
- If model files are absent, `is_ready` stays `False` → falls through to Gemini silently

**`_build_from_bucket()`** in `intent_service.py`:
- Constructs the full actions dict from the classifier bucket + regex-extracted time/date/reminder flags
- Bypasses LLM entirely for routine captures; still runs rule-based extraction for `event_time`, `due_date`, `is_reminder`
- **Also detects list format for ALL buckets** (not just To-Do): named header + bullets/colon/bare-lines/inline-comma → `save_as_list=True`. This is bucket-agnostic — "Places to visit in Goa:\n- Baga\n- Anjuna" → Remember bucket + is_list=True.
- Neutral headers (todo/tasks/today/tomorrow) + To-Do → split to individual task rows, NOT a list.

**List format detection in `_build_from_bucket()` — four signals (priority order):**
1. Bullets/numbered lines (`\n- item` or `\n1. item`) — highest confidence
2. Colon at end of header line (`Header:\nitem1\nitem2`)
3. Bare short lines ≤5 words each (requires named header — filters prose)
4. Single-line inline (`Header: item1, item2, item3`) — 2+ items required

**Slow path** (Gemini 2.5 Flash Lite, paid):
- `CerebrasClient(provider="gemini", model="gemini-2.5-flash-lite")` — default provider and model
- Full structured prompt → JSON response with bucket, summary, entities, time, reminder flag
- Retry: 3 attempts, exponential backoff on 429/500/503

### Search Pipeline — Two-Tier Architecture

iOS makes two calls per search query:

**Tier 1 — `fast=True` (~5ms, shown immediately):**
- Keyword ILIKE match only — no embedding, no LLM
- Returns the top text-match result instantly

**Tiers 2+3 — `fast=False` (~300ms, replaces tier-1 result):**
- Tier 2: `embed(query)` → pgvector cosine similarity — starts immediately
- Tier 3: LLM `_expand_query()` → richer keywords — starts in parallel with Tier 2, skipped if ≤3 words
- Both run concurrently via `asyncio.create_task()`; results merged and ranked

**Scoring (`_score()` in `search_service.py`):**
- Primary text: `rapidfuzz.token_set_ratio(query, content) × 0.60` (max +60)
  - Handles word order, partial words, minor typos — "people call" = "people to call" = 100
- Semantic: `embedding_similarity × 15.0` (max +15, secondary to text)
- Secondary fields: subtasks ×0.15, split_from ×0.12, original_dump ×0.12
- Expansion terms, entities, bucket hints, due-date, recency: unchanged
- `natural_response` is always `""` (LLM summary generation commented out, reserved for future)

### Other LLM Use Cases — all use Gemini 2.5 Flash Lite (paid)
1. Reminder + recurrence temporal parsing (`recurrence_service.parse_temporal`) — one LLM call extracts time, date, recurrence rule, multi-day patterns
2. Search query expansion for long queries >3 words (`search_service._expand_query`)
3. Morning briefing generation (`briefing_service`)
4. Category suggestions (`category_manager`)
5. Subtask breakdown (`subtask_service`)

### Embeddings
Gemini `text-embedding-004` via REST, 1536 dims, stored in pgvector column on `messages`.

### Rule-Based Fallback (LLM completely down)
`message_processor._sniff_buckets_fast()` + `_extract_time_mention()` extract time/date/reminder signals from regex even when both paths fail. `_full_analysis()` promotes `fast_time`/`fast_date` into the response when the LLM returns nothing.

---

## Things to Always Do

- Use `async with get_db() as db:` (or the FastAPI `Depends(get_db)` pattern) — never create a session manually.
- Run through `cerebras_client` for all LLM calls — it handles retries and uses Gemini 2.5 Flash Lite (paid). Never instantiate `CerebrasClient` without `provider="gemini"`.
- Invalidate the relevant Redis cache keys when mutating messages or groups.
- Respect the 6 bucket names exactly: `"Remember"`, `"To-Do"`, `"Ideas"`, `"Track"`, `"Events"`, `"Random"`. `"List"` is **not a bucket** — it is a format flag (`tags.is_list=true`).
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
