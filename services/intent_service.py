"""
Intent Service — Multi-Action Parser
─────────────────────────────────────────────────────────────
A single message can trigger MULTIPLE actions simultaneously.

Examples:
  "remind me to call mom at 10pm"
    → save_as_todo: true   (it's a task)
    → set_reminder: true   (it needs a reminder)
    → NOT two separate intents — one message, two actions

  "dentist appointment Friday 3pm"
    → save_as_todo: true   (something to act on)
    → save_as_event: true  (it's a scheduled event)
    → set_reminder: true   (remind before it happens)

  "dmart shopping:\n- milk\n- eggs"
    → save_as_list: true   (named list with items)
    → save_as_todo: false  (not a task)
    → set_reminder: false

  "weight 74kg, slept 7 hours"
    → save_as_track: true
    → save_as_todo: false

Output schema — always returns all action flags:
{
  "actions": {
    "save_as_todo":  bool,   # save in To-Do bucket
    "save_as_event": bool,   # save in Events bucket  
    "save_as_note":  bool,   # save in Remember bucket
    "save_as_idea":  bool,   # save in Ideas bucket
    "save_as_track": bool,   # save in Track bucket
    "save_as_list":  bool,   # save as named list
    "set_reminder":  bool,   # create a reminder
    "is_query":      bool,   # user wants to retrieve something
  },
  "tasks": [                 # present when save_as_todo=true
    {
      "task": "clean description",
      "due_date": "YYYY-MM-DD",
      "time": "HH:MM or null",
      "priority": "normal|high|urgent"
    }
  ],
  "reminder": {              # present when set_reminder=true
    "due_date": "YYYY-MM-DD",
    "time": "HH:MM",
    "priority": "normal|high|urgent"
  },
  "event": {                 # present when save_as_event=true
    "title": "short title",
    "due_date": "YYYY-MM-DD",
    "time": "HH:MM or null",
    "people": []
  },
  "list": {                  # present when save_as_list=true
    "list_name": "Dmart Shopping List",
    "list_type": "shopping|bag|packing|reading|watching|custom",
    "items": ["item1", "item2"]
  },
  "track": {                 # present when save_as_track=true
    "logs": [{"metric": "weight", "value": "74", "unit": "kg"}]
  },
  "note": {                  # present when save_as_note=true
    "content": "the fact",
    "keywords": []
  },
  "idea": {                  # present when save_as_idea=true
    "content": "the idea",
    "keywords": []
  },
  "query": {                 # present when is_query=true
    "query_text": "what they want to find",
    "date_hint": "today|tomorrow|this_week|null",
    "list_name": "if asking for a named list, else null"
  },
  "people": [],              # all people mentioned
  "priority": "normal",      # overall message priority
  "essence": "one sentence summary"
}
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from cerebras_client import CerebrasClient

IST = ZoneInfo("Asia/Kolkata")

COMMAND_PREFIXES = {
    "/start", "/link", "/status", "briefing:", "project:",
    "subtask:", "search:", "find:", "get:",
}

def _is_command(content: str) -> bool:
    lc = content.lower().strip()
    return any(lc.startswith(p) for p in COMMAND_PREFIXES)

def _is_trivial(content: str) -> bool:
    lc = content.lower().strip()
    trivials = {"hi", "hey", "heyy", "hello", "ok", "okay", "lol", "test", "hmm", "hm"}
    return len(lc) < 4 or lc in trivials


# ─────────────────────────────────────────────────────────────────────────────
# Default empty result
# ─────────────────────────────────────────────────────────────────────────────

def _empty_actions() -> Dict:
    return {
        "save_as_todo":  False,
        "save_as_event": False,
        "save_as_note":  False,
        "save_as_idea":  False,
        "save_as_track": False,
        "save_as_list":  False,
        "set_reminder":  False,
        "is_query":      False,
    }

def _default_result() -> Dict:
    return {
        "actions":  _empty_actions(),
        "tasks":    [],
        "reminder": None,
        "event":    None,
        "list":     None,
        "track":    None,
        "note":     None,
        "idea":     None,
        "query":    None,
        "people":   [],
        "priority": "normal",
        "essence":  "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Intent Service
# ─────────────────────────────────────────────────────────────────────────────

class IntentService:

    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client

    async def parse(
        self,
        content: str,
        user_name: str,
        user_timezone: str = "Asia/Kolkata",
    ) -> Dict:
        """
        Parse a message into multi-action structured output.
        Never raises — returns safe default on failure.
        """
        if _is_command(content):
            result = _default_result()
            result["_is_command"] = True
            return result

        if _is_trivial(content):
            return _default_result()

        now_local = datetime.now(ZoneInfo(user_timezone or "Asia/Kolkata"))
        today     = now_local.strftime("%Y-%m-%d")
        tomorrow  = (now_local + timedelta(days=1)).strftime("%Y-%m-%d")
        now_str   = now_local.strftime("%Y-%m-%d %H:%M (%A)")
        day_map   = {
            (now_local + timedelta(days=i+1)).strftime("%A").lower():
            (now_local + timedelta(days=i+1)).strftime("%Y-%m-%d")
            for i in range(7)
        }

        try:
            return await self._llm_parse(
                content, user_name, now_str, today, tomorrow, day_map
            )
        except Exception as e:
            print(f"[intent] LLM parse failed: {e}")
            return _default_result()

    async def _llm_parse(
        self,
        content: str,
        user_name: str,
        now_str: str,
        today: str,
        tomorrow: str,
        day_map: Dict,
    ) -> Dict:

        prompt = (
            f"You are parsing a message from {user_name} for their personal knowledge base.\n\n"
            f"NOW: {now_str}\n"
            f"TODAY: {today}\n"
            f"TOMORROW: {tomorrow}\n"
            f"NEXT 7 DAYS: {day_map}\n\n"
            f"MESSAGE:\n\"\"\"\n{content}\n\"\"\"\n\n"
            "TASK: Determine ALL actions this message should trigger. Multiple actions can be true simultaneously.\n\n"
            "ACTION FLAGS:\n"
            "  save_as_todo  — message contains one or more tasks/things to do\n"
            "  save_as_event — message is a scheduled appointment/meeting with a specific time\n"
            "  save_as_note  — message is a fact/info to remember (credential, location, info)\n"
            "  save_as_idea  — message is a creative thought/insight/concept\n"
            "  save_as_track — message logs health/habit data (weight, steps, mood, sleep)\n"
            "  save_as_list  — message is a named list (shopping, bag, packing) with items\n"
            "  set_reminder  — message has a specific time → should create a reminder\n"
            "  is_query      — user wants to retrieve/search something they saved before\n\n"
            "KEY RULES:\n"
            "  - A reminder IS ALWAYS also a todo → both save_as_todo AND set_reminder = true\n"
            "  - An event with time IS ALWAYS also a todo → both save_as_event AND save_as_todo = true\n"
            "  - set_reminder = true ONLY when a specific time is mentioned\n"
            "  - Time without explicit date → assume TODAY\n"
            "  - Extract ALL tasks even from unstructured prose\n"
            "  - Priority: urgent/asap/critical/important → high, else normal\n\n"
            "  NAMED HEADER RULE (most important for Header:\\n- items structure):\n"
            "  A header has its OWN IDENTITY if it refers to a specific named thing:\n"
            "  a project, place, brand, person, product, or feature set.\n"
            "  A header is NEUTRAL if it's just a time/category label with no specific identity.\n\n"
            "  Named header + bullet items = save_as_list=true\n"
            "    (items belong to that named thing as a collection)\n"
            "  Named header + bullet items WITH time/date = save_as_list=true AND save_as_todo=true\n"
            "    (items are both a collection AND actionable tasks)\n"
            "  Neutral header + bullet items = save_as_todo=true (NOT save_as_list)\n\n"
            "  NAMED headers (have own identity) → save_as_list:\n"
            "    'Extended minds changes', 'Japan trip', 'Grocery', 'Dmart',\n"
            "    'House renovation', 'Birthday party', 'Project X', 'Client ABC'\n"
            "  NEUTRAL headers (time/category labels) → save_as_todo:\n"
            "    'Todo', 'Tasks', 'Today', 'Tomorrow', 'This week',\n"
            "    'Office tasks', 'Urgent', 'Things to do', 'New todo list'\n\n"
            "Return ONLY this JSON (no markdown):\n"
            "{\n"
            '  "actions": {\n'
            '    "save_as_todo": bool,\n'
            '    "save_as_event": bool,\n'
            '    "save_as_note": bool,\n'
            '    "save_as_idea": bool,\n'
            '    "save_as_track": bool,\n'
            '    "save_as_list": bool,\n'
            '    "set_reminder": bool,\n'
            '    "is_query": bool\n'
            "  },\n"
            '  "tasks": [{"task": "...", "due_date": "YYYY-MM-DD", "time": "HH:MM|null", "priority": "normal|high|urgent"}],\n'
            '  "reminder": {"due_date": "YYYY-MM-DD", "time": "HH:MM", "priority": "normal"} | null,\n'
            '  "event": {"title": "...", "due_date": "YYYY-MM-DD", "time": "HH:MM|null", "people": []} | null,\n'
            '  "list": {"list_name": "...", "list_type": "shopping|bag|packing|reading|watching|custom", "items": []} | null,\n'
            '  "track": {"logs": [{"metric": "...", "value": "...", "unit": "..."}]} | null,\n'
            '  "note": {"content": "...", "keywords": []} | null,\n'
            '  "idea": {"content": "...", "keywords": []} | null,\n'
            '  "query": {"query_text": "...", "date_hint": "today|tomorrow|this_week|null", "list_name": "...|null"} | null,\n'
            '  "people": [],\n'
            '  "priority": "normal|high|urgent",\n'
            '  "essence": "one sentence summary"\n'
            "}\n\n"
            "EXAMPLES:\n\n"

            # ── Reminder ──────────────────────────────────────────────────
            f'Message: "remind me to call mom at 10pm"\n'
            f'→ actions: save_as_todo=true, set_reminder=true\n'
            f'  tasks: [{{"task":"call mom","due_date":"{today}","time":"22:00","priority":"normal"}}]\n'
            f'  reminder: {{"due_date":"{today}","time":"22:00","priority":"normal"}}\n\n'

            # ── Event ─────────────────────────────────────────────────────
            f'Message: "dentist appointment Friday 3pm"\n'
            f'→ actions: save_as_todo=true, save_as_event=true, set_reminder=true\n'
            f'  tasks: [{{"task":"dentist appointment","due_date":"{day_map.get("friday", tomorrow)}","time":"15:00","priority":"normal"}}]\n'
            f'  event: {{"title":"dentist appointment","due_date":"{day_map.get("friday", tomorrow)}","time":"15:00","people":[]}}\n'
            f'  reminder: {{"due_date":"{day_map.get("friday", tomorrow)}","time":"15:00","priority":"normal"}}\n\n'

            # ── Neutral header → Todo (NOT list) ──────────────────────────
            f'Message: "New todo list for tomorrow:\\n- Work on meraki 4hrs\\n- Follow up with Aditi\\n- Order clothes"\n'
            f'→ actions: save_as_todo=true (NOT save_as_list)\n'
            f'  reasoning: "New todo list for tomorrow" is a NEUTRAL label — time reference, no specific identity\n'
            f'  tasks: [{{"task":"Work on meraki matter for 4 hours","due_date":"{tomorrow}","time":null,"priority":"normal"}},{{"task":"Follow up with Aditi","due_date":"{tomorrow}","time":null,"priority":"normal"}},{{"task":"Order clothes","due_date":"{tomorrow}","time":null,"priority":"normal"}}]\n\n'

            # ── Neutral header with times → Todo ──────────────────────────
            f'Message: "My todo for today: - do x task at 3 pm, -do y task at 7 pm"\n'
            f'→ actions: save_as_todo=true, set_reminder=true\n'
            f'  reasoning: "My todo for today" is a NEUTRAL time label\n'
            f'  tasks: [{{"task":"do x task","due_date":"{today}","time":"15:00","priority":"normal"}},{{"task":"do y task","due_date":"{today}","time":"19:00","priority":"normal"}}]\n'
            f'  reminder: null (multiple tasks each have own time)\n\n'

            # ── Named header → List (shopping) ────────────────────────────
            f'Message: "dmart shopping:\\n- Pen\\n- coffee mug\\n- glue"\n'
            f'→ actions: save_as_list=true\n'
            f'  reasoning: "dmart shopping" is a NAMED thing — specific store identity\n'
            f'  list: {{"list_name":"Dmart Shopping List","list_type":"shopping","items":["Pen","coffee mug","glue"]}}\n\n'

            # ── Named header → List (custom/project) ──────────────────────
            f'Message: "Extended minds changes:\\n- Image with space\\n- improve search message"\n'
            f'→ actions: save_as_list=true\n'
            f'  reasoning: "Extended minds changes" is a NAMED project — has its own identity\n'
            f'  list: {{"list_name":"Extended Minds Changes List","list_type":"custom","items":["Image with space","improve search message"]}}\n\n'

            # ── Named header + actionable items with dates → List AND Todo ─
            f'Message: "Japan trip:\\n- book flights\\n- get visa\\n- pack bags"\n'
            f'→ actions: save_as_list=true, save_as_todo=true\n'
            f'  reasoning: "Japan trip" is a NAMED thing AND items are also actionable tasks\n'
            f'  list: {{"list_name":"Japan Trip List","list_type":"packing","items":["book flights","get visa","pack bags"]}}\n'
            f'  tasks: [{{"task":"book flights","due_date":null,"time":null,"priority":"normal"}},{{"task":"get visa","due_date":null,"time":null,"priority":"normal"}},{{"task":"pack bags","due_date":null,"time":null,"priority":"normal"}}]\n\n'

            # ── Named header + items with explicit time → List AND Todo ───
            f'Message: "Client ABC meeting prep:\\n- send agenda by 9am\\n- book conference room"\n'
            f'→ actions: save_as_list=true, save_as_todo=true, set_reminder=true\n'
            f'  reasoning: "Client ABC meeting prep" is a NAMED thing AND items have time ref\n'
            f'  list: {{"list_name":"Client ABC Meeting Prep List","list_type":"custom","items":["send agenda by 9am","book conference room"]}}\n'
            f'  tasks: [{{"task":"send agenda","due_date":"{today}","time":"09:00","priority":"normal"}},{{"task":"book conference room","due_date":"{today}","time":null,"priority":"normal"}}]\n'
            f'  reminder: {{"due_date":"{today}","time":"09:00","priority":"normal"}}\n\n'

            # ── Single task with time ──────────────────────────────────────
            f'Message: "pay sabziwala at 10pm"\n'
            f'→ actions: save_as_todo=true, set_reminder=true\n'
            f'  tasks: [{{"task":"pay sabziwala","due_date":"{today}","time":"22:00","priority":"normal"}}]\n'
            f'  reminder: {{"due_date":"{today}","time":"22:00","priority":"normal"}}\n\n'

            # ── Query: todo ───────────────────────────────────────────────
            f'Message: "Todo list for tomorrow?"\n'
            f'→ actions: is_query=true\n'
            f'  query: {{"query_text":"todos for tomorrow","date_hint":"tomorrow","list_name":null}}\n\n'

            # ── Query: named list ─────────────────────────────────────────
            f'Message: "Show dmart list?"\n'
            f'→ actions: is_query=true\n'
            f'  query: {{"query_text":"dmart shopping list","date_hint":null,"list_name":"Dmart Shopping List"}}\n\n'

            # ── Track ─────────────────────────────────────────────────────
            f'Message: "weight 74kg, slept 7 hours, ran 3km"\n'
            f'→ actions: save_as_track=true\n'
            f'  track: {{"logs":[{{"metric":"weight","value":"74","unit":"kg"}},{{"metric":"sleep","value":"7","unit":"hours"}},{{"metric":"run","value":"3","unit":"km"}}]}}\n\n'

            # ── Note ──────────────────────────────────────────────────────
            f'Message: "wifi password is airtel123"\n'
            f'→ actions: save_as_note=true\n'
            f'  note: {{"content":"wifi password is airtel123","keywords":["wifi","password"]}}\n\n'
        )

        response = await self.cerebras.chat(prompt, max_tokens=1000)

        return self._validate(response)

    def _validate(self, raw: Dict) -> Dict:
        """Validate and sanitise LLM output. Fill missing fields with safe defaults."""
        result = _default_result()

        # Actions
        actions_raw = raw.get("actions", {})
        if isinstance(actions_raw, dict):
            for key in result["actions"]:
                val = actions_raw.get(key, False)
                result["actions"][key] = bool(val)

        # Enforce action coherence
        # If set_reminder=true, save_as_todo must also be true
        if result["actions"]["set_reminder"]:
            result["actions"]["save_as_todo"] = True
        # If save_as_event=true, save_as_todo must also be true
        if result["actions"]["save_as_event"]:
            result["actions"]["save_as_todo"] = True

        # Tasks
        tasks_raw = raw.get("tasks", [])
        if isinstance(tasks_raw, list):
            for t in tasks_raw:
                if not isinstance(t, dict):
                    continue
                task_text = str(t.get("task", "")).strip()
                if not task_text:
                    continue
                due_date = t.get("due_date")
                if not due_date or not _DATE_RE.match(str(due_date)):
                    due_date = _today_str()
                result["tasks"].append({
                    "task":     task_text,
                    "due_date": due_date,
                    "time":     t.get("time"),
                    "priority": t.get("priority", "normal")
                    if t.get("priority") in ("normal", "high", "urgent") else "normal",
                })

        # Reminder
        rem = raw.get("reminder")
        if isinstance(rem, dict) and result["actions"]["set_reminder"]:
            due = rem.get("due_date")
            if not due or not _DATE_RE.match(str(due)):
                due = _today_str()
            result["reminder"] = {
                "due_date": due,
                "time":     rem.get("time"),
                "priority": rem.get("priority", "normal"),
            }

        # Event
        evt = raw.get("event")
        if isinstance(evt, dict) and result["actions"]["save_as_event"]:
            due = evt.get("due_date")
            if not due or not _DATE_RE.match(str(due)):
                due = _today_str()
            result["event"] = {
                "title":    str(evt.get("title", ""))[:80],
                "due_date": due,
                "time":     evt.get("time"),
                "people":   evt.get("people", []) if isinstance(evt.get("people"), list) else [],
            }

        # List
        lst = raw.get("list")
        if isinstance(lst, dict) and result["actions"]["save_as_list"]:
            items = lst.get("items", [])
            if isinstance(items, list) and items:
                result["list"] = {
                    "list_name": str(lst.get("list_name", "My List")),
                    "list_type": lst.get("list_type", "custom")
                    if lst.get("list_type") in ("shopping","bag","packing","reading","watching","custom")
                    else "custom",
                    "items": [str(i).strip() for i in items if str(i).strip()],
                }

        # Track
        trk = raw.get("track")
        if isinstance(trk, dict) and result["actions"]["save_as_track"]:
            logs = trk.get("logs", [])
            if isinstance(logs, list):
                result["track"] = {"logs": [l for l in logs if isinstance(l, dict)]}

        # Note
        note = raw.get("note")
        if isinstance(note, dict) and result["actions"]["save_as_note"]:
            result["note"] = {
                "content":  str(note.get("content", ""))[:500],
                "keywords": note.get("keywords", []) if isinstance(note.get("keywords"), list) else [],
            }

        # Idea
        idea = raw.get("idea")
        if isinstance(idea, dict) and result["actions"]["save_as_idea"]:
            result["idea"] = {
                "content":  str(idea.get("content", ""))[:500],
                "keywords": idea.get("keywords", []) if isinstance(idea.get("keywords"), list) else [],
            }

        # Query
        qry = raw.get("query")
        if isinstance(qry, dict) and result["actions"]["is_query"]:
            result["query"] = {
                "query_text": str(qry.get("query_text", ""))[:200],
                "date_hint":  qry.get("date_hint") if qry.get("date_hint") in
                              ("today", "tomorrow", "this_week", None) else None,
                "list_name":  qry.get("list_name"),
            }

        # People, priority, essence
        result["people"]   = raw.get("people", []) if isinstance(raw.get("people"), list) else []
        result["priority"] = raw.get("priority", "normal") \
            if raw.get("priority") in ("normal", "high", "urgent") else "normal"
        result["essence"]  = str(raw.get("essence", ""))[:200]

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (used in validation)
# ─────────────────────────────────────────────────────────────────────────────

import re
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _today_str() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_intent_service: Optional[IntentService] = None

def get_intent_service(cerebras_client: CerebrasClient) -> IntentService:
    global _intent_service
    if _intent_service is None:
        _intent_service = IntentService(cerebras_client)
    return _intent_service
