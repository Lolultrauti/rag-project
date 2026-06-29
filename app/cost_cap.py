"""
cost_cap.py  --  hard daily ceiling on /query calls.

Rate limiting (slowapi, per-IP) throttles any single client, but it does not
bound *total* spend: an attacker rotating IPs, or just a viral spike of
distinct users, can still drive the Gemini bill arbitrarily high. This is the
real backstop -- one global counter that, once it hits the cap, stops serving
LLM-backed requests for the rest of the UTC day regardless of who's asking.

Implementation tradeoff (in-memory, documented on purpose):
We keep the counter in process memory rather than in Postgres. Pros: zero DB
round-trip on the hot path, dead simple. Con: the count resets to zero on app
restart/redeploy, so a crash-loop or frequent deploys could let total usage
exceed the nominal daily cap. For a single-process portfolio demo on Railway
that's an accepted tradeoff. If this ever needs to be exact across restarts or
multiple workers, move the counter into a Postgres row keyed by UTC date
(date UNIQUE, count) and increment with an UPSERT -- the call site below would
not have to change.

Concurrency: the API runs request handlers across threadpool workers, so the
counter is guarded by a Lock to keep check-and-increment atomic.
"""

import threading
from datetime import datetime, timezone

from app.config import settings

_lock = threading.Lock()
_state = {"date": None, "count": 0}


def check_and_increment() -> bool:
    """
    Atomically: roll the counter over if the UTC day changed, then either
    consume one unit of budget (return True) or report the cap is reached
    (return False) without consuming.

    Must be called BEFORE doing any paid embedding/generation work so we never
    spend budget we've already decided we're out of.
    """
    today = datetime.now(timezone.utc).date()
    with _lock:
        if _state["date"] != today:
            _state["date"] = today
            _state["count"] = 0
        if _state["count"] >= settings.daily_cost_cap:
            return False
        _state["count"] += 1
        return True
