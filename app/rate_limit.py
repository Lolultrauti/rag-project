"""
rate_limit.py  --  shared slowapi limiter instance.

The Limiter lives in its own module so both the route layer (which decorates
/query with it) and the app factory in main.py (which registers it on
app.state and wires the 429 handler) import the SAME instance, without a
circular import between routes.py and main.py.

Keyed by client IP (get_remote_address): the limit is "per IP" per the
public-deploy spec. Behind Railway's proxy the client IP arrives via
X-Forwarded-For; slowapi's get_remote_address reads it from the ASGI scope.
The actual limit value comes from settings.rate_limit so it's tunable via
environment, not hardcoded.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
