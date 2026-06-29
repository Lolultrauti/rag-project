"""
config.py  --  single source of truth for all configuration.

Before this existed, every module called load_dotenv() and os.getenv()
independently. That's fragile: a typo'd env var name fails silently (returns
None) deep inside whichever module reads it, and there's no one place to see
what the system actually needs to run. pydantic-settings fixes both:

  - It validates on startup. A missing required var (DATABASE_URL,
    GEMINI_API_KEY) raises immediately and loudly, not 50 requests later.
  - It documents the full config surface in one typed class.
  - The operational knobs that Phase 1 adds (rate limit, daily cap) live here
    as real settings, so they're tunable via environment on the deploy
    platform instead of being hardcoded magic numbers in business logic.

Import the singleton `settings` instance; do not instantiate Settings() again
elsewhere.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_file is read for local dev; in production (Railway) the same names
    # are provided as real environment variables, which take precedence.
    # Env var matching is case-insensitive, so DATABASE_URL maps to
    # database_url below. extra="ignore" lets the .env hold vars we don't
    # model here (e.g. POSTGRES_USER) without erroring.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- required: no safe default, the app cannot run without these ---
    database_url: str
    gemini_api_key: str

    # --- operational guards (Phase 1), overridable via environment ---
    # slowapi limit string, "<count>/<period>" form.
    rate_limit: str = "10/minute"
    # Hard ceiling on /query calls per UTC day -- the real cost backstop.
    daily_cost_cap: int = 300


# Single shared instance. Importing modules use `from app.config import settings`.
settings = Settings()
