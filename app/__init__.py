# AI Lesson Builder Service
import os

from dotenv import load_dotenv

# Must stay in the package __init__: it is the only module guaranteed to run
# before any other `app.*` import, and several of them read os.getenv() (or
# instantiate Settings) at module level - run.py imports app.config before
# app.main, so loading from an entry point is too late.
#
# Locally .env is the source of truth and overrides stale shell exports.
# Production ships no .env file and injects config as real env vars (compose
# env_file:/environment:, Dockerfile ENV); those must never be clobbered.
_LOCAL_ENVS = {"dev", "development", "local", "test"}
_is_local = os.getenv("ENVIRONMENT", "dev").strip().lower() in _LOCAL_ENVS

load_dotenv(override=_is_local)
