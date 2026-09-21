"""Paths and tunables. Everything can be overridden by environment variables."""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("AUTODIAG_HOME", Path.home() / ".autodiag"))
KB_PATH = Path(os.environ.get("AUTODIAG_KB", HOME / "kb.sqlite"))
QUEUE_DIR = Path(os.environ.get("AUTODIAG_QUEUE", HOME / "queue"))
REPORTS_DIR = Path(os.environ.get("AUTODIAG_REPORTS", HOME / "reports"))

MODEL = os.environ.get("AUTODIAG_MODEL", "claude-opus-5")
EFFORT = os.environ.get("AUTODIAG_EFFORT", "high")
MAX_TOKENS = int(os.environ.get("AUTODIAG_MAX_TOKENS", "16000"))

# Server-side refusal fallbacks need a beta header. Gateways that proxy the
# Anthropic API (LiteLLM, internal routers with model IDs like
# "azure/anthropic/claude-opus-5") often reject unknown betas; set
# AUTODIAG_FALLBACKS=0 to send a plain request.
FALLBACKS_ENABLED = os.environ.get("AUTODIAG_FALLBACKS", "1") not in ("0", "false", "no")

# Host the connectivity probe checks. Follows ANTHROPIC_BASE_URL so a gateway
# setup probes the gateway, not api.anthropic.com.
API_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# Backend: "claude" (default) or "ollama" for a local model.
BACKEND = os.environ.get("AUTODIAG_BACKEND", "claude")
# When the Claude host is unreachable and Ollama is running, use it instead of
# the rules-only report. Off by default: local output is weaker and should be
# an explicit choice.
LOCAL_FALLBACK = os.environ.get("AUTODIAG_LOCAL_FALLBACK", "0") in ("1", "true", "yes")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("AUTODIAG_LOCAL_MODEL", "qwen2.5:14b")
LOCAL_NUM_CTX = int(os.environ.get("AUTODIAG_LOCAL_NUM_CTX", "16384"))
LOCAL_TIMEOUT = float(os.environ.get("AUTODIAG_LOCAL_TIMEOUT", "600"))

# Retrieval
CHUNK_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
PER_QUERY_LIMIT = 20
TOP_K = 12
RRF_K = 60
DTC_QUERY_WEIGHT = 2.0

# Connectivity probe before the API call (seconds)
CONNECT_PROBE_TIMEOUT = 3.0

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SEED_KNOWLEDGE_DIR = REPO_ROOT / "knowledge" / "seed"
FIXTURES_DIR = REPO_ROOT / "fixtures"


def ensure_dirs() -> None:
    for p in (HOME, QUEUE_DIR, REPORTS_DIR):
        p.mkdir(parents=True, exist_ok=True)
