"""
Monkey-patch toolshield to use LiteLLM instead of the hardcoded OpenRouter client.

Import this module **before** any other toolshield imports to ensure the patch is
applied. It does two things:

1. Sets a dummy OPENROUTER_API_KEY so toolshield's top-level OpenAI client creation
   doesn't crash (it reads the env var eagerly at import time).
2. Replaces ``toolshield.tree_generation.client`` with a thin LiteLLM adapter so
   all ``client.chat.completions.create(...)`` calls are forwarded to
   ``litellm.completion(...)`` — no code changes needed inside toolshield itself.

This keeps the installed toolshield package unmodified.

Usage (at the top of any Baseline script that uses toolshield)::

    import Baseline.toolshield_patch  # noqa: F401  — must be first
    from toolshield.tree_generation import run_generation
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Step 1: Ensure OPENROUTER_API_KEY exists so toolshield's eager OpenAI(...)
# constructor doesn't raise. The key value doesn't matter — we replace the
# client immediately after import.
# ---------------------------------------------------------------------------
_had_key = "OPENROUTER_API_KEY" in os.environ
if not _had_key:
    os.environ["OPENROUTER_API_KEY"] = "unused-placeholder"

# ---------------------------------------------------------------------------
# Step 2: Import toolshield (this triggers the OpenAI client creation with
# the dummy key above).
# ---------------------------------------------------------------------------
from toolshield import tree_generation  # noqa: E402

# ---------------------------------------------------------------------------
# Step 3: Replace the client with a LiteLLM adapter.
# ---------------------------------------------------------------------------
try:
    import litellm

    class _LiteLLMCompletions:
        """Adapter: forward ``client.chat.completions.create(...)`` to
        ``litellm.completion(...)``."""

        @staticmethod
        def create(**kwargs):
            kwargs.pop("timeout", None)
            return litellm.completion(**kwargs)

    class _LiteLLMChat:
        completions = _LiteLLMCompletions()

    class _LiteLLMClient:
        chat = _LiteLLMChat()

    tree_generation.client = _LiteLLMClient()

except ImportError:
    print(
        "Warning: litellm not found. ToolShield will fall back to its "
        "default OpenRouter client. Install with: pip install litellm"
    )

# ---------------------------------------------------------------------------
# Step 4: Clean up the dummy env var if we set it.
# ---------------------------------------------------------------------------
if not _had_key:
    os.environ.pop("OPENROUTER_API_KEY", None)
