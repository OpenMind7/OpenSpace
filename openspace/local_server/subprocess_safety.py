"""Subprocess safety utilities for the local server and local connector.

Provides:
  - Environment variable sanitization (blocks LD_PRELOAD, DYLD_*, PATH hijack)
  - Secure temporary file creation (mkstemp with restricted permissions)
  - Resource limit preambles shared between execution paths

This module centralises subprocess hardening so that both
``local_server/main.py`` (HTTP mode) and ``local_connector.py`` (direct mode)
use identical protections.
"""

from __future__ import annotations

import os
import stat
import tempfile
from typing import Dict, FrozenSet, Optional, Tuple

from openspace.utils.logging import Logger

logger = Logger.get_logger(__name__)


# ── Dangerous environment variables ──────────────────────────────────────
# These can hijack child process execution via library injection, PATH
# manipulation, or Python import hooks.  User-provided env dicts MUST be
# stripped of these before passing to subprocess.

_DANGEROUS_ENV_VARS: FrozenSet[str] = frozenset({
    # Library injection (Linux)
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "LD_PROFILE",
    # Library injection (macOS)
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    # Python import hijack
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONHOME",
    # Execution hijack
    "PATH",
    "SHELL",
    "BASH_ENV",
    "ENV",
    "CDPATH",
    # Debugger / instrumentation injection
    "DEBUGINFOD_URLS",
    "MALLOC_CHECK_",
    "ASAN_OPTIONS",
    "TSAN_OPTIONS",
    "MSAN_OPTIONS",
    "UBSAN_OPTIONS",
    # Ruby / Node / Perl equivalents (defense in depth)
    "RUBYLIB",
    "NODE_PATH",
    "NODE_OPTIONS",
    "PERL5LIB",
    "PERL5OPT",
})


def sanitize_env(user_env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Merge *user_env* into ``os.environ`` after stripping dangerous keys.

    Returns a **new** dict (never mutates the original).  Logs a warning for
    every blocked key so that callers can audit.
    """
    base = os.environ.copy()
    if not user_env:
        return base

    blocked = []
    for key, value in user_env.items():
        upper_key = key.upper()
        if upper_key in _DANGEROUS_ENV_VARS or upper_key.startswith("LD_") or upper_key.startswith("DYLD_"):
            blocked.append(key)
        else:
            base[key] = value

    if blocked:
        logger.warning(
            "Blocked %d dangerous env var(s) from user request: %s",
            len(blocked),
            ", ".join(sorted(blocked)),
        )

    return base


# ── Secure temporary files ───────────────────────────────────────────────

def create_secure_temp_file(
    suffix: str = ".py",
    prefix: str = "openspace_exec_",
) -> Tuple[int, str]:
    """Create a temp file with restricted permissions (owner-only rwx).

    Uses ``tempfile.mkstemp`` (O_EXCL under the hood) to prevent symlink
    attacks and race conditions that manual ``/tmp/<uuid>.py`` patterns are
    vulnerable to.

    Returns:
        (fd, path) — caller is responsible for closing *fd* and unlinking *path*.
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    # Restrict to owner read+write+execute (scripts need +x)
    os.fchmod(fd, stat.S_IRWXU)
    return fd, path


# ── Resource limit preambles ─────────────────────────────────────────────
# Injected at the top of temp scripts to constrain child processes.
# These are the SINGLE SOURCE OF TRUTH — local_connector.py re-exports them.

BASH_RLIMIT_PREAMBLE = """\
# --- OpenSpace resource limits ---
ulimit -t 300 2>/dev/null    # CPU time (seconds)
ulimit -v 2097152 2>/dev/null  # Virtual memory (KB) — 2 GB
ulimit -f 512000 2>/dev/null   # Max file size (KB blocks) — 500 MB
ulimit -n 1024 2>/dev/null     # Open file descriptors
# --- end resource limits ---
"""

PYTHON_RLIMIT_PREAMBLE = """\
# --- OpenSpace resource limits ---
import resource as _os_rlimit
for _res, _lim in [
    (_os_rlimit.RLIMIT_CPU, (300, 300)),
    (_os_rlimit.RLIMIT_FSIZE, (500 * 1024 * 1024, 500 * 1024 * 1024)),
    (_os_rlimit.RLIMIT_NOFILE, (1024, 1024)),
]:
    try:
        _os_rlimit.setrlimit(_res, _lim)
    except (ValueError, OSError):
        pass
# RLIMIT_AS is unsupported on macOS — only set on Linux
import sys as _os_sys
if _os_sys.platform == "linux":
    try:
        _os_rlimit.setrlimit(_os_rlimit.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    except (ValueError, OSError):
        pass
del _os_rlimit, _os_sys, _res, _lim
# --- end resource limits ---
"""
