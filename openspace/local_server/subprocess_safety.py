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
import re
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

# W14: Injection-class vars that should be stripped from inherited parent env.
# Excludes PATH/SHELL which are needed for basic subprocess execution.
_INJECTION_ENV_VARS: FrozenSet[str] = frozenset({
    "BASH_ENV", "ENV", "CDPATH",
    "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME",
    "DEBUGINFOD_URLS", "MALLOC_CHECK_",
    "ASAN_OPTIONS", "TSAN_OPTIONS", "MSAN_OPTIONS", "UBSAN_OPTIONS",
    "RUBYLIB", "NODE_PATH", "NODE_OPTIONS",
    "PERL5LIB", "PERL5OPT",
})

# W14: Conda environment name validation (Codex CRIT — shell injection via conda_env)
# W15.2: Use fullmatch — re.match() with $ accepts trailing newline (Codex LOW)
_CONDA_ENV_RE = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9._-]*')


def validate_conda_env(conda_env: Optional[str]) -> Optional[str]:
    """Validate conda environment name against shell injection.

    Returns the validated name, or None if empty/None.
    Raises ValueError if the name contains injection characters.
    Raises TypeError if *conda_env* is not a string.
    """
    if conda_env is None:
        return None
    if not isinstance(conda_env, str):
        raise TypeError(
            f"conda_env must be str or None, got {type(conda_env).__name__}"
        )
    if not conda_env:
        return None
    if len(conda_env) > 128:
        raise ValueError(f"Conda environment name too long: {len(conda_env)} chars")
    if not _CONDA_ENV_RE.fullmatch(conda_env):
        raise ValueError(
            f"Invalid conda environment name: {conda_env!r} — "
            "must match [a-zA-Z0-9._-]+"
        )
    return conda_env


def sanitize_env(user_env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Merge *user_env* into ``os.environ`` after stripping dangerous keys.

    Returns a **new** dict (never mutates the original).  Logs a warning for
    every blocked key so that callers can audit.

    W14: Also strips injection-class vars from the inherited parent environment
    (LD_*, DYLD_*, BASH_FUNC_*, PYTHON*, BASH_ENV, etc.). PATH/SHELL are
    preserved in the base since they're needed for subprocess execution.
    """
    base = os.environ.copy()

    # W14: Scrub injection-class vars from inherited parent env (Codex HIGH)
    inherited_blocked = []
    for key in list(base.keys()):
        upper_key = key.upper()
        if (
            upper_key in _INJECTION_ENV_VARS
            or upper_key.startswith("LD_")
            or upper_key.startswith("DYLD_")
            or upper_key.startswith("BASH_FUNC_")
        ):
            del base[key]
            inherited_blocked.append(key)
    if inherited_blocked:
        logger.warning(
            "Stripped %d dangerous inherited env var(s): %s",
            len(inherited_blocked),
            ", ".join(sorted(inherited_blocked)),
        )

    if not user_env:
        return base

    blocked = []
    for key, value in user_env.items():
        upper_key = key.upper()
        if (
            upper_key in _DANGEROUS_ENV_VARS
            or upper_key.startswith("LD_")
            or upper_key.startswith("DYLD_")
            # W13.2: shell function injection (BASH_FUNC_*) and interpreter hooks
            or upper_key.startswith("BASH_FUNC_")
            or upper_key.startswith("PYTHON")  # catches PYTHONDONTWRITEBYTECODE etc.
        ):
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
ulimit -t 300 2>/dev/null || echo "[openspace:rlimit] WARN: CPU time limit not enforced" >&2
ulimit -v 2097152 2>/dev/null || echo "[openspace:rlimit] WARN: virtual memory limit not enforced" >&2
ulimit -f 512000 2>/dev/null || echo "[openspace:rlimit] WARN: file size limit not enforced" >&2
ulimit -n 1024 2>/dev/null || echo "[openspace:rlimit] WARN: file descriptor limit not enforced" >&2
ulimit -u 64 2>/dev/null || echo "[openspace:rlimit] WARN: max process limit not enforced" >&2
# --- end resource limits ---
"""

PYTHON_RLIMIT_PREAMBLE = """\
# --- OpenSpace resource limits ---
import resource as _os_rlimit
import sys as _os_sys
for _res, _lim in [
    (_os_rlimit.RLIMIT_CPU, (300, 300)),
    (_os_rlimit.RLIMIT_FSIZE, (500 * 1024 * 1024, 500 * 1024 * 1024)),
    (_os_rlimit.RLIMIT_NOFILE, (1024, 1024)),
]:
    try:
        _os_rlimit.setrlimit(_res, _lim)
    except (ValueError, OSError) as _os_e:
        print(f"[openspace:rlimit] WARN: setrlimit({_res}) failed: {_os_e}", file=_os_sys.stderr)
# RLIMIT_AS and RLIMIT_NPROC are unsupported on macOS — only set on Linux
if _os_sys.platform == "linux":
    for _res, _lim, _name in [
        (_os_rlimit.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3), "RLIMIT_AS"),
        (_os_rlimit.RLIMIT_NPROC, (64, 64), "RLIMIT_NPROC"),
    ]:
        try:
            _os_rlimit.setrlimit(_res, _lim)
        except (ValueError, OSError) as _os_e:
            print(f"[openspace:rlimit] WARN: setrlimit({_name}) failed: {_os_e}", file=_os_sys.stderr)
    del _name
del _os_rlimit, _os_sys, _res, _lim
# --- end resource limits ---
"""
