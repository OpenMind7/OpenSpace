"""Shared utility functions for the skill engine.

Provides:
  - YAML frontmatter parsing/manipulation (unified across registry, evolver, etc.)
  - LLM output cleaning (markdown fence stripping, change summary extraction)
  - Skill content safety checking (regex-based moderation)
  - Skill directory validation
  - Text truncation
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from openspace.utils.logging import Logger

logger = Logger.get_logger(__name__)

SKILL_FILENAME = "SKILL.md"

# File extensions scanned for safety violations in multi-file skills.
# Executables, scripts, and markup that could contain injected payloads.
_SCANNABLE_EXTENSIONS: frozenset = frozenset({
    ".py", ".pyw", ".sh", ".bash", ".zsh", ".js", ".jsx", ".ts", ".tsx", ".md",
})

_SAFETY_RULES = [
    ("blocked.malware",         re.compile(r"(ClawdAuthenticatorTool)", re.IGNORECASE)),
    # Shell injection: detect patterns that execute arbitrary shell commands
    ("blocked.shell_injection", re.compile(
        r"(?:"
        r"(?:bash|sh|zsh)\s+-c\s+"          # bash -c "..."
        r"|subprocess\.(?:call|run|Popen)"   # Python subprocess
        r"|os\.(?:system|popen|exec[lv]?p?)" # os.system / os.popen / os.exec*
        r"|\beval\s*\([^)]*(?:input|argv|arg|param|request|query)"  # eval(user_input)
        r"|\bexec\s*\([^)]*(?:input|argv|arg|param|request|query)"  # exec(user_input)
        r"|__import__\s*\(\s*['\"](?:subprocess|shutil|ctypes)"     # __import__('subprocess')
        r")",
        re.IGNORECASE,
    )),
    # Credential exfiltration: detect attempts to read secrets or env vars
    ("blocked.credential_exfil", re.compile(
        r"(?:"
        r"os\.environ\s*\[.*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)"  # os.environ["API_KEY"]
        r"|open\s*\(.*(?:\.ssh|\.aws|\.gnupg|credentials|shadow|passwd)"  # open("~/.ssh/...")
        r"|(?:curl|wget|fetch|requests\.(?:get|post))\s*.*(?:webhook|exfil|ngrok|burp)"  # exfiltration via HTTP
        r"|base64\.b64encode.*(?:key|token|secret|password)"  # base64 encode secrets
        r")",
        re.IGNORECASE,
    )),
    ("suspicious.keyword",      re.compile(r"(malware|stealer|phish|phishing|keylogger)", re.IGNORECASE)),
    ("suspicious.secrets",      re.compile(r"(api[-_ ]?key|token|password|private key|secret)", re.IGNORECASE)),
    ("suspicious.crypto",       re.compile(r"(wallet|seed phrase|mnemonic|crypto)", re.IGNORECASE)),
    ("suspicious.webhook",      re.compile(r"(discord\.gg|webhook|hooks\.slack)", re.IGNORECASE)),
    ("suspicious.script",       re.compile(r"(curl[^\n]+\|\s*(sh|bash))", re.IGNORECASE)),
    ("suspicious.url_shortener", re.compile(r"(bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd)", re.IGNORECASE)),
]

_BLOCKING_FLAGS = frozenset({
    "blocked.malware",
    "blocked.shell_injection",
    "blocked.credential_exfil",
    "blocked.unreadable_file",        # fail-closed: unreadable helper file
    "blocked.unreadable_directory",   # fail-closed: unreadable skill directory
    "blocked.unparseable_code",       # S1: fail-closed on SyntaxError (polyglot bypass)
})


def check_skill_safety(text: str) -> List[str]:
    """Check *text* against safety rules, return list of triggered flag names.

    Uses AST-based analysis for Python code blocks (catches evasion vectors
    like alias tracking, dynamic imports, attribute chains) plus regex rules
    for non-Python content.  Returns an empty list if no rules match (= safe).
    """
    from .ast_safety import check_python_blocks_safety

    ast_flags = check_python_blocks_safety(text)
    regex_flags = [flag for flag, pat in _SAFETY_RULES if pat.search(text)]
    # merge + dedup, AST first (AST is more precise)
    return list(dict.fromkeys(ast_flags + regex_flags))


def check_skill_directory_safety(skill_dir: Path) -> List[str]:
    """Scan every text file in *skill_dir* for safety violations.

    Extends ``check_skill_safety`` to cover multi-file skills: a malicious
    helper bundled alongside ``SKILL.md`` would previously bypass the safety
    filter (which only checked the markdown file).

    Returns a deduplicated list of all triggered flag names across every
    scannable file in the directory.  Fail-closed: returns a blocking flag if
    the directory or any individual file is unreadable.

    Only files whose extension is in ``_SCANNABLE_EXTENSIONS`` are read;
    binary files and unknown extensions are skipped silently.
    """
    all_flags: List[str] = []
    _MAX_DEPTH = 5  # Prevent symlink loop abuse in deeply nested dirs
    if not skill_dir.is_dir():
        return ["blocked.unreadable_directory"]
    try:
        for fpath in sorted(skill_dir.rglob("*")):
            if not fpath.is_file():
                continue
            # Depth guard: reject files nested beyond _MAX_DEPTH levels
            try:
                rel = fpath.relative_to(skill_dir)
            except ValueError:
                continue  # Outside skill_dir (e.g. symlink escape)
            if len(rel.parts) > _MAX_DEPTH:
                all_flags.append("blocked.excessive_nesting")
                continue
            if fpath.suffix.lower() not in _SCANNABLE_EXTENSIONS:
                # W15: Also detect shebang scripts without known extensions
                try:
                    with open(fpath, "rb") as bf:
                        head = bf.read(64)
                    if head.startswith(b"#!") and any(
                        interp in head
                        for interp in (b"python", b"bash", b"sh", b"node", b"ruby", b"perl")
                    ):
                        text = fpath.read_text(encoding="utf-8", errors="replace")
                        all_flags.extend(check_skill_safety(text))
                except OSError:
                    pass
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                all_flags.extend(check_skill_safety(text))
            except OSError:
                all_flags.append("blocked.unreadable_file")
    except OSError:
        all_flags.append("blocked.unreadable_directory")

    # Deduplicate while preserving first-occurrence order.
    return list(dict.fromkeys(all_flags))


def is_skill_safe(flags: List[str]) -> bool:
    """Return True if *flags* contain no blocking flag.

    ``suspicious.*`` flags are informational (logged / attached to search
    results) but do NOT block.  Only ``blocked.*`` flags cause rejection.
    """
    return not any(f in _BLOCKING_FLAGS for f in flags)

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)

# Characters that require YAML value quoting (colon-space, hash-space,
# or values starting with special YAML indicators).
_YAML_NEEDS_QUOTE_RE = re.compile(r"[:\#\[\]{}&*!|>'\"%@`]")


def _yaml_quote(value: str) -> str:
    """Quote a YAML scalar value if it contains special characters."""
    if not value or not _YAML_NEEDS_QUOTE_RE.search(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _yaml_unquote(value: str) -> str:
    """Strip surrounding quotes and unescape a YAML scalar value."""
    if len(value) >= 2:
        if (value[0] == '"' and value[-1] == '"') or \
           (value[0] == "'" and value[-1] == "'"):
            inner = value[1:-1]
            if value[0] == '"':
                inner = inner.replace('\\"', '"').replace("\\\\", "\\")
            return inner
    return value


def parse_frontmatter(content: str) -> Dict[str, Any]:
    """Parse YAML frontmatter into a flat dict.

    Simple line-by-line parser (no PyYAML dependency).
    Handles both quoted and unquoted values.
    Returns ``{}`` if no valid frontmatter is found.

    **First-wins semantics**: if a key appears more than once, only the
    first occurrence is kept (consistent with :func:`get_frontmatter_field`).
    Duplicates are logged as a warning — they may indicate an injection
    attempt where the attacker hopes the second value overrides the first.
    """
    if not content.startswith("---"):
        return {}
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}
    fm: Dict[str, Any] = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if key:
                if key in fm:
                    logger.warning(
                        "Duplicate frontmatter key '%s' — keeping first occurrence "
                        "(possible injection attempt)",
                        key,
                    )
                    continue  # first-wins, consistent with get_frontmatter_field
                fm[key] = _yaml_unquote(value.strip())
    return fm


def get_frontmatter_field(content: str, field_name: str) -> Optional[str]:
    """Extract a single field value from YAML frontmatter.

    Returns ``None`` if the field is absent or content has no frontmatter.
    """
    if not content.startswith("---"):
        return None
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return None
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() == field_name:
                return _yaml_unquote(value.strip())
    return None


def set_frontmatter_field(content: str, field_name: str, value: str) -> str:
    """Set (or insert) a field in YAML frontmatter.

    Values containing YAML special characters (``:``, ``#``, etc.) are
    automatically double-quoted to produce valid YAML.

    If *content* has no frontmatter, a new one is prepended.
    """
    quoted = _yaml_quote(value)
    if not content.startswith("---"):
        return f"---\n{field_name}: {quoted}\n---\n{content}"

    match = _FRONTMATTER_RE.match(content)
    if not match:
        return content

    fm_text = match.group(1)
    new_line = f"{field_name}: {quoted}"
    found = False
    new_lines = []
    for line in fm_text.split("\n"):
        if ":" in line and line.split(":", 1)[0].strip() == field_name:
            if not found:
                new_lines.append(new_line)
                found = True
            else:
                # W18.1: Drop duplicate keys instead of replacing all
                # (which would produce multiple identical lines).
                logger.warning(
                    "set_frontmatter_field: dropping duplicate key '%s' "
                    "(possible injection attempt)",
                    field_name,
                )
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(new_line)

    new_fm = "\n".join(new_lines)
    return f"---\n{new_fm}\n---{content[match.end():]}"


def normalize_frontmatter(content: str) -> str:
    """Re-serialize frontmatter with proper YAML quoting.

    Parses the existing frontmatter, then re-writes each value through
    :func:`_yaml_quote` so that colons, hashes, and other special
    characters are safely double-quoted.  The body after ``---`` is
    preserved verbatim.

    Returns *content* unchanged if no frontmatter is found.
    """
    if not content.startswith("---"):
        return content
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return content

    fm = parse_frontmatter(content)
    if not fm:
        return content

    safe_lines = [f"{k}: {_yaml_quote(v)}" for k, v in fm.items()]
    new_fm = "\n".join(safe_lines)
    return f"---\n{new_fm}\n---{content[match.end():]}"


def strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    if content.startswith("---"):
        match = re.match(r"^---\n.*?\n---\n?", content, re.DOTALL)
        if match:
            return content[match.end():].strip()
    return content

def strip_markdown_fences(text: str) -> str:
    """Remove surrounding markdown code fences if present.

    Handles common LLM wrapping patterns:
      - ````` ```markdown ```, ````` ```md ```, ````` ``` ```, ````` ```text `````
      - Nested triple-backtick pairs (outermost only)
      - Leading/trailing whitespace around fences
    """
    text = text.strip()

    # Pattern: opening ``` with optional language tag, content, closing ```
    m = re.match(
        r"^```(?:markdown|md|text|yaml|diff|patch)?\s*\n(.*?)\n```\s*$",
        text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()

    # Some LLMs emit ``````` (4+ backticks) as outer fence
    m = re.match(
        r"^`{3,}(?:\w+)?\s*\n(.*?)\n`{3,}\s*$",
        text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()

    return text


_CHANGE_SUMMARY_RE = re.compile(
    r"^[\s*_]*(?:CHANGE[\s_-]?SUMMARY)\s*[:：]\s*(.+)",
    re.IGNORECASE,
)


def extract_change_summary(content: str) -> tuple[str, str]:
    """Extract ``CHANGE_SUMMARY`` from LLM output.

    Returns ``(clean_content, change_summary)``.
    """
    lines = content.split("\n")

    # Find the first non-blank line
    first_nonblank = -1
    for i, line in enumerate(lines):
        if line.strip():
            first_nonblank = i
            break

    if first_nonblank == -1:
        return content, ""

    m = _CHANGE_SUMMARY_RE.match(lines[first_nonblank])
    if not m:
        return content, ""

    # Strip markdown bold/italic markers (** or __) from both ends
    summary = m.group(1).strip().strip("*_").strip()

    # Skip blank lines after the summary line to find content start
    content_start = first_nonblank + 1
    while content_start < len(lines) and not lines[content_start].strip():
        content_start += 1

    rest = "\n".join(lines[content_start:])
    return rest.strip(), summary

def validate_skill_dir(skill_dir: Path) -> Optional[str]:
    """Validate a skill directory after edit application.

    Returns None if valid, or an error message string.
    Checks:
      1. Directory exists
      2. SKILL.md exists and is non-empty
      3. SKILL.md has valid YAML frontmatter with ``name`` field
      4. No empty files (warning-level, not blocking)
    """
    if not skill_dir.exists():
        return f"Skill directory does not exist: {skill_dir}"

    skill_file = skill_dir / SKILL_FILENAME
    if not skill_file.exists():
        return f"SKILL.md not found in {skill_dir}"

    try:
        content = skill_file.read_text(encoding="utf-8")
    except Exception as e:
        return f"Cannot read SKILL.md: {e}"

    if not content.strip():
        return "SKILL.md is empty"

    # Check frontmatter
    if not content.startswith("---"):
        return "SKILL.md missing YAML frontmatter (should start with '---')"

    m = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return "SKILL.md has malformed YAML frontmatter (missing closing '---')"

    # Check for required 'name' field in frontmatter
    name = get_frontmatter_field(content, "name")
    if not name:
        return "SKILL.md frontmatter missing 'name' field"

    # Non-blocking checks: log warnings for empty auxiliary files
    for p in skill_dir.rglob("*"):
        if p.is_file() and p != skill_file:
            try:
                if p.stat().st_size == 0:
                    logger.warning(f"Validation: empty auxiliary file: {p.relative_to(skill_dir)}")
            except OSError:
                pass

    return None


def truncate(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars* with an ellipsis marker."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [truncated at {max_chars} chars]"


# ---------------------------------------------------------------------------
# Skill Capability Manifest (Gate Patch 5)
# ---------------------------------------------------------------------------

VALID_CAPABILITIES = frozenset({
    "network",        # HTTP / websocket / DNS access
    "filesystem",     # Read / write local files beyond skill dir
    "subprocess",     # Shell / process execution
    "env_vars",       # Environment variable access
    "cloud_api",      # External cloud API calls (OpenAI, etc.)
    "gpu",            # GPU compute
})

_CAPABILITY_DETECTORS: List[tuple[str, re.Pattern]] = [
    ("network", re.compile(
        r"(?:requests\.|urllib|httpx|aiohttp|fetch\(|curl\s|wget\s)",
        re.IGNORECASE,
    )),
    ("subprocess", re.compile(
        r"(?:subprocess\.|os\.system|os\.popen|os\.exec[lv]?p?|Popen)",
        re.IGNORECASE,
    )),
    ("env_vars", re.compile(
        r"(?:os\.environ|os\.getenv|dotenv\.load|load_dotenv)",
        re.IGNORECASE,
    )),
    ("filesystem", re.compile(
        r"(?:open\s*\(|pathlib\.Path|shutil\.|os\.remove|os\.mkdir)",
        re.IGNORECASE,
    )),
]


def _is_near_capabilities(key: str) -> bool:
    """Return True if *key* looks like a typo of ``capabilities``.

    W18.1: Detects frontmatter keys that are near-misses (edit distance <= 2
    or ``capab`` prefix) to prevent fail-open bypass where a misspelled KEY
    (e.g. ``capabilites:``) causes the manifest validator to treat the skill
    as legacy (no capabilities = no restrictions).
    """
    target = "capabilities"
    if key == target:
        return False  # exact match — not a typo
    key_lower = key.lower()
    # W20 FP3: Exclude keys with extra suffix segments (_, -)
    # e.g., "capability_notes", "capability-map" are not typos
    if "_" in key_lower or "-" in key_lower:
        return False
    # Prefix check: covers capabilites, capablities, capability, etc.
    if key_lower.startswith("capab"):
        return True
    # Edit distance check for non-prefix typos (e.g. "cpabilities")
    if abs(len(key_lower) - len(target)) > 2:
        return False
    # Levenshtein distance (O(mn), but strings are ~12 chars — trivial)
    m, n = len(key_lower), len(target)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if key_lower[i - 1] == target[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] <= 2


def parse_capabilities(content: str) -> frozenset[str]:
    """Parse ``capabilities`` field from SKILL.md frontmatter.

    Expected format::

        capabilities: network,filesystem,subprocess

    Returns a frozenset of validated capability names.
    Unknown capabilities are logged and ignored for backward compatibility.
    Use :func:`validate_capability_manifest` for strict (fail-closed) validation.
    """
    raw = get_frontmatter_field(content, "capabilities")
    if not raw:
        return frozenset()
    caps = frozenset(c.strip().lower() for c in raw.split(",") if c.strip())
    invalid = caps - VALID_CAPABILITIES
    if invalid:
        logger.warning("Unknown capabilities ignored: %s", invalid)
    return caps & VALID_CAPABILITIES


def validate_capability_manifest(content: str) -> Optional[str]:
    """Validate the ``capabilities`` field in SKILL.md frontmatter (fail-closed).

    Returns ``None`` if valid (or absent — legacy skills are OK).
    Returns an error message string if any declared capability is unknown,
    which likely indicates a typo that would silently grant excess permissions.

    W18: Fixes the fail-open gap where typos like ``netwerk`` instead of
    ``network`` were silently ignored, causing the skill to run without
    the intended capability restriction.

    W18.1: Also detects near-miss FIELD-NAME typos (e.g. ``capabilites:``
    instead of ``capabilities:``) that previously caused the validator to
    treat the skill as legacy (no restrictions).
    """
    # W18.1: Check for near-miss field-name typos BEFORE checking values.
    # A misspelled KEY (e.g. "capabilites:") causes get_frontmatter_field
    # to return None, making the skill appear legacy (no restrictions).
    fm = parse_frontmatter(content)
    for key in fm:
        if _is_near_capabilities(key):
            return (
                f"Frontmatter key '{key}' looks like a typo of 'capabilities' — "
                f"please use the exact key 'capabilities'. "
                f"Blocking skill to prevent fail-open capability bypass."
            )

    raw = get_frontmatter_field(content, "capabilities")
    if not raw:
        return None  # no capabilities declared = legacy, OK
    caps = frozenset(c.strip().lower() for c in raw.split(",") if c.strip())
    if not caps:
        return None  # empty after stripping = same as absent
    invalid = caps - VALID_CAPABILITIES
    if invalid:
        return (
            f"Unknown capabilities declared: {sorted(invalid)} — "
            f"valid capabilities are: {sorted(VALID_CAPABILITIES)}. "
            f"Fix typos or the skill will be blocked (fail-closed)."
        )
    return None


def check_capability_violations(
    declared: frozenset[str],
    body: str,
) -> List[str]:
    """Check if skill body uses capabilities not declared in its manifest.

    Returns a list of violation descriptions (empty = clean).
    """
    violations: List[str] = []
    for cap, pattern in _CAPABILITY_DETECTORS:
        if cap not in declared and pattern.search(body):
            violations.append(
                f"Uses {cap} APIs but '{cap}' capability not declared in manifest"
            )
    return violations


# ---------------------------------------------------------------------------
# Capability-to-Backend mapping (Gate Patch 5 — Phase 2)
# ---------------------------------------------------------------------------

CAPABILITY_TO_BACKENDS: Dict[str, frozenset[str]] = {
    "network":    frozenset({"mcp"}),
    "filesystem": frozenset({"shell", "mcp"}),
    "subprocess": frozenset({"shell"}),
    "env_vars":   frozenset({"shell"}),
    "cloud_api":  frozenset({"mcp"}),
    "gpu":        frozenset({"shell", "mcp"}),
}

# Capabilities that require the shell backend to be auto-added.
_SHELL_REQUIRING_CAPABILITIES = frozenset({"filesystem", "subprocess", "env_vars", "gpu"})


def capabilities_need_shell(capabilities: frozenset[str]) -> bool:
    """Return True if *capabilities* require the shell backend.

    Fail-open: empty capabilities (legacy skills) return True so that
    existing skills continue to work.  Only explicit network-only or
    cloud_api-only skills skip shell auto-add.
    """
    if not capabilities:
        logger.warning(
            "capabilities_need_shell: empty capability set — fail-open to True "
            "(legacy skill without manifest). Consider adding capability declarations.",
        )
        return True  # legacy / no manifest → fail-open
    return bool(capabilities & _SHELL_REQUIRING_CAPABILITIES)


def allowed_backends_for_capabilities(
    capabilities: frozenset[str],
) -> Optional[frozenset[str]]:
    """Return the set of backend names a skill may use, or None (allow all).

    Returns None when *capabilities* is empty (legacy fail-open) so callers
    can skip filtering entirely for backward compatibility.
    """
    if not capabilities:
        return None  # legacy — no restriction
    backends: set[str] = set()
    for cap in capabilities:
        mapped = CAPABILITY_TO_BACKENDS.get(cap)
        if mapped:
            backends.update(mapped)
    # system backend is always allowed (internal tools like retrieve_skill)
    backends.add("system")
    return frozenset(backends)


def filter_tools_by_capabilities(
    tools: list,
    capabilities: frozenset[str],
) -> list:
    """Remove tools whose backend is not permitted by *capabilities*.

    Legacy skills (empty capabilities) pass all tools through unchanged.
    Tools with backend_type NOT_SET or SYSTEM always pass.
    """
    allowed = allowed_backends_for_capabilities(capabilities)
    if allowed is None:
        return tools  # legacy — no filtering

    filtered = []
    for tool in tools:
        bt = getattr(tool, "backend_type", None)
        if bt is None:
            filtered.append(tool)  # unknown structure — pass through
            continue
        backend_value = bt.value if hasattr(bt, "value") else str(bt)
        if backend_value in allowed or backend_value == "not_set":
            filtered.append(tool)
        else:
            logger.info(
                f"Capability filter: dropped tool '{getattr(tool, 'name', '?')}' "
                f"(backend={backend_value}, allowed={sorted(allowed)})"
            )
    return filtered

