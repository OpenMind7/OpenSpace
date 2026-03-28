"""AST-based safety checker for skill Python code.

Replaces regex-only screening (Finding #2) with structural analysis that
catches evasion vectors: import aliasing, dynamic imports, attribute chains,
and dangerous stdlib calls.

Design principles:
  - Single O(n) AST walk per source — no repeated traversals.
  - Alias-aware: tracks ``import X as Y`` and ``from X import Y as Z``.
  - Fail-closed on ambiguity: dynamic attrs on dangerous modules → block.
  - Fail-open on SyntaxError: skip unparseable blocks (regex still runs).
  - Reuses existing flag names: ``blocked.shell_injection``,
    ``blocked.credential_exfil``.
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Dangerous module/function sets
# ---------------------------------------------------------------------------

_DANGEROUS_EXEC_MODULES: frozenset[str] = frozenset({
    "subprocess", "shutil", "ctypes", "pty", "commands",
    # Security review additions (CRITICAL):
    "importlib", "importlib.util",       # import_module() bypasses __import__ blocking
    "pickle", "cPickle", "shelve",       # deserialization → arbitrary code exec via __reduce__
    "marshal",                            # bytecode deserialization
    "code", "codeop",                     # interactive code execution
    "multiprocessing",                    # Process(target=os.system)
})

_DANGEROUS_EXFIL_MODULES: frozenset[str] = frozenset({
    "urllib", "urllib.request", "http.client",
    "requests", "httpx", "aiohttp",
    # Security review additions (CRITICAL):
    "socket", "ssl",                     # raw TCP/UDP exfiltration
    "smtplib", "ftplib",                 # email/FTP exfiltration
    "xmlrpc.client", "xmlrpc.server",   # XML-RPC to arbitrary servers
    "http.server",                       # expose local files via HTTP
})

# All modules considered dangerous (union for general checks)
_ALL_DANGEROUS_MODULES: frozenset[str] = _DANGEROUS_EXEC_MODULES | _DANGEROUS_EXFIL_MODULES

_DANGEROUS_OS_FUNCS: frozenset[str] = frozenset({
    "system", "popen", "popen2", "popen3", "popen4",
    "execl", "execle", "execlp", "execlpe",
    "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
})

_DANGEROUS_SUBPROCESS_FUNCS: frozenset[str] = frozenset({
    "call", "run", "Popen", "check_call", "check_output",
    "getoutput", "getstatusoutput",
})

_DANGEROUS_ASYNCIO_FUNCS: frozenset[str] = frozenset({
    "create_subprocess_exec", "create_subprocess_shell",
})

_DANGEROUS_SHUTIL_FUNCS: frozenset[str] = frozenset({
    "rmtree", "move", "copy2", "copytree",
})

_SENSITIVE_PATH_FRAGMENTS: frozenset[str] = frozenset({
    ".ssh", ".aws", ".gnupg", "credentials", "shadow",
    "id_rsa", "id_ed25519", ".env", "passwd",
    "private_key", ".kube/config", ".docker/config",
})

# Builtins that are blanket-blocked (no legitimate use in skill code)
_BLANKET_BLOCKED_BUILTINS: frozenset[str] = frozenset({
    "eval", "exec", "compile",
})


# ---------------------------------------------------------------------------
# Markdown Python block extraction
# ---------------------------------------------------------------------------

# Fenced code blocks: ```python or ```py
_PYTHON_FENCE_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL,
)

# Heredoc Python in bash blocks: python3 <<'EOF' ... EOF
_HEREDOC_PYTHON_RE = re.compile(
    r"python[23]?\s+<<['\"]?(\w+)['\"]?\s*\n(.*?)\n\1",
    re.DOTALL,
)


def extract_python_blocks(markdown_text: str) -> List[str]:
    """Extract Python code blocks from markdown content.

    Finds:
      - Fenced ``python`` / ``py`` code blocks
      - Heredoc Python in bash blocks (``python3 <<'EOF' ... EOF``)

    Returns a list of source strings (may be empty).
    """
    blocks: List[str] = []

    for m in _PYTHON_FENCE_RE.finditer(markdown_text):
        code = m.group(1).strip()
        if code:
            blocks.append(code)

    for m in _HEREDOC_PYTHON_RE.finditer(markdown_text):
        code = m.group(2).strip()
        if code:
            blocks.append(code)

    return blocks


# ---------------------------------------------------------------------------
# AST visitor — single O(n) walk with alias tracking
# ---------------------------------------------------------------------------

class DangerousNodeVisitor(ast.NodeVisitor):
    """Walk AST once, collecting safety flags.

    Tracks import aliases so ``import os as o; o.system('x')`` is caught.
    """

    def __init__(self) -> None:
        self.flags: List[str] = []
        # Track module aliases: alias_name -> canonical_module_name
        self._module_aliases: Dict[str, str] = {}
        # Track name aliases: alias_name -> "module.name" or just "name"
        self._name_aliases: Dict[str, str] = {}

    # --- Import tracking ---

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            mod_name = alias.name
            local_name = alias.asname or alias.name
            self._module_aliases[local_name] = mod_name

            # Flag dangerous module imports (but don't block just the import)
            # Blocking happens at CALL sites, not import sites.
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod_name = node.module or ""
        for alias in node.names:
            local_name = alias.asname or alias.name
            # Record as "module.name" so we can resolve calls later
            self._name_aliases[local_name] = f"{mod_name}.{alias.name}"
        self.generic_visit(node)

    # --- Call analysis ---

    def _check_simple_call(self, name: str, node: ast.Call) -> None:
        """Check a simple function call like ``eval(...)`` or ``run(...)``."""

        # Blanket-blocked builtins
        if name in _BLANKET_BLOCKED_BUILTINS:
            self.flags.append("blocked.shell_injection")
            return

        # __import__ — block if non-constant arg or dangerous module
        if name == "__import__":
            arg = _get_constant_arg(node, 0)
            if arg is None:
                # Non-constant: fail-closed
                self.flags.append("blocked.shell_injection")
            elif arg in _DANGEROUS_EXEC_MODULES or arg in {"os"}:
                self.flags.append("blocked.shell_injection")
            elif arg in _DANGEROUS_EXFIL_MODULES:
                self.flags.append("blocked.credential_exfil")
            return

        # getattr on a dangerous module: getattr(os, "system")
        if name == "getattr" and len(node.args) >= 2:
            self._check_getattr(node)
            return

        # Check if this name is an alias for a dangerous function
        if name in self._name_aliases:
            qualified = self._name_aliases[name]
            self._check_qualified_call(qualified)

    def _check_attribute_call(self, func: ast.Attribute, node: ast.Call) -> None:
        """Check a dotted call like ``os.system(...)`` or ``sp.Popen(...)``."""
        chain = _resolve_attr_chain(func)
        if chain is None:
            return

        # Resolve the root through module aliases
        parts = chain.split(".", 1)
        root = parts[0]
        if root in self._module_aliases:
            resolved = self._module_aliases[root]
            chain = f"{resolved}.{parts[1]}" if len(parts) > 1 else resolved

        self._check_qualified_call(chain)

    def _check_qualified_call(self, qualified: str) -> None:
        """Check a fully-qualified call path like ``os.system`` or ``subprocess.run``."""
        parts = qualified.rsplit(".", 1)
        if len(parts) != 2:
            return
        module_part, func_name = parts

        # builtins.eval(), builtins.exec(), builtins.__import__()
        if module_part == "builtins" and func_name in _BLANKET_BLOCKED_BUILTINS:
            self.flags.append("blocked.shell_injection")
            return
        if module_part == "builtins" and func_name == "__import__":
            self.flags.append("blocked.shell_injection")
            return

        # os.dangerous_func()
        if module_part == "os" and func_name in _DANGEROUS_OS_FUNCS:
            self.flags.append("blocked.shell_injection")
            return

        # subprocess.dangerous_func()
        if module_part == "subprocess" and func_name in _DANGEROUS_SUBPROCESS_FUNCS:
            self.flags.append("blocked.shell_injection")
            return

        # shutil.dangerous_func()
        if module_part == "shutil" and func_name in _DANGEROUS_SHUTIL_FUNCS:
            self.flags.append("blocked.shell_injection")
            return

        # asyncio.create_subprocess_*
        if module_part == "asyncio" and func_name in _DANGEROUS_ASYNCIO_FUNCS:
            self.flags.append("blocked.shell_injection")
            return

        # ctypes / pty / commands — any call on these modules (or submodules) is dangerous
        _any_call_blocked = {"ctypes", "pty", "commands"}
        if module_part in _any_call_blocked or any(
            module_part.startswith(m + ".") for m in _any_call_blocked
        ):
            self.flags.append("blocked.shell_injection")
            return

        # Any call on a dangerous exec/exfil module's submodule (e.g., importlib.util.*)
        root_module = module_part.split(".")[0]
        if root_module in _DANGEROUS_EXEC_MODULES:
            self.flags.append("blocked.shell_injection")
            return
        if root_module in _DANGEROUS_EXFIL_MODULES:
            self.flags.append("blocked.credential_exfil")
            return

        # open() with sensitive path constant
        if func_name == "open" or qualified == "builtins.open":
            # Handled in visit_Call for bare open() — skip here
            pass

    def _check_getattr(self, node: ast.Call) -> None:
        """Check ``getattr(module, attr)`` for dangerous module + dynamic attr."""
        if len(node.args) < 2:
            return

        obj = node.args[0]
        if not isinstance(obj, ast.Name):
            return

        obj_name = obj.id
        # Resolve through aliases
        resolved_mod = self._module_aliases.get(obj_name, obj_name)

        if resolved_mod in _ALL_DANGEROUS_MODULES or resolved_mod == "os":
            attr_arg = _get_constant_arg(node, 1)
            if attr_arg is None:
                # Dynamic attr on dangerous module — fail-closed
                self.flags.append("blocked.shell_injection")
            elif resolved_mod == "os" and attr_arg in _DANGEROUS_OS_FUNCS:
                self.flags.append("blocked.shell_injection")
            elif resolved_mod == "subprocess" and attr_arg in _DANGEROUS_SUBPROCESS_FUNCS:
                self.flags.append("blocked.shell_injection")

    # --- Main visit_Call (single entry point for all call analysis) ---

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # Check open() for sensitive paths
        if isinstance(func, ast.Name) and func.id == "open":
            self._check_open_sensitive(node)
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            self._check_open_sensitive(node)

        # Main dispatch
        if isinstance(func, ast.Name):
            self._check_simple_call(func.id, node)
        elif isinstance(func, ast.Attribute):
            self._check_attribute_call(func, node)

        self.generic_visit(node)

    def _check_open_sensitive(self, node: ast.Call) -> None:
        """Flag open() calls with constant paths pointing to sensitive files."""
        path_arg = _get_constant_arg(node, 0)
        if path_arg is None:
            return  # Non-constant arg — allow (too many false positives)

        path_lower = path_arg.lower()
        for fragment in _SENSITIVE_PATH_FRAGMENTS:
            if fragment in path_lower:
                self.flags.append("blocked.credential_exfil")
                return


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resolve_attr_chain(node: ast.expr, max_depth: int = 5) -> Optional[str]:
    """Resolve a dotted attribute chain to a string like ``os.path.join``.

    Returns None if the chain exceeds *max_depth* or contains non-Name/Attribute
    nodes (e.g., function calls in the chain).
    """
    parts: List[str] = []
    current = node
    for _ in range(max_depth):
        if isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        elif isinstance(current, ast.Name):
            parts.append(current.id)
            parts.reverse()
            return ".".join(parts)
        else:
            return None
    return None


def _get_constant_arg(call: ast.Call, pos: int) -> Optional[str]:
    """Extract a string constant from a call's positional arguments.

    Returns the string value if ``call.args[pos]`` is a ``ast.Constant``
    with a ``str`` value.  Returns None otherwise (non-constant, missing,
    or non-string).
    """
    if pos >= len(call.args):
        return None
    arg = call.args[pos]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_ast_safety(source: str) -> List[str]:
    """Parse a single Python source string and check for dangerous operations.

    Returns a list of triggered flag names (empty = safe).
    On ``SyntaxError``, returns an empty list (fall through to regex).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    visitor = DangerousNodeVisitor()
    visitor.visit(tree)

    # Deduplicate while preserving order
    seen: set[str] = set()
    return [f for f in visitor.flags if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]


def check_python_blocks_safety(text: str) -> List[str]:
    """Check all Python code blocks in markdown/text for dangerous operations.

    Extracts fenced Python blocks and heredoc Python from the text.
    If no fenced blocks are found, tries to parse the entire text as Python
    (fast path for plain .py files).

    Returns a deduplicated list of triggered flag names.
    """
    blocks = extract_python_blocks(text)

    if not blocks:
        # No fenced blocks found — try parsing the entire text as Python
        return check_ast_safety(text)

    all_flags: List[str] = []
    for block in blocks:
        all_flags.extend(check_ast_safety(block))

    # Deduplicate while preserving order
    seen: set[str] = set()
    return [f for f in all_flags if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]
