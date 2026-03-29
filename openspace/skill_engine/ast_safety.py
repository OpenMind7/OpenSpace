"""AST-based safety checker for skill Python code.

Replaces regex-only screening (Finding #2) with structural analysis that
catches evasion vectors: import aliasing, dynamic imports, attribute chains,
and dangerous stdlib calls.

Design principles:
  - Single O(n) AST walk per source — no repeated traversals.
  - Alias-aware: tracks ``import X as Y`` and ``from X import Y as Z``.
  - Fail-closed on ambiguity: dynamic attrs on dangerous modules → block.
  - Fail-closed on SyntaxError: unparseable code is blocked (S1 fix).
  - Reuses existing flag names: ``blocked.shell_injection``,
    ``blocked.credential_exfil``.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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
    # W12.1: Additional dangerous modules from security review
    "signal",                             # intercept termination signals
    "zipimport",                          # load code from zip archives
    "gc",                                 # gc.get_objects() enumerates live objects in memory
})

_DANGEROUS_EXFIL_MODULES: frozenset[str] = frozenset({
    "urllib", "urllib.request", "http.client",
    "requests", "httpx", "aiohttp",
    # Security review additions (CRITICAL):
    "socket", "ssl",                     # raw TCP/UDP exfiltration
    "smtplib", "ftplib",                 # email/FTP exfiltration
    "xmlrpc.client", "xmlrpc.server",   # XML-RPC to arbitrary servers
    "http.server",                       # expose local files via HTTP
    # W12.1: exfiltration without network imports
    "webbrowser",                        # webbrowser.open() exfil via URL
})

# S2: Modules that enable type/code construction
_DANGEROUS_META_MODULES: frozenset[str] = frozenset({
    "types",       # types.FunctionType / types.CodeType → construct functions from bytecode
    "dis",         # bytecode disassembly (info leak, paired with types → dangerous)
})

# All modules considered dangerous (union for general checks)
_ALL_DANGEROUS_MODULES: frozenset[str] = (
    _DANGEROUS_EXEC_MODULES | _DANGEROUS_EXFIL_MODULES | _DANGEROUS_META_MODULES
)

# Modules where ANY call is dangerous (promoted from inline set — H2 fix)
_ANY_CALL_BLOCKED_MODULES: frozenset[str] = frozenset({"ctypes", "pty", "commands"})

_DANGEROUS_OS_FUNCS: frozenset[str] = frozenset({
    "system", "popen", "popen2", "popen3", "popen4",
    "execl", "execle", "execlp", "execlpe",
    "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
})

# W12.1: os.environ method calls that access credentials
_DANGEROUS_OS_ENVIRON_METHODS: frozenset[str] = frozenset({
    "get", "pop", "setdefault",
    # W12.3: whole-environment copy paths (Codex finding)
    "copy", "items", "values", "keys",
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
    "breakpoint",  # S2: drops into interactive debugger → shell access
    # W12.1: globals()/locals() enable __import__ via dict subscription
    "globals", "locals", "vars",
})

# W12.1: Dangerous dunder attributes — sandbox escape via introspection
_DANGEROUS_DUNDER_ATTRS: frozenset[str] = frozenset({
    "__subclasses__",  # ().__class__.__bases__[0].__subclasses__() → reach Popen
    "__bases__",       # type traversal to reach base classes
    "__mro__",         # method resolution order traversal
    "__globals__",     # function.__globals__ → access module namespace
    "__builtins__",    # reach __import__ via builtins dict
    "__code__",        # bytecode manipulation → construct functions
    "__import__",      # dunder import as attribute access
    # W12.2: builtins.__dict__["__import__"] bypass (Codex finding)
    "__dict__",        # dict access → builtins.__dict__["__import__"] → arbitrary import
    # W12.3: builtins.__getattribute__('__import__') bypass (Codex finding)
    "__getattribute__",  # introspection → reaches any attr including __import__
})

# S2: sys module dangerous attributes (checked via visit_Attribute)
_DANGEROUS_SYS_ATTRS: frozenset[str] = frozenset({
    "modules",     # sys.modules — module manipulation / hijacking
    "_getframe",   # stack frame access → inspect caller code
})


# ---------------------------------------------------------------------------
# Markdown Python block extraction
# ---------------------------------------------------------------------------

# Fenced code blocks: ```python or ```py
_PYTHON_FENCE_RE = re.compile(
    r"```(?:python|py|python3|python2)\s*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# S6: Heredoc Python in bash blocks — matches python3, /usr/bin/python3.11, etc.
_HEREDOC_PYTHON_RE = re.compile(
    r"(?:/[\w/.-]*)?python[23]?(?:\.\d+)?\s+<<['\"]?(\w+)['\"]?\s*\n(.*?)\n\1",
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
        # W17 C4: Track function return taint for decorator analysis
        # func_name -> resolved dangerous return value
        self._func_return_taint: Dict[str, str] = {}

    # --- Import tracking + assignment taint ---

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

    # --- W15.3/W16: Assignment-based taint propagation ---

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track assignment-based alias propagation.

        W16 CRIT-2: Handles chained assignment (a = b = subprocess) by
        iterating ALL targets, and destructuring ((a, b) = ...) by recursing
        into Tuple/List/Starred leaves.
        W17 C2: Also detects dangerous attribute assignments (cls.attr = os.system).
        """
        for target in node.targets:
            self._propagate_taint_target(target, node.value)
            # W17 C2: Flag attribute assignments where value is dangerous
            self._check_dangerous_attr_assign(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Track annotated assignment taint: ``x: Any = os.environ``.

        W19 C1: Also checks dangerous attribute assignments (e.g.
        ``cls.pwn: object = os.system``), matching visit_Assign behaviour.
        """
        if node.value:
            self._propagate_taint_target(node.target, node.value)
            # W19 C1: Annotated attr assignments must also be checked
            self._check_dangerous_attr_assign(node.target, node.value)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Track walrus operator taint: ``(x := os.environ)``."""
        self._propagate_taint_target(node.target, node.value)
        self.generic_visit(node)

    # W16 HIGH-4: Non-assignment taint bindings

    def visit_For(self, node: ast.For) -> None:
        """Track for-loop taint: ``for env in [os.environ]: ...``

        W17 H2: Propagate taint from ALL elements of literal iterables,
        not just single-element ones.
        """
        if isinstance(node.iter, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.iter.elts:
                self._propagate_taint_target(node.target, elt)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Same as visit_For for async for loops."""
        if isinstance(node.iter, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.iter.elts:
                self._propagate_taint_target(node.target, elt)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Track with-statement taint: ``with open(...) as f: ...``"""
        for item in node.items:
            if item.optional_vars is not None:
                self._propagate_taint_target(item.optional_vars, item.context_expr)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Same as visit_With for async with."""
        for item in node.items:
            if item.optional_vars is not None:
                self._propagate_taint_target(item.optional_vars, item.context_expr)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track dangerous default arguments and decorator return taint.

        W17 C4: Records function return taint and applies decorator rebinding.
        """
        self._propagate_default_args(node)
        self._scan_function_returns(node)
        self._check_decorator_taint(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Same as visit_FunctionDef for async functions."""
        self._propagate_default_args(node)
        self._scan_function_returns(node)
        self._check_decorator_taint(node)
        self.generic_visit(node)

    def _propagate_default_args(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """Taint function parameters whose defaults resolve to dangerous values.

        W17 H2: Also handles positional-only args (posonlyargs). Defaults are
        right-aligned to the full positional list (posonlyargs + args).
        """
        args = node.args
        # W17 H2: Combine posonlyargs + args for right-aligned default matching
        all_positional = list(args.posonlyargs) + list(args.args)
        n_defaults = len(args.defaults)
        if n_defaults:
            defaulted_args = all_positional[-n_defaults:]
            for arg, default in zip(defaulted_args, args.defaults):
                self._propagate_taint(arg.arg, default)
        # kw-only defaults
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                self._propagate_taint(arg.arg, default)

    def _propagate_taint_target(
        self, target: ast.expr, value: ast.expr
    ) -> None:
        """Recursively propagate taint through target patterns.

        W16 CRIT-2: Handles:
          - ast.Name: direct binding
          - ast.Tuple/ast.List: destructuring (each element gets the value)
          - ast.Starred: starred assignment
        """
        if isinstance(target, ast.Name):
            self._propagate_taint(target.id, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # For destructuring, propagate to each element.
            # W17 H1: Check for Starred elements — starred makes exact-length
            # matching unreliable, so propagate all value elements to all targets.
            has_starred = any(isinstance(e, ast.Starred) for e in target.elts)
            if (
                not has_starred
                and isinstance(value, (ast.Tuple, ast.List))
                and len(value.elts) == len(target.elts)
            ):
                for t, v in zip(target.elts, value.elts):
                    self._propagate_taint_target(t, v)
            elif isinstance(value, (ast.Tuple, ast.List)):
                # W17 H1: Try each value element per target; stop at first
                # that resolves to avoid stale-taint clearing from
                # non-resolving elements wiping valid taint.
                for elt in target.elts:
                    for v in value.elts:
                        r = self._resolve_rhs(v)
                        if r is None:
                            r = self._peel_to_resolve(v)
                        if r is not None:
                            self._propagate_taint_target(elt, v)
                            break
            else:
                # Can't decompose value — propagate full value to each name leaf
                for elt in target.elts:
                    self._propagate_taint_target(elt, value)
        elif isinstance(target, ast.Starred):
            self._propagate_taint_target(target.value, value)

    def _propagate_taint(self, target_name: str, value: ast.expr) -> None:
        """Resolve *value* and record *target_name* as an alias.

        If value resolves to a module name → ``_module_aliases``.
        If value resolves to a qualified path → ``_name_aliases``.
        If value is a string constant → ``_name_aliases`` (for getattr
        constant propagation in W16 HIGH-3).
        If value cannot be resolved → no taint (conservative, not fail-closed
        on assignment — we only fail-closed at USE sites).

        W16: Also clears stale aliases on rebind before resolving new RHS.
        """
        # W16 LOW: clear stale taint before re-resolving
        self._module_aliases.pop(target_name, None)
        self._name_aliases.pop(target_name, None)

        # W16 HIGH-3: string constant assignment (a = "__globals__")
        # Store in _name_aliases so _check_getattr can resolve it.
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            self._name_aliases[target_name] = value.value
            return

        resolved = self._resolve_rhs(value)
        if resolved is None:
            # W17 C1: Fall back to _peel_to_resolve for wrapped values.
            # Catches: f = (lambda y: y)(os.system), f = staticmethod(os.system)
            resolved = self._peel_to_resolve(value)
        if resolved is None:
            return

        # Determine if resolved is a module (goes to _module_aliases)
        # or a qualified name (goes to _name_aliases)
        if (
            resolved in self._module_aliases.values()
            or resolved in _ALL_DANGEROUS_MODULES
            or resolved in {"os", "sys", "builtins"}
        ):
            self._module_aliases[target_name] = resolved
        else:
            self._name_aliases[target_name] = resolved

    def _resolve_rhs(self, node: ast.expr) -> Optional[str]:
        """Resolve an expression to a qualified name string.

        W16 CRIT-1: Also resolves dangerous builtins (eval, exec, globals,
        vars, __import__, __builtins__) as taintable names.

        W16 HIGH-5: Recurses through identity-preserving wrappers:
        BoolOp (or/and), IfExp, NamedExpr.
        """
        if isinstance(node, ast.Name):
            name = node.id
            if name in self._module_aliases:
                return self._module_aliases[name]
            if name in self._name_aliases:
                return self._name_aliases[name]
            # W16 CRIT-1: dangerous builtins are taintable
            if name in _BLANKET_BLOCKED_BUILTINS:
                return f"builtins.{name}"
            if name == "__import__":
                return "builtins.__import__"
            if name == "__builtins__":
                return "__builtins__"
            return None

        if isinstance(node, ast.Attribute):
            chain = _resolve_attr_chain(node)
            if chain is None:
                return None
            parts = chain.split(".", 1)
            root = parts[0]
            if root in self._module_aliases:
                resolved = self._module_aliases[root]
                return f"{resolved}.{parts[1]}" if len(parts) > 1 else resolved
            if root in self._name_aliases:
                resolved = self._name_aliases[root]
                return f"{resolved}.{parts[1]}" if len(parts) > 1 else resolved
            return chain

        # W16 HIGH-5: identity-preserving wrappers — resolve through them
        if isinstance(node, ast.BoolOp):
            # `x or {}` / `x and y` — try each value, return first resolved
            for val in node.values:
                r = self._resolve_rhs(val)
                if r is not None:
                    return r
            return None

        if isinstance(node, ast.IfExp):
            # `x if cond else y` — try body then orelse
            return self._resolve_rhs(node.body) or self._resolve_rhs(node.orelse)

        if isinstance(node, ast.NamedExpr):
            # `(x := dangerous)` — resolve the value
            return self._resolve_rhs(node.value)

        return None

    def _peel_to_resolve(self, node: ast.expr) -> Optional[str]:
        """Recursively peel wrapper expressions to resolve a qualified name.

        W16 HIGH-6: Handles identity-preserving wrappers that break
        _resolve_attr_chain but preserve the object at runtime:
          - Call: ``(lambda x: x)(os.environ)`` → resolve arg
          - NamedExpr: ``(x := os.environ)`` → resolve value
          - Subscript on dict literal: ``{"k": os.environ}["k"]``
        Falls back to _resolve_rhs for Name/Attribute/BoolOp/IfExp.
        """
        # Direct resolution via _resolve_rhs (handles Name, Attribute, BoolOp, IfExp)
        r = self._resolve_rhs(node)
        if r is not None:
            return r

        # Call: check return taint first (W19 C2), then try first arg (W17 C1)
        if isinstance(node, ast.Call):
            # W19 C2: If callee has recorded return taint, use it.
            # Catches: def make(): return os.system; f = make(); f("id")
            callee_name = None
            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee_name = _resolve_attr_chain(node.func)
            if callee_name:
                # Resolve through aliases in case callee is aliased
                resolved_callee = self._name_aliases.get(callee_name, callee_name)
                taint = (
                    self._func_return_taint.get(resolved_callee)
                    or self._func_return_taint.get(callee_name)
                )
                if taint is not None:
                    return taint
            # Existing W17 C1: try to resolve first positional argument
            if node.args:
                return self._peel_to_resolve(node.args[0])

        # NamedExpr: (x := dangerous)
        if isinstance(node, ast.NamedExpr):
            return self._peel_to_resolve(node.value)

        # W19 C4: Lambda — resolve body expression
        # Catches: return (lambda: os.system) where the lambda wraps a dangerous value
        if isinstance(node, ast.Lambda):
            return self._peel_to_resolve(node.body)

        # W19 C3: Dict with dangerous values — taint propagates through the dict
        # Catches: ns = {"run": os.system}; type("C", (), ns)
        if isinstance(node, ast.Dict):
            for val in node.values:
                if val is not None:
                    r = self._peel_to_resolve(val)
                    if r is not None and self._is_dangerous_resolved(r):
                        return r

        # W17 H3: Subscript on dict literal: {"k": os.environ}["k"]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Dict):
            key = None
            if isinstance(node.slice, ast.Constant):
                key = node.slice.value
            if key is not None:
                dict_node = node.value
                for k, v in zip(dict_node.keys, dict_node.values):
                    if (
                        isinstance(k, ast.Constant)
                        and k.value == key
                        and v is not None
                    ):
                        return self._peel_to_resolve(v)

        # W19 H5: Subscript on Name that's aliased to a dangerous value
        # Catches: d = {"k": os.environ}; d["k"].get("PASSWORD")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            resolved = (
                self._name_aliases.get(node.value.id)
                or self._module_aliases.get(node.value.id)
            )
            if resolved is not None and self._is_dangerous_resolved(resolved):
                return resolved

        return None

    # --- W17: Helper methods for new bypass detection ---

    def _is_dangerous_resolved(self, resolved: str) -> bool:
        """Check if a resolved qualified name refers to a dangerous callable."""
        parts = resolved.rsplit(".", 1)
        if len(parts) == 2:
            module_part, func_name = parts
            if module_part == "os" and func_name in _DANGEROUS_OS_FUNCS:
                return True
            if module_part == "os" and func_name in {"environ", "getenv"}:
                return True
            if module_part == "os.environ":
                return True
            if module_part == "subprocess" and func_name in _DANGEROUS_SUBPROCESS_FUNCS:
                return True
            if module_part == "shutil" and func_name in _DANGEROUS_SHUTIL_FUNCS:
                return True
            if module_part == "asyncio" and func_name in _DANGEROUS_ASYNCIO_FUNCS:
                return True
            if module_part == "builtins" and (
                func_name in _BLANKET_BLOCKED_BUILTINS or func_name == "__import__"
            ):
                return True
            root = module_part.split(".")[0]
            if root in _ALL_DANGEROUS_MODULES:
                return True
        else:
            if resolved in _ALL_DANGEROUS_MODULES or resolved in {"os", "sys"}:
                return True
        return False

    def _flag_for_resolved(self, resolved: str) -> Optional[str]:
        """Return the appropriate flag for a dangerous resolved value."""
        parts = resolved.rsplit(".", 1)
        if len(parts) == 2:
            module_part, func_name = parts
            if module_part == "os" and func_name in {"environ", "getenv"}:
                return "blocked.credential_exfil"
            if module_part == "os.environ":
                return "blocked.credential_exfil"
            if module_part in _DANGEROUS_EXFIL_MODULES:
                return "blocked.credential_exfil"
            root = module_part.split(".")[0]
            if root in _DANGEROUS_EXFIL_MODULES:
                return "blocked.credential_exfil"
        return "blocked.shell_injection"

    def _check_dangerous_attr_assign(
        self, target: ast.expr, value: ast.expr
    ) -> None:
        """W17 C2: Flag attribute assignments where value is a dangerous callable.

        Catches: cls.pwn = staticmethod(os.system), self.run = subprocess.call

        W19 C1: Also descends into Tuple/List destructuring to find
        attribute leaves.  ``(cls.pwn,) = (os.system,)`` is now caught.
        """
        if isinstance(target, ast.Attribute):
            resolved = self._peel_to_resolve(value)
            if resolved is not None and self._is_dangerous_resolved(resolved):
                self.flags.append(self._flag_for_resolved(resolved))
        elif isinstance(target, (ast.Tuple, ast.List)):
            # W19 C1: Pair target elements with value elements if both are sequences
            if (
                isinstance(value, (ast.Tuple, ast.List))
                and len(value.elts) == len(target.elts)
            ):
                for t, v in zip(target.elts, value.elts):
                    elt = t.value if isinstance(t, ast.Starred) else t
                    self._check_dangerous_attr_assign(elt, v)
            else:
                for t in target.elts:
                    elt = t.value if isinstance(t, ast.Starred) else t
                    self._check_dangerous_attr_assign(elt, value)

    def _scan_function_returns(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """W17 C4: Scan function body for return statements with dangerous values."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                resolved = self._peel_to_resolve(child.value)
                if resolved is not None and self._is_dangerous_resolved(resolved):
                    self._func_return_taint[node.name] = resolved
                    return

    def _check_decorator_taint(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """W17 C4: If a decorator returns dangerous values, taint the decorated name.

        W19 C4: Also handles Call decorators (``@deco_factory()``,
        ``@X.dec()``) and resolves aliases through ``_name_aliases``.
        """
        for deco in node.decorator_list:
            deco_name = None
            if isinstance(deco, ast.Name):
                deco_name = deco.id
            elif isinstance(deco, ast.Attribute):
                deco_name = _resolve_attr_chain(deco)
            # W19 C4: Call decorators: @deco_factory(), @X.dec()
            elif isinstance(deco, ast.Call):
                if isinstance(deco.func, ast.Name):
                    deco_name = deco.func.id
                elif isinstance(deco.func, ast.Attribute):
                    deco_name = _resolve_attr_chain(deco.func)
            if deco_name is None:
                continue
            # W19 C4: Resolve through name aliases (e.g. dec_alias = make)
            resolved_name = self._name_aliases.get(deco_name, deco_name)
            dangerous_val = (
                self._func_return_taint.get(resolved_name)
                or self._func_return_taint.get(deco_name)
            )
            if dangerous_val is None:
                continue
            # Use same routing as _propagate_taint for alias storage
            if (
                dangerous_val in self._module_aliases.values()
                or dangerous_val in _ALL_DANGEROUS_MODULES
                or dangerous_val in {"os", "sys", "builtins"}
            ):
                self._module_aliases[node.name] = dangerous_val
            else:
                self._name_aliases[node.name] = dangerous_val

    def _resolve_string_expr(self, node: ast.expr) -> Optional[str]:
        """W17 H4: Resolve simple string expressions (constant or aliased name).

        W19 H6: Recursive BinOp(Add) folding for nested concatenation.
        Catches: ``"__" + "glo" + "bals__"`` (parsed as nested BinOp).
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            val = self._name_aliases.get(node.id)
            if val is not None:
                return val
        # W19 H6: Recursive BinOp(Add) folding
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._resolve_string_expr(node.left)
            right = self._resolve_string_expr(node.right)
            if left is not None and right is not None:
                return left + right
        return None

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

        # W17 C3: type() with 3 args — dynamic class synthesis
        # Inspect dict values for dangerous callables
        if name == "type" and len(node.args) == 3:
            dict_arg = node.args[2]
            if isinstance(dict_arg, ast.Dict):
                for val in dict_arg.values:
                    if val is not None:
                        resolved = self._peel_to_resolve(val)
                        if resolved is not None and self._is_dangerous_resolved(
                            resolved
                        ):
                            self.flags.append(self._flag_for_resolved(resolved))
                            return
            # W19 C3: Resolve Name reference to aliased dicts
            # Catches: ns = {"run": os.system}; type("C", (), ns)
            elif isinstance(dict_arg, ast.Name):
                resolved = (
                    self._name_aliases.get(dict_arg.id)
                    or self._module_aliases.get(dict_arg.id)
                )
                if resolved is not None and self._is_dangerous_resolved(resolved):
                    self.flags.append(self._flag_for_resolved(resolved))
                    return
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
            # S7: Chain contains subscripts/calls (e.g., foo()[0].bar.system()).
            # Fallback: check the final attr name against known dangerous funcs.
            if func.attr in _DANGEROUS_OS_FUNCS | _DANGEROUS_SUBPROCESS_FUNCS | _DANGEROUS_ASYNCIO_FUNCS:
                self.flags.append("blocked.shell_injection")
            # W16 HIGH-6: Peel simple wrappers to detect os.environ method calls.
            # Catches: (lambda x: x)(os.environ).pop("PW"),
            #          (env := os.environ).get("PW")
            elif func.attr in _DANGEROUS_OS_ENVIRON_METHODS:
                receiver_resolved = self._peel_to_resolve(func.value)
                if receiver_resolved == "os.environ":
                    self.flags.append("blocked.credential_exfil")
            return

        # Resolve the root through module aliases
        parts = chain.split(".", 1)
        root = parts[0]
        if root in self._module_aliases:
            resolved = self._module_aliases[root]
            chain = f"{resolved}.{parts[1]}" if len(parts) > 1 else resolved
        elif root in self._name_aliases:
            # W12.2: Resolve from-import aliases (e.g., `from os import environ`)
            # _name_aliases maps "environ" → "os.environ"
            resolved = self._name_aliases[root]
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
        # W12.3: builtins.__getattribute__() and other dunder methods (Codex finding)
        if module_part == "builtins" and func_name in _DANGEROUS_DUNDER_ATTRS:
            self.flags.append("blocked.shell_injection")
            return
        # W14: __builtins__ is a dangerous mapping — any method call enables
        # sandbox escape (e.g., __builtins__.get('__import__'), .pop(), .values())
        if module_part == "__builtins__":
            self.flags.append("blocked.shell_injection")
            return

        # os.dangerous_func() — shell execution
        if module_part == "os" and func_name in _DANGEROUS_OS_FUNCS:
            self.flags.append("blocked.shell_injection")
            return

        # os.getenv() — credential exfiltration (not shell injection)
        if module_part == "os" and func_name == "getenv":
            self.flags.append("blocked.credential_exfil")
            return

        # os.environ.get/pop/setdefault — credential exfiltration
        if module_part == "os.environ" and func_name in _DANGEROUS_OS_ENVIRON_METHODS:
            self.flags.append("blocked.credential_exfil")
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
        if module_part in _ANY_CALL_BLOCKED_MODULES or any(
            module_part.startswith(m + ".") for m in _ANY_CALL_BLOCKED_MODULES
        ):
            self.flags.append("blocked.shell_injection")
            return

        # Any call on a dangerous exec/meta module's submodule (e.g., importlib.util.*)
        root_module = module_part.split(".")[0]
        if root_module in _DANGEROUS_EXEC_MODULES or root_module in _DANGEROUS_META_MODULES:
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
        """Check ``getattr(module, attr)`` for dangerous module + dynamic attr.

        S8: Handles both ``getattr(os, ...)`` (ast.Name) and
        ``getattr(some.module, ...)`` (ast.Attribute).
        W12.2: Also blocks getattr(ANY_object, dunder_attr) for sandbox escapes.
        """
        if len(node.args) < 2:
            return

        # W12.2+W16 HIGH-3: Block getattr(ANY_obj, dunder_attr) regardless of
        # receiver type.  W16: also resolve Name→Constant via alias tables to
        # catch `a = "__globals__"; getattr(f, a)`.
        attr_arg = _get_constant_arg(node, 1)
        if attr_arg is None:
            # W16 HIGH-3: try constant propagation through name aliases
            second = node.args[1]
            if isinstance(second, ast.Name):
                attr_arg = self._name_aliases.get(second.id)
                if attr_arg is None and second.id in self._module_aliases:
                    attr_arg = self._module_aliases[second.id]
            # W17 H4: try constant folding for string concatenation
            elif isinstance(second, ast.BinOp) and isinstance(second.op, ast.Add):
                left = self._resolve_string_expr(second.left)
                right = self._resolve_string_expr(second.right)
                if left is not None and right is not None:
                    attr_arg = left + right
        if attr_arg is not None and attr_arg in _DANGEROUS_DUNDER_ATTRS:
            self.flags.append("blocked.shell_injection")
            return

        obj = node.args[0]
        resolved_mod: Optional[str] = None

        if isinstance(obj, ast.Name):
            # W15.2: Also resolve through _name_aliases (Codex HIGH)
            # e.g., `from os import environ as e` → _name_aliases["e"] = "os.environ"
            resolved_mod = self._module_aliases.get(obj.id)
            if resolved_mod is None:
                resolved_mod = self._name_aliases.get(obj.id, obj.id)
        elif isinstance(obj, ast.Attribute):
            # S8: getattr(some.module, "func") — resolve the attribute chain
            chain = _resolve_attr_chain(obj)
            if chain is not None:
                parts = chain.split(".", 1)
                root = parts[0]
                if root in self._module_aliases:
                    resolved_mod = f"{self._module_aliases[root]}.{parts[1]}" if len(parts) > 1 else self._module_aliases[root]
                else:
                    resolved_mod = chain

        if resolved_mod is None:
            return

        # W14: __builtins__ is a dangerous mapping — any getattr is shell injection
        if resolved_mod == "__builtins__":
            self.flags.append("blocked.shell_injection")
            return

        # W14: getattr(sys, attr) — check _DANGEROUS_SYS_ATTRS (Codex CRIT)
        if resolved_mod == "sys":
            if attr_arg is None:
                self.flags.append("blocked.shell_injection")
            elif attr_arg in _DANGEROUS_SYS_ATTRS:
                self.flags.append("blocked.shell_injection")
            return

        # W14: getattr(os.environ, method) — check environ methods (Codex HIGH)
        if resolved_mod == "os.environ":
            if attr_arg is None:
                self.flags.append("blocked.credential_exfil")
            elif attr_arg in _DANGEROUS_OS_ENVIRON_METHODS:
                self.flags.append("blocked.credential_exfil")
            return

        if resolved_mod in _ALL_DANGEROUS_MODULES or resolved_mod == "os":
            if attr_arg is None:
                # Dynamic attr on dangerous module — fail-closed
                flag = ("blocked.credential_exfil"
                        if resolved_mod in _DANGEROUS_EXFIL_MODULES
                        else "blocked.shell_injection")
                self.flags.append(flag)
            elif resolved_mod == "os" and attr_arg in _DANGEROUS_OS_FUNCS:
                self.flags.append("blocked.shell_injection")
            elif resolved_mod == "os" and attr_arg == "getenv":
                # W12.2: getattr(os, "getenv") — credential exfiltration
                self.flags.append("blocked.credential_exfil")
            elif resolved_mod == "os" and attr_arg == "environ":
                # W12.2: getattr(os, "environ") — gateway to credentials
                self.flags.append("blocked.credential_exfil")
            elif resolved_mod == "subprocess" and attr_arg in _DANGEROUS_SUBPROCESS_FUNCS:
                self.flags.append("blocked.shell_injection")

    # --- S2: Attribute access on dangerous modules (non-call) ---

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Detect dangerous attribute access even without a call.

        Catches: sys.modules, sys._getframe (S2), dunder escape attrs (W12.1).
        """
        # W12.1: Dangerous dunder attributes — sandbox escape via introspection
        if node.attr in _DANGEROUS_DUNDER_ATTRS:
            self.flags.append("blocked.shell_injection")

        if isinstance(node.value, ast.Name):
            obj_name = node.value.id
            resolved = self._module_aliases.get(obj_name, obj_name)
            if resolved == "sys" and node.attr in _DANGEROUS_SYS_ATTRS:
                self.flags.append("blocked.shell_injection")
        self.generic_visit(node)

    # --- M3: Subscript access on os.environ ---

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Detect dangerous subscript access patterns.

        Handles:
        - os.environ[...] — credential exfiltration (M3/W12.2)
        - __builtins__[...] — sandbox escape via dict-style import (W12.3)
        """
        # W12.3+W16: __builtins__['__import__'] etc. — also via alias
        if isinstance(node.value, ast.Name):
            builtins_name = node.value.id
            if builtins_name == "__builtins__" or self._name_aliases.get(builtins_name) == "__builtins__":
                self.flags.append("blocked.shell_injection")
                self.generic_visit(node)
                return

        is_os_environ = False

        if isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
            if isinstance(node.value.value, ast.Name):
                obj_name = node.value.value.id
                resolved = self._module_aliases.get(obj_name, obj_name)
                if resolved == "os":
                    is_os_environ = True
        elif isinstance(node.value, ast.Name):
            # W12.2: from os import environ (as alias); environ["KEY"]
            name = node.value.id
            resolved = self._name_aliases.get(name, "")
            if resolved == "os.environ":
                is_os_environ = True

        if is_os_environ:
            self._check_environ_subscript_key(node)

        self.generic_visit(node)

    def _check_environ_subscript_key(self, node: ast.Subscript) -> None:
        """Check an os.environ[key] subscript for sensitive or dynamic keys."""
        key = None
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            key = node.slice.value
        if key is not None:
            key_upper = key.upper()
            sensitive_keywords = {"KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PRIVATE"}
            if any(kw in key_upper for kw in sensitive_keywords):
                self.flags.append("blocked.credential_exfil")
        else:
            # Non-constant key — fail-closed
            self.flags.append("blocked.credential_exfil")

    # --- Main visit_Call (single entry point for all call analysis) ---

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # Check open() for sensitive paths
        if isinstance(func, ast.Name) and func.id == "open":
            self._check_open_sensitive(node)
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            self._check_open_sensitive(node)

        # W12.3: dict(os.environ) / list(os.environ.items()) — whole-env copy
        if isinstance(func, ast.Name) and func.id in ("dict", "list"):
            self._check_environ_containerization(node)

        # Main dispatch
        if isinstance(func, ast.Name):
            self._check_simple_call(func.id, node)
        elif isinstance(func, ast.Attribute):
            self._check_attribute_call(func, node)

        self.generic_visit(node)

    def _check_environ_containerization(self, node: ast.Call) -> None:
        """W12.3: Detect dict(os.environ), list(os.environ.items()), etc.
        W14: Also detect dict(__builtins__) — sandbox escape via mapping copy.

        Catches whole-environment copy patterns that bypass per-key checks.
        """
        if not node.args:
            return
        arg = node.args[0]

        # W14: dict(__builtins__) / list(__builtins__) — sandbox escape (Codex CRIT)
        if isinstance(arg, ast.Name) and arg.id == "__builtins__":
            self.flags.append("blocked.shell_injection")
            return

        # dict(os.environ) or list(os.environ)
        if self._is_os_environ_expr(arg):
            self.flags.append("blocked.credential_exfil")
            return

        # list(os.environ.items()) / list(os.environ.values())
        if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute)
                and arg.func.attr in ("items", "values", "keys")):
            if self._is_os_environ_expr(arg.func.value):
                self.flags.append("blocked.credential_exfil")

    def _is_os_environ_expr(self, node: ast.expr) -> bool:
        """Check if *node* resolves to ``os.environ``."""
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            if isinstance(node.value, ast.Name):
                resolved = self._module_aliases.get(node.value.id, node.value.id)
                return resolved == "os"
        if isinstance(node, ast.Name):
            resolved = self._name_aliases.get(node.id, "")
            return resolved == "os.environ"
        return False

    def _check_open_sensitive(self, node: ast.Call) -> None:
        """Flag open() calls with sensitive or non-constant paths.

        S3: Non-constant paths are fail-closed (could be variable-based exfil).
        Constant paths are checked against _SENSITIVE_PATH_FRAGMENTS.
        """
        path_arg = _get_constant_arg(node, 0)
        if path_arg is None:
            # S3: Non-constant path — fail-closed (could exfiltrate via variable)
            self.flags.append("blocked.credential_exfil")
            return

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

def check_ast_safety(source: str, *, fail_closed: bool = True) -> List[str]:
    """Parse a single Python source string and check for dangerous operations.

    Returns a list of triggered flag names (empty = safe).

    Args:
        source: Python source code to analyze.
        fail_closed: S1 fix — if True (default), SyntaxError returns
            ``["blocked.unparseable_code"]``.  Set to False for speculative
            parsing (e.g., trying markdown as Python).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        if fail_closed:
            logger.warning("AST parse failed — fail-closed (S1): %s", exc)
            return ["blocked.unparseable_code"]
        logger.debug("AST parse failed — fail-open (speculative): %s", exc)
        return []

    visitor = DangerousNodeVisitor()
    visitor.visit(tree)

    # Deduplicate while preserving order
    return list(dict.fromkeys(visitor.flags))


def check_python_blocks_safety(text: str) -> List[str]:
    """Check all Python code blocks in markdown/text for dangerous operations.

    Extracts fenced Python blocks and heredoc Python from the text.
    If no fenced blocks are found, tries to parse the entire text as Python
    (fast path for plain .py files — fail-open since it's speculative).

    Returns a deduplicated list of triggered flag names.
    """
    blocks = extract_python_blocks(text)

    if not blocks:
        # No fenced blocks found — try parsing the entire text as Python.
        # Speculative: fail-open since this might be markdown, not Python.
        return check_ast_safety(text, fail_closed=False)

    all_flags: List[str] = []
    for block in blocks:
        all_flags.extend(check_ast_safety(block))

    # Deduplicate while preserving order
    return list(dict.fromkeys(all_flags))
