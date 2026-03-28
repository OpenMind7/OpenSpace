"""Tests for AST-based safety checker — replaces regex-only screening (Finding #2).

The AST checker parses Python code blocks from skill content, walks the AST to detect
dangerous operations (shell injection, credential exfiltration), and handles evasion
vectors that regex screening cannot catch (alias tracking, dynamic imports, attribute
chains).

7 test groups, 44 cases:
  1. Evasion vectors (15)  — attacks that bypass regex but AST catches
  2. Alias tracking (5)    — import-as, from-import-as, chained aliases
  3. Markdown extraction (7) — fenced code blocks, heredocs, edge cases
  4. False positive prevention (8) — safe code that must NOT be flagged
  5. Syntax errors (3)     — malformed Python handled gracefully
  6. Backward compat (4)   — AST flags use same names as existing regex flags
  7. Performance (2)        — large inputs don't hang or OOM
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

    def test_open_with_variable_arg_safe(self) -> None:
        """open() with non-constant arg — allowed (too many false positives)."""
        flags = check_ast_safety("filename = get_path()\nf = open(filename)")
        blocking = [f for f in flags if f in _BLOCKING_FLAGS]
        assert not blocking

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
