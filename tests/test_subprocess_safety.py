"""Tests for openspace.local_server.subprocess_safety module.

W13 hardening: verifies environment variable sanitization, secure temp file
creation, and resource limit preamble contents.
"""

from __future__ import annotations

import os
import stat
import tempfile
from unittest import mock

import pytest

from openspace.local_server.subprocess_safety import (
    BASH_RLIMIT_PREAMBLE,
    PYTHON_RLIMIT_PREAMBLE,
    _DANGEROUS_ENV_VARS,
    create_secure_temp_file,
    sanitize_env,
    validate_conda_env,
)


# ---------------------------------------------------------------------------
# sanitize_env
# ---------------------------------------------------------------------------

class TestSanitizeEnv:
    """Environment variable sanitization."""

    def test_returns_new_dict(self):
        """Must return a new dict, never mutate os.environ."""
        result = sanitize_env(None)
        assert result is not os.environ

    def test_none_returns_copy_of_environ(self):
        result = sanitize_env(None)
        assert "PATH" in result  # inherits os.environ

    def test_empty_dict_returns_copy_of_environ(self):
        result = sanitize_env({})
        assert "PATH" in result

    def test_safe_vars_merged(self):
        result = sanitize_env({"MY_CUSTOM_VAR": "hello"})
        assert result["MY_CUSTOM_VAR"] == "hello"

    def test_blocks_ld_preload(self):
        result = sanitize_env({"LD_PRELOAD": "/evil.so"})
        assert "LD_PRELOAD" not in result

    def test_blocks_dyld_insert_libraries(self):
        result = sanitize_env({"DYLD_INSERT_LIBRARIES": "/evil.dylib"})
        assert "DYLD_INSERT_LIBRARIES" not in result

    def test_blocks_pythonpath(self):
        result = sanitize_env({"PYTHONPATH": "/evil"})
        assert "PYTHONPATH" not in result

    def test_blocks_node_options(self):
        result = sanitize_env({"NODE_OPTIONS": "--require /evil.js"})
        assert "NODE_OPTIONS" not in result

    def test_blocks_path_hijack(self):
        result = sanitize_env({"PATH": "/evil/bin:/usr/bin"})
        # PATH is in _DANGEROUS_ENV_VARS so user-provided PATH is blocked.
        # The os.environ base copy already provides a safe PATH.
        assert result.get("PATH") != "/evil/bin:/usr/bin"

    def test_blocks_all_explicit_dangerous_vars(self):
        """Every var in _DANGEROUS_ENV_VARS must be stripped."""
        user_env = {var: "malicious" for var in _DANGEROUS_ENV_VARS}
        result = sanitize_env(user_env)
        for var in _DANGEROUS_ENV_VARS:
            assert result.get(var) != "malicious", f"{var} was not blocked"

    def test_blocks_unknown_ld_prefix(self):
        """Any LD_* prefix should be blocked (catch-all)."""
        result = sanitize_env({"LD_UNKNOWN_FUTURE": "/evil"})
        assert "LD_UNKNOWN_FUTURE" not in result

    def test_blocks_unknown_dyld_prefix(self):
        """Any DYLD_* prefix should be blocked (catch-all)."""
        result = sanitize_env({"DYLD_UNKNOWN_FUTURE": "/evil"})
        assert "DYLD_UNKNOWN_FUTURE" not in result

    def test_blocks_bash_func_injection(self):
        """W13.2: BASH_FUNC_* shell function injection must be blocked."""
        result = sanitize_env({"BASH_FUNC_evil%%": "() { evil; }"})
        assert "BASH_FUNC_evil%%" not in result

    def test_blocks_python_prefixed_vars(self):
        """W13.2: PYTHON* vars (import hooks, debug) must be blocked."""
        result = sanitize_env({"PYTHONDONTWRITEBYTECODE": "1"})
        assert result.get("PYTHONDONTWRITEBYTECODE") != "1"

    def test_case_insensitive_blocking(self):
        """Blocking compares upper-cased keys."""
        result = sanitize_env({"ld_preload": "/evil.so"})
        assert "ld_preload" not in result

    def test_safe_and_dangerous_mixed(self):
        """Safe vars survive while dangerous are stripped."""
        result = sanitize_env({
            "SAFE_VAR": "ok",
            "LD_PRELOAD": "/evil",
            "ANOTHER_SAFE": "fine",
        })
        assert result["SAFE_VAR"] == "ok"
        assert result["ANOTHER_SAFE"] == "fine"
        assert "LD_PRELOAD" not in result

    def test_logs_warning_for_blocked_vars(self):
        with mock.patch("openspace.local_server.subprocess_safety.logger") as mock_logger:
            sanitize_env({"LD_PRELOAD": "/evil", "NODE_OPTIONS": "--bad"})
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "2" in str(call_args)  # blocked count


# ---------------------------------------------------------------------------
# create_secure_temp_file
# ---------------------------------------------------------------------------

class TestCreateSecureTempFile:
    """Secure temporary file creation."""

    def test_returns_fd_and_path(self):
        fd, path = create_secure_temp_file()
        try:
            assert isinstance(fd, int)
            assert isinstance(path, str)
            assert os.path.exists(path)
        finally:
            os.close(fd)
            os.unlink(path)

    def test_permissions_owner_only(self):
        fd, path = create_secure_temp_file()
        try:
            st = os.fstat(fd)
            mode = stat.S_IMODE(st.st_mode)
            assert mode == stat.S_IRWXU, f"Expected 0o700, got {oct(mode)}"
        finally:
            os.close(fd)
            os.unlink(path)

    def test_suffix_respected(self):
        fd, path = create_secure_temp_file(suffix=".sh")
        try:
            assert path.endswith(".sh")
        finally:
            os.close(fd)
            os.unlink(path)

    def test_prefix_respected(self):
        fd, path = create_secure_temp_file(prefix="test_exec_")
        try:
            assert "test_exec_" in os.path.basename(path)
        finally:
            os.close(fd)
            os.unlink(path)

    def test_file_is_writable(self):
        fd, path = create_secure_temp_file()
        try:
            os.write(fd, b"#!/usr/bin/env python3\nprint('hello')\n")
        finally:
            os.close(fd)
            os.unlink(path)

    def test_no_symlink_race(self):
        """mkstemp uses O_EXCL — verify we get a real file, not a symlink."""
        fd, path = create_secure_temp_file()
        try:
            assert not os.path.islink(path)
        finally:
            os.close(fd)
            os.unlink(path)


# ---------------------------------------------------------------------------
# Resource limit preambles
# ---------------------------------------------------------------------------

class TestResourceLimitPreambles:
    """Preamble strings contain expected resource constraints."""

    def test_bash_preamble_has_cpu_limit(self):
        assert "ulimit -t" in BASH_RLIMIT_PREAMBLE

    def test_bash_preamble_has_memory_limit(self):
        assert "ulimit -v" in BASH_RLIMIT_PREAMBLE

    def test_bash_preamble_has_file_size_limit(self):
        assert "ulimit -f" in BASH_RLIMIT_PREAMBLE

    def test_bash_preamble_has_fd_limit(self):
        assert "ulimit -n" in BASH_RLIMIT_PREAMBLE

    def test_python_preamble_has_cpu_limit(self):
        assert "RLIMIT_CPU" in PYTHON_RLIMIT_PREAMBLE

    def test_python_preamble_has_file_size_limit(self):
        assert "RLIMIT_FSIZE" in PYTHON_RLIMIT_PREAMBLE

    def test_python_preamble_has_fd_limit(self):
        assert "RLIMIT_NOFILE" in PYTHON_RLIMIT_PREAMBLE

    def test_python_preamble_linux_only_rlimit_as(self):
        """RLIMIT_AS should only be set on Linux (macOS unsupported)."""
        assert "platform == \"linux\"" in PYTHON_RLIMIT_PREAMBLE
        assert "RLIMIT_AS" in PYTHON_RLIMIT_PREAMBLE

    def test_python_preamble_cleans_up_imports(self):
        """Preamble should delete its helper imports to avoid namespace pollution."""
        assert "del _os_rlimit" in PYTHON_RLIMIT_PREAMBLE


# ---------------------------------------------------------------------------
# _DANGEROUS_ENV_VARS completeness
# ---------------------------------------------------------------------------

class TestDangerousEnvVarsCompleteness:
    """Verify the blocklist covers known attack vectors."""

    @pytest.mark.parametrize("var", [
        "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT",
        "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
        "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME",
        "PATH", "SHELL", "BASH_ENV", "ENV",
        "NODE_PATH", "NODE_OPTIONS",
        "RUBYLIB", "PERL5LIB", "PERL5OPT",
    ])
    def test_critical_var_in_blocklist(self, var: str):
        assert var in _DANGEROUS_ENV_VARS, f"{var} missing from _DANGEROUS_ENV_VARS"


# ---------------------------------------------------------------------------
# W14: Inherited parent env stripping
# ---------------------------------------------------------------------------

class TestInheritedEnvStripping:
    """W14: sanitize_env must strip injection-class vars from inherited env."""

    def test_strips_inherited_ld_preload(self, monkeypatch):
        """CRIT: LD_PRELOAD in parent env must be stripped."""
        monkeypatch.setenv("LD_PRELOAD", "/tmp/evil.so")
        result = sanitize_env(None)
        assert "LD_PRELOAD" not in result

    def test_strips_inherited_bash_env(self, monkeypatch):
        """HIGH: BASH_ENV in parent env must be stripped."""
        monkeypatch.setenv("BASH_ENV", "/tmp/evil.sh")
        result = sanitize_env(None)
        assert "BASH_ENV" not in result

    def test_strips_inherited_dyld(self, monkeypatch):
        """CRIT: DYLD_INSERT_LIBRARIES in parent env must be stripped."""
        monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/tmp/evil.dylib")
        result = sanitize_env(None)
        assert "DYLD_INSERT_LIBRARIES" not in result

    def test_strips_inherited_bash_func(self, monkeypatch):
        """HIGH: BASH_FUNC_* in parent env must be stripped."""
        monkeypatch.setenv("BASH_FUNC_evil%%", "() { cat /etc/passwd; }")
        result = sanitize_env(None)
        assert "BASH_FUNC_evil%%" not in result

    def test_strips_inherited_pythonpath(self, monkeypatch):
        """HIGH: PYTHONPATH in parent env must be stripped."""
        monkeypatch.setenv("PYTHONPATH", "/tmp/evil")
        result = sanitize_env(None)
        assert "PYTHONPATH" not in result

    def test_strips_inherited_node_options(self, monkeypatch):
        """HIGH: NODE_OPTIONS in parent env must be stripped."""
        monkeypatch.setenv("NODE_OPTIONS", "--require=/tmp/evil.js")
        result = sanitize_env(None)
        assert "NODE_OPTIONS" not in result

    def test_preserves_path_in_base(self, monkeypatch):
        """PATH must be preserved in the base environment."""
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        result = sanitize_env(None)
        assert "PATH" in result

    def test_preserves_shell_in_base(self, monkeypatch):
        """SHELL must be preserved in the base environment."""
        monkeypatch.setenv("SHELL", "/bin/bash")
        result = sanitize_env(None)
        assert "SHELL" in result

    def test_preserves_home_in_base(self, monkeypatch):
        """HOME must be preserved in the base environment."""
        monkeypatch.setenv("HOME", "/home/user")
        result = sanitize_env(None)
        assert "HOME" in result


# ---------------------------------------------------------------------------
# W14: Conda environment name validation
# ---------------------------------------------------------------------------

class TestCondaEnvValidation:
    """W14: validate_conda_env must prevent shell injection."""

    def test_valid_simple_name(self):
        assert validate_conda_env("myenv") == "myenv"

    def test_valid_name_with_dots_dashes(self):
        assert validate_conda_env("my-env.2.0") == "my-env.2.0"

    def test_valid_name_with_underscores(self):
        assert validate_conda_env("my_env_v3") == "my_env_v3"

    def test_empty_returns_none(self):
        assert validate_conda_env("") is None
        assert validate_conda_env(None) is None

    def test_rejects_semicolon_injection(self):
        with pytest.raises(ValueError):
            validate_conda_env("myenv; curl http://evil.com")

    def test_rejects_pipe_injection(self):
        with pytest.raises(ValueError):
            validate_conda_env("myenv | cat /etc/passwd")

    def test_rejects_backtick_injection(self):
        with pytest.raises(ValueError):
            validate_conda_env("myenv`whoami`")

    def test_rejects_dollar_injection(self):
        with pytest.raises(ValueError):
            validate_conda_env("myenv$(whoami)")

    def test_rejects_space_injection(self):
        with pytest.raises(ValueError):
            validate_conda_env("my env")

    def test_rejects_newline_injection(self):
        with pytest.raises(ValueError):
            validate_conda_env("myenv\nwhoami")

    def test_rejects_slash_traversal(self):
        with pytest.raises(ValueError):
            validate_conda_env("../../../etc/passwd")

    def test_rejects_too_long_name(self):
        with pytest.raises(ValueError):
            validate_conda_env("a" * 129)

    def test_rejects_leading_dash(self):
        """Leading dash could be interpreted as option flag."""
        with pytest.raises(ValueError):
            validate_conda_env("-malicious")

    def test_rejects_trailing_newline_only(self):
        """W15.2 LOW: 'myenv\\n' must be rejected.

        Before W15.2, _CONDA_ENV_RE.match() with $ anchor accepted trailing
        newline (Python re quirk: $ matches before final \\n).  Fix: use
        fullmatch() which requires the ENTIRE string to match, no trailing
        newline allowed.
        """
        with pytest.raises(ValueError):
            validate_conda_env("myenv\n")

    def test_rejects_carriage_return_injection(self):
        """W15.2: Carriage return in conda env name must be rejected."""
        with pytest.raises(ValueError):
            validate_conda_env("myenv\r")

    def test_rejects_null_byte_injection(self):
        """W15.2: Null byte in conda env name must be rejected."""
        with pytest.raises(ValueError):
            validate_conda_env("myenv\x00")


# ---------------------------------------------------------------------------
# W18: validate_conda_env type guard
# ---------------------------------------------------------------------------

class TestCondaEnvTypeGuard:
    """W18: validate_conda_env must reject non-string types with TypeError."""

    def test_int_raises_type_error(self):
        with pytest.raises(TypeError, match="must be str"):
            validate_conda_env(123)

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError, match="must be str"):
            validate_conda_env(["myenv"])

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="must be str"):
            validate_conda_env({"name": "myenv"})

    def test_bool_raises_type_error(self):
        """bool is a subclass of int in Python — must still reject."""
        with pytest.raises(TypeError, match="must be str"):
            validate_conda_env(True)

    def test_bytes_raises_type_error(self):
        with pytest.raises(TypeError, match="must be str"):
            validate_conda_env(b"myenv")

    def test_none_returns_none(self):
        """None should still return None (not TypeError)."""
        assert validate_conda_env(None) is None

    def test_empty_string_returns_none(self):
        """Empty string should still return None (not TypeError)."""
        assert validate_conda_env("") is None

    def test_valid_string_passes(self):
        """Normal string should still work."""
        assert validate_conda_env("myenv") == "myenv"


# ---------------------------------------------------------------------------
# W18: RLIMIT fail-open telemetry
# ---------------------------------------------------------------------------

class TestRlimitTelemetry:
    """W18: RLIMIT preambles must emit warnings when limits fail to apply,
    instead of silently passing (fail-open telemetry gap)."""

    def test_bash_preamble_has_failure_warnings(self):
        """Bash preamble must log when ulimit fails."""
        assert "WARN" in BASH_RLIMIT_PREAMBLE
        # Each ulimit line should have a failure handler
        assert "CPU time limit not enforced" in BASH_RLIMIT_PREAMBLE
        assert "virtual memory limit not enforced" in BASH_RLIMIT_PREAMBLE
        assert "file size limit not enforced" in BASH_RLIMIT_PREAMBLE
        assert "file descriptor limit not enforced" in BASH_RLIMIT_PREAMBLE

    def test_bash_preamble_warnings_go_to_stderr(self):
        """Warnings must go to stderr, not stdout (to not corrupt script output)."""
        for line in BASH_RLIMIT_PREAMBLE.split("\n"):
            if "WARN" in line:
                assert ">&2" in line, f"Warning not redirected to stderr: {line}"

    def test_python_preamble_has_failure_warnings(self):
        """Python preamble must log when setrlimit fails."""
        assert "WARN" in PYTHON_RLIMIT_PREAMBLE
        assert "setrlimit" in PYTHON_RLIMIT_PREAMBLE
        assert "stderr" in PYTHON_RLIMIT_PREAMBLE

    def test_python_preamble_no_silent_pass(self):
        """Python preamble must NOT have bare 'pass' in except blocks."""
        # Check that no except block has a bare 'pass' (old fail-open pattern)
        lines = PYTHON_RLIMIT_PREAMBLE.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "pass":
                # Check if prev line is an except — that's the fail-open pattern
                if i > 0 and "except" in lines[i - 1]:
                    pytest.fail(
                        f"Silent 'pass' in except block at line {i}: "
                        f"{lines[i-1].strip()} / {stripped}"
                    )

    def test_python_preamble_still_compiles(self):
        """Preamble must still be valid Python after telemetry changes."""
        compile(PYTHON_RLIMIT_PREAMBLE, "<preamble>", "exec")

    def test_python_preamble_cleans_up_all_names(self):
        """Preamble must still delete temp names after telemetry additions."""
        assert "del _os_rlimit" in PYTHON_RLIMIT_PREAMBLE
        assert "del" in PYTHON_RLIMIT_PREAMBLE
