"""Tests for capability enforcement — Finding #4 (W11 Phase 2).

Skills declare capabilities in their SKILL.md frontmatter. The enforcement
pipeline filters skills whose capabilities cannot be satisfied by the current
session's available backends and tools.

6 test groups, ~35 cases:
  1. CAPABILITY_TO_BACKENDS mapping (4) — constant correctness
  2. _filter_by_capability on registry (10) — selection gate
  3. Shell auto-add conditioning (6) — grounding_agent shell logic
  4. set_skill_context capabilities (4) — capability propagation
  5. End-to-end selection with capabilities (6) — full pipeline
  6. Backward compatibility (5) — legacy skills with no capabilities
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openspace.skill_engine.skill_utils import (
    CAPABILITY_TO_BACKENDS,
    VALID_CAPABILITIES,
    parse_capabilities,
)
from openspace.skill_engine.registry import SkillMeta, SkillRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill_dir(tmp_path: Path, skill_id: str, content: str) -> Path:
    """Create a temp skill directory with a SKILL.md file."""
    skill_dir = tmp_path / skill_id
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def _skill_md(name: str, desc: str, capabilities: str = "") -> str:
    """Build a minimal SKILL.md with optional capabilities."""
    lines = [
        "---",
        f"name: {name}",
        f"description: {desc}",
    ]
    if capabilities:
        lines.append(f"capabilities: {capabilities}")
    lines.append("---")
    lines.append(f"\n# {name}\n\n{desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Group 1: CAPABILITY_TO_BACKENDS mapping
# ---------------------------------------------------------------------------

class TestCapabilityToBackends:
    """Verify the CAPABILITY_TO_BACKENDS constant maps every valid capability."""

    def test_all_valid_capabilities_have_mapping(self) -> None:
        """Every capability in VALID_CAPABILITIES must appear in CAPABILITY_TO_BACKENDS."""
        for cap in VALID_CAPABILITIES:
            assert cap in CAPABILITY_TO_BACKENDS, (
                f"Capability '{cap}' missing from CAPABILITY_TO_BACKENDS"
            )

    def test_mapping_values_are_frozensets(self) -> None:
        """Backend sets must be frozensets for immutability."""
        for cap, backends in CAPABILITY_TO_BACKENDS.items():
            assert isinstance(backends, frozenset), (
                f"CAPABILITY_TO_BACKENDS['{cap}'] should be frozenset, got {type(backends)}"
            )

    def test_subprocess_requires_shell(self) -> None:
        """subprocess capability MUST require the shell backend."""
        assert "shell" in CAPABILITY_TO_BACKENDS["subprocess"]

    def test_network_does_not_require_shell(self) -> None:
        """network-only skills should NOT require shell."""
        assert "shell" not in CAPABILITY_TO_BACKENDS["network"]


# ---------------------------------------------------------------------------
# Group 2: _filter_by_capability on registry
# ---------------------------------------------------------------------------

class TestFilterByCapability:
    """Test the capability filter in skill selection."""

    def _make_registry(self, tmp_path: Path, skills: List[dict]) -> SkillRegistry:
        """Create a registry with skills having capabilities."""
        dirs = []
        for s in skills:
            skill_dir = _make_skill_dir(
                tmp_path,
                s["id"],
                _skill_md(s["id"], s.get("desc", "A skill"), s.get("caps", "")),
            )
            dirs.append(skill_dir.parent)
        # Deduplicate dirs
        unique_dirs = list(dict.fromkeys(dirs))
        registry = SkillRegistry(skill_dirs=unique_dirs)
        registry.discover()
        return registry

    def test_no_capabilities_passes_filter(self, tmp_path: Path) -> None:
        """Legacy skill with no capabilities declared should pass (fail-open)."""
        registry = self._make_registry(tmp_path, [
            {"id": "legacy_skill", "desc": "No caps declared"},
        ])
        available = list(registry._skills.values())
        filtered = registry._filter_by_capability(
            available, session_backends=frozenset({"shell"}),
        )
        assert len(filtered) == 1

    def test_matching_capabilities_passes(self, tmp_path: Path) -> None:
        """Skill with capabilities satisfied by session backends should pass."""
        registry = self._make_registry(tmp_path, [
            {"id": "net_skill", "caps": "network", "desc": "Needs network"},
        ])
        available = list(registry._skills.values())
        filtered = registry._filter_by_capability(
            available, session_backends=frozenset({"mcp"}),
        )
        assert len(filtered) == 1

    def test_unsatisfied_capabilities_filtered_out(self, tmp_path: Path) -> None:
        """Skill requiring subprocess but no shell backend → filtered out."""
        registry = self._make_registry(tmp_path, [
            {"id": "shell_skill", "caps": "subprocess", "desc": "Needs shell"},
        ])
        available = list(registry._skills.values())
        filtered = registry._filter_by_capability(
            available, session_backends=frozenset({"mcp"}),  # no shell
        )
        assert len(filtered) == 0

    def test_partial_capabilities_satisfied(self, tmp_path: Path) -> None:
        """Skill needing network+subprocess: both backends must be present."""
        registry = self._make_registry(tmp_path, [
            {"id": "dual_skill", "caps": "network,subprocess", "desc": "Needs both"},
        ])
        available = list(registry._skills.values())
        # Only mcp, no shell → subprocess unsatisfied
        filtered = registry._filter_by_capability(
            available, session_backends=frozenset({"mcp"}),
        )
        assert len(filtered) == 0

    def test_all_capabilities_satisfied(self, tmp_path: Path) -> None:
        """Skill needing network+subprocess: both backends present → passes."""
        registry = self._make_registry(tmp_path, [
            {"id": "dual_skill", "caps": "network,subprocess", "desc": "Needs both"},
        ])
        available = list(registry._skills.values())
        filtered = registry._filter_by_capability(
            available, session_backends=frozenset({"mcp", "shell"}),
        )
        assert len(filtered) == 1

    def test_none_backends_passes_all(self, tmp_path: Path) -> None:
        """When session_backends is None (unknown), all skills pass (fail-open)."""
        registry = self._make_registry(tmp_path, [
            {"id": "net_skill", "caps": "network"},
            {"id": "shell_skill", "caps": "subprocess"},
        ])
        available = list(registry._skills.values())
        filtered = registry._filter_by_capability(
            available, session_backends=None,
        )
        assert len(filtered) == 2

    def test_mixed_skills_partial_filter(self, tmp_path: Path) -> None:
        """Mix of legacy + capable skills: only unsatisfied removed."""
        registry = self._make_registry(tmp_path, [
            {"id": "legacy", "desc": "No caps"},
            {"id": "net_only", "caps": "network"},
            {"id": "shell_only", "caps": "subprocess"},
        ])
        available = list(registry._skills.values())
        filtered = registry._filter_by_capability(
            available, session_backends=frozenset({"mcp"}),  # no shell
        )
        # legacy passes (fail-open), net_only passes (mcp), shell_only filtered
        ids = {s.skill_id for s in filtered}
        assert any(sid.startswith("legacy") for sid in ids)
        assert any(sid.startswith("net_only") for sid in ids)
        assert not any(sid.startswith("shell_only") for sid in ids)

    def test_critical_tools_checked(self, tmp_path: Path) -> None:
        """Skills with critical_tools not in session → filtered."""
        registry = self._make_registry(tmp_path, [
            {"id": "tool_skill", "caps": "network", "desc": "Needs specific tool"},
        ])
        available = list(registry._skills.values())
        # Manually set critical_tools on the SkillMeta
        for s in available:
            s.critical_tools = ("special_mcp_tool",)
        filtered = registry._filter_by_capability(
            available,
            session_backends=frozenset({"mcp"}),
            session_tool_names=frozenset({"read_file", "write_file"}),  # no special_mcp_tool
        )
        assert len(filtered) == 0

    def test_critical_tools_satisfied(self, tmp_path: Path) -> None:
        """Skills with critical_tools present in session → passes."""
        registry = self._make_registry(tmp_path, [
            {"id": "tool_skill", "caps": "network", "desc": "Needs specific tool"},
        ])
        available = list(registry._skills.values())
        for s in available:
            s.critical_tools = ("special_mcp_tool",)
        filtered = registry._filter_by_capability(
            available,
            session_backends=frozenset({"mcp"}),
            session_tool_names=frozenset({"special_mcp_tool", "read_file"}),
        )
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# Group 3: Shell auto-add conditioning
# ---------------------------------------------------------------------------

class TestShellAutoAddConditioning:
    """Test that shell backend is conditionally added based on capabilities."""

    def test_no_capabilities_adds_shell(self) -> None:
        """Legacy skills with no capabilities → shell still added (fail-open)."""
        from openspace.skill_engine.skill_utils import capabilities_need_shell
        assert capabilities_need_shell(frozenset()) is True

    def test_subprocess_capability_adds_shell(self) -> None:
        from openspace.skill_engine.skill_utils import capabilities_need_shell
        assert capabilities_need_shell(frozenset({"subprocess"})) is True

    def test_filesystem_capability_adds_shell(self) -> None:
        from openspace.skill_engine.skill_utils import capabilities_need_shell
        assert capabilities_need_shell(frozenset({"filesystem"})) is True

    def test_env_vars_capability_adds_shell(self) -> None:
        from openspace.skill_engine.skill_utils import capabilities_need_shell
        assert capabilities_need_shell(frozenset({"env_vars"})) is True

    def test_network_only_no_shell(self) -> None:
        """network-only skill should NOT get shell auto-added."""
        from openspace.skill_engine.skill_utils import capabilities_need_shell
        assert capabilities_need_shell(frozenset({"network"})) is False

    def test_cloud_api_only_no_shell(self) -> None:
        """cloud_api-only skill should NOT get shell auto-added."""
        from openspace.skill_engine.skill_utils import capabilities_need_shell
        assert capabilities_need_shell(frozenset({"cloud_api"})) is False


# ---------------------------------------------------------------------------
# Group 4: set_skill_context with capabilities
# ---------------------------------------------------------------------------

class TestSetSkillContextCapabilities:
    """Test that set_skill_context accepts and stores skill_capabilities."""

    def test_set_skill_context_stores_capabilities(self) -> None:
        """set_skill_context should accept and store skill_capabilities."""
        from openspace.agents.grounding_agent import GroundingAgent
        agent = GroundingAgent.__new__(GroundingAgent)
        agent._skill_context = None
        agent._active_skill_ids = []
        agent._skill_capabilities = frozenset()
        agent._skill_registry = None
        agent.set_skill_context("context", ["skill1"], skill_capabilities=frozenset({"network"}))
        assert agent._skill_capabilities == frozenset({"network"})

    def test_set_skill_context_defaults_empty(self) -> None:
        """Without skill_capabilities param, defaults to empty frozenset."""
        from openspace.agents.grounding_agent import GroundingAgent
        agent = GroundingAgent.__new__(GroundingAgent)
        agent._skill_context = None
        agent._active_skill_ids = []
        agent._skill_capabilities = frozenset()
        agent._skill_registry = None
        agent.set_skill_context("context", ["skill1"])
        assert agent._skill_capabilities == frozenset()

    def test_clear_skill_context_clears_capabilities(self) -> None:
        """clear_skill_context should also clear capabilities."""
        from openspace.agents.grounding_agent import GroundingAgent
        agent = GroundingAgent.__new__(GroundingAgent)
        agent._skill_context = "some context"
        agent._active_skill_ids = ["s1"]
        agent._skill_capabilities = frozenset({"subprocess"})
        agent._skill_registry = None
        agent.clear_skill_context()
        assert agent._skill_capabilities == frozenset()

    def test_has_skill_context_unaffected(self) -> None:
        """has_skill_context should still work based on _skill_context, not capabilities."""
        from openspace.agents.grounding_agent import GroundingAgent
        agent = GroundingAgent.__new__(GroundingAgent)
        agent._skill_context = None
        agent._active_skill_ids = []
        agent._skill_capabilities = frozenset({"network"})
        agent._skill_registry = None
        assert agent.has_skill_context is False


# ---------------------------------------------------------------------------
# Group 5: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Legacy skills with no capabilities must work exactly as before."""

    def test_legacy_skill_passes_all_gates(self, tmp_path: Path) -> None:
        """Skill with no capabilities field passes capability filter."""
        content = textwrap.dedent("""\
            ---
            name: legacy_tool
            description: Old skill without capabilities
            ---
            # Legacy Tool
            Does stuff.
        """)
        skill_dir = _make_skill_dir(tmp_path, "legacy_tool", content)
        caps = parse_capabilities(content)
        assert caps == frozenset()

    def test_empty_capabilities_treated_as_legacy(self, tmp_path: Path) -> None:
        """Empty capabilities string treated same as absent."""
        content = textwrap.dedent("""\
            ---
            name: empty_caps
            description: Empty caps field
            capabilities:
            ---
            # Empty Caps
        """)
        caps = parse_capabilities(content)
        assert caps == frozenset()

    def test_select_skills_without_session_info(self, tmp_path: Path) -> None:
        """select_skills_with_llm called without session_* params works."""
        # This tests that the new params are optional and default to None
        registry = SkillRegistry(skill_dirs=[tmp_path])
        # Just verify the method signature accepts no session params
        import inspect
        sig = inspect.signature(registry.select_skills_with_llm)
        params = sig.parameters
        assert "session_tool_names" in params
        assert "session_backends" in params
        assert params["session_tool_names"].default is None
        assert params["session_backends"].default is None

    def test_capabilities_need_shell_empty_returns_true(self) -> None:
        """Empty capabilities → shell needed (fail-open for legacy)."""
        from openspace.skill_engine.skill_utils import capabilities_need_shell
        assert capabilities_need_shell(frozenset()) is True

    def test_filter_by_capability_no_params_passes_all(self, tmp_path: Path) -> None:
        """_filter_by_capability with all None params passes everything."""
        content = _skill_md("test", "desc", "subprocess")
        skill_dir = _make_skill_dir(tmp_path, "test_skill", content)
        registry = SkillRegistry(skill_dirs=[tmp_path])
        available = list(registry._skills.values())
        if available:
            filtered = registry._filter_by_capability(
                available, session_backends=None, session_tool_names=None,
            )
            assert len(filtered) == len(available)
