# Capability Enforcement Design — Gate Patch 7

**Status**: Design complete, ready for implementation
**Finding**: Codex Finding #4 (HIGH) — skills declare capabilities but they are never enforced
**Date**: 2026-03-28

---

## 1. Problem Statement

Skills declare `capabilities` in SKILL.md frontmatter (parsed by `skill_utils.parse_capabilities()`),
and `critical_tools`/`tool_dependencies` are stored in `SkillRecord` (via `store.py`). However, this
data is **never enforced** at any point in the pipeline:

1. **Selection** (`registry.py`) scores by relevance/quality only — never checks whether the
   session actually has the tools or backends a skill requires.
2. **Tool dispatch** (`grounding_agent.py:_get_available_tools`) does not restrict tools based
   on the selected skill's declared capabilities.
3. **Shell auto-add** (`grounding_agent.py:556-559`) unconditionally adds the Shell backend
   whenever `has_skill_context=True`, regardless of whether the skill declares `subprocess`
   or `filesystem` capabilities.

A skill declaring `capabilities: network` can therefore use filesystem, subprocess, or any tool
because nothing gates execution to the declared scope.

---

## 2. Architecture Snapshot (Current)

```
tool_layer.py                          registry.py                      grounding_agent.py
─────────────                          ───────────                      ──────────────────
_select_and_inject_skills()            select_skills_with_llm()         _get_available_tools()
  │                                      │                                │
  ├─ get quality metrics from store      ├─ quality filter               ├─ builds backends list
  ├─ call registry.select_skills_*()     ├─ _prefilter_skills()          ├─ if has_skill_context:
  ├─ ts_blend_reorder()                  ├─ LLM selection                │     auto-add Shell  ← PROBLEM
  ├─ build_context_injection()           ├─ parse response               ├─ get_tools_with_auto_search()
  └─ set_skill_context()                 └─ return SkillMeta list        └─ return tools
```

**Data that exists but never flows:**
- `skill_utils.parse_capabilities()` → returns `frozenset[str]` from frontmatter
- `SkillRecord.critical_tools` → `List[str]` of must-have tool keys in `skill_tool_deps`
- `SkillRecord.tool_dependencies` → `List[str]` of all tool keys in `skill_tool_deps`
- `VALID_CAPABILITIES` → `{"network", "filesystem", "subprocess", "env_vars", "cloud_api", "gpu"}`

**Key types:**
- `SkillMeta` — lightweight (skill_id, name, description, path). No capabilities.
- `SkillRecord` — full profile (has critical_tools, tool_dependencies, but no capabilities field)
- `BaseTool.name` → tool's string identifier (e.g., `"shell_agent"`, `"read_file"`, `"web_search"`)
- `BackendType` → enum: `SHELL`, `GUI`, `MCP`, `WEB`, `SYSTEM`

---

## 3. Design

### 3.1 Expand SkillMeta with Capabilities

`SkillMeta` is the data structure that flows through the entire selection pipeline. It currently
lacks capability data, so the first change is to add it.

**File: `openspace/skill_engine/registry.py`**

```python
@dataclass
class SkillMeta:
    skill_id: str
    name: str
    description: str
    path: Path
    # NEW: parsed from frontmatter during discover()
    capabilities: frozenset[str] = field(default_factory=frozenset)
    critical_tools: List[str] = field(default_factory=list)
```

**In `_parse_skill()`** — parse capabilities from content at discovery time:

```python
@staticmethod
def _parse_skill(
    dir_name: str,
    skill_dir: Path,
    skill_file: Path,
    content: str,
) -> SkillMeta:
    from .skill_utils import parse_capabilities, get_frontmatter_field

    frontmatter = parse_frontmatter(content)
    name = frontmatter.get("name", dir_name)
    description = frontmatter.get("description", name)
    skill_id = _read_or_create_skill_id(name, skill_dir)

    # NEW: parse capabilities and critical_tools from frontmatter
    capabilities = parse_capabilities(content)
    raw_ct = get_frontmatter_field(content, "critical_tools")
    critical_tools = (
        [t.strip() for t in raw_ct.split(",") if t.strip()]
        if raw_ct else []
    )

    return SkillMeta(
        skill_id=skill_id,
        name=name,
        description=description,
        path=skill_file,
        capabilities=capabilities,
        critical_tools=critical_tools,
    )
```

### 3.2 Add Capability-to-Backend Mapping

**File: `openspace/skill_engine/skill_utils.py`** — new constant after `VALID_CAPABILITIES`:

```python
# Maps declared capabilities to the BackendType(s) that provide them.
# Used by enforcement gates to determine which backends a skill actually needs.
CAPABILITY_TO_BACKENDS: Dict[str, frozenset[str]] = {
    "network":     frozenset({"web", "mcp"}),       # web backend or MCP servers
    "filesystem":  frozenset({"shell", "system"}),   # shell tools or system tools
    "subprocess":  frozenset({"shell"}),             # shell_agent
    "env_vars":    frozenset({"shell"}),             # shell env access
    "cloud_api":   frozenset({"mcp", "web"}),        # external API via MCP or web
    "gpu":         frozenset({"shell"}),             # GPU compute via shell
}


def required_backends_for_capabilities(
    capabilities: frozenset[str],
) -> frozenset[str]:
    """Return the union of backend names required by the given capabilities."""
    result: set[str] = set()
    for cap in capabilities:
        result.update(CAPABILITY_TO_BACKENDS.get(cap, set()))
    return frozenset(result)
```

### 3.3 Selection Gate (registry.py)

**Where**: After quality-based filtering, before LLM selection — inside
`select_skills_with_llm()` at the point where `available` is the live list.

**What**: Remove skills whose `critical_tools` are not satisfiable by the session's
available tool names. Skills with **no** declared `critical_tools` pass through (fail-open).

**New parameter** on `select_skills_with_llm()`:

```python
async def select_skills_with_llm(
    self,
    task_description: str,
    llm_client: "LLMClient",
    max_skills: int = 2,
    model: Optional[str] = None,
    skill_quality: Optional[Dict[str, Dict[str, Any]]] = None,
    store: Optional["SkillStore"] = None,
    # NEW: tool names available in the current session
    session_tool_names: Optional[frozenset[str]] = None,
    # NEW: backend names active in the current session
    session_backends: Optional[frozenset[str]] = None,
) -> tuple[List[SkillMeta], Optional[Dict[str, Any]]]:
```

**Gate logic** — new method in `SkillRegistry`:

```python
def _filter_by_capability(
    self,
    available: List[SkillMeta],
    session_tool_names: Optional[frozenset[str]],
    session_backends: Optional[frozenset[str]],
) -> tuple[List[SkillMeta], List[str]]:
    """Remove skills whose critical_tools are missing from the session.

    Fail-open: skills with no declared critical_tools are kept.
    Fail-open: if session_tool_names is None (caller didn't provide), all pass.

    Returns:
        (kept, excluded_ids)
    """
    if session_tool_names is None and session_backends is None:
        return available, []

    kept: List[SkillMeta] = []
    excluded: List[str] = []

    for skill in available:
        # Check 1: critical_tools must be present in session
        if skill.critical_tools and session_tool_names is not None:
            missing = [t for t in skill.critical_tools if t not in session_tool_names]
            if missing:
                logger.info(
                    f"Capability gate: excluding {skill.skill_id} — "
                    f"missing critical tools: {missing}"
                )
                excluded.append(skill.skill_id)
                continue

        # Check 2: declared capabilities must map to available backends
        if skill.capabilities and session_backends is not None:
            from .skill_utils import required_backends_for_capabilities
            needed = required_backends_for_capabilities(skill.capabilities)
            if needed and not needed & session_backends:
                logger.info(
                    f"Capability gate: excluding {skill.skill_id} — "
                    f"needs backends {needed}, session has {session_backends}"
                )
                excluded.append(skill.skill_id)
                continue

        kept.append(skill)

    return kept, excluded
```

**Call site** in `select_skills_with_llm()` — insert after quality filter, before prefilter:

```python
    # NEW: Capability gate — exclude skills with unsatisfiable requirements
    capability_excluded: List[str] = []
    if session_tool_names is not None or session_backends is not None:
        available, capability_excluded = self._filter_by_capability(
            available, session_tool_names, session_backends
        )
        if capability_excluded:
            filtered_out.extend(capability_excluded)
```

### 3.4 Tool Dispatch Gate (grounding_agent.py)

**Where**: In `_get_available_tools()`, after tools are fetched, before returning.

**What**: When a skill declares capabilities, filter the returned tool list to only include
tools from backends that are allowed by the skill's declared capabilities. This is a **soft
gate** — it logs warnings and restricts the tool list, but does not crash.

**New state on GroundingAgent**:

```python
def set_skill_context(
    self,
    context: str,
    skill_ids: Optional[List[str]] = None,
    # NEW: capabilities declared by the active skill(s)
    skill_capabilities: Optional[frozenset[str]] = None,
) -> None:
    self._skill_context = context if context else None
    self._active_skill_ids = skill_ids or []
    # NEW
    self._active_skill_capabilities = skill_capabilities or frozenset()
```

**New filter in `_get_available_tools()`** — after the tool list is built, before return:

```python
    # NEW: Capability-based tool filtering
    if self.has_skill_context and self._active_skill_capabilities:
        from openspace.skill_engine.skill_utils import (
            CAPABILITY_TO_BACKENDS,
            required_backends_for_capabilities,
        )
        allowed_backends = required_backends_for_capabilities(
            self._active_skill_capabilities
        )
        # Always allow system backend (for retrieve_skill, etc.)
        allowed_backends = allowed_backends | {"system"}

        if allowed_backends:
            pre_count = len(tools)
            tools = [
                t for t in tools
                if not hasattr(t, '_runtime_info')
                or t._runtime_info is None
                or t._runtime_info.backend.value in allowed_backends
            ]
            if len(tools) < pre_count:
                logger.warning(
                    f"Capability gate: filtered {pre_count - len(tools)} tools "
                    f"outside allowed backends {allowed_backends} "
                    f"(capabilities: {self._active_skill_capabilities})"
                )
```

**IMPORTANT**: This is a **logged warning + filter**, not a hard crash. If a skill
declares `capabilities: network` but the agent tries to use shell tools, those tools
simply will not be in the tool list, so the LLM cannot call them.

### 3.5 Shell Backend Auto-Add — Conditional (grounding_agent.py)

**Where**: `_get_available_tools()` lines 554-560

**Current code** (unconditional):
```python
if self.has_skill_context:
    shell_bt = BackendType.SHELL
    if shell_bt not in backends:
        backends = list(backends) + [shell_bt]
        logger.info("Added Shell backend to scope for skill file I/O")
```

**New code** (capability-aware):
```python
if self.has_skill_context:
    # Only auto-add Shell if the skill declares capabilities that need it,
    # or if the skill declares NO capabilities (fail-open for legacy skills).
    needs_shell = (
        not self._active_skill_capabilities  # fail-open: legacy skill, no caps declared
        or self._active_skill_capabilities & {"filesystem", "subprocess", "env_vars", "gpu"}
    )
    if needs_shell:
        shell_bt = BackendType.SHELL
        if shell_bt not in backends:
            backends = list(backends) + [shell_bt]
            logger.info("Added Shell backend to scope for skill (capability-justified)")
    else:
        logger.info(
            f"Shell backend NOT auto-added — skill capabilities "
            f"{self._active_skill_capabilities} do not require it"
        )
```

### 3.6 Data Flow — How Session Tools Reach the Selector

**In `tool_layer.py:_select_and_inject_skills()`**:

The tool_layer already has `self._grounding_agent` and `self._grounding_client`. Before calling
`select_skills_with_llm()`, we gather the session's tool inventory:

```python
async def _select_and_inject_skills(
    self,
    task: str,
    task_id: str = "",
) -> bool:
    if not self._skill_registry or not self._grounding_agent:
        return False

    # NEW: Gather session tool names and backend names for capability gate
    session_tool_names: Optional[frozenset[str]] = None
    session_backends: Optional[frozenset[str]] = None
    try:
        agent_backends = self._grounding_agent.backend_scope
        session_backends = frozenset(agent_backends) if agent_backends else None

        # Get tool names from all active backends (lightweight — uses cache)
        if self._grounding_client:
            all_tools = []
            for bn in agent_backends:
                try:
                    bt = BackendType(bn)
                    tools = await self._grounding_client.list_tools(backend=bt)
                    all_tools.extend(tools)
                except Exception:
                    pass
            session_tool_names = frozenset(t.name for t in all_tools)
    except Exception as e:
        logger.debug(f"Could not gather session tools for capability gate: {e}")

    # ... existing code ...

    # Pass to selector
    if skill_llm:
        selected, selection_record = await self._skill_registry.select_skills_with_llm(
            task,
            llm_client=skill_llm,
            max_skills=ts_pool_size,
            skill_quality=skill_quality,
            store=self._skill_store,
            session_tool_names=session_tool_names,      # NEW
            session_backends=session_backends,           # NEW
        )
```

**Passing capabilities to the agent** — after selection, when calling `set_skill_context()`:

```python
    # Merge capabilities from all selected skills
    merged_capabilities: frozenset[str] = frozenset()
    for s in selected:
        merged_capabilities = merged_capabilities | s.capabilities

    # Inject
    self._grounding_agent.set_skill_context(
        context_text,
        skill_ids,
        skill_capabilities=merged_capabilities,   # NEW
    )
```

---

## 4. Backward Compatibility

### Fail-open policy for legacy skills

Many existing skills do not declare `capabilities` or `critical_tools`. The design
handles this with a **fail-open with warning** approach:

| Condition | Behavior |
|-----------|----------|
| Skill has no `capabilities` field | Passes all capability gates; shell auto-add still happens |
| Skill has no `critical_tools` field | Passes critical_tools check |
| Skill has `capabilities` but session_tool_names not provided | Passes (gate is no-op) |
| Skill has `capabilities: network` only | Shell NOT auto-added; tool list filtered to web/mcp backends |
| Skill has `capabilities: network,filesystem` | Shell auto-added; both web and shell tools available |

### Migration path

1. **Phase 1 (this design)**: Fail-open. Legacy skills work unchanged. Warnings logged when
   a skill appears to use undeclared capabilities (via `check_capability_violations()`).
2. **Phase 2 (future)**: Log metric — track how many skills pass without capabilities.
   Auto-detect capabilities during `discover()` by scanning skill body with `_CAPABILITY_DETECTORS`.
3. **Phase 3 (future)**: Fail-closed option in config. Skills without capabilities are
   excluded from selection if `strict_capability_mode: true` in grounding config.

---

## 5. Complete Data Flow Diagram

```
                   discover()
                      │
                      ▼
              ┌───────────────┐
              │  SKILL.md     │
              │  frontmatter  │
              │  ─────────    │
              │  capabilities │──→ parse_capabilities() ──→ SkillMeta.capabilities
              │  critical_tools│──→ split(",")          ──→ SkillMeta.critical_tools
              └───────────────┘

tool_layer._select_and_inject_skills()
  │
  ├─ Gather session_tool_names (from grounding_client.list_tools per backend)
  ├─ Gather session_backends (from agent.backend_scope)
  │
  ▼
registry.select_skills_with_llm(session_tool_names, session_backends)
  │
  ├─ Quality filter (existing)
  ├─ _filter_by_capability(available, session_tool_names, session_backends)  ← NEW GATE 1
  │     ├─ Exclude if critical_tools not in session_tool_names
  │     └─ Exclude if capability backends not in session_backends
  ├─ _prefilter_skills() (existing BM25+embedding)
  ├─ LLM selection (existing)
  └─ ts_blend_reorder() (existing)
  │
  ▼
tool_layer: merge capabilities from selected skills
  │
  ▼
agent.set_skill_context(text, ids, skill_capabilities)
  │
  ▼
agent._get_available_tools()
  │
  ├─ Shell auto-add: CONDITIONAL on capabilities           ← NEW GATE 2
  │     ├─ No capabilities declared → auto-add (fail-open)
  │     └─ Capabilities declared → only if needs shell
  │
  ├─ Fetch tools from backends
  │
  └─ Filter tools to allowed backends from capabilities    ← NEW GATE 3
        ├─ System backend always allowed
        └─ Other backends must match CAPABILITY_TO_BACKENDS mapping
```

---

## 6. Function Signatures Summary

### New/Modified in `skill_utils.py`

```python
CAPABILITY_TO_BACKENDS: Dict[str, frozenset[str]]  # NEW constant

def required_backends_for_capabilities(
    capabilities: frozenset[str],
) -> frozenset[str]:                                # NEW function
```

### Modified in `registry.py`

```python
@dataclass
class SkillMeta:
    skill_id: str
    name: str
    description: str
    path: Path
    capabilities: frozenset[str] = field(default_factory=frozenset)   # NEW
    critical_tools: List[str] = field(default_factory=list)            # NEW

class SkillRegistry:
    def _filter_by_capability(                                         # NEW
        self,
        available: List[SkillMeta],
        session_tool_names: Optional[frozenset[str]],
        session_backends: Optional[frozenset[str]],
    ) -> tuple[List[SkillMeta], List[str]]: ...

    async def select_skills_with_llm(
        self,
        ...,
        session_tool_names: Optional[frozenset[str]] = None,           # NEW param
        session_backends: Optional[frozenset[str]] = None,             # NEW param
    ) -> tuple[List[SkillMeta], Optional[Dict[str, Any]]]: ...

    @staticmethod
    def _parse_skill(...) -> SkillMeta:  # Modified to populate new fields
```

### Modified in `grounding_agent.py`

```python
class GroundingAgent:
    _active_skill_capabilities: frozenset[str]                         # NEW field

    def set_skill_context(
        self,
        context: str,
        skill_ids: Optional[List[str]] = None,
        skill_capabilities: Optional[frozenset[str]] = None,           # NEW param
    ) -> None: ...

    async def _get_available_tools(self, task_description) -> List:
        # Modified: conditional shell auto-add + capability-based tool filter
```

### Modified in `tool_layer.py`

```python
class OpenSpace:
    async def _select_and_inject_skills(self, task, task_id) -> bool:
        # Modified: gathers session_tool_names + session_backends
        # Passes them to select_skills_with_llm()
        # Passes merged capabilities to set_skill_context()
```

---

## 7. Test Scenarios

### 7.1 Selection Gate Tests (`tests/test_capability_enforcement.py`)

```python
class TestCapabilitySelectionGate:
    """Tests for _filter_by_capability in SkillRegistry."""

    def test_skill_with_no_capabilities_passes(self):
        """Legacy skill with no capabilities declared should always pass."""
        skill = SkillMeta(
            skill_id="legacy__imp_abc",
            name="legacy",
            description="No caps",
            path=Path("/skills/legacy/SKILL.md"),
            capabilities=frozenset(),
            critical_tools=[],
        )
        registry = SkillRegistry()
        kept, excluded = registry._filter_by_capability(
            [skill],
            session_tool_names=frozenset({"web_search"}),
            session_backends=frozenset({"web"}),
        )
        assert len(kept) == 1
        assert len(excluded) == 0

    def test_skill_with_missing_critical_tool_excluded(self):
        """Skill requiring 'pandoc_convert' excluded when tool not in session."""
        skill = SkillMeta(
            skill_id="pdf__imp_abc",
            name="pdf-gen",
            description="PDF generation",
            path=Path("/skills/pdf/SKILL.md"),
            capabilities=frozenset({"filesystem"}),
            critical_tools=["pandoc_convert"],
        )
        registry = SkillRegistry()
        kept, excluded = registry._filter_by_capability(
            [skill],
            session_tool_names=frozenset({"shell_agent", "read_file"}),
            session_backends=frozenset({"shell"}),
        )
        assert len(kept) == 0
        assert "pdf__imp_abc" in excluded

    def test_skill_with_satisfied_critical_tools_passes(self):
        """Skill passes when all critical_tools are present."""
        skill = SkillMeta(
            skill_id="pdf__imp_abc",
            name="pdf-gen",
            description="PDF generation",
            path=Path("/skills/pdf/SKILL.md"),
            capabilities=frozenset({"filesystem"}),
            critical_tools=["pandoc_convert"],
        )
        registry = SkillRegistry()
        kept, excluded = registry._filter_by_capability(
            [skill],
            session_tool_names=frozenset({"pandoc_convert", "shell_agent"}),
            session_backends=frozenset({"shell", "mcp"}),
        )
        assert len(kept) == 1

    def test_skill_with_network_cap_excluded_when_no_web_backend(self):
        """Skill declaring 'network' excluded when only shell backend active."""
        skill = SkillMeta(
            skill_id="api__imp_abc",
            name="api-caller",
            description="Calls REST APIs",
            path=Path("/skills/api/SKILL.md"),
            capabilities=frozenset({"network"}),
            critical_tools=[],
        )
        registry = SkillRegistry()
        kept, excluded = registry._filter_by_capability(
            [skill],
            session_tool_names=frozenset({"shell_agent"}),
            session_backends=frozenset({"shell"}),
        )
        assert len(kept) == 0

    def test_none_session_tools_passes_all(self):
        """When session tools are None (not provided), all skills pass."""
        skill = SkillMeta(
            skill_id="any__imp_abc",
            name="any",
            description="Anything",
            path=Path("/skills/any/SKILL.md"),
            capabilities=frozenset({"network", "subprocess"}),
            critical_tools=["exotic_tool"],
        )
        registry = SkillRegistry()
        kept, excluded = registry._filter_by_capability(
            [skill],
            session_tool_names=None,
            session_backends=None,
        )
        assert len(kept) == 1

    def test_mixed_skills_partial_exclusion(self):
        """Mix of satisfiable and unsatisfiable skills — correct partition."""
        good = SkillMeta(
            skill_id="good__imp_1",
            name="good",
            description="Has what it needs",
            path=Path("/skills/good/SKILL.md"),
            capabilities=frozenset({"filesystem"}),
            critical_tools=["read_file"],
        )
        bad = SkillMeta(
            skill_id="bad__imp_2",
            name="bad",
            description="Missing tool",
            path=Path("/skills/bad/SKILL.md"),
            capabilities=frozenset({"network"}),
            critical_tools=["http_client"],
        )
        registry = SkillRegistry()
        kept, excluded = registry._filter_by_capability(
            [good, bad],
            session_tool_names=frozenset({"read_file", "shell_agent"}),
            session_backends=frozenset({"shell"}),
        )
        assert [s.skill_id for s in kept] == ["good__imp_1"]
        assert excluded == ["bad__imp_2"]
```

### 7.2 Shell Auto-Add Tests

```python
class TestShellAutoAddConditional:
    """Tests for conditional shell backend addition."""

    def test_no_capabilities_adds_shell(self):
        """Legacy skill with no capabilities: shell auto-added (fail-open)."""
        caps = frozenset()
        needs_shell = not caps or caps & {"filesystem", "subprocess", "env_vars", "gpu"}
        assert needs_shell is True  # empty caps -> fail-open

    def test_network_only_no_shell(self):
        """Skill declaring only 'network': shell NOT added."""
        caps = frozenset({"network"})
        needs_shell = not caps or caps & {"filesystem", "subprocess", "env_vars", "gpu"}
        assert needs_shell == frozenset()  # falsy

    def test_filesystem_adds_shell(self):
        """Skill declaring 'filesystem': shell added."""
        caps = frozenset({"filesystem"})
        needs_shell = not caps or caps & {"filesystem", "subprocess", "env_vars", "gpu"}
        assert needs_shell  # truthy (non-empty intersection)

    def test_network_plus_subprocess_adds_shell(self):
        """Skill declaring 'network,subprocess': shell added."""
        caps = frozenset({"network", "subprocess"})
        needs_shell = not caps or caps & {"filesystem", "subprocess", "env_vars", "gpu"}
        assert needs_shell  # truthy
```

### 7.3 Tool Dispatch Filter Tests

```python
class TestToolDispatchFilter:
    """Tests for capability-based tool filtering in _get_available_tools."""

    def test_network_skill_excludes_shell_tools(self):
        """A skill with only 'network' capability should not see shell tools."""
        from openspace.skill_engine.skill_utils import required_backends_for_capabilities
        caps = frozenset({"network"})
        allowed = required_backends_for_capabilities(caps) | {"system"}
        assert "shell" not in allowed
        assert "web" in allowed or "mcp" in allowed

    def test_filesystem_skill_includes_shell_tools(self):
        from openspace.skill_engine.skill_utils import required_backends_for_capabilities
        caps = frozenset({"filesystem"})
        allowed = required_backends_for_capabilities(caps) | {"system"}
        assert "shell" in allowed

    def test_empty_capabilities_no_filter(self):
        """No capabilities declared: no filtering applied (fail-open)."""
        caps = frozenset()
        # Gate only activates when capabilities are non-empty
        should_filter = bool(caps)
        assert should_filter is False

    def test_system_backend_always_allowed(self):
        """System backend (for retrieve_skill etc.) always in allowed set."""
        from openspace.skill_engine.skill_utils import required_backends_for_capabilities
        caps = frozenset({"network"})
        allowed = required_backends_for_capabilities(caps) | {"system"}
        assert "system" in allowed
```

### 7.4 Integration / End-to-End Scenario

```python
class TestCapabilityEnforcementE2E:
    """End-to-end scenario: skill declares network-only, verify shell is gated."""

    async def test_network_skill_cannot_use_shell(self):
        """
        Given: A skill with 'capabilities: network' and no critical_tools
        And: Session has shell + web backends
        When: Skill is selected
        Then: Shell is NOT auto-added
        And: Tool list excludes shell tools
        """
        # This test would require mocking GroundingAgent + GroundingClient
        # Validates the full chain: selection -> injection -> tool filtering
        pass  # Implementation requires fixtures from existing test infra

    async def test_legacy_skill_retains_full_access(self):
        """
        Given: A skill with NO capabilities declared
        And: Session has shell + web backends
        When: Skill is selected
        Then: Shell IS auto-added (fail-open)
        And: Tool list is NOT filtered
        """
        pass  # Implementation requires fixtures from existing test infra
```

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Legacy skills break due to unexpected filtering | Low | High | Fail-open: no capabilities = no filtering |
| Selection gate is too aggressive | Medium | Medium | Only gates on critical_tools + backend mapping, not all tool_dependencies |
| Tool gathering in tool_layer adds latency | Low | Low | Uses cached tool lists; list_tools() is already fast |
| Capability-to-backend mapping is incomplete | Medium | Low | System backend always allowed; mapping can be extended |
| Skills declare wrong capabilities | Medium | Medium | check_capability_violations() already detects mismatches; add to CI |

---

## 9. Implementation Order

1. **skill_utils.py**: Add `CAPABILITY_TO_BACKENDS` and `required_backends_for_capabilities()`
2. **registry.py**: Expand `SkillMeta` dataclass, modify `_parse_skill()`, add `_filter_by_capability()`,
   modify `select_skills_with_llm()` signature
3. **grounding_agent.py**: Add `_active_skill_capabilities` field, modify `set_skill_context()`,
   conditional shell auto-add, tool dispatch filter in `_get_available_tools()`
4. **tool_layer.py**: Gather session tools/backends, pass to selector, pass capabilities to agent
5. **Tests**: `tests/test_capability_enforcement.py` with all scenarios from Section 7
6. **Verify**: Run existing test suite to confirm no regressions

Estimated implementation: ~200 lines of production code, ~250 lines of tests.
