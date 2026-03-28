"""Data types for skill quality tracking and evolution."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional


class SkillCategory(str, Enum):
    """Skill primary category."""

    TOOL_GUIDE = "tool_guide"    # Tool guide
    WORKFLOW   = "workflow"      # End-to-end workflow
    REFERENCE  = "reference"     # Reference knowledge


class SkillVisibility(str, Enum):
    """Cloud visibility of a skill. (`Group` is managed by the cloud platform)"""

    PRIVATE = "private"  # Only visible to the creator
    PUBLIC  = "public"   # Visible to all users on the cloud


class EvolutionType(str, Enum):
    FIX      = "fix"       # Repair broken / outdated skill instructions
    DERIVED  = "derived"   # Enhance / specialize an existing skill
    CAPTURED = "captured"  # Capture a novel reusable pattern

    def to_origin(self) -> "SkillOrigin":
        """Convert this evolution action to the corresponding SkillOrigin."""
        return _EVOLUTION_TO_ORIGIN[self]


class SkillOrigin(str, Enum):
    """How this skill was created / entered the system.

    Version DAG model — every change creates a new SkillRecord node::

    Lineage rules:
      IMPORTED / CAPTURED → root node, no parent (parent_skill_ids = [])
      DERIVED             → 1+ parents, new skill name, new directory
      FIXED               → exactly 1 parent (previous version of same skill),
                            same ``name`` & ``path``, new ``skill_id``.
                            Files on disk are updated in-place; old directory
                            content (all files) is preserved via
                            ``content_snapshot`` dict in the DB.
                            Only the latest version is ``is_active=True``.
    """

    IMPORTED = "imported"  # Initial import, no parent
    CAPTURED = "captured"  # Captured from a successful execution with no parent skill involved
    DERIVED  = "derived"   # Derived from existing skill(s) (upgrade, wrap, compose, etc.)
    FIXED    = "fixed"     # Fix of existing skill — new record, parent = previous version


_EVOLUTION_TO_ORIGIN: Dict["EvolutionType", "SkillOrigin"] = {
    EvolutionType.FIX:      SkillOrigin.FIXED,
    EvolutionType.DERIVED:  SkillOrigin.DERIVED,
    EvolutionType.CAPTURED: SkillOrigin.CAPTURED,
}

_ORIGIN_TO_EVOLUTION: Dict["SkillOrigin", "EvolutionType"] = {
    v: k for k, v in _EVOLUTION_TO_ORIGIN.items()
}


@dataclass
class SkillLineage:
    """Tracks the evolutionary lineage of a skill.

    ``parent_skill_ids`` may contain multiple parents for DERIVED.
    FIXED always has exactly one parent (the previous version).
    IMPORTED / CAPTURED have no parents.

    ─── generation ─────────────────────────────────────────────────

    Distance from root in the version DAG. Set by the evolution logic
    when creating a new skill record:

      - IMPORTED / CAPTURED → ``generation = 0``  (root node)
      - FIXED               → ``parent.generation + 1``
      - DERIVED             → ``max(p.generation for p in parents) + 1``

    ─── change_summary ─────────────────────────────────────────────

    LLM-generated free-text description of what changed vs. the parent.
    Produced by the evolution LLM when creating FIXED or DERIVED skills.
    Examples:
      - FIXED:   "Fixed curl parameter format in step 3"
      - DERIVED: "Composed weather + geocoding guides into an
                  end-to-end location-aware forecast workflow"
      - IMPORTED / CAPTURED: typically empty or a brief import note.

    ─── content_diff / content_snapshot ────────────────────────────

    ``content_snapshot`` stores the **full directory snapshot** at this
    version as a ``Dict[str, str]`` mapping relative file paths to their
    text content.

    ``content_diff`` stores a combined unified diff (``git diff``
    format) covering **all** files in the skill directory.
    Policy by parent count:

      - **0 parents** (IMPORTED / CAPTURED):
        add-all diff — every line prefixed with ``+``
        (like ``git diff /dev/null`` for each file).
      - **1 parent** (FIXED, or single-parent DERIVED):
        normal unified diff between the parent's directory content
        and this version's directory content, covering all files.
      - **N parents** (multi-parent DERIVED):
        ``""`` (empty string). A multi-parent composition is a
        creative act, not a patch — per-parent diffs are large and
        unhelpful.  The composition intent is captured in
        ``change_summary`` instead.  Individual parent content can
        be retrieved via ``parent_skill_ids`` → each parent's
        ``content_snapshot``.
    """

    origin: SkillOrigin
    generation: int = 0                                    # Distance from root (see docstring)
    parent_skill_ids: List[str] = field(default_factory=list)  # [] for IMPORTED / CAPTURED
    source_task_id: Optional[str] = None                   # Task that triggered evolution / capture
    change_summary: str = ""                               # LLM-generated description of changes
    content_diff: str = ""                                 # Combined unified diff of all files (empty for multi-parent DERIVED)
    content_snapshot: Dict[str, str] = field(default_factory=dict)  # {relative_path: content} full directory snapshot
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""                                   # "human" | model name (version-level actor)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin.value,
            "generation": self.generation,
            "parent_skill_ids": self.parent_skill_ids,
            "source_task_id": self.source_task_id,
            "change_summary": self.change_summary,
            "content_diff": self.content_diff,
            "content_snapshot": self.content_snapshot,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillLineage":
        return cls(
            origin=SkillOrigin(data["origin"]),
            generation=data.get("generation", 0),
            parent_skill_ids=data.get("parent_skill_ids", []),
            source_task_id=data.get("source_task_id"),
            change_summary=data.get("change_summary", ""),
            content_diff=data.get("content_diff", ""),
            content_snapshot=data.get("content_snapshot", {}),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at") else datetime.now()
            ),
            created_by=data.get("created_by", ""),
        )


# Per-skill judgment within a task analysis
@dataclass
class SkillJudgment:
    """Per-skill assessment within an :class:`ExecutionAnalysis`.

    One ``ExecutionAnalysis`` (per task) contains zero or more
    ``SkillJudgment`` entries — one for each skill that was selected
    for that task.

    ``skill_applied`` semantics depend on skill category:
      - WORKFLOW:   agent followed the prescribed steps
      - TOOL_GUIDE: agent used the described tool / approach
      - REFERENCE:  knowledge influenced agent decisions
    """

    skill_id: str
    skill_applied: bool = False    # Whether the skill was actually applied
    note: str = ""                 # Per-skill observation (deviation, usage, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_applied": self.skill_applied,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillJudgment":
        return cls(
            skill_id=data["skill_id"],
            skill_applied=data.get("skill_applied", False),
            note=data.get("note", ""),
        )


@dataclass
class EvolutionSuggestion:
    """One evolution action suggested by the analysis LLM.

    ``target_skill_ids`` lists the parent skill(s) this action targets
    using **true skill_id** values (e.g. ``weather__imp_a1b2c3d4``):
      - FIX: exactly 1 parent (the skill to repair in-place)
      - DERIVED: 1+ parents (single parent → enhance; multi → merge/fuse)
      - CAPTURED: empty list (brand-new skill, no parents)
    """

    evolution_type: EvolutionType
    target_skill_ids: List[str] = field(default_factory=list)  # True skill_id(s)
    category: Optional[SkillCategory] = None                   # Desired category of the result
    direction: str = ""                                        # Free-text: what to evolve / capture

    @property
    def target_skill_id(self) -> str:
        """Primary (or only) target skill_id.  Empty string if none."""
        return self.target_skill_ids[0] if self.target_skill_ids else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.evolution_type.value,
            "target_skills": self.target_skill_ids,
            # Keep legacy singular key for backward compat with stored analyses
            "target_skill": self.target_skill_id,
            "category": self.category.value if self.category else None,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionSuggestion":
        cat = None
        if data.get("category"):
            try:
                cat = SkillCategory(data["category"])
            except ValueError:
                pass
        # Support both new list format and legacy single-string format
        raw_targets = data.get("target_skills")
        if isinstance(raw_targets, list):
            targets = [t for t in raw_targets if t]
        else:
            legacy = data.get("target_skill", "")
            targets = [legacy] if legacy else []
        return cls(
            evolution_type=EvolutionType(data["type"]),
            target_skill_ids=targets,
            category=cat,
            direction=data.get("direction", ""),
        )


# Task-level execution analysis (1 per task)
@dataclass
class ExecutionAnalysis:
    """LLM-produced analysis of a single task execution."""

    task_id: str
    timestamp: datetime

    # Task-level LLM judgments
    task_completed: bool = False           # Whether the task completed successfully
    execution_note: str = ""               # Task-level observation
    tool_issues: List[str] = field(default_factory=list)  # Tool keys that had issues

    # Per-skill judgments (one per selected skill; empty = no skill involved)
    skill_judgments: List[SkillJudgment] = field(default_factory=list)

    # Evolution suggestions — 0-N per analysis, each fully specifies an action
    evolution_suggestions: List[EvolutionSuggestion] = field(default_factory=list)

    # Analysis metadata
    analyzed_by: str = ""                  # Model name used for analysis
    analyzed_at: datetime = field(default_factory=datetime.now)

    def get_judgment(self, skill_id: str) -> Optional[SkillJudgment]:
        """Find the judgment for a specific skill, or None."""
        for j in self.skill_judgments:
            if j.skill_id == skill_id:
                return j
        return None

    @property
    def skill_ids(self) -> List[str]:
        """List of skill_ids that were judged in this analysis."""
        return [j.skill_id for j in self.skill_judgments]

    @property
    def candidate_for_evolution(self) -> bool:
        """Whether any evolution suggestions exist."""
        return len(self.evolution_suggestions) > 0

    def suggestions_by_type(self, evo_type: EvolutionType) -> List[EvolutionSuggestion]:
        """Filter evolution suggestions by type."""
        return [s for s in self.evolution_suggestions if s.evolution_type == evo_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "task_completed": self.task_completed,
            "execution_note": self.execution_note,
            "tool_issues": self.tool_issues,
            "skill_judgments": [j.to_dict() for j in self.skill_judgments],
            "evolution_suggestions": [s.to_dict() for s in self.evolution_suggestions],
            "analyzed_by": self.analyzed_by,
            "analyzed_at": self.analyzed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionAnalysis":
        return cls(
            task_id=data["task_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            task_completed=data.get("task_completed", False),
            execution_note=data.get("execution_note", ""),
            tool_issues=data.get("tool_issues", []),
            skill_judgments=[
                SkillJudgment.from_dict(j)
                for j in data.get("skill_judgments", [])
            ],
            evolution_suggestions=[
                EvolutionSuggestion.from_dict(s)
                for s in data.get("evolution_suggestions", [])
            ],
            analyzed_by=data.get("analyzed_by", ""),
            analyzed_at=(
                datetime.fromisoformat(data["analyzed_at"])
                if data.get("analyzed_at") else datetime.now()
            ),
        )


_FAILURE_MODES = frozenset({
    "wrong_tool_sequence", "api_misuse", "missing_prerequisite",
    "scope_creep", "auth_error", "rate_limit", "data_format", "other",
})


@dataclass
class FailureLesson:
    """Distilled lesson from a failed task — injected as NEGATIVE guidance."""

    lesson_id: str                                        # uuid hex
    task_id: str                                          # source ExecutionAnalysis.task_id
    skill_ids: List[str] = field(default_factory=list)   # skills active at failure
    task_summary: str = ""                               # what was attempted
    failure_mode: str = "other"                          # one of _FAILURE_MODES
    lesson_text: str = ""                                # "avoid X because Y"
    tool_culprits: List[str] = field(default_factory=list)
    confidence: float = 0.7                              # gate: skip if < 0.7
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None                # 30-day TTL from created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "task_id": self.task_id,
            "skill_ids": self.skill_ids,
            "task_summary": self.task_summary,
            "failure_mode": self.failure_mode,
            "lesson_text": self.lesson_text,
            "tool_culprits": self.tool_culprits,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureLesson":
        expires = None
        if data.get("expires_at"):
            expires = datetime.fromisoformat(data["expires_at"])
        return cls(
            lesson_id=data["lesson_id"],
            task_id=data["task_id"],
            skill_ids=data.get("skill_ids", []),
            task_summary=data.get("task_summary", ""),
            failure_mode=data.get("failure_mode", "other"),
            lesson_text=data.get("lesson_text", ""),
            tool_culprits=data.get("tool_culprits", []),
            confidence=data.get("confidence", 0.7),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at") else datetime.now()
            ),
            expires_at=expires,
        )


@dataclass
class SkillBanditStats:
    """Thompson Sampling bandit state for one skill.

    Models dispatch quality as Beta(alpha, beta).  At selection time,
    ``sample()`` draws from this posterior — higher samples rank the skill
    higher, creating exploration/exploitation balance.

    Prior seeding: alpha/beta may be initialised from semantic similarity
    to existing high-performing skills so cold-start skills are not
    penalised too early.
    """

    skill_id: str
    alpha: float = 1.0          # successes + 1  (Beta parameter a)
    beta: float = 1.0           # failures  + 1  (Beta parameter b)
    prior_confidence: float = 0.5   # 0=cold-start, 1=seeded from similar skill
    total_dispatches: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def sample(self) -> float:
        """Draw a sample from Beta(alpha, beta) via random.betavariate."""
        return random.betavariate(max(self.alpha, 1e-6), max(self.beta, 1e-6))

    def updated(self, *, success: bool) -> "SkillBanditStats":
        """Return a NEW SkillBanditStats with alpha or beta incremented (immutable)."""
        return SkillBanditStats(
            skill_id=self.skill_id,
            alpha=self.alpha + (1.0 if success else 0.0),
            beta=self.beta + (0.0 if success else 1.0),
            prior_confidence=self.prior_confidence,
            total_dispatches=self.total_dispatches + 1,
            last_updated=datetime.now(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "alpha": self.alpha,
            "beta": self.beta,
            "prior_confidence": self.prior_confidence,
            "total_dispatches": self.total_dispatches,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillBanditStats":
        return cls(
            skill_id=data["skill_id"],
            alpha=float(data.get("alpha", 1.0)),
            beta=float(data.get("beta", 1.0)),
            prior_confidence=float(data.get("prior_confidence", 0.5)),
            total_dispatches=int(data.get("total_dispatches", 0)),
            last_updated=(
                datetime.fromisoformat(data["last_updated"])
                if data.get("last_updated") else datetime.now()
            ),
        )


# Full skill profile (identity + lineage + deps + quality)
@dataclass
class SkillRecord:
    """Comprehensive record for a skill: identity + lineage + quality.

    This is the full profile for a skill within the quality / evolution system.
    The lightweight SkillMeta is still used for discovery; SkillRecord is managed by
    ExecutionAnalyzer.
    """

    skill_id: str                            # Unique identifier
    name: str                                # Logical skill name (shared across versions)
    description: str
    path: str = ""                           # Path to SKILL.md (shared across FIXED versions)

    is_active: bool = True                   # Only the latest version is active

    # Category & tags
    category: SkillCategory = SkillCategory.WORKFLOW
    tags: List[str] = field(default_factory=list)  # Auxiliary tags generated by LLM

    # Ownership & visibility (for cloud sync)
    visibility: SkillVisibility = SkillVisibility.PRIVATE  # Cloud visibility
    creator_id: str = ""                     # User ID of the skill owner / creator

    # Lineage
    lineage: SkillLineage = field(
        default_factory=lambda: SkillLineage(origin=SkillOrigin.IMPORTED)
    )

    # Tool dependencies
    tool_dependencies: List[str] = field(default_factory=list)  # All involved tool keys
    critical_tools: List[str] = field(default_factory=list)     # Required (must-have) tool keys

    # Execution stats (updated by add_analysis or atomically in store)
    total_selections: int = 0    # Times this skill was selected by the LLM
    total_applied: int = 0       # Times the skill was actually applied by the agent
    total_completions: int = 0   # Times task completed when skill was applied
    total_fallbacks: int = 0     # Times skill was not applied and task failed

    # Recent analysis history (rolling window of analyses involving this skill)
    recent_analyses: List[ExecutionAnalysis] = field(default_factory=list)
    MAX_RECENT: ClassVar[int] = 50

    # Metadata
    first_seen: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def applied_rate(self) -> float:
        """Ratio of selections where the skill was actually applied."""
        return self.total_applied / self.total_selections if self.total_selections else 0.0

    @property
    def completion_rate(self) -> float:
        """Ratio of applied uses that led to task completion."""
        return self.total_completions / self.total_applied if self.total_applied else 0.0

    @property
    def effective_rate(self) -> float:
        """End-to-end effectiveness: selected → applied → completed."""
        return self.total_completions / self.total_selections if self.total_selections else 0.0

    @property
    def fallback_rate(self) -> float:
        """Ratio of selections that fell back (skill unusable signal)."""
        return self.total_fallbacks / self.total_selections if self.total_selections else 0.0

    # NOTE: Counter updates (total_selections, total_applied, etc.) are
    # performed atomically in SQL by SkillStore.record_analysis().
    # Do NOT duplicate that logic here in Python.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "is_active": self.is_active,
            "category": self.category.value,
            "tags": self.tags,
            "visibility": self.visibility.value,
            "creator_id": self.creator_id,
            "lineage": self.lineage.to_dict(),
            "tool_dependencies": self.tool_dependencies,
            "critical_tools": self.critical_tools,
            "total_selections": self.total_selections,
            "total_applied": self.total_applied,
            "total_completions": self.total_completions,
            "total_fallbacks": self.total_fallbacks,
            "recent_analyses": [a.to_dict() for a in self.recent_analyses],
            "first_seen": self.first_seen.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillRecord":
        record = cls(
            skill_id=data["skill_id"],
            name=data["name"],
            description=data.get("description", ""),
            path=data.get("path", ""),
            is_active=data.get("is_active", True),
            category=SkillCategory(data["category"]) if data.get("category") else SkillCategory.WORKFLOW,
            tags=data.get("tags", []),
            visibility=(
                SkillVisibility(data["visibility"])
                if data.get("visibility") else SkillVisibility.PRIVATE
            ),
            creator_id=data.get("creator_id", ""),
            lineage=(
                SkillLineage.from_dict(data["lineage"])
                if data.get("lineage")
                else SkillLineage(origin=SkillOrigin.IMPORTED)
            ),
            tool_dependencies=data.get("tool_dependencies", []),
            critical_tools=data.get("critical_tools", []),
            total_selections=data.get("total_selections", 0),
            total_applied=data.get("total_applied", 0),
            total_completions=data.get("total_completions", 0),
            total_fallbacks=data.get("total_fallbacks", 0),
            first_seen=(
                datetime.fromisoformat(data["first_seen"])
                if data.get("first_seen") else datetime.now()
            ),
            last_updated=(
                datetime.fromisoformat(data["last_updated"])
                if data.get("last_updated") else datetime.now()
            ),
        )
        for a in data.get("recent_analyses", []):
            record.recent_analyses.append(ExecutionAnalysis.from_dict(a))
        return record
