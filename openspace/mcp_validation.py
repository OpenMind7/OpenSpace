"""Pydantic models for MCP tool input validation.

All MCP tool inputs pass through these models before execution,
rejecting malformed, oversized, or suspicious input at the boundary.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

_MAX_TASK_LENGTH = 50_000      # 50K chars max for task descriptions
_MAX_DIR_PATH_LENGTH = 4096    # Filesystem path limit
_MAX_QUERY_LENGTH = 10_000     # Search query limit
_PATH_TRAVERSAL_RE = re.compile(r"\.\.[/\\]")


def _reject_path_traversal(path: str) -> str:
    """Raise ValueError if path contains '..' traversal sequences."""
    if _PATH_TRAVERSAL_RE.search(path):
        raise ValueError(f"Path traversal not allowed: {path}")
    return path


class ExecuteTaskInput(BaseModel):
    """Validates input for the execute_task MCP tool."""

    task: str = Field(..., min_length=1, max_length=_MAX_TASK_LENGTH)
    workspace_dir: Optional[str] = Field(None, max_length=_MAX_DIR_PATH_LENGTH)
    max_iterations: Optional[int] = Field(None, ge=1, le=100)
    skill_dirs: Optional[List[str]] = Field(None)
    search_scope: str = Field("all")

    @field_validator("search_scope")
    @classmethod
    def validate_search_scope(cls, v: str) -> str:
        if v not in ("all", "local"):
            raise ValueError(f"search_scope must be 'all' or 'local', got '{v}'")
        return v

    @field_validator("workspace_dir")
    @classmethod
    def validate_workspace_dir(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            _reject_path_traversal(v)
        return v

    @field_validator("skill_dirs")
    @classmethod
    def validate_skill_dirs(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            if len(v) > 20:
                raise ValueError(f"Too many skill_dirs ({len(v)}), max 20")
            for p in v:
                _reject_path_traversal(p)
        return v


class SearchSkillsInput(BaseModel):
    """Validates input for the search_skills MCP tool."""

    query: str = Field(..., min_length=1, max_length=_MAX_QUERY_LENGTH)
    source: str = Field("all")
    limit: int = Field(20, ge=1, le=100)
    auto_import: bool = True

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in ("all", "local", "cloud"):
            raise ValueError(f"source must be 'all', 'local', or 'cloud', got '{v}'")
        return v


class FixSkillInput(BaseModel):
    """Validates input for the fix_skill MCP tool."""

    skill_dir: str = Field(..., min_length=1, max_length=_MAX_DIR_PATH_LENGTH)
    direction: str = Field(..., min_length=1, max_length=_MAX_TASK_LENGTH)

    @field_validator("skill_dir")
    @classmethod
    def validate_skill_dir(cls, v: str) -> str:
        return _reject_path_traversal(v)


class UploadSkillInput(BaseModel):
    """Validates input for the upload_skill MCP tool."""

    skill_dir: str = Field(..., min_length=1, max_length=_MAX_DIR_PATH_LENGTH)
    visibility: str = Field("public")
    origin: Optional[str] = Field(None, max_length=50)
    parent_skill_ids: Optional[List[str]] = Field(None)
    tags: Optional[List[str]] = Field(None)
    created_by: Optional[str] = Field(None, max_length=200)
    change_summary: Optional[str] = Field(None, max_length=5_000)

    @field_validator("visibility")
    @classmethod
    def validate_visibility(cls, v: str) -> str:
        if v not in ("public", "private"):
            raise ValueError(f"visibility must be 'public' or 'private', got '{v}'")
        return v

    @field_validator("skill_dir")
    @classmethod
    def validate_skill_dir(cls, v: str) -> str:
        return _reject_path_traversal(v)

    @field_validator("parent_skill_ids")
    @classmethod
    def validate_parent_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None and len(v) > 10:
            raise ValueError(f"Too many parent_skill_ids ({len(v)}), max 10")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None and len(v) > 20:
            raise ValueError(f"Too many tags ({len(v)}), max 20")
        return v
