"""Nexus Knowledge Base bridge for OpenSpace skill search.

Provides direct SQLite access to nexus-kb for:
  - Layer 0 search: FTS5 query before OpenSpace BM25 pipeline
  - Feedback loop: record skill usefulness back to KB
  - Auto-distill: persist captured skill patterns as draft KB entries

The bridge is non-blocking and fail-safe — all functions degrade
gracefully when nexus-kb is unavailable.

Environment:
  NEXUS_KB_DB_PATH — override default database path
  Default: ~/.nexus-kb/nexus-kb.db
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openspace.nexus_kb_bridge")

_DEFAULT_DB_PATH = Path.home() / ".nexus-kb" / "nexus-kb.db"


def _get_db_path() -> Path:
    """Resolve nexus-kb database path."""
    custom = os.environ.get("NEXUS_KB_DB_PATH")
    if custom:
        return Path(custom)
    return _DEFAULT_DB_PATH


def _connect_readonly() -> Optional[sqlite3.Connection]:
    """Open a read-only connection to nexus-kb. Returns None if unavailable."""
    db_path = _get_db_path()
    if not db_path.exists() or db_path.stat().st_size == 0:
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.debug("nexus-kb read-only connect failed: %s", e)
        return None


_wal_initialized: bool = False


def _connect_readwrite() -> Optional[sqlite3.Connection]:
    """Open a read-write connection to nexus-kb. Returns None if unavailable.

    WAL journal mode is set once per process lifetime to avoid redundant
    PRAGMA calls on every connection (WAL is a database-level setting that
    persists across connections).
    """
    global _wal_initialized
    db_path = _get_db_path()
    if not db_path.exists() or db_path.stat().st_size == 0:
        return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        if not _wal_initialized:
            conn.execute("PRAGMA journal_mode=WAL")
            _wal_initialized = True
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.debug("nexus-kb read-write connect failed: %s", e)
        return None


def _escape_fts5_query(query: str) -> str:
    """Build an FTS5 query from free-text input.

    Wraps each non-empty token in double quotes to avoid FTS5 syntax
    errors from special characters. Joins with OR for broad matching.
    """
    tokens = []
    for token in query.strip().split():
        cleaned = token.strip('"\'(){}[]')
        if cleaned and len(cleaned) >= 2:
            escaped = cleaned.replace('"', '""')
            tokens.append(f'"{escaped}"')
    return " OR ".join(tokens)


def search(
    query: str,
    *,
    category: Optional[str] = None,
    limit: int = 10,
    min_confidence: float = 0.3,
) -> List[Dict[str, Any]]:
    """Search nexus-kb via FTS5 for knowledge relevant to the query.

    Returns results scored in [0, 1] for merging with OpenSpace search.
    Gracefully returns empty list if nexus-kb is unavailable.

    Args:
        query: Free-text search query.
        category: Optional category filter (e.g. 'code_pattern', 'tool_usage').
        limit: Maximum results.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of result dicts with keys: entry_id, title, summary, category,
        tags, confidence, score, source='nexus-kb'.
    """
    # Clamp limit to prevent unbounded FTS fetch (LIMIT -1 = no limit in SQLite)
    limit = max(1, min(limit, 100))

    conn = _connect_readonly()
    if conn is None:
        return []

    try:
        fts_query = _escape_fts5_query(query)
        if not fts_query:
            conn.close()
            return []

        sql = """
            SELECT e.id, e.title, e.summary, e.category, e.subcategory,
                   e.tags, e.confidence, e.relevance_score, e.project,
                   rank
            FROM entry_fts
            JOIN entries e ON e.id = entry_fts.rowid
            WHERE entry_fts MATCH ?
              AND e.is_active = 1
              AND e.lifecycle_state = 'active'
              AND e.confidence >= ?
        """
        params: list = [fts_query, min_confidence]

        if category:
            sql += " AND e.category = ?"
            params.append(category)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            # FTS5 BM25 rank is negative (lower = better match).
            # Normalize to [0, 1]: typical range is [-50, 0].
            raw_rank = row["rank"]
            bm25_score = min(1.0, max(0.0, (-raw_rank) / 25.0))

            # Combine BM25 relevance with confidence for final score.
            combined = round(bm25_score * 0.6 + row["confidence"] * 0.25
                             + row["relevance_score"] * 0.15, 4)

            results.append({
                "entry_id": row["id"],
                "title": row["title"],
                "summary": row["summary"] or "",
                "category": row["category"],
                "subcategory": row["subcategory"] or "",
                "tags": _parse_tags(row["tags"]),
                "confidence": row["confidence"],
                "project": row["project"] or "",
                "source": "nexus-kb",
                "score": combined,
            })

        return results

    except Exception as e:
        logger.warning("nexus-kb search failed (non-fatal): %s", e)
        return []
    finally:
        conn.close()


def get_related(entry_id: int) -> List[Dict[str, Any]]:
    """Find entries related to a given entry via the knowledge graph.

    Returns relation metadata for cross-domain discovery.
    """
    conn = _connect_readonly()
    if conn is None:
        return []

    try:
        cursor = conn.execute("""
            SELECT er.to_id, er.relation_type, er.strength,
                   e.title, e.category, e.confidence
            FROM entry_relations er
            JOIN entries e ON e.id = er.to_id
            WHERE er.from_id = ?
              AND e.is_active = 1
              AND e.lifecycle_state = 'active'
            UNION ALL
            SELECT er.from_id, er.relation_type, er.strength,
                   e.title, e.category, e.confidence
            FROM entry_relations er
            JOIN entries e ON e.id = er.from_id
            WHERE er.to_id = ?
              AND e.is_active = 1
              AND e.lifecycle_state = 'active'
        """, (entry_id, entry_id))

        return [
            {
                "entry_id": row["to_id"],
                "relation_type": row["relation_type"],
                "strength": row["strength"],
                "title": row["title"],
                "category": row["category"],
                "confidence": row["confidence"],
            }
            for row in cursor.fetchall()
        ]
    except Exception as e:
        logger.debug("nexus-kb get_related failed: %s", e)
        return []
    finally:
        conn.close()


def record_feedback(entry_id: int, useful: bool) -> bool:
    """Record usefulness feedback on a nexus-kb entry.

    Updates hit_count/miss_count and adjusts confidence.
    Returns True if feedback was recorded successfully.
    """
    conn = _connect_readwrite()
    if conn is None:
        return False

    try:
        if useful:
            cursor = conn.execute("""
                UPDATE entries
                SET hit_count = hit_count + 1,
                    confidence = MIN(1.0, confidence + 0.02),
                    relevance_score = MIN(1.0, relevance_score + 0.03),
                    last_accessed_at = datetime('now'),
                    access_count = access_count + 1,
                    updated_at = datetime('now')
                WHERE id = ? AND is_active = 1
            """, (entry_id,))
        else:
            cursor = conn.execute("""
                UPDATE entries
                SET miss_count = miss_count + 1,
                    confidence = MAX(0.1, confidence - 0.05),
                    relevance_score = MAX(0.1, relevance_score - 0.03),
                    last_accessed_at = datetime('now'),
                    access_count = access_count + 1,
                    updated_at = datetime('now')
                WHERE id = ? AND is_active = 1
            """, (entry_id,))

        if cursor.rowcount != 1:
            logger.warning(
                "nexus-kb feedback: entry_id=%d not found or inactive (rowcount=%d)",
                entry_id, cursor.rowcount,
            )
            conn.rollback()
            return False

        conn.commit()
        logger.debug(
            "nexus-kb feedback: entry_id=%d useful=%s", entry_id, useful
        )
        return True

    except Exception as e:
        logger.warning("nexus-kb feedback failed (non-fatal): %s", e)
        return False
    finally:
        conn.close()


def distill_pattern(
    title: str,
    content: str,
    domain: str,
    *,
    tags: Optional[List[str]] = None,
    project: Optional[str] = None,
    task_context: Optional[str] = None,
) -> Optional[int]:
    """Distill a captured skill pattern into nexus-kb as a draft entry.

    Creates a code_pattern entry with distillation_state='draft' and
    confidence=0.55. The entry auto-promotes on positive feedback
    (validated at 2+ useful, promoted at 4+ useful, confidence 0.70+).

    Args:
        title: Pattern title (e.g. "OPENSPACE-001 Retry on Rate Limit").
        content: Full pattern description (approach, insight, why it works).
        domain: Domain tag (e.g. 'automation', 'search', 'evolution').
        tags: Additional searchable tags.
        project: Associated project name.
        task_context: What task produced this pattern.

    Returns:
        Entry ID if created, None on failure.
    """
    conn = _connect_readwrite()
    if conn is None:
        return None

    try:
        tag_list = list(set((tags or []) + ["openspace", "auto-distilled", domain]))

        full_content = content
        if task_context:
            full_content += f"\n\n## Task Context\n{task_context}"

        cursor = conn.execute("""
            INSERT INTO entries (
                title, content, summary, category, subcategory,
                tags, source_type, project, confidence,
                relevance_score, distillation_state,
                ttl_days, source_agent
            ) VALUES (
                ?, ?, ?, 'code_pattern', ?,
                ?, 'session_learning', ?, 0.55,
                0.5, 'draft',
                365, 'openspace-evolver'
            )
        """, (
            title,
            full_content,
            f"Auto-distilled from OpenSpace skill evolution: {title}",
            domain,
            json.dumps(tag_list),
            project,
        ))

        entry_id = cursor.lastrowid

        # Sync to FTS5 index.
        conn.execute("""
            INSERT INTO entry_fts(rowid, title, content, summary, tags,
                                  category, subcategory)
            VALUES (?, ?, ?, ?, ?, 'code_pattern', ?)
        """, (
            entry_id,
            title,
            full_content,
            f"Auto-distilled from OpenSpace skill evolution: {title}",
            json.dumps(tag_list),
            domain,
        ))

        conn.commit()
        logger.info(
            "Distilled pattern to nexus-kb: id=%d title=%s", entry_id, title
        )
        return entry_id

    except Exception as e:
        logger.warning("nexus-kb distill failed (non-fatal): %s", e)
        return None
    finally:
        conn.close()


def extract_keywords(results: List[Dict[str, Any]]) -> set:
    """Extract keyword tokens from nexus-kb results for cross-validation.

    Used to boost OpenSpace candidates that match nexus-kb knowledge.
    """
    keywords: set = set()
    for r in results:
        # Tokenize title
        for word in r.get("title", "").lower().split():
            cleaned = word.strip("[]()-:;,.'\"")
            if len(cleaned) >= 3:
                keywords.add(cleaned)
        # Add tags
        for tag in r.get("tags", []):
            if isinstance(tag, str) and len(tag) >= 3:
                keywords.add(tag.lower())
    return keywords


def _parse_tags(raw: Any) -> List[str]:
    """Parse tags field — may be JSON string or list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return []
