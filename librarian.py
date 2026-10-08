#!/usr/bin/env python3
"""
Muninn Librarian — The Collection Manager

Runs daily to maintain the Muninn memory collection:
1. Audit & Recategorise — fix misfiled facts, re-evaluate PDS codes
2. Duplicate Detection & Merging — find and merge cross-session duplicate facts
3. Pruning — archive cold/superseded facts, delete truly dead ones
4. Entity Reconciliation — merge entity aliases, clean up orphans

Designed to run at 4 AM AEST (18:00 UTC) after sleep cycle (2 AM AEST) and dreaming (3 AM AEST).

Usage:
    python3 librarian.py                    # Full librarian run
    python3 librarian.py --audit-only       # Only audit & recategorise
    python3 librarian.py --merge-only       # Only duplicate merging
    python3 librarian.py --prune-only       # Only pruning
    python3 librarian.py --reconcile-only   # Only entity reconciliation
"""

import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher

# Import from muninn.py
MUNINN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MUNINN_DIR)

from muninn import (
    get_db, get_llm_client, get_embedding, get_config, set_config,
    index_fact_fts, index_session_fts,
    EXTRACTION_MODEL, COOL_THRESHOLD, DECAY_HALF_LIFE_DAYS,
    DB_PATH, MEMORY_MD, WORKSPACE
)

# Override get_db to add busy_timeout for librarian concurrency
import sqlite3 as _sqlite3
import sqlite_vec as _sqlite_vec

def get_db_safe():
    """Get a DB connection with busy_timeout for librarian concurrency."""
    conn = _sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")  # 30 second timeout
    conn.enable_load_extension(True)
    _sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = _sqlite3.Row
    return conn

# Librarian config
PRUNE_ARCHIVE_DAYS = 60       # Facts below cool threshold for this many days get archived
PRUNE_DELETE_DAYS = 120       # Archived facts older than this get hard deleted
ENTITY_SIMILARITY_THRESHOLD = 0.85  # Name similarity for entity merge candidates
DUPLICATE_SIMILARITY_THRESHOLD = 0.90  # Object similarity for duplicate fact detection

# PDS domain descriptions for LLM audit prompt
PDS_REFERENCE = """
1000 — Internal State (identity, health, mood, preferences, personal traits)
  1100 Identity & Self, 1200 Health & Body, 1300 Mood & Mental, 1400 Preferences
2000 — Relational Orbit (family, friends, colleagues, professional relationships)
  2100 Immediate Kin, 2200 Extended Family, 2300 Social Circle, 2400 Professional
3000 — Instrumental (projects, career, infrastructure, finance, tools)
  3100 Projects, 3200 Career, 3300 Infrastructure, 3400 Finance
4000 — Chronological (events, duration, routine, origins, schedules)
  4100 Fixed Schedule, 4200 Specific Events, 4300 Origins & History
5000 — Conceptual (beliefs, mental models, philosophical positions, values)
  5100 Beliefs & Values, 5200 Mental Models, 5300 Learning & Growth
"""



# ─── JSON Parsing Helpers ────────────────────────────────────────────────────

def _strip_code_fences(text):
    """Remove markdown code fences from JSON responses."""
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    return text


def _safe_json_parse(raw, is_array=True):
    """Robust JSON parsing that handles code fences and partial responses."""
    if not raw or not raw.strip():
        return None
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    pattern = r'\[.*\]' if is_array else r'\{.*\}'
    import re as _re
    match = _re.search(pattern, cleaned, _re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ─── 1. Audit & Recategorise ─────────────────────────────────────────────────

def audit_and_recategorise():
    """
    Scan facts with low confidence or suspicious PDS codes.
    Use LLM to re-evaluate and correct PDS categorisation.
    """
    conn = get_db_safe()
    print("\n📚 [1/4] Audit & Recategorise")

    # Find facts that are candidates for recategorisation:
    # - Low confidence (< 0.7)
    # - Missing or suspicious PDS codes (not matching known subdomains)
    # - PDS domain doesn't match the subdomain prefix
    valid_subdomains = set()
    subdomains = conn.execute("SELECT code FROM pds_subdomains").fetchall()
    valid_subdomains = {s["code"] for s in subdomains}

    candidates = conn.execute(
        """SELECT f.id, f.predicate, f.object_value, f.pds_decimal, f.pds_domain,
                  f.confidence, e.name as subject_name, eo.name as object_name
           FROM facts f
           JOIN entities e ON f.subject_entity_id = e.id
           LEFT JOIN entities eo ON f.object_entity_id = eo.id
           WHERE f.valid_until IS NULL
           AND (f.confidence < 0.7
                OR f.pds_decimal IS NULL
                OR f.pds_decimal NOT IN ({})
                OR (f.pds_domain IS NOT NULL AND f.pds_decimal IS NOT NULL
                    AND substr(f.pds_decimal, 1, 1) || '000' != f.pds_domain))
           LIMIT 50"""
        .format(",".join(f"'{s}'" for s in valid_subdomains))
    ).fetchall()

    if not candidates:
        print("  ✓ All facts properly categorised")
        conn.close()
        return 0

    print(f"  → Found {len(candidates)} facts to audit")

    # Batch into groups of 20 for LLM review
    client = get_llm_client()
    recategorised = 0

    for i in range(0, len(candidates), 20):
        batch = candidates[i:i+20]
        fact_lines = []
        for f in batch:
            obj = f["object_name"] or f["object_value"] or ""
            fact_lines.append(
                f"ID:{f['id']} | {f['subject_name']} {f['predicate']} {obj} | "
                f"PDS:{f['pds_decimal']} Domain:{f['pds_domain']} | "
                f"Confidence:{f['confidence']}"
            )

        prompt = f"""You are a memory librarian auditing fact categorisation.

PDS Reference:
{PDS_REFERENCE}

Review these facts and determine if their PDS categorisation is correct.
For each fact that needs recategorisation, return the corrected PDS code.

Return a JSON array of objects with:
- "id": the fact ID
- "pds_decimal": the corrected 4-digit PDS subdomain code
- "pds_domain": the corrected 3-digit domain code (first digit + "000")
- "reason": brief reason for the change
- "confidence": updated confidence score (0.0-1.0), or null to keep existing

Only include facts that need changes. Return [] if all are correct.

Facts to audit:
{chr(10).join(fact_lines)}
"""

        try:
            resp = client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=1000,
            )
            raw = resp.choices[0].message.content.strip()
            corrections = _safe_json_parse(raw, is_array=True)
        except Exception as e:
            print(f"  ✗ Audit batch {i//20} failed: {e}")
            continue

        if not corrections:
            continue
        for corr in corrections:
            fid = corr.get("id")
            new_pds = corr.get("pds_decimal")
            new_domain = corr.get("pds_domain")
            new_conf = corr.get("confidence")
            reason = corr.get("reason", "")

            if not fid or not new_pds:
                continue

            # Verify the new PDS code is valid
            if new_pds not in valid_subdomains:
                print(f"  ✗ Skipping fact #{fid}: invalid PDS code '{new_pds}'")
                continue

            updates = []
            params = []
            if new_pds:
                updates.append("pds_decimal = ?")
                params.append(new_pds)
            if new_domain:
                updates.append("pds_domain = ?")
                params.append(new_domain)
            if new_conf is not None:
                updates.append("confidence = ?")
                params.append(new_conf)
            params.append(fid)

            if updates:
                conn.execute(
                    f"UPDATE facts SET {', '.join(updates)} WHERE id = ?",
                    params
                )
                # Re-index in FTS5
                fact = conn.execute(
                    """SELECT f.predicate, f.object_value, e.name as subject_name,
                       eo.name as object_name
                       FROM facts f
                       JOIN entities e ON f.subject_entity_id = e.id
                       LEFT JOIN entities eo ON f.object_entity_id = eo.id
                       WHERE f.id = ?""",
                    (fid,)
                ).fetchone()
                if fact:
                    index_fact_fts(conn, fid, fact["subject_name"], fact["predicate"],
                                   fact["object_name"], fact["object_value"],
                                   new_domain or "", new_pds)
                recategorised += 1
                print(f"  📁 Fact #{fid}: {reason}")

    conn.commit()
    conn.close()
    print(f"  ✓ Recategorised {recategorised} facts")
    return recategorised


# ─── 2. Duplicate Detection & Merging ────────────────────────────────────────

def find_duplicate_facts():
    """
    Find cross-session duplicate facts (same subject + predicate, similar objects).
    Unlike contradiction detection (which handles singular predicates with different objects),
    this catches duplicates where the object is the same but worded differently.
    """
    conn = get_db_safe()
    print("\n📚 [2/4] Duplicate Detection & Merging")

    # Group active facts by subject + predicate
    groups = conn.execute(
        """SELECT f.id, f.predicate, f.object_value, f.object_entity_id,
                  f.pds_decimal, f.confidence, f.source_session_id,
                  e.name as subject_name, eo.name as object_name,
                  f.created_at
           FROM facts f
           JOIN entities e ON f.subject_entity_id = e.id
           LEFT JOIN entities eo ON f.object_entity_id = eo.id
           WHERE f.valid_until IS NULL AND f.is_composite = 0
           ORDER BY f.subject_entity_id, f.predicate, f.created_at"""
    ).fetchall()

    # Group by subject_name + predicate
    grouped = {}
    for f in groups:
        key = (f["subject_name"].lower(), f["predicate"].lower())
        grouped.setdefault(key, []).append(f)

    duplicates_found = 0
    merged = 0

    for (subj, pred), facts in grouped.items():
        if len(facts) < 2:
            continue

        # Compare all pairs within the group
        for i, f1 in enumerate(facts):
            for j, f2 in enumerate(facts):
                if j <= i:
                    continue

                obj1 = (f1["object_name"] or f1["object_value"] or "").lower().strip()
                obj2 = (f2["object_name"] or f2["object_value"] or "").lower().strip()

                if not obj1 or not obj2:
                    continue

                # Check if objects are the same entity
                same_entity = (f1["object_entity_id"] and f2["object_entity_id"]
                               and f1["object_entity_id"] == f2["object_entity_id"])

                # Check string similarity
                similarity = SequenceMatcher(None, obj1, obj2).ratio()

                if same_entity or similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
                    duplicates_found += 1
                    # Keep the one with higher confidence, or the older one if equal
                    keep, remove = (f1, f2) if f1["confidence"] >= f2["confidence"] else (f2, f1)

                    # Don't merge if they're the exact same fact pointing to same entity
                    if same_entity and f1["object_entity_id"] == f2["object_entity_id"]:
                        # Mark the newer one as superseded by the older one
                        now = datetime.now(timezone.utc).isoformat()
                        conn.execute(
                            "UPDATE facts SET valid_until = ?, superseded_by = ? WHERE id = ?",
                            (now, keep["id"], remove["id"])
                        )
                        merged += 1
                        print(f"  🔀 Merged: {subj} {pred} '{obj1}' == '{obj2}' → kept #{keep['id']}")
                    elif similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
                        # Use LLM to determine if these are truly the same fact
                        is_dup = llm_check_duplicate(
                            subj, pred,
                            f1["object_name"] or f1["object_value"],
                            f2["object_name"] or f2["object_value"]
                        )
                        if is_dup:
                            now = datetime.now(timezone.utc).isoformat()
                            conn.execute(
                                "UPDATE facts SET valid_until = ?, superseded_by = ? WHERE id = ?",
                                (now, keep["id"], remove["id"])
                            )
                            merged += 1
                            print(f"  🔀 Merged (LLM confirmed): {subj} {pred} '{obj1}' ≈ '{obj2}' → kept #{keep['id']}")

    conn.commit()
    conn.close()
    print(f"  ✓ Found {duplicates_found} duplicate pairs, merged {merged}")
    return merged


def llm_check_duplicate(subject, predicate, obj1, obj2):
    """Use LLM to determine if two facts are semantically identical."""
    client = get_llm_client()
    prompt = f"""Are these two facts describing the same thing?

Fact 1: {subject} {predicate} {obj1}
Fact 2: {subject} {predicate} {obj2}

Answer with ONLY "true" or "false". True if they describe the same information, false if they are different facts."""

    try:
        resp = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=300,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        return "true" in answer
    except Exception:
        return False


# ─── 3. Pruning ───────────────────────────────────────────────────────────────

def prune_cold_facts():
    """
    Archive facts that have decayed below the cold threshold and haven't been
    accessed in PRUNE_ARCHIVE_DAYS. Hard delete facts that are both cold
    AND superseded AND older than PRUNE_DELETE_DAYS.
    """
    conn = get_db_safe()
    print("\n📚 [3/4] Pruning")

    now = datetime.now(timezone.utc)
    archive_cutoff = (now - timedelta(days=PRUNE_ARCHIVE_DAYS)).isoformat()
    delete_cutoff = (now - timedelta(days=PRUNE_DELETE_DAYS)).isoformat()

    # Find facts to archive: cold activation, not accessed in 60 days, not superseded
    archive_candidates = conn.execute(
        """SELECT id, subject_entity_id, predicate, activation, last_accessed_at, created_at
           FROM facts
           WHERE activation < ? AND valid_until IS NULL AND superseded_by IS NULL
           AND (last_accessed_at IS NULL OR last_accessed_at < ?)
           AND created_at < ?""",
        (COOL_THRESHOLD, archive_cutoff, archive_cutoff)
    ).fetchall()

    archived = 0
    for f in archive_candidates:
        # Mark as archived (set valid_until but don't supersede)
        conn.execute(
            "UPDATE facts SET valid_until = ? WHERE id = ?",
            (now.isoformat(), f["id"])
        )
        archived += 1

    print(f"  📦 Archived {archived} cold facts (activation < {COOL_THRESHOLD}, untouched > {PRUNE_ARCHIVE_DAYS} days)")

    # Find facts to hard delete: superseded AND archived AND very old
    delete_candidates = conn.execute(
        """SELECT id FROM facts
           WHERE superseded_by IS NOT NULL
           AND valid_until IS NOT NULL
           AND valid_until < ?""",
        (delete_cutoff,)
    ).fetchall()

    deleted = 0
    for f in delete_candidates:
        # Remove from FTS5 index
        conn.execute("DELETE FROM facts_fts WHERE fact_id = ?", (f["id"],))
        # Remove from fact embeddings if they exist
        try:
            map_row = conn.execute(
                "SELECT rowid FROM fact_embedding_map WHERE fact_id = ?", (f["id"],)
            ).fetchone()
            if map_row:
                conn.execute("DELETE FROM fact_embeddings WHERE rowid = ?", (map_row["rowid"],))
                conn.execute("DELETE FROM fact_embedding_map WHERE rowid = ?", (map_row["rowid"],))
        except Exception:
            pass
        # Delete the fact
        conn.execute("DELETE FROM facts WHERE id = ?", (f["id"],))
        deleted += 1

    print(f"  🗑️  Hard deleted {deleted} superseded+archived facts (older than {PRUNE_DELETE_DAYS} days)")

    # Clean up orphaned consolidated memories (very old, never accessed)
    old_memories = conn.execute(
        """SELECT m.id FROM consolidated_memories m
           WHERE m.created_at < ?
           AND m.id NOT IN (SELECT DISTINCT mm.memory_id FROM memory_embedding_map mm
                           JOIN memory_embeddings v ON v.rowid = mm.rowid)""",
        (delete_cutoff,)
    ).fetchall()

    mem_deleted = 0
    for m in old_memories:
        conn.execute("DELETE FROM consolidated_memories WHERE id = ?", (m["id"],))
        mem_deleted += 1

    if mem_deleted:
        print(f"  🗑️  Cleaned up {mem_deleted} orphaned consolidated memories")

    conn.commit()
    conn.close()
    print(f"  ✓ Pruning complete: {archived} archived, {deleted} deleted")
    return archived + deleted


# ─── 4. Entity Reconciliation ─────────────────────────────────────────────────

def reconcile_entities():
    """
    Find and merge entity duplicates (different name variants for the same entity).
    Uses string similarity and LLM confirmation.
    """
    conn = get_db_safe()
    print("\n📚 [4/4] Entity Reconciliation")

    entities = conn.execute(
        "SELECT id, name, type, aliases FROM entities ORDER BY name"
    ).fetchall()

    if len(entities) < 2:
        print("  ✓ Not enough entities to reconcile")
        conn.close()
        return 0

    # Find potential merge candidates by name similarity
    merge_candidates = []
    for i, e1 in enumerate(entities):
        for j, e2 in enumerate(entities):
            if j <= i:
                continue
            name1 = e1["name"].lower().strip()
            name2 = e2["name"].lower().strip()

            # Skip if one is a substring of the other and they're different lengths
            # (e.g., "Phil" vs "Alex" — likely same person)
            similarity = SequenceMatcher(None, name1, name2).ratio()

            # Also check if one name is a prefix/abbreviation of the other
            is_prefix = (name1.startswith(name2) or name2.startswith(name1)) and min(len(name1), len(name2)) >= 3

            if similarity >= ENTITY_SIMILARITY_THRESHOLD or is_prefix:
                merge_candidates.append((e1, e2, similarity))

    if not merge_candidates:
        print("  ✓ No entity merge candidates found")
        conn.close()
        return 0

    # Sort by similarity (highest first) and cap to avoid timeout
    merge_candidates.sort(key=lambda x: -x[2])
    MAX_MERGE_CHECKS = 50  # Cap LLM calls per run
    merge_candidates = merge_candidates[:MAX_MERGE_CHECKS]
    print(f"  → Checking top {len(merge_candidates)} merge candidates (capped at {MAX_MERGE_CHECKS})")

    client = get_llm_client()
    merged = 0

    for e1, e2, sim in merge_candidates:
        # Determine which entity to keep (prefer the one with more facts)
        count1 = conn.execute(
            "SELECT COUNT(*) as c FROM facts WHERE subject_entity_id = ? AND valid_until IS NULL",
            (e1["id"],)
        ).fetchone()["c"]
        count2 = conn.execute(
            "SELECT COUNT(*) as c FROM facts WHERE subject_entity_id = ? AND valid_until IS NULL",
            (e2["id"],)
        ).fetchone()["c"]

        keep, remove = (e1, e2) if count1 >= count2 else (e2, e1)

        # LLM confirmation
        prompt = f"""Are these two entity names referring to the same thing?

Entity 1: "{e1['name']}" (type: {e1['type'] or 'unknown'})
Entity 2: "{e2['name']}" (type: {e2['type'] or 'unknown'})

Consider: name similarity, common abbreviations, nicknames, and context.
Answer with ONLY "true" or "false". True if they are the same entity."""

        try:
            resp = client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=300,
            )
            answer = resp.choices[0].message.content.strip().lower()
            should_merge = "true" in answer
        except Exception:
            should_merge = sim >= 0.95  # Fall back to high threshold

        if should_merge:
            # Merge: move all facts from remove → keep
            # Update subject_entity_id
            conn.execute(
                "UPDATE facts SET subject_entity_id = ? WHERE subject_entity_id = ?",
                (keep["id"], remove["id"])
            )
            # Update object_entity_id
            conn.execute(
                "UPDATE facts SET object_entity_id = ? WHERE object_entity_id = ?",
                (keep["id"], remove["id"])
            )
            # Update consolidated_memories entity_ids (JSON arrays)
            memories = conn.execute(
                "SELECT id, entity_ids FROM consolidated_memories WHERE entity_ids IS NOT NULL"
            ).fetchall()
            for m in memories:
                try:
                    ids = json.loads(m["entity_ids"])
                    if remove["id"] in ids:
                        ids = [keep["id"] if x == remove["id"] else x for x in ids]
                        # Deduplicate
                        ids = list(set(ids))
                        conn.execute(
                            "UPDATE consolidated_memories SET entity_ids = ? WHERE id = ?",
                            (json.dumps(ids), m["id"])
                        )
                except (json.JSONDecodeError, TypeError):
                    pass

            # Add the removed entity's name as an alias of the kept entity
            existing_aliases = []
            if keep["aliases"]:
                try:
                    existing_aliases = json.loads(keep["aliases"])
                except (json.JSONDecodeError, TypeError):
                    pass
            existing_aliases.append(remove["name"])
            # Also carry over any aliases from the removed entity
            if remove["aliases"]:
                try:
                    removed_aliases = json.loads(remove["aliases"])
                    existing_aliases.extend(removed_aliases)
                except (json.JSONDecodeError, TypeError):
                    pass
            # Deduplicate aliases
            existing_aliases = list(set(existing_aliases))
            conn.execute(
                "UPDATE entities SET aliases = ? WHERE id = ?",
                (json.dumps(existing_aliases), keep["id"])
            )

            # Delete the merged entity
            conn.execute("DELETE FROM entities WHERE id = ?", (remove["id"],))

            merged += 1
            print(f"  🔗 Merged: '{remove['name']}' → '{keep['name']}' (similarity: {sim:.2f})")

    # Clean up orphaned entities (no facts referencing them)
    orphaned = conn.execute(
        """SELECT id, name FROM entities
           WHERE id NOT IN (SELECT DISTINCT subject_entity_id FROM facts WHERE valid_until IS NULL)
           AND id NOT IN (SELECT DISTINCT object_entity_id FROM facts WHERE valid_until IS NULL AND object_entity_id IS NOT NULL)"""
    ).fetchall()

    orphan_count = 0
    for o in orphaned:
        conn.execute("DELETE FROM entities WHERE id = ?", (o["id"],))
        orphan_count += 1

    if orphan_count:
        print(f"  🧹 Cleaned up {orphan_count} orphaned entities")

    conn.commit()
    conn.close()
    print(f"  ✓ Merged {merged} entities, removed {orphan_count} orphans")
    return merged + orphan_count


# ─── 5. Confidence Recalibration ─────────────────────────────────────────────

# Predicates that are multi-valued — a subject can have multiple different values
# without them being contradictions (e.g. Alex worked_at multiple companies)
MULTI_VALUED_PREDICATES = {
    'worked_at', 'has_role', 'has_tool_proficiency', 'achievement',
    'has_education', 'has_certification', 'communicated_in_channel',
    'has_contact', 'has_skill', 'managed_vendors', 'managed_workforce',
    'explored', 'wants', 'likes', 'has_interest', 'has_hobby',
    'has_project', 'owns_project', 'has_child', 'has_family_member',
    'had_role', 'previous_role', 'has_previous_experience',
}

# Predicates that are singular — two different values ARE a contradiction
# (e.g. Alex has_name X vs Alex has_name Y)
SINGULAR_PREDICATES = {
    'has_name', 'has_age', 'has_birthday', 'has_address', 'has_phone',
    'has_email', 'has_timezone', 'has_pronouns', 'has_employer',
    'current_role', 'has_title', 'hosted_at', 'has_location',
}


def _check_contradictions(conn, fact_id, subject_name, predicate, obj_value):
    """Tier 1: Check if this fact contradicts any other active fact in the DB.

    Only flags contradictions for singular predicates — where two different
    values for the same subject+predicate are genuinely mutually exclusive.
    Multi-valued predicates (worked_at, has_role, has_tool_proficiency, etc.)
    are NOT contradictions — a person can have multiple employers, roles, or skills.
    """
    # Skip contradiction check for multi-valued predicates
    if predicate in MULTI_VALUED_PREDICATES:
        return None

    # For singular predicates, find other facts with same subject+predicate but different value
    contradictions = conn.execute(
        """SELECT f.id, f.object_value, f.confidence, eo.name as object_name
           FROM facts f
           JOIN entities e ON f.subject_entity_id = e.id
           LEFT JOIN entities eo ON f.object_entity_id = eo.id
           WHERE e.name = ? AND f.predicate = ? AND f.id != ?
           AND f.valid_until IS NULL AND f.superseded_by IS NULL""",
        (subject_name, predicate, fact_id)
    ).fetchall()

    if not contradictions:
        return None

    current_val = (obj_value or "").strip().lower()
    for c in contradictions:
        other_val = (c["object_name"] or c["object_value"] or "").strip().lower()
        if other_val and other_val != current_val:
            return {
                "contradicts_id": c["id"],
                "contradicts_value": c["object_name"] or c["object_value"],
                "contradicts_confidence": c["confidence"]
            }
    return None


def _check_corroboration(conn, fact_id, subject_name, predicate, obj_value):
    """Tier 2: Check if other facts corroborate this one (same subject, related predicates)."""
    # Count other active facts about the same subject
    related = conn.execute(
        """SELECT COUNT(*) as c FROM facts f
           JOIN entities e ON f.subject_entity_id = e.id
           WHERE e.name = ? AND f.id != ?
           AND f.valid_until IS NULL AND f.superseded_by IS NULL""",
        (subject_name, fact_id)
    ).fetchone()

    return related["c"] if related else 0


def recalibrate_confidence():
    """
    Three-tier confidence recalibration:
    1. Self-consistency: Does this fact contradict other facts in the DB?
    2. Source confidence weighting: Trust high-confidence extractions unless contradicted.
    3. LLM verification with wide context: Only for suspicious facts, using full session text.

    Guard rule: Never drop a fact below 0.5 in a single pass. This prevents
    false-negative nukes from cascading — a second pass or future session can recover.
    """
    conn = get_db_safe()
    print("\n📚 [5/5] Confidence Recalibration (Three-Tier)")

    # Get candidate facts: low confidence to verify, high confidence old ones to spot-check
    candidates = conn.execute(
        """SELECT f.id, f.predicate, f.object_value, f.confidence, f.pds_decimal,
                  e.name as subject_name, eo.name as object_name,
                  f.source_session_id, s.content as session_content
           FROM facts f
           JOIN entities e ON f.subject_entity_id = e.id
           LEFT JOIN entities eo ON f.object_entity_id = eo.id
           LEFT JOIN raw_sessions s ON f.source_session_id = s.id
           WHERE f.valid_until IS NULL AND f.is_composite = 0
           AND (f.confidence < 0.8 OR (f.confidence >= 0.9 AND f.created_at < datetime('now', '-30 days')))
           ORDER BY f.confidence ASC
           LIMIT 30"""
    ).fetchall()

    if not candidates:
        print("  ✓ No facts need recalibration")
        conn.close()
        return 0

    print(f"  → Checking {len(candidates)} facts (three-tier verification)")

    client = get_llm_client()
    recalibrated = 0
    tier1_hits = 0  # Contradictions found
    tier2_skipped = 0  # Trusted via source confidence + corroboration
    tier3_verified = 0  # LLM verified with wide context

    # ─── Tier 1: Self-consistency check ────────────────────────────────
    # ─── Tier 2: Source confidence weighting ───────────────────────────
    # Process all candidates through tiers 1 & 2 first
    needs_llm = []  # Facts that survive tiers 1-2 and need LLM verification

    for f in candidates:
        obj = f["object_name"] or f["object_value"] or ""

        # Tier 1: Check for contradictions
        contradiction = _check_contradictions(conn, f["id"], f["subject_name"], f["predicate"], obj)
        if contradiction:
            # Fact contradicts another — flag it down to 0.5 (guard rule)
            new_conf = 0.50
            old_conf = f["confidence"]
            if abs(new_conf - old_conf) > 0.1:
                conn.execute("UPDATE facts SET confidence = ? WHERE id = ?", (new_conf, f["id"]))
                print(f"  ⚠ Fact #{f['id']}: {old_conf:.2f} → {new_conf:.2f} — contradicts fact #{contradiction['contradicts_id']} ({f['subject_name']} {f['predicate']} {contradiction['contradicts_value']})")
                recalibrated += 1
                tier1_hits += 1
            continue  # Don't LLM-verify contradicted facts

        # Tier 2: Source confidence weighting
        # If the fact was extracted with high confidence and has corroboration, trust it
        corroborating = _check_corroboration(conn, f["id"], f["subject_name"], f["predicate"], obj)
        if f["confidence"] >= 0.85 and corroborating >= 2:
            # High confidence + multiple corroborating facts = trust the extraction
            tier2_skipped += 1
            continue

        # Survived tiers 1-2, needs LLM verification
        needs_llm.append(f)

    if needs_llm:
        print(f"  → Tier 1: {tier1_hits} contradictions found")
        print(f"  → Tier 2: {tier2_skipped} facts trusted via source confidence + corroboration")
        print(f"  → Tier 3: {len(needs_llm)} facts need LLM verification with wide context")

    # ─── Tier 3: LLM verification with wide context ────────────────────
    # Use full session content (up to 3000 chars) instead of narrow snippets
    # Batch into groups of 5 (smaller batches = more context per fact)
    for i in range(0, len(needs_llm), 5):
        batch = needs_llm[i:i+5]
        fact_lines = []
        for f in batch:
            obj = f["object_name"] or f["object_value"] or ""
            # Use wide context: 3000 chars from the source session
            session_text = (f["session_content"] or "")[:3000]
            fact_lines.append(
                f"ID:{f['id']} | {f['subject_name']} {f['predicate']} {obj} | "
                f"Current confidence:{f['confidence']} | "
                f"Full source context (up to 3000 chars): {session_text}"
            )

        prompt = f"""You are a fact verification system. For each fact, check if it is supported by its source transcript.

Important guidelines:
- The source context may be a long conversation. The fact may be supported by content anywhere in that context, not just the beginning.
- If the context is a conversation between humans and an AI assistant, facts may be derived from the overall discussion, not just explicit statements.
- If you cannot find evidence but the fact seems reasonable and was extracted with high confidence, do not score below 0.5 — it may have been derived from context not captured in this excerpt.
- Only score below 0.5 if you find clear evidence the fact is WRONG or CONTRADICTED.

For each fact, return a JSON object with:
- "id": the fact ID
- "confidence": new confidence score (0.0-1.0)
  - 1.0: directly stated in the transcript
  - 0.8: clearly implied or strongly supported by the discussion
  - 0.6: partially supported, reasonable inference
  - 0.5: cannot verify but no evidence against it — keep as provisional
  - 0.3: appears unsupported and possibly wrong
  - 0.0: directly contradicted by the transcript
- "reason": brief explanation

Facts to verify:
{chr(10).join(fact_lines)}

Return ONLY a JSON array of objects."""

        try:
            resp = client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=1000,
            )
            raw = resp.choices[0].message.content.strip()
            results = _safe_json_parse(raw, is_array=True)
        except Exception as e:
            print(f"  ✗ Recalibration batch {i//5} failed: {e}")
            continue

        if not results:
            continue
        for r in results:
            fid = r.get("id")
            new_conf = r.get("confidence")
            reason = r.get("reason", "")

            if not fid or new_conf is None:
                continue

            new_conf = max(0.0, min(1.0, float(new_conf)))

            # Guard rule: never drop below 0.5 in a single pass
            # (prevents false-negative cascades from wiping valid facts)
            old_conf = next((f["confidence"] for f in batch if f["id"] == fid), None)
            if old_conf and new_conf < 0.5 and old_conf >= 0.5:
                new_conf = 0.5  # Floor at 0.5 for this pass
                reason += " (floored at 0.5 by guard rule)"

            # Only update if confidence changed significantly (delta > 0.1)
            if old_conf and abs(new_conf - old_conf) > 0.1:
                conn.execute(
                    "UPDATE facts SET confidence = ? WHERE id = ?",
                    (new_conf, fid)
                )
                direction = "↑" if new_conf > old_conf else "↓"
                print(f"  {direction} Fact #{fid}: {old_conf:.2f} → {new_conf:.2f} — {reason[:80]}")
                recalibrated += 1
                tier3_verified += 1

    conn.commit()
    conn.close()
    print(f"  ✓ Recalibrated {recalibrated} facts (T1:{tier1_hits} contradictions, T2:{tier2_skipped} trusted, T3:{tier3_verified} LLM-verified)")
    return recalibrated


# ─── Summary Report ───────────────────────────────────────────────────────────

def librarian_report():
    """Generate a summary report of the librarian's work."""
    conn = get_db_safe()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "before": {
            "facts": conn.execute("SELECT COUNT(*) as c FROM facts WHERE valid_until IS NULL").fetchone()["c"],
            "entities": conn.execute("SELECT COUNT(*) as c FROM entities").fetchone()["c"],
            "superseded": conn.execute("SELECT COUNT(*) as c FROM facts WHERE superseded_by IS NOT NULL").fetchone()["c"],
            "archived": conn.execute("SELECT COUNT(*) as c FROM facts WHERE valid_until IS NOT NULL AND superseded_by IS NULL").fetchone()["c"],
        }
    }
    conn.close()
    return report


def run_full_librarian():
    """Run the full librarian cycle: audit → merge → prune → reconcile → recalibrate → digest."""
    print("╔══════════════════════════════════════════════╗")
    print("║     Muninn Librarian — Daily Maintenance     ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")

    before = librarian_report()

    audit_and_recategorise()
    find_duplicate_facts()
    prune_cold_facts()
    reconcile_entities()
    recalibrate_confidence()

    # Export fact digest so OpenClaw's memory_search can index it
    try:
        from muninn import export_fact_digest
        print("\n📚 [Digest] Exporting fact digest...")
        export_fact_digest()
    except Exception as e:
        print(f"  ⚠ Fact digest export failed: {e}")

    after = librarian_report()

    print("\n╔══════════════════════════════════════════════╗")
    print("║     Librarian Summary                        ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  Active facts:  {before['before']['facts']:>5} → {after['before']['facts']:>5}            ║")
    print(f"║  Entities:      {before['before']['entities']:>5} → {after['before']['entities']:>5}            ║")
    print(f"║  Superseded:    {before['before']['superseded']:>5} → {after['before']['superseded']:>5}            ║")
    print(f"║  Archived:      {before['before']['archived']:>5} → {after['before']['archived']:>5}            ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Completed: {datetime.now(timezone.utc).isoformat()}")

    # Write report to daily memory
    report_path = os.path.join(WORKSPACE, "memory", datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".md")
    entry = f"\n## Muninn Librarian — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC\n"
    entry += f"- Active facts: {before['before']['facts']} → {after['before']['facts']}\n"
    entry += f"- Entities: {before['before']['entities']} → {after['before']['entities']}\n"
    entry += f"- Superseded: {before['before']['superseded']} → {after['before']['superseded']}\n"
    entry += f"- Archived: {before['before']['archived']} → {after['before']['archived']}\n"

    try:
        with open(report_path, "a") as f:
            f.write(entry)
    except Exception as e:
        print(f"  ⚠ Could not write to daily memory: {e}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Muninn Librarian — Memory Collection Manager")
    parser.add_argument("--audit-only", action="store_true", help="Only run audit & recategorise")
    parser.add_argument("--merge-only", action="store_true", help="Only run duplicate merging")
    parser.add_argument("--prune-only", action="store_true", help="Only run pruning")
    parser.add_argument("--reconcile-only", action="store_true", help="Only run entity reconciliation")
    parser.add_argument("--recalibrate-only", action="store_true", help="Only run confidence recalibration")
    args = parser.parse_args()

    if args.audit_only:
        audit_and_recategorise()
    elif args.merge_only:
        find_duplicate_facts()
    elif args.prune_only:
        prune_cold_facts()
    elif args.reconcile_only:
        reconcile_entities()
    elif args.recalibrate_only:
        recalibrate_confidence()
    else:
        run_full_librarian()