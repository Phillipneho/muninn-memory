#!/usr/bin/env python3
"""
Muninn Local v3 — Memory as evolving reality
Inspired by Muninn + coolmanns' OpenClaw memory architecture + SimpleMem

New in v3:
- Hybrid retrieval (BM25 via FTS5 + semantic via sqlite-vec + structured PDS filtering)
- Intent-aware retrieval planning (LLM generates retrieval plan before searching)
- Online semantic synthesis (merge related facts during WRITE phase)
- EvolveMem self-tuning loop (Evaluate → Diagnose → Propose → Guard)
- Pre-answer recall hook (callable from OpenClaw's memory pipeline)
- FTS5 lexical search for sessions and facts
- Reciprocal Rank Fusion (RRF) for merging semantic + lexical results

Usage:
    ./muninn.sh init
    ./muninn.sh ingest --content "..." --source discord --speakers '["Alex","Leo"]'
    ./muninn.sh ingest-transcript --file session.json
    ./muninn.sh search "query" --top-k 10
    ./muninn.sh hybrid-search "query" [--top-k 10] [--plan]
    ./muninn.sh plan-search "query"
    ./muninn.sh extract --all-unprocessed
    ./muninn.sh synthesize [--entity "Alex"]
    ./muninn.sh facts --entity "Alex" [--pds 2100]
    ./muninn.sh recall "what were we working on"
    ./muninn.sh recall-context "user's question"   # JSON context bundle for pre-answer hook
    ./muninn.sh decay
    ./muninn.sh sleep-cycle
    ./muninn.sh evolve                               # Run EvolveMem self-tuning loop
    ./muninn.sh evaluate                             # Evaluate retrieval quality
    ./muninn.sh status
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import sqlite_vec
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("MUNINN_DB_PATH",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "muninn.db"))
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

# Embeddings: local Ollama (nomic-embed-text)
EMBEDDING_BASE_URL = os.environ.get("MUNINN_EMBED_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("MUNINN_EMBED_MODEL", "nomic-embed-text")
EMBEDDING_DIMS = 768

# LLM: Ollama Cloud (GLM-5.2) for fact extraction & consolidation
LLM_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com")
LLM_API_KEY = os.environ.get("OLLAMA_API_KEY", os.environ.get("ANTHROPIC_AUTH_TOKEN", ""))
EXTRACTION_MODEL = os.environ.get("MUNINN_LLM_MODEL", "glm-5.3-flash:cloud")

# OpenClaw workspace paths
WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
MEMORY_MD = os.path.join(WORKSPACE, "MEMORY.md")
DAILY_DIR = os.path.join(WORKSPACE, "memory")

# Decay config
DECAY_HALF_LIFE_DAYS = 30  # activation halves every 30 days without access
HOT_THRESHOLD = 0.5
WARM_THRESHOLD = 0.2
COOL_THRESHOLD = 0.05  # below this, facts are candidates for forgetting

# RRF constant (standard value from TREC literature)
RRF_K = 60

# ─── Database ─────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    # Vector embedding tables (sqlite-vec)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS session_embeddings
        USING vec0(embedding float[{EMBEDDING_DIMS}])
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_embedding_map (
            rowid INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL
        )
    """)
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings
        USING vec0(embedding float[{EMBEDDING_DIMS}])
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_embedding_map (
            rowid INTEGER PRIMARY KEY,
            memory_id INTEGER NOT NULL
        )
    """)

    # v3: Fact embeddings for semantic fact search
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS fact_embeddings
        USING vec0(embedding float[{EMBEDDING_DIMS}])
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact_embedding_map (
            rowid INTEGER PRIMARY KEY,
            fact_id INTEGER NOT NULL
        )
    """)

    # v3: Migration — add new columns to existing tables if they don't exist
    # SQLite doesn't support ADD COLUMN IF NOT EXISTS, so we check pragma first
    def column_exists(conn, table, column):
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(c["name"] == column for c in cols)

    # Add is_composite and component_fact_ids to facts table
    if not column_exists(conn, "facts", "is_composite"):
        try:
            conn.execute("ALTER TABLE facts ADD COLUMN is_composite INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
    if not column_exists(conn, "facts", "component_fact_ids"):
        try:
            conn.execute("ALTER TABLE facts ADD COLUMN component_fact_ids TEXT")
        except sqlite3.OperationalError:
            pass

    # Create composite index now that the column exists
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_composite ON facts(is_composite) WHERE is_composite = 1")
    except sqlite3.OperationalError:
        pass

    # v3.1: query_logs.kind — separates real user queries from evolve eval probes.
    # One-time relabel: everything logged before this migration came from
    # evaluate_retrieval probes (no other writer existed), so mark it 'eval'.
    if not column_exists(conn, "query_logs", "kind"):
        try:
            conn.execute("ALTER TABLE query_logs ADD COLUMN kind TEXT NOT NULL DEFAULT 'real'")
            conn.execute("UPDATE query_logs SET kind = 'eval'")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()
    print(f"✓ Database initialised at {DB_PATH}")


def get_config(conn, key, default=None):
    """Get a retrieval config value."""
    row = conn.execute("SELECT value, type FROM retrieval_config WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    val = row["value"]
    if row["type"] == "float":
        return float(val)
    elif row["type"] == "int":
        return int(val)
    elif row["type"] == "json":
        return json.loads(val)
    return val


def set_config(conn, key, value, updated_by="system"):
    """Set a retrieval config value."""
    if isinstance(value, float):
        val_str = str(value)
        vtype = "float"
    elif isinstance(value, int):
        val_str = str(value)
        vtype = "int"
    elif isinstance(value, (dict, list)):
        val_str = json.dumps(value)
        vtype = "json"
    else:
        val_str = str(value)
        vtype = "string"
    conn.execute(
        "INSERT OR REPLACE INTO retrieval_config (key, value, type, updated_at, updated_by) VALUES (?, ?, ?, ?, ?)",
        (key, val_str, vtype, datetime.now(timezone.utc).isoformat(), updated_by)
    )



# ─── JSON Parsing Helpers ────────────────────────────────────────────────────

def _strip_code_fences(text):
    """Remove markdown code fences from JSON responses."""
    text = text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        lines = lines[1:]
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
    match = re.search(pattern, cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return None

# ─── FTS5 Index Management ────────────────────────────────────────────────────

def index_session_fts(conn, session_id, content, session_date, source):
    """Add or update a session in the FTS5 index."""
    # Remove old entry
    conn.execute("DELETE FROM sessions_fts WHERE session_id = ?", (session_id,))
    conn.execute(
        "INSERT INTO sessions_fts (content, session_id, session_date, source) VALUES (?, ?, ?, ?)",
        (content, session_id, session_date, source)
    )


def index_fact_fts(conn, fact_id, subject_name, predicate, object_name, object_value, pds_domain, pds_decimal):
    """Add or update a fact in the FTS5 index."""
    search_text = f"{subject_name} {predicate} {object_name or object_value or ''}"
    conn.execute("DELETE FROM facts_fts WHERE fact_id = ?", (fact_id,))
    conn.execute(
        "INSERT INTO facts_fts (search_text, fact_id, pds_domain, pds_decimal) VALUES (?, ?, ?, ?)",
        (search_text, fact_id, pds_domain or "", pds_decimal or "")
    )


def rebuild_fts_indexes():
    """Rebuild FTS5 indexes from existing data."""
    conn = get_db()
    print("→ Rebuilding FTS5 indexes...")

    # Rebuild sessions FTS
    sessions = conn.execute("SELECT id, content, session_date, source FROM raw_sessions").fetchall()
    conn.execute("DELETE FROM sessions_fts")
    for s in sessions:
        index_session_fts(conn, s["id"], s["content"], s["session_date"], s["source"])
    print(f"  ✓ Indexed {len(sessions)} sessions")

    # Rebuild facts FTS
    facts = conn.execute(
        """SELECT f.id, e.name as subject_name, f.predicate, eo.name as object_name,
           f.object_value, f.pds_domain, f.pds_decimal
           FROM facts f
           JOIN entities e ON f.subject_entity_id = e.id
           LEFT JOIN entities eo ON f.object_entity_id = eo.id"""
    ).fetchall()
    conn.execute("DELETE FROM facts_fts")
    for f in facts:
        index_fact_fts(conn, f["id"], f["subject_name"], f["predicate"],
                       f["object_name"], f["object_value"], f["pds_domain"], f["pds_decimal"])
    print(f"  ✓ Indexed {len(facts)} facts")

    conn.commit()
    conn.close()
    print("✓ FTS5 indexes rebuilt")


# ─── Embeddings ───────────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    import requests
    text = text[:8000]
    resp = requests.post(
        f"{EMBEDDING_BASE_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=30
    )
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)


def _llm_api_key():
    """Key resolution: env OLLAMA_API_KEY first, then OpenClaw auth store
    (authProfiles.store primary DB, agent sqlite fallback). Import paths
    without the wrapper env (grind probes, scripts that skip muninn.sh)
    hit a 401 with the env-only key — expansion/extraction silently died
    returning [] (2026-10-01 cycle-42: audit showed _llm_expand [] on every
    query, LLM 401)."""
    import sqlite3 as _sq, json as _sj
    if LLM_API_KEY:
        return LLM_API_KEY
    for db_path, table, col, outer, inner, where in (
        ('/home/homelab/.openclaw/state/openclaw.sqlite',
         'config_machine_state', 'value_json', 'profiles', 'ollama-cloud:default',
         "state_key='authProfiles.store'"),
        ('/home/homelab/.openclaw/agents/main/agent/openclaw-agent.sqlite',
         'auth_profile_store', 'store_json', 'profiles', 'ollama-cloud:default',
         "store_key='primary'"),
    ):
        try:
            db = _sq.connect(db_path)
            row = db.execute(
                f"SELECT {col} FROM {table} WHERE {where}" +
                (" ORDER BY rowid DESC LIMIT 1" if not where.endswith("='primary'") else ""),
            ).fetchone()
            db.close()
            if row:
                key = _sj.loads(row[0]).get(outer, {}).get(inner, {}).get('key', '')
                if key:
                    return key
        except Exception:
            continue
    return "dummy"


def get_llm_client():
    return OpenAI(base_url=LLM_BASE_URL + "/v1", api_key=_llm_api_key(), timeout=240, max_retries=1)


# ─── Ingestion ───────────────────────────────────────────────────────────────

def ingest_session(content, source="openclaw", speakers=None, channel=None, session_date=None):
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    if session_date is None:
        session_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if speakers is None:
        speakers = []

    conn = get_db()
    conn.execute(
        """INSERT INTO raw_sessions (id, content, session_date, source, speakers, channel)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (session_id, content, session_date, source, json.dumps(speakers), channel)
    )

    # Index in FTS5
    index_session_fts(conn, session_id, content, session_date, source)

    try:
        emb = get_embedding(content)
        cur = conn.execute("INSERT INTO session_embeddings (embedding) VALUES (?)", (emb.tobytes(),))
        rowid = cur.lastrowid
        conn.execute("INSERT INTO session_embedding_map (rowid, session_id) VALUES (?, ?)", (rowid, session_id))
        print(f"✓ Session {session_id} ingested ({len(emb)} dims, rowid={rowid})")
    except Exception as e:
        print(f"✓ Session {session_id} ingested (no embedding: {e})")

    conn.commit()
    conn.close()
    return session_id


def ingest_transcript(filepath):
    """Ingest a JSON session transcript from OpenClaw."""
    with open(filepath) as f:
        data = json.load(f)

    messages = data.get("messages", data.get("transcript", []))
    content_parts = []
    speakers = set()
    for msg in messages:
        role = msg.get("role", msg.get("speaker", "unknown"))
        text = msg.get("content", msg.get("text", ""))
        if text:
            content_parts.append(f"[{role}]: {text}")
            speakers.add(role)

    content = "\n".join(content_parts)
    source = data.get("source", "openclaw")
    channel = data.get("channel", data.get("session_type"))
    session_date = data.get("date", data.get("session_date", datetime.now(timezone.utc).strftime("%Y-%m-%d")))

    return ingest_session(content, source, list(speakers), channel, session_date)


# Backward-compatible aliases
def search_sessions(query, top_k=10):
    """Backward-compatible semantic search (delegates to semantic_search_sessions)."""
    return semantic_search_sessions(query, top_k)


def keyword_search(query, top_k=10):
    """Backward-compatible keyword search (delegates to lexical_search_sessions)."""
    return lexical_search_sessions(query, top_k)


def search_memories(query, top_k=5):
    """Backward-compatible memory search (delegates to semantic_search_memories)."""
    return semantic_search_memories(query, top_k)


# ─── Lexical Search (FTS5 / BM25) ─────────────────────────────────────────────

def lexical_search_sessions(query, top_k=10):
    """BM25 lexical search over sessions using FTS5."""
    conn = get_db()
    try:
        # FTS5 BM25 scoring (negative scores; more negative = more relevant)
        results = conn.execute(
            """SELECT s.id, s.content, s.session_date, s.source, s.speakers, s.channel,
               bm25(sessions_fts) as bm25_score
               FROM sessions_fts
               JOIN raw_sessions s ON s.id = sessions_fts.session_id
               WHERE sessions_fts MATCH ?
               ORDER BY bm25_score
               LIMIT ?""",
            (query, top_k)
        ).fetchall()
        conn.close()
        # Convert BM25 scores to ranks (0-based; lower BM25 = better = lower rank)
        return [{"id": r["id"], "content": r["content"][:500], "session_date": r["session_date"],
                 "source": r["source"], "bm25_score": r["bm25_score"], "rank": i}
                for i, r in enumerate(results)]
    except Exception as e:
        conn.close()
        return []



_SYN = {"kids": ["children", "kid"], "children": ["kids", "child"],
            "kid": ["children", "kids"], "books": ["book", "read", "reading"],
            "book": ["books", "read", "reading"], "read": ["book", "books"],
            "camped": ["camping", "camp", "hiking", "hike"], "camping": ["camp", "camped", "hiking"],
            "hiking": ["hike", "camping", "mountains"], "hike": ["hiking", "mountains"],
            "partake": ["do", "enjoys", "enjoy", "swimming"], "activities": ["enjoys", "enjoy", "hobby", "swimming"],
            "hobbies": ["enjoys", "enjoy"], "hobby": ["enjoys", "enjoy"],
            "status": ["relationship", "parent", "marit"], "relationship": ["status", "single", "married"],
            "like": ["love", "loves", "enjoys", "enjoy"], "likes": ["love", "loves", "enjoys"],
            "seen": ["concert", "show", "saw", "attended"], "saw": ["concert", "show", "seen"],
            "artists": ["concert", "band", "musician"], "bands": ["band", "concert", "concerts"],
            "artist": ["concert", "band", "musician"], "band": ["concert", "bands", "concerts"],
            "art": ["artwork", "painting", "abstract", "watercolor"],
            "artwork": ["art", "painting", "abstract"], "painting": ["paintings", "artwork"],
            "abstract": ["art", "painting"]}

_FTS_STOPWORDS = {"when","did","does","do","what","where","who","whom","is","are","was",
                  "were","the","a","an","of","to","in","on","at","for","and","or","how",
                  "why","has","have","had","with","that","this","it","its"}
def _fts_query(query):
    """FTS5 MATCH preprocessing: natural-language questions fail implicit-AND match
    (stopword/content tokens absent from index -> 0 rows). OR-union over content
    tokens; bm25 ranking keeps relevance. Tokens quoted to neutralize syntax chars."""
    import re as _re
    toks = [t for t in _re.sub(r"['\"?!,.;:()]|[-]{2}", " ", str(query)).split()
            if t.lower() not in _FTS_STOPWORDS and len(t) > 1]
    # Synonym expansion: question vocabulary rarely overlaps fact vocabulary
    # ('kids' vs 'children attended dinosaur exhibit'). Expand before OR-union;
    # bm25 keeps relevance among expanded hits.
    expanded = list(toks)
    for t in toks:
        for s in _SYN.get(t.lower(), []):
            if s not in expanded:
                expanded.append(s)
    if not toks:
        toks = [str(query).replace("?", "").strip()] or ["."]
    return " OR ".join('"%s"' % t.replace('"', '""') for t in expanded)

def lexical_search_facts(query, top_k=10, pds_filter=None):
    """BM25 lexical search over facts using FTS5."""
    conn = get_db()
    try:
        if pds_filter:
            results = conn.execute(
                """SELECT f.id, f.predicate, f.pds_decimal, f.pds_domain, f.activation,
                   e.name as subject_name, eo.name as object_name, f.object_value,
                   bm25(facts_fts) as bm25_score
                   FROM facts_fts
                   JOIN facts f ON f.id = facts_fts.fact_id
                   JOIN entities e ON f.subject_entity_id = e.id
                   LEFT JOIN entities eo ON f.object_entity_id = eo.id
                   WHERE facts_fts MATCH ? AND facts_fts.pds_domain = ?
                   AND f.valid_until IS NULL
                   ORDER BY bm25_score
                   LIMIT ?""",
                (_fts_query(query), str(pds_filter), top_k)
            ).fetchall()
        else:
            results = conn.execute(
                """SELECT f.id, f.predicate, f.pds_decimal, f.pds_domain, f.activation,
                   e.name as subject_name, eo.name as object_name, f.object_value,
                   bm25(facts_fts) as bm25_score
                   FROM facts_fts
                   JOIN facts f ON f.id = facts_fts.fact_id
                   JOIN entities e ON f.subject_entity_id = e.id
                   LEFT JOIN entities eo ON f.object_entity_id = eo.id
                   WHERE facts_fts MATCH ? AND f.valid_until IS NULL
                   ORDER BY bm25_score
                   LIMIT ?""",
                (_fts_query(query), top_k)
            ).fetchall()
        conn.close()
        return [{"id": r["id"], "subject": r["subject_name"], "predicate": r["predicate"],
                 "object": r["object_name"] or r["object_value"], "pds": r["pds_decimal"],
                 "activation": round(r["activation"], 3), "bm25_score": r["bm25_score"], "rank": i}
                for i, r in enumerate(results)]
    except Exception as e:
        conn.close()
        return []


# ─── Semantic Search ──────────────────────────────────────────────────────────

def semantic_search_sessions(query, top_k=10):
    """Dense vector search over sessions."""
    conn = get_db()
    try:
        query_emb = get_embedding(query)
        results = conn.execute(
            """SELECT s.id, s.content, s.session_date, s.source, v.distance
               FROM session_embeddings v
               JOIN session_embedding_map m ON m.rowid = v.rowid
               JOIN raw_sessions s ON s.id = m.session_id
               WHERE v.embedding MATCH ? AND v.k = ?
               ORDER BY v.distance""",
            (query_emb.tobytes(), top_k)
        ).fetchall()
        conn.close()
        return [{"id": r["id"], "content": r["content"][:500], "session_date": r["session_date"],
                 "source": r["source"], "distance": r["distance"],
                 "similarity": round(max(0.0, min(1.0, 1.0 - (r["distance"] or 0) / 25.0)), 3), "rank": i}
                for i, r in enumerate(results)]
    except Exception:
        conn.close()
        return []


def semantic_search_facts(query, top_k=10):
    """Dense vector search over facts using fact_embeddings."""
    conn = get_db()
    try:
        query_emb = get_embedding(query)
        results = conn.execute(
            """SELECT f.id, f.predicate, f.pds_decimal, f.activation,
                   e.name as subject_name, eo.name as object_name, f.object_value, v.distance
               FROM fact_embeddings v
               JOIN fact_embedding_map fm ON fm.rowid = v.rowid
               JOIN facts f ON f.id = fm.fact_id
               JOIN entities e ON f.subject_entity_id = e.id
               LEFT JOIN entities eo ON f.object_entity_id = eo.id
               WHERE v.embedding MATCH ? AND v.k = ? AND f.valid_until IS NULL
               ORDER BY v.distance""",
            (query_emb.tobytes(), top_k)
        ).fetchall()
        conn.close()
        return [{"id": r["id"], "subject": r["subject_name"], "predicate": r["predicate"],
                 "object": r["object_name"] or r["object_value"], "pds": r["pds_decimal"],
                 "activation": round(r["activation"], 3),
                 "similarity": round(max(0.0, min(1.0, 1.0 - (r["distance"] or 0) / 25.0)), 3), "rank": i}
                for i, r in enumerate(results)]
    except Exception:
        conn.close()
        return []


def semantic_search_memories(query, top_k=5):
    """Dense vector search over consolidated memories."""
    conn = get_db()
    try:
        query_emb = get_embedding(query)
        results = conn.execute(
            """SELECT m.id, m.content, m.type, m.created_at, v.distance
               FROM memory_embeddings v
               JOIN memory_embedding_map mm ON mm.rowid = v.rowid
               JOIN consolidated_memories m ON m.id = mm.memory_id
               WHERE v.embedding MATCH ? AND v.k = ?
               ORDER BY v.distance""",
            (query_emb.tobytes(), top_k)
        ).fetchall()
        conn.close()
        return [{"id": r["id"], "content": r["content"], "type": r["type"],
                 "created_at": r["created_at"],
                 "similarity": round(max(0.0, min(1.0, 1.0 - (r["distance"] or 0) / 25.0)), 3), "rank": i}
                for i, r in enumerate(results)]
    except Exception:
        conn.close()
        return []


# ─── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def reciprocal_rank_fusion(result_lists, k=RRF_K):
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank_i)) for each list where the item appears.

    Args:
        result_lists: list of lists, each list is a ranked result list of dicts
                      each dict must have an "id" key and a "rank" key (0-based)
        k: RRF constant (default 60)

    Returns: merged list of dicts sorted by RRF score, each with "id", "rrf_score", and merged data
    """
    scores = {}
    data_map = {}

    for result_list in result_lists:
        for item in result_list:
            item_id = item["id"]
            rank = item.get("rank", 0)
            rrf_score = 1.0 / (k + rank)
            if item_id in scores:
                scores[item_id] += rrf_score
                # Merge data (prefer non-empty values)
                for key, val in item.items():
                    if key not in data_map[item_id] or not data_map[item_id][key]:
                        data_map[item_id][key] = val
            else:
                scores[item_id] = rrf_score
                data_map[item_id] = dict(item)

    # Sort by RRF score descending
    merged = []
    for item_id, score in sorted(scores.items(), key=lambda x: -x[1]):
        item = data_map[item_id]
        item["rrf_score"] = round(score, 6)
        merged.append(item)

    return merged


def weighted_fusion(result_lists, weights=None):
    """
    Merge result lists using weighted score fusion.
    Normalizes scores within each list to [0, 1] then applies weights.
    """
    if weights is None:
        weights = [1.0 / len(result_lists)] * len(result_lists)

    scores = {}
    data_map = {}

    for result_list, weight in zip(result_lists, weights):
        if not result_list:
            continue
        # Normalize scores to [0, 1]
        max_score = max((item.get("similarity", item.get("bm25_score", 0)) for item in result_list), default=1)
        min_score = min((item.get("similarity", item.get("bm25_score", 0)) for item in result_list), default=0)
        score_range = max_score - min_score if max_score != min_score else 1.0

        for item in result_list:
            item_id = item["id"]
            raw_score = item.get("similarity", item.get("bm25_score", 0))
            norm_score = (raw_score - min_score) / score_range if score_range else 0.5
            weighted = norm_score * weight

            if item_id in scores:
                scores[item_id] += weighted
                for key, val in item.items():
                    if key not in data_map[item_id] or not data_map[item_id][key]:
                        data_map[item_id][key] = val
            else:
                scores[item_id] = weighted
                data_map[item_id] = dict(item)

    merged = []
    for item_id, score in sorted(scores.items(), key=lambda x: -x[1]):
        item = data_map[item_id]
        item["fused_score"] = round(score, 6)
        merged.append(item)

    return merged


# ─── Hybrid Retrieval (Semantic + Lexical + Structured) ──────────────────────

def hybrid_search(query, top_k=10, use_plan=False, pds_filter=None):
    """
    Hybrid retrieval combining semantic, lexical, and structured search.
    Uses RRF by default, or weighted fusion based on config.
    """
    conn = get_db()
    fusion_mode = get_config(conn, "fusion_mode", "rrf")
    sem_weight = get_config(conn, "semantic_weight", 0.5)
    lex_weight = get_config(conn, "lexical_weight", 0.5)
    fact_search_enabled = get_config(conn, "fact_search_enabled", 1)
    rrf_k = get_config(conn, "rrf_k", RRF_K)
    conn.close()

    # If plan is requested, generate a retrieval plan first
    plan = None
    if use_plan:
        plan = plan_retrieval(query)
        if plan:
            top_k = plan.get("top_k", top_k)
            sub_queries = plan.get("query_decomposition", [])
            if sub_queries and len(sub_queries) > 1:
                # Execute sub-queries and fuse
                all_session_results = []
                all_fact_results = []
                for sq in sub_queries:
                    sem_s = semantic_search_sessions(sq, top_k)
                    lex_s = lexical_search_sessions(sq, top_k)
                    all_session_results.extend(sem_s)
                    all_fact_results.extend(lexical_search_facts(sq, top_k, pds_filter))
                # Fuse all results
                sessions = reciprocal_rank_fusion([all_session_results], rrf_k)[:top_k]
                facts = reciprocal_rank_fusion([all_fact_results], rrf_k)[:top_k]
                memories = semantic_search_memories(query, top_k)
                return _assemble_hybrid_result(sessions, facts, memories, plan)

    # Standard hybrid search
    sem_sessions = semantic_search_sessions(query, top_k)
    lex_sessions = lexical_search_sessions(query, top_k)

    # Fuse session results
    if fusion_mode == "rrf":
        session_lists = [sem_sessions, lex_sessions]
        sessions = reciprocal_rank_fusion(session_lists, rrf_k)[:top_k]
    elif fusion_mode == "weighted":
        session_lists = [sem_sessions, lex_sessions]
        sessions = weighted_fusion(session_lists, [sem_weight, lex_weight])[:top_k]
    else:  # sum
        session_lists = [sem_sessions, lex_sessions]
        sessions = weighted_fusion(session_lists, [1.0, 1.0])[:top_k]

    # Facts
    facts = []
    if fact_search_enabled:
        lex_facts = [f for f in lexical_search_facts(query, top_k * 2, pds_filter)
                     if not str(f.get("predicate", "")).startswith("said_in")]
        lex_facts = lex_facts[:top_k]
        sem_facts = semantic_search_facts(query, top_k * 2)  # dense lane: bridges vocabulary gaps
        for _f in sem_facts:
            _f["via"] = f"{_f.get('via','x')}+dense"
        if fusion_mode == "rrf":
            fact_lists = [l for l in [lex_facts, sem_facts] if l]
            if fact_lists:
                facts = reciprocal_rank_fusion(fact_lists, rrf_k)[:top_k]
        elif fusion_mode == "weighted":
            fact_lists = [l for l in [lex_facts, sem_facts] if l]
            if fact_lists:
                facts = weighted_fusion(fact_lists, [lex_weight, sem_weight])[:top_k]
        else:
            facts = lex_facts[:top_k]

    # Memories (semantic only)
    memories = semantic_search_memories(query, top_k)

    # Touch accessed facts
    if facts:
        conn = get_db()
        for f in facts:
            if "id" in f:
                touch_fact(conn, f["id"])
        conn.commit()
        conn.close()

    return _assemble_hybrid_result(sessions, facts, memories, plan)


def _assemble_hybrid_result(sessions, facts, memories, plan=None):
    return {
        "sessions": sessions,
        "facts": facts,
        "memories": memories,
        "plan": plan,
        "summary": f"Hybrid search: {len(sessions)} sessions, {len(facts)} facts, {len(memories)} memories"
    }


# ─── Intent-Aware Retrieval Planning ─────────────────────────────────────────

def plan_retrieval(query):
    """
    Use LLM to generate a retrieval plan before searching.
    Infers intent, views, depth, and decomposition.
    """
    conn = get_db()
    plan_enabled = get_config(conn, "plan_enabled", 1)
    conn.close()

    if not plan_enabled:
        return None

    prompt = f"""You are a memory retrieval planner. Given a user query, generate a structured retrieval plan.

Analyze the query and determine:
1. "intent": What kind of search is needed?
   - "simple": Direct fact lookup (e.g., "What's Alex's birthday?")
   - "complex": Multi-faceted question requiring synthesis (e.g., "What were we working on last week?")
   - "aggregation": Requires combining multiple facts (e.g., "How many projects are active?")

2. "views": Which retrieval views to use? (array of: "semantic", "lexical", "structured")
   - "semantic": For conceptual/thematic queries
   - "lexical": For exact keyword matches
   - "structured": For PDS-domain-filtered queries

3. "query_decomposition": If the query is multi-hop, split into sub-queries (array of strings).
   Only decompose if genuinely multi-hop. Otherwise return [].

4. "top_k": How many results to retrieve per view? (integer, 5-20)

5. "context_budget": Max tokens of context to return? (integer, 1000-8000)

6. "pds_filter": If the query clearly relates to a specific PDS domain, provide the 4-digit code. Otherwise null.

Return ONLY a JSON object with these fields. No explanation.

Query: "{query}"
"""

    client = get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=3000,
        )
        raw = resp.choices[0].message.content.strip()
        plan = _safe_json_parse(raw, is_array=False)

        # Validate and fill defaults
        plan.setdefault("intent", "simple")
        plan.setdefault("views", ["semantic", "lexical"])
        plan.setdefault("query_decomposition", [])
        plan.setdefault("top_k", 10)
        plan.setdefault("context_budget", 4000)
        plan.setdefault("pds_filter", None)

        return plan
    except Exception as e:
        print(f"⚠ Retrieval planning failed: {e}")
        return None


# ─── Online Semantic Synthesis ────────────────────────────────────────────────

def synthesize_facts(entity_name=None):
    """
    Find related facts (same subject, complementary predicates) and merge them
    into higher-level composite facts using LLM synthesis.

    This keeps the fact store compact and reduces redundancy.
    """
    conn = get_db()
    synth_enabled = get_config(conn, "synthesis_enabled", 1)
    if not synth_enabled:
        print("✗ Synthesis is disabled in config")
        conn.close()
        return 0

    # Find candidate groups: same subject_entity_id, multiple active non-composite facts
    if entity_name:
        entity = conn.execute("SELECT id FROM entities WHERE name = ?", (entity_name,)).fetchone()
        if not entity:
            print(f"✗ Entity '{entity_name}' not found")
            conn.close()
            return 0
        candidates = conn.execute(
            """SELECT subject_entity_id, COUNT(*) as cnt FROM facts
               WHERE subject_entity_id = ? AND valid_until IS NULL AND is_composite = 0
               GROUP BY subject_entity_id HAVING cnt >= 2""",
            (entity["id"],)
        ).fetchall()
    else:
        candidates = conn.execute(
            """SELECT subject_entity_id, COUNT(*) as cnt FROM facts
               WHERE valid_until IS NULL AND is_composite = 0
               GROUP BY subject_entity_id HAVING cnt >= 2"""
        ).fetchall()

    if not candidates:
        print("✓ No synthesis candidates found")
        conn.close()
        return 0

    print(f"→ Found {len(candidates)} entity groups with composable facts")
    client = get_llm_client()
    total_synthesized = 0

    for cand in candidates:
        entity_id = cand["subject_entity_id"]
        entity_row = conn.execute("SELECT name FROM entities WHERE id = ?", (entity_id,)).fetchone()
        entity_name = entity_row["name"]

        # Get active non-composite facts for this entity
        facts = conn.execute(
            """SELECT f.id, f.predicate, f.object_value, f.pds_decimal, f.pds_domain,
               eo.name as object_name, f.confidence, f.source_session_id
               FROM facts f
               LEFT JOIN entities eo ON f.object_entity_id = eo.id
               WHERE f.subject_entity_id = ? AND f.valid_until IS NULL AND f.is_composite = 0
               ORDER BY f.pds_decimal, f.predicate""",
            (entity_id,)
        ).fetchall()

        if len(facts) < 2:
            continue

        # Group by PDS subdomain for more targeted synthesis
        pds_groups = {}
        for f in facts:
            pds = f["pds_decimal"] or "0000"
            pds_groups.setdefault(pds, []).append(f)

        for pds, group_facts in pds_groups.items():
            if len(group_facts) < 2:
                continue

            # Check if we already have a composite for this exact set
            fact_ids = [str(f["id"]) for f in group_facts]
            existing_composite = conn.execute(
                """SELECT id FROM synthesis_log WHERE source_fact_ids = ?""",
                (json.dumps(fact_ids),)
            ).fetchone()
            if existing_composite:
                continue

            # Build synthesis prompt
            fact_lines = []
            for f in group_facts:
                obj = f["object_name"] or f["object_value"] or ""
                fact_lines.append(f"- {entity_name} {f['predicate']} {obj}")

            prompt = f"""You are a memory synthesis system. Merge these related facts into a single composite fact.

Input facts:
{chr(10).join(fact_lines)}

Create a single composite fact that captures all the information. Return a JSON object with:
- "predicate": A higher-level predicate that encompasses the individual facts
- "object_value": A concise value that combines the individual objects
- "pds_decimal": The PDS subdomain code (use the most specific one from the input facts)
- "pds_domain": The PDS domain code (use the corresponding domain)
- "confidence": Confidence score (0.0-1.0, typically the minimum of input confidences)

Return ONLY the JSON object. No explanation."""

            try:
                resp = client.chat.completions.create(
                    model=EXTRACTION_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1, max_tokens=3000,
                )
                raw = resp.choices[0].message.content.strip()
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    raw = json_match.group(0)
                composite = json.loads(raw)
            except Exception as e:
                print(f"  ✗ Synthesis failed for {entity_name} ({pds}): {e}")
                continue

            # Store the composite fact
            conn.execute(
                """INSERT INTO facts (subject_entity_id, predicate, object_value,
                   pds_decimal, pds_domain, confidence, source_session_id, activation,
                   is_composite, component_fact_ids)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, 1, ?)""",
                (entity_id, composite.get("predicate", "composite"),
                 composite.get("object_value", ""),
                 composite.get("pds_decimal", pds),
                 composite.get("pds_domain", pds[:1] + "000" if pds else "1000"),
                 composite.get("confidence", 0.8),
                 group_facts[0]["source_session_id"],
                 json.dumps(fact_ids))
            )
            composite_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Log the synthesis
            conn.execute(
                "INSERT INTO synthesis_log (composite_fact_id, source_fact_ids, synthesis_prompt) VALUES (?, ?, ?)",
                (composite_id, json.dumps(fact_ids), prompt[:500])
            )

            # Index the composite fact in FTS5
            index_fact_fts(conn, composite_id, entity_name,
                          composite.get("predicate", "composite"),
                          None, composite.get("object_value", ""),
                          composite.get("pds_domain", ""),
                          composite.get("pds_decimal", pds))

            total_synthesized += 1
            print(f"  ✓ {entity_name} ({pds}): {composite.get('predicate', '')} → {composite.get('object_value', '')[:80]}")

    conn.commit()
    conn.close()
    print(f"✓ Synthesized {total_synthesized} composite facts")
    return total_synthesized


# ─── EvolveMem Self-Tuning Loop ──────────────────────────────────────────────

GOLDEN_EVAL_QUERIES = [
    "What projects is Alex currently working on?",
    "What is Alex's preferred working style and how should Leo communicate?",
    "What infrastructure runs in Alex's home lab?",
    "Who are Alex's family members?",
    "What is Alex's role at Acme Staffing and what does it involve?",
]

# Sanity bounds for LLM-proposed retrieval config changes (v3.1 hardening).
CONFIG_BOUNDS = {
    "fusion_mode":     {"type": "enum",  "choices": ["rrf", "weighted", "sum"]},
    "semantic_weight": {"type": "float", "min": 0.0,  "max": 1.0},
    "lexical_weight":  {"type": "float", "min": 0.0,  "max": 1.0},
    "top_k":           {"type": "int",   "min": 5,    "max": 20},
    "context_budget":  {"type": "int",   "min": 1000, "max": 8000},
}


def _build_eval_set(conn):
    """Build the EvolveMem eval set: golden queries + recent real user queries."""
    eval_queries = list(GOLDEN_EVAL_QUERIES)
    try:
        real_rows = conn.execute(
            """SELECT DISTINCT query FROM query_logs
               WHERE kind = 'real' ORDER BY created_at DESC LIMIT 5"""
        ).fetchall()
        eval_queries += [r["query"] for r in real_rows]
    except sqlite3.OperationalError:
        pass  # pre-migration DB without kind column
    return eval_queries or ["What projects is Alex working on?"]


def _sanitise_config_changes(changes, current_config):
    """
    Filter LLM-proposed config changes before they are applied:
    unknown keys dropped, out-of-bounds values clamped, no-op changes dropped.
    Returns (clean_changes dict, rejected list of human-readable reasons).
    """
    clean, rejected = {}, []
    for key, new_val in (changes or {}).items():
        bounds = CONFIG_BOUNDS.get(key)
        if bounds is None:
            rejected.append(f"{key}: unknown parameter")
            continue
        try:
            if bounds["type"] == "enum":
                if str(new_val) not in bounds["choices"]:
                    rejected.append(f"{key}: '{new_val}' not in {bounds['choices']}")
                    continue
                new_val = str(new_val)
            elif bounds["type"] == "float":
                new_val = float(new_val)
                new_val = max(bounds["min"], min(bounds["max"], new_val))
            else:
                new_val = int(round(float(new_val)))
                new_val = max(bounds["min"], min(bounds["max"], new_val))
        except (TypeError, ValueError):
            rejected.append(f"{key}: non-numeric value '{new_val}'")
            continue
        cur = current_config.get(key)
        if cur is not None and str(cur) == str(new_val):
            rejected.append(f"{key}: no change ({new_val})")
            continue
        clean[key] = new_val
    return clean, rejected


def _llm_judge_relevance(query, sessions, facts, memories):
    """
    Judge retrieval quality for one query with the extraction LLM.
    Returns (score 0.0-1.0, via) where via is 'llm' or 'fallback'.
    Temperature 0 + coarse rubric keeps generation-to-generation noise well
    under the Guard's 10% regression threshold.
    """
    def _fmt(items, fields, limit=5, chars=160):
        lines = []
        for it in (items or [])[:limit]:
            if isinstance(it, dict):
                parts = [str(it.get(f, ""))[:chars] for f in fields if it.get(f)]
                lines.append(" | ".join(parts) if parts else json.dumps(it, default=str)[:chars])
            else:
                lines.append(str(it)[:chars])
        return "\n".join(lines) if lines else "(none)"

    prompt = f"""You are a retrieval quality judge. Given a query and the results returned by a
memory system, rate how well the result set answers the query.

Rating rubric:
- 1.0  results directly and centrally answer the query
- 0.75 results clearly relevant, they mostly answer it
- 0.5  results are on-topic but only partially relevant
- 0.25 results are mostly tangential, at least one useful fragment
- 0.0  results are unrelated junk, or nothing useful at all

Return ONLY JSON: {{"score": <one of 0.0, 0.25, 0.5, 0.75, 1.0>}}

Query: {query}

Sessions (top):
{_fmt(sessions, ['session_date', 'content'])}

Facts (top):
{_fmt(facts, ['subject', 'predicate', 'object'])}

Memories (top):
{_fmt(memories, ['type', 'content'])}
"""
    try:
        client = get_llm_client()
        resp = client.chat.completions.create(
            model=EXTRACTION_MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=2000,
        )
        parsed = _safe_json_parse(resp.choices[0].message.content, is_array=False)
        if isinstance(parsed, dict) and parsed.get("score") is not None:
            score = float(parsed["score"])
            if 0.0 <= score <= 1.0:
                return score, "llm"
    except Exception:
        pass
    # Fallback: legacy heuristic — ratio of non-empty result sets
    non_empty = sum(1 for r in [sessions, facts, memories] if r)
    return (non_empty / 3.0), "fallback"


def evaluate_retrieval(test_queries=None):
    """
    Run test queries and log results for EvolveMem evaluation.
    Eval set: golden queries + up to 5 recent real user queries
    (query_logs.kind='real'). Each result set is scored by an LLM relevance
    judge; probes are logged with kind='eval'.
    """
    conn = get_db()

    if test_queries is None:
        test_queries = _build_eval_set(conn)

    config_snapshot = {}
    for key in CONFIG_BOUNDS:
        row = conn.execute("SELECT value, type FROM retrieval_config WHERE key = ?", (key,)).fetchone()
        if row:
            config_snapshot[key] = row["value"]

    total_score = 0.0
    for query in test_queries:
        result = hybrid_search(query, top_k=get_config(conn, "top_k", 10))
        sessions = result.get("sessions", [])
        facts = result.get("facts", [])
        memories = result.get("memories", [])

        # LLM relevance judge (falls back to legacy heuristic on failure)
        quality, judged_by = _llm_judge_relevance(query, sessions, facts, memories)

        # Log the query
        conn.execute(
            """INSERT INTO query_logs (query, intent, views_used, results_count, results_quality, config_snapshot, kind)
               VALUES (?, ?, ?, ?, ?, ?, 'eval')""",
            (query, result.get("plan", {}).get("intent", "simple") if result.get("plan") else "simple",
             json.dumps(["semantic", "lexical"] + (["structured"] if any("pds" in f for f in facts) else [])),
             len(sessions) + len(facts) + len(memories),
             json.dumps({"quality_score": quality, "judged_by": judged_by,
                         "session_count": len(sessions),
                         "fact_count": len(facts), "memory_count": len(memories)}),
             json.dumps(config_snapshot))
        )
        total_score += quality

    avg_score = total_score / len(test_queries) if test_queries else 0
    conn.commit()
    conn.close()
    print(f"✓ Evaluated {len(test_queries)} queries, avg quality: {avg_score:.3f}")
    return avg_score


def diagnose_failures():
    """
    Use LLM to read recent failure logs and propose config changes.
    Considers scored failures from both real user queries and eval probes
    (kind IN ('real','eval')). Returns proposed changes dict — already
    sanitised via _sanitise_config_changes (unknown keys, out-of-bounds and
    no-op changes removed).
    """
    conn = get_db()
    try:
        recent_failures = conn.execute(
            """SELECT query, results_quality, config_snapshot FROM query_logs
               WHERE kind IN ('real', 'eval')
               AND CAST(JSON_EXTRACT(results_quality, '$.quality_score') AS REAL) < 0.5
               ORDER BY created_at DESC LIMIT 10"""
        ).fetchall()
    except sqlite3.OperationalError:
        # Pre-migration DB without kind column
        recent_failures = conn.execute(
            """SELECT query, results_quality, config_snapshot FROM query_logs
               WHERE JSON_EXTRACT(results_quality, '$.quality_score') < 0.5
               ORDER BY created_at DESC LIMIT 10"""
        ).fetchall()

    if not recent_failures:
        print("✓ No failures to diagnose")
        conn.close()
        return {}

    current_config = {}
    for key in CONFIG_BOUNDS:
        row = conn.execute("SELECT value FROM retrieval_config WHERE key = ?", (key,)).fetchone()
        if row:
            current_config[key] = row["value"]

    failure_descriptions = []
    for f in recent_failures:
        try:
            quality = json.loads(f["results_quality"]) if f["results_quality"] else {}
        except (TypeError, ValueError):
            quality = {}
        failure_descriptions.append(
            f"Query: '{f['query']}' → quality={quality.get('quality_score', 0)}, "
            f"sessions={quality.get('session_count', 0)}, facts={quality.get('fact_count', 0)}"
        )

    prompt = f"""You are a retrieval system tuner. Analyze query failures and propose config changes.

Current config:
{json.dumps(current_config, indent=2)}

Recent failures:
{chr(10).join(failure_descriptions)}

Propose changes to improve retrieval quality. Only change parameters that seem problematic.
Return a JSON object where keys are config names and values are the proposed new values.
Only include parameters you want to change. Return {{}} if no changes needed.

Available parameters:
- fusion_mode: "rrf", "weighted", or "sum"
- semantic_weight: 0.0-1.0 (weight for semantic scores)
- lexical_weight: 0.0-1.0 (weight for lexical scores)
- top_k: 5-20 (results per view)
- context_budget: 1000-8000 (max tokens in context bundle)

Return ONLY the JSON object."""

    client = get_llm_client()
    try:
        resp = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=3000,
        )
        raw = resp.choices[0].message.content.strip()
        proposed = _safe_json_parse(raw, is_array=False) or {}
    except Exception as e:
        print(f"✗ Diagnosis failed: {e}")
        conn.close()
        return {}

    conn.close()
    changes, rejected = _sanitise_config_changes(proposed, current_config)
    if rejected:
        print(f"✓ Sanitiser dropped {len(rejected)} proposal(s): {'; '.join(rejected)}")
    return changes


def run_evolution(generations=1):
    """
    Run the EvolveMem closed-loop: Evaluate → Diagnose → Propose → Guard.
    Uses one fixed eval set per generation so the Guard's before/after
    comparison measures the config change, not eval-set drift. Diagnose
    proposals arrive pre-sanitised; a second CONFIG_BOUNDS check runs at
    apply time (defense in depth).
    """
    conn = get_db()
    evolve_enabled = get_config(conn, "evolve_enabled", 0)
    conn.close()

    if not evolve_enabled:
        print("⚠ EvolveMem is disabled. Enable with: config --set evolve_enabled --value 1")
        print("  Running one evaluation cycle anyway for diagnostics...")

    for gen in range(generations):
        print(f"\n── EvolveMem Generation {gen + 1} ──")

        # Fixed eval set for this generation (before/after comparable)
        _c = get_db()
        gen_queries = _build_eval_set(_c)
        _c.close()

        # 1. Evaluate
        metric_before = evaluate_retrieval(test_queries=gen_queries)

        # 2. Diagnose
        changes = diagnose_failures()
        if not changes:
            print("✓ No changes proposed")
            continue

        print(f"→ Proposed changes: {json.dumps(changes)}")

        # 2.5 EXTERNAL GATE — LLM-judged evals can drift (same model scores
        # before and after). Guard rails: auto-apply only when ALL hold:
        #   a) small change: <= 2 config keys, each a bounded numeric tweak
        #   b) no consecutive self-applied generations: 2 in a row -> pending
        #   c) pending proposals exist -> new ones queue, nothing self-applies
        # Anything else lands in retrieval_config pending_*, for Alex to
        # approve via: python3 muninn.py config --set <key> --value <v>
        import json as _json
        pending_count = 0
        _gc = get_db()
        row = _gc.execute("SELECT COUNT(*) n FROM evolution_history "
                          "WHERE applied=0 AND reverted=0").fetchone()
        pending_count = row["n"]
        last_two = _gc.execute("SELECT applied FROM evolution_history "
                               "ORDER BY generation DESC LIMIT 2").fetchall()
        _gc.close()
        consecutive_self = (len(last_two) >= 2 and
                            all(r["applied"] == 1 for r in last_two))
        small = (len(changes) <= 2 and
                 all(isinstance(v, (int, float)) for v in changes.values()))
        gated = pending_count > 0 or consecutive_self or not small
        if gated:
            conn = get_db()
            last_gen = conn.execute("SELECT MAX(generation) g FROM evolution_history").fetchone()["g"]
            next_gen = (last_gen or 0) + 1
            conn.execute(
                """INSERT INTO evolution_history (generation, diagnosis,
                   proposed_changes, applied, metric_before)
                   VALUES (?, ?, ?, 0, ?)""",
                (next_gen, "external-gate: pending Alex's approval",
                 _json.dumps(changes), metric_before))
            conn.commit()
            conn.close()
            print("⏸ GATED: proposal queued for approval (not self-applied)")
            print(f"  reason: {'consecutive self-applies' if consecutive_self else 'pending queue' if pending_count else 'non-small change'}")
            print(f"  approve with: python3 muninn.py config --set <key> --value <v>")
            continue

        # 3. Apply (Propose) — defense-in-depth bounds check before writing
        conn = get_db()
        old_values = {}
        applied_changes = {}
        for key, new_val in changes.items():
            if key not in CONFIG_BOUNDS:
                print(f"⚠ Skipping unknown config key '{key}'")
                continue
            old_row = conn.execute("SELECT value FROM retrieval_config WHERE key = ?", (key,)).fetchone()
            old_values[key] = old_row["value"] if old_row else None
            set_config(conn, key, new_val, updated_by="evolve")
            applied_changes[key] = new_val

        # Get next generation number
        last_gen = conn.execute("SELECT MAX(generation) as g FROM evolution_history").fetchone()["g"]
        next_gen = (last_gen or 0) + 1

        conn.execute(
            """INSERT INTO evolution_history (generation, diagnosis, proposed_changes, applied, metric_before)
               VALUES (?, ?, ?, 1, ?)""",
            (next_gen, json.dumps(applied_changes), json.dumps({"old": old_values, "new": applied_changes}), metric_before)
        )
        evo_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()

        # 4. Guard — re-evaluate on the SAME query set and check for regression
        metric_after = evaluate_retrieval(test_queries=gen_queries)

        conn = get_db()
        conn.execute("UPDATE evolution_history SET metric_after = ? WHERE id = ?", (metric_after, evo_id))

        # Revert if regression > 10%
        # autoresearch doctrine (karpathy/autoresearch): every generation is
        # a logged experiment — keep or discard is decided ONLY by the
        # measured metric, and the discard reason is written down.
        if metric_after < metric_before * 0.9:
            print(f"⚠ Regression detected: {metric_before:.3f} → {metric_after:.3f}")
            print(f"  DISCARD (regression {(1-metric_after/max(metric_before,1e-9))*100:.1f}% > 10% threshold). Reverting...")
            for key, old_val in old_values.items():
                if old_val is not None:
                    set_config(conn, key, old_val, updated_by="evolve-revert")
            conn.execute("UPDATE evolution_history SET reverted = 1 WHERE id = ?", (evo_id,))
            print("  ✓ Reverted")
        else:
            print(f"✓ Improvement: {metric_before:.3f} → {metric_after:.3f}")

        conn.commit()
        conn.close()


# ─── Fact Extraction (with contradiction detection) ─────────────────────────

def extract_facts_from_session(session_id):
    conn = get_db()
    session = conn.execute("SELECT * FROM raw_sessions WHERE id = ?", (session_id,)).fetchone()
    if not session:
        print(f"✗ Session {session_id} not found")
        conn.close()
        return []

    content = session["content"]
    source = session["source"] or "openclaw"

    # ── Session quality gate: skip trivial greeting-only sessions ──
    # All source types are processed — the LLM extraction prompt handles
    # noise filtering, and the Librarian cron cleans up low-value facts.
    GREETING_PATTERNS = [
        "hello?", "hey phillip", "what's up", "morning leo", "wake up leo",
        "you there", "leo?", "ping", "test", "hi leo", "g'day leo"
    ]

    is_greeting_only = (
        len(content) < 800 and
        any(p in content.lower() for p in GREETING_PATTERNS) and
        not any(keyword in content.lower() for keyword in
                ["project", "build", "fix", "update", "create", "design",
                 "deploy", "muninn", "memory", "config", "server", "database"])
    )

    if is_greeting_only:
        # Mark as extracted (skip) but don't produce facts
        conn.execute("UPDATE raw_sessions SET extracted_at = datetime('now') WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
        print(f"⏭ Session {session_id}: skipped (greeting-only, {len(content)} chars)")
        return []

    prompt = """You are a fact extraction system. Extract structured facts from the following conversation.

Return a JSON array of facts. Each fact should have:
- "subject": entity name (who the fact is about)
- "predicate": relationship or attribute (e.g. "works_at", "prefers", "attended")
- "object": entity or value (what the subject relates to)
- "pds_domain": one of "1000","2000","3000","4000","5000"
- "pds_subdomain": 4-digit code (e.g. "2101", "3200") — must match a real subdomain
- "valid_from": date if known, else null
- "valid_until": date if applicable, else null
- "confidence": 0.0-1.0

Valid PDS subdomains:
1100 (Identity), 1200 (Health), 1300 (Mood), 1400 (Preferences)
2100 (Immediate Kin), 2200 (Extended Family), 2300 (Social), 2400 (Professional)
3100 (Projects), 3200 (Career), 3300 (Infrastructure), 3400 (Finance)
4100 (Fixed Schedule), 4200 (Specific Events), 4300 (Origins)
5100 (Beliefs), 5200 (Mental Models), 5300 (Learning)

IMPORTANT — NOISE FILTERING RULES:
- Do NOT extract facts about system operations: cron jobs, scheduled tasks, heartbeat checks, auto-ingestion runs, sleep cycles, or pipeline executions
- Do NOT extract facts about tool usage: "Alex triggered_cron X", "Alex uses_tool Y", "Alex ran_command Z"
- Do NOT extract facts about session metadata: session IDs, token counts, model names, or API calls
- Do NOT extract facts about infrastructure operations: database rebuilds, index creation, config changes, or deployment steps
- ONLY extract facts that a human would find useful to remember about another person, project, or topic
- When in doubt, ask: "Would someone care about this a month from now?" If no, skip it.

ONLY extract clear, stated facts. Do not infer or fabricate.

ACTIVITY COVERAGE RAIL (do not skip):
- When a speaker states a present, past, or planned activity in first person ("I'm off to go swimming", "we went camping last weekend", "I'm taking up pottery"), emit a fact for it: subject = the speaker, predicate = "engages_in" (present/recurring) or "participated_in" / "plans_to" (past/planned), object = the activity, valid_from = the resolved date. These count as clear stated facts — the noise rules above do NOT exclude them.
- Brief utterances count. A single sentence stating an activity is enough; do not require elaboration.

Conversation:
---
{CONTENT_PLACEHOLDER}
---
"""  # plain string (NOT f-string): {CONTENT_PLACEHOLDER} must survive for the retry ladder

    client = get_llm_client()

    # Session-date anchor for relative-date resolution (ledger v13: orphan "last Saturday"
    # refs made cat-2 dates unresolvable). The extractor must convert relative anchors
    # into ISO dates using this timestamp, never emit bare "yesterday"/"last week".
    sess_dt = session["created_at"] or session["updated_at"] if "created_at" in session.keys() else None
    date_anchor = ""
    if sess_dt:
        date_anchor = f"\nSession Timestamp: {str(sess_dt)[:10]}\nWhen a speaker uses a relative date (e.g. 'yesterday', 'last Saturday'), resolve it to the absolute ISO date using this timestamp and put the ISO date in valid_from. Append the original relative wording in parentheses.\n"
    prompt += date_anchor

    # Reasoning-model extraction guardrail (ledger v13/v14): glm reasoning models can burn
    # the entire completion budget on hidden reasoning (finish_reason=length, empty content).
    # Assistant prefill ("```json\n") breaks the spiral; empty results retried with trimmed
    # context — silent zero-fact extraction is banned.
    facts = []
    for attempt, (mtk, cut) in enumerate([(9000, 6000), (12000, 4000), (12000, 2500)]):
        try:
            resp = client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[{"role": "user", "content": prompt.replace("{CONTENT_PLACEHOLDER}", content[:cut])},
                          {"role": "assistant", "content": "```json\n"}],
                temperature=0.1, max_tokens=mtk,
            )
            raw_output = (resp.choices[0].message.content or "").strip()
            if not raw_output:
                print(f"⚠ Extract burn (attempt {attempt+1}) for {session_id}: finish={resp.choices[0].finish_reason}")
                continue
            facts = _safe_json_parse(raw_output, is_array=True) or []
            if facts:
                break
        except Exception as e:
            print(f"✗ Extraction attempt {attempt+1} failed for {session_id}: {e}")
    if not facts:
        print(f"✗ Extraction produced 0 facts after prefill ladder for {session_id}")
        conn.close()
        return []

    stored = []
    for fact in facts:
        subj_name = fact.get("subject", "unknown")
        subj_id = get_or_create_entity(conn, subj_name)

        obj_entity_id = None
        obj_value = None
        obj = fact.get("object")
        if obj:
            if len(str(obj)) > 2 and str(obj)[0].isupper():
                obj_entity_id = get_or_create_entity(conn, str(obj))
            else:
                obj_value = str(obj)

        pds = fact.get("pds_subdomain", fact.get("pds_domain", "1000"))
        domain = fact.get("pds_domain", pds[:1] + "000" if pds else "1000")

        # ── Contradiction detection ──
        SINGULAR_PREDICATES = {
            "works_at", "lives_in", "lives_at", "job_title", "partner",
            "spouse", "employer", "role", "title", "manages",
            "located_in", "based_in", "reports_to", "birthplace",
            "nationality", "occupation", "education", "degree"
        }
        predicate_str = fact.get("predicate", "related_to")
        is_singular = predicate_str.lower() in SINGULAR_PREDICATES

        existing = conn.execute(
            """SELECT id, object_value, object_entity_id FROM facts
               WHERE subject_entity_id = ? AND predicate = ? AND valid_until IS NULL""",
            (subj_id, predicate_str)
        ).fetchall()

        new_fact_id = None
        contradicted = False

        for ex in existing:
            ex_obj = ex["object_value"] or (conn.execute("SELECT name FROM entities WHERE id = ?", (ex["object_entity_id"],)).fetchone() or [None])[0] or ""
            new_obj = obj_value or (conn.execute("SELECT name FROM entities WHERE id = ?", (obj_entity_id,)).fetchone() or [None])[0] or ""

            if is_singular and ex_obj.lower() != new_obj.lower():
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "UPDATE facts SET valid_until = ?, superseded_by = NULL WHERE id = ?",
                    (now, ex["id"])
                )
                conn.execute(
                    "INSERT INTO facts (subject_entity_id, predicate, object_entity_id, object_value, "
                    "pds_decimal, pds_domain, valid_from, valid_until, confidence, source_session_id, activation) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0)",
                    (subj_id, predicate_str, obj_entity_id, obj_value,
                     pds, domain, fact.get("valid_from"), fact.get("valid_until"),
                     fact.get("confidence", 0.9), session_id)
                )
                new_fact_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute("UPDATE facts SET superseded_by = ? WHERE id = ?", (new_fact_id, ex["id"]))
                contradicted = True
                print(f"  ↳ Contradiction: {subj_name} {predicate_str} {ex_obj} → {new_obj} (superseded fact #{ex['id']})")
                break

        if not contradicted:
            is_dup = any(
                (obj_value and ex["object_value"] and obj_value.lower() == ex["object_value"].lower())
                or (obj_entity_id and ex["object_entity_id"] and obj_entity_id == ex["object_entity_id"])
                for ex in existing
            )
            if not is_dup:
                conn.execute(
                    "INSERT INTO facts (subject_entity_id, predicate, object_entity_id, object_value, "
                    "pds_decimal, pds_domain, valid_from, valid_until, confidence, source_session_id, activation) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0)",
                    (subj_id, predicate_str, obj_entity_id, obj_value,
                     pds, domain, fact.get("valid_from"), fact.get("valid_until"),
                     fact.get("confidence", 0.9), session_id)
                )
                new_fact_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        if new_fact_id:
            # Index in FTS5
            # Resolve entity-object name; bare None left entity objects (e.g.
            # "Charlotte's Web") out of the search index entirely.
            obj_name_for_idx = obj_value
            if obj_entity_id:
                obj_name_for_idx = (conn.execute("SELECT name FROM entities WHERE id=?",
                                                 (obj_entity_id,)).fetchone() or [None])[0] or obj_value
            index_fact_fts(conn, new_fact_id, subj_name, predicate_str,
                           obj_name_for_idx, obj_value, domain, pds)
            stored.append(fact)

    conn.execute("UPDATE raw_sessions SET extracted_at = datetime('now') WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    print(f"✓ Extracted {len(stored)} facts from {session_id}")
    return stored


def learn_fact(subject, predicate, obj, pds=3000, confidence=0.8,
               source_session_id=None):
    """Public single write path for external clients (e.g. Huginn).
    Handles entity resolution, same-subject+predicate supersession, and FTS
    indexing in one call — clients never hand-roll SQL. Returns dict."""
    import sqlite3
    conn = get_db()
    conn.row_factory = sqlite3.Row
    try:
        subj_id = get_or_create_entity(conn, str(subject))
        obj_entity_id = None
        obj_value = None
        # literal values (with spaces/type hints) vs entity names: entity if
        # a matching entity exists, else store as value
        row = conn.execute("SELECT id FROM entities WHERE name = ?",
                           (str(obj),)).fetchone()
        if row and " " not in str(obj).strip():
            obj_entity_id = row["id"]
        else:
            obj_value = str(obj)
        domain = int(pds)
        now = datetime.now(timezone.utc).isoformat()
        old = conn.execute(
            "SELECT id FROM facts WHERE subject_entity_id=? AND predicate=? "
            "AND superseded_by IS NULL AND valid_until IS NULL",
            (subj_id, predicate)).fetchall()
        cur = conn.execute(
            "INSERT INTO facts (subject_entity_id, predicate, object_entity_id, "
            "object_value, pds_decimal, pds_domain, valid_from, confidence, "
            "source_session_id, activation) VALUES (?,?,?,?,?,?,?,?,?,1.0)",
            (subj_id, predicate, obj_entity_id, obj_value, domain, domain,
             now, float(confidence), None))
        new_id = cur.lastrowid
        if old and not predicate.startswith("said_in"):
            conn.executemany("UPDATE facts SET superseded_by=? WHERE id=?",
                             [(new_id, o["id"]) for o in old])
        subj_name = conn.execute("SELECT name FROM entities WHERE id=?",
                                 (subj_id,)).fetchone()["name"]
        obj_name = None
        if obj_entity_id:
            obj_name = conn.execute("SELECT name FROM entities WHERE id=?",
                                    (obj_entity_id,)).fetchone()["name"]
        index_fact_fts(conn, new_id, subj_name, predicate, obj_name,
                       obj_value, domain, domain)
        conn.commit()
        return {"ok": True, "fact_id": new_id, "superseded": len(old)}
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()


def get_or_create_entity(conn, name, entity_type=None):
    existing = conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
    if existing:
        return existing["id"]
    conn.execute("INSERT INTO entities (name, type) VALUES (?, ?)", (name, entity_type))
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def extract_all_unprocessed():
    conn = get_db()
    sessions = conn.execute("SELECT id FROM raw_sessions WHERE extracted_at IS NULL").fetchall()
    conn.close()
    if not sessions:
        print("✓ No unprocessed sessions")
        return
    print(f"→ Processing {len(sessions)} sessions...")
    for s in sessions:
        extract_facts_from_session(s["id"])

    # v3: Run synthesis after extraction
    if sessions:
        print("→ Running online semantic synthesis...")
        synthesize_facts()


# ─── Activation / Decay ──────────────────────────────────────────────────────

def apply_decay():
    """
    Decay activation scores for all active facts.
    activation *= 0.5 ^ (days_since_last_access / half_life)
    """
    conn = get_db()
    now = datetime.now(timezone.utc)
    facts = conn.execute(
        "SELECT id, activation, last_accessed_at, created_at FROM facts WHERE valid_until IS NULL"
    ).fetchall()

    decayed = 0
    for f in facts:
        ref_date = f["last_accessed_at"] or f["created_at"]
        if ref_date:
            try:
                ref = datetime.fromisoformat(ref_date.replace("Z", "+00:00"))
                days_since = (now - ref).total_seconds() / 86400
                decay_factor = 0.5 ** (days_since / DECAY_HALF_LIFE_DAYS)
                new_activation = max(0.0, f["activation"] * decay_factor)
                conn.execute("UPDATE facts SET activation = ? WHERE id = ?", (new_activation, f["id"]))
                decayed += 1
            except Exception:
                pass

    conn.commit()
    conn.close()
    print(f"✓ Decayed {decayed} facts")


def touch_fact(conn, fact_id):
    """Boost activation when a fact is accessed."""
    conn.execute(
        "UPDATE facts SET activation = MIN(activation + 0.1, 1.0), access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), fact_id)
    )


def get_facts_for_entity(name, pds_filter=None):
    conn = get_db()
    entity = conn.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()
    if not entity:
        print(f"✗ Entity '{name}' not found")
        conn.close()
        return []

    query = """
        SELECT f.*, e.name as subject_name, eo.name as object_name
        FROM facts f
        JOIN entities e ON f.subject_entity_id = e.id
        LEFT JOIN entities eo ON f.object_entity_id = eo.id
        WHERE f.subject_entity_id = ? AND f.valid_until IS NULL
    """
    params = [entity["id"]]
    if pds_filter:
        query += " AND f.pds_domain = ?"
        params.append(str(pds_filter))
    query += " ORDER BY f.activation DESC, f.created_at DESC"

    results = conn.execute(query, params).fetchall()

    for r in results:
        touch_fact(conn, r["id"])

    conn.commit()
    conn.close()
    return [dict(r) for r in results]


# ─── Recall (combined search for pre-answer context) ─────────────────────────

_FACT_SYNONYMS = {
    "kids": "child", "kid": "child", "children": "child",
    "employer": "works", "company": "works", "family": "child",
    "tool": "uses_tool", "tools": "uses_tool",
    "dog": "pet", "dogs": "pet", "cat": "pet", "cats": "pet",
    "pet": "pet", "pets": "pet",
    "birthday": "date_of_birth", "born": "born", "birthdate": "date_of_birth",
    "age": "date_of_birth",
    "bjj": "brazilian jiu-jitsu", "jiu-jitsu": "brazilian jiu-jitsu",
    "jitsu": "brazilian jiu-jitsu", "martial": "brazilian jiu-jitsu",
    "gym": "gym", "kelpie": "dog", "kelpies": "dog",
    "partner": "partner", "philosophy": "philosophy",
    "memory": "memory", "homelab": "homelab",
}

_RECALL_STOPWORDS = frozenset(
    "what who where when how why which are was were is the a an my your me "
    "you i do does did have has and or for from with about that this these "
    "those tell show list give any all out can could would should get got".split())

# v2.0 Phase C: deterministic query-keyword -> compiled slot-class map.
# Used ONLY when the entity_state_read_enabled flag is on; a MISS falls
# straight through to the unchanged 3-lane retrieval below.
_STATE_SLOT_KEYWORDS = {
    "children": ("child", "children", "kid", "kids", "son", "daughter",
                  "step-daughter", "stepdaughter", "bonus", "family"),
    "employers": ("employer", "employers", "company", "works", "work", "worked",
                  "works at", "work at", "employed", "contract"),
    "residences": ("live", "lives", "living", "residence", "home", "where",
                   "currently live", "located", "location"),
    "job_titles": ("role", "job", "title", "position", "works as", "focus"),
    "preferences": ("prefer", "prefers", "preference", "preferences", "likes", "favourite", "favorite", "enjoy",
                    "exercise", "hobbies", "hobby", "interests"),
    "contact": ("contact", "phone", "email", "reach"),
    "memberships": ("member", "membership"),
    "owned_items": ("owns", "owned", "own"),
    "pets": ("pet", "pets", "dog", "dogs", "cat", "cats", "puppy", "kitty", "kitten"),
    "schooling": ("school", "study", "studies", "studied", "qualification",
                  "qualifications", "education", "diploma", "cert", "certs"),
    "gym": ("gym", "fitness", "membership", "trainer", "training"),
    "vehicles": ("car", "vehicle", "drive", "drives", "daily driver", "truck"),
}


def _compiled_state_span(conn, query, probes):
    """Phase C canary: deterministic (entity, slot) point lookup against
    entity_state. Returns a synthetic pinned fact dict rendered from the
    compiled state, or None on MISS/no-flag (caller falls through to the
    unchanged 3-lane path). Read-only; provenance-tagged."""
    try:
        if not int(get_config(conn, "entity_state_read_enabled", 0)):
            return None
    except Exception:
        return None
    q = query.lower().replace("'s", "")
    q_terms = {t.strip("?.,!") for t in q.split()}
    # First-person queries ("Where do I live?", "my employer") refer to the
    # principal user; map them to Alex so the compiled front door fires.
    # Word-token match only — never substring ("current" contains no "i "
    # but " i " naively substring-matches inside words).
    q_words = set(q.strip("?.,!").split())
    if {"i", "my", "me", "mine", "myself"} & q_words:
        ent = "Alex"
    else:
        # pick the probe entity whose name literally appears in the query
        ent = next((n for n in sorted(probes, key=len, reverse=True)
                    if n.lower() in q), None)
        if ent is None:
            # fallback: exact multi-word entity names appearing verbatim in the
            # query (single-token capitalize probes miss "Sam Clark"). Longest
            # name wins; only when flag is already on, so zero cost when off.
            ent = next((r["name"] for r in sorted(
                           conn.execute("SELECT name FROM entities").fetchall(),
                           key=lambda r: len(r["name"]), reverse=True)
                        if len(r["name"]) > 3 and r["name"].lower() in q), None)
    if ent is None:
        return None
    def _kw_hit(k):
        # word-token match with light stemming: word == k, or word starts
        # with k and the remainder is <= 3 chars ("prefer"->"preferences"),
        # but never bare substring ("cat" must not hit "qualifications")
        if " " in k:
            return k in q
        return any(w == k or (w.startswith(k) and len(w) - len(k) <= 3)
                   for w in q_words)
    slot = next((s for s, kws in _STATE_SLOT_KEYWORDS.items()
                 if any(_kw_hit(k) for k in kws)), None)
    if slot is None:
        return None
    row = conn.execute(
        """SELECT s.attribute, s.state_json, s.evidence_json, s.compiled_at,
                  s.source_span_count
           FROM entity_state s JOIN entities e ON s.entity_id = e.id
           WHERE e.name = ? AND s.attribute = ?""", (ent, slot)).fetchone()
    if not row:
        return None  # MISS: fall straight through
    try:
        state = json.loads(row["state_json"])
    except Exception:
        return None
    # render human-readable span text from the compiled state
    if "current" in state and state.get("current") not in (None, ""):
        text = f"{ent} — {slot.replace('_', ' ')}: {state['current']}"
        if state.get("superseded"):
            sup = [s for s in state["superseded"]
                   if s and not str(s).startswith("[unresolved")]
            if sup:
                text += f" (superseded: {'; '.join(sup[:3])})"
    else:
        items = state.get("items", [])
        if not items:
            return None
        text = f"{ent} — {slot.replace('_', ' ')} ({state.get('distinct_count', len(items))}): " + \
               "; ".join(str(i) for i in items[:8])
    sources = []
    try:
        ev = json.loads(row["evidence_json"] or "[]")
        sources = [e.get("fact_id") for e in ev if e.get("fact_id") is not None]
    except Exception:
        pass
    return {
        "id": None,
        "subject": ent,
        "predicate": f"compiled_state:{slot}",
        "object": text,
        "pds": 3000,
        "activation": 2.0,
        "compiled_state": True,
        "origin": "entity_state",
        "compiled_at": row["compiled_at"],
        "source_span_count": row["source_span_count"],
        "fact_ids": sources,
    }


def recall_facts(query, top_k=5):
    """Engine-level structured fact recall for turn injection.
    Lexical FTS + entity probes + synonym expansion + predicate-exact boost,
    session-echo chatter (said_in_*) filtered. No LLM needed."""
    import re as _re
    if not query or not query.strip():
        return []
    conn = get_db()
    terms = [t for t in _re.findall(r"[a-z0-9]{3,}", query.lower())
             if t not in _RECALL_STOPWORDS][:8]
    expanded = list({t for t in terms} |
                    {_FACT_SYNONYMS[t] for t in terms if t in _FACT_SYNONYMS})
    # Tier 1: lexical FTS hits, predicate-exact boosted
    scored = {}
    for t in expanded:
        for r in lexical_search_facts(t, top_k=15):
            v = scored.setdefault(r["id"], {"row": r, "score": 0.0})
            v["score"] += 1.0
    # probe set computed up front (subject tiebreak depends on it)
    names = {r["name"] for r in conn.execute("SELECT name FROM entities")}
    probes = ({t.capitalize() for t in terms} | {"Alex", "Huginn"}) & names
    for fid, v in scored.items():
        pred = str(v["row"]["predicate"]).lower()
        if any(t in pred for t in expanded):
            v["score"] += 2.0
        # subject-priority tiebreak: a probed-entity subject outranks ties
        if str(v["row"].get("subject", "")).capitalize() in probes:
            v["score"] += 0.5
    tier1 = [v["row"] for _, v in sorted(scored.items(),
            key=lambda kv: (-kv[1]["score"], str(kv[1]["row"].get("subject", ""))))]
    # Tier 2: entity-probe facts appended after lexical winners (proven
    # client ordering — entity rows never outrank a lexical hit)
    # Tier 1.5: semantic (vector) paraphrase matches — catches wording the
    # lexical path misses ("Who does Alex work for?" -> employer fact).
    # Graceful no-op if local Ollama is down or fact_embeddings is empty.
    try:
        sem_rows = semantic_search_facts(query, top_k=max(top_k * 2, 10))
    except Exception:
        sem_rows = []
    # entity-probe facts split by relevance: term-matching rows rank after
    # tier 1, zero-term filler (name/role boilerplate) ranks LAST
    ent_matched, ent_filler = [], []
    terms_set = set(expanded)
    for e in sorted(probes)[:6]:
        for r in get_facts_for_entity(e):
            text = (str(r.get("predicate", "")) + " " +
                    str(r.get("object_name") or r.get("object_value") or "")).lower()
            words = {w.strip(".,!?") for w in text.split()}
            (ent_matched if terms_set & words else ent_filler).append(r)
    sem_strong = [r for r in sem_rows
                  if (r.get("similarity") or 0) >= 0.40
                  and not str(r.get("predicate", "")).startswith("said_in")]
    sem_weak = [r for r in sem_rows
                if 0.25 <= (r.get("similarity") or 0) < 0.40
                and not str(r.get("predicate", "")).startswith("said_in")]
    # merge order: lexical winners -> term-matched entity facts ->
    # above-gate semantic paraphrases -> weak semantic tail (only if the
    # window is short) -> zero-term entity filler last.
    # Never-empty without displacement. Lexical cap: when strong semantic
    # hits exist, lexical keeps at most half the window.
    merged, have = [], set()

    def _push(r):
        if r.get("id") is None or r.get("id") not in have:
            merged.append(r)
            have.add(r.get("id"))

    lex_cap = top_k if not sem_strong else max(2, (top_k + 1) // 2)
    for r in tier1[:lex_cap]:
        _push(r)
        if len(merged) >= top_k:
            break
    for r in sem_strong + ent_matched:  # paraphrase evidence beats probe hits
        if len(merged) >= top_k:
            break
        _push(r)
    if len(merged) < top_k:
        for r in sem_weak:
            if len(merged) >= top_k:
                break
            _push(r)
    for r in ent_filler:
        if len(merged) >= top_k:
            break
        _push(r)
    stream = merged
    seen, out = set(), []
    for r in stream:
        if str(r.get("predicate", "")).startswith("said_in"):
            continue
        if r.get("superseded_by") is not None or r.get("valid_until") is not None:
            continue
        # canonical shape: lexical rows (subject/object) and entity-probe rows
        # (subject_name/object_name/object_value) normalized to one form
        norm = {
            "id": r.get("id"),
            "subject": r.get("subject") or r.get("subject_name") or "?",
            "predicate": r.get("predicate", ""),
            "object": r.get("object") if r.get("object") is not None
                      else (r.get("object_name") if r.get("object_name") is not None
                            else r.get("object_value")),
            "pds": r.get("pds"),
            "activation": r.get("activation"),
        }
        fid = norm["id"]
        if fid is not None and fid in seen:
            continue
        seen.add(fid)
        out.append(norm)
        if len(out) >= top_k:
            break
    # Phase C: pin a provenance-tagged compiled-state span at the TOP of
    # the context block on HIT; MISS/no-flag changes nothing.
    span = _compiled_state_span(conn, query, probes)
    if span is not None:
        out.insert(0, span)
    # Maintenance item: abstention floor (Cat-5 lesson). If nothing in the
    # window shows real evidence — no compiled span, no >=gate semantic
    # paraphrase, and no row matching >=2 distinct query terms across
    # predicate+object — prepend an explicit signal instead of silently
    # returning filler rows (name/role boilerplate) that tempt the answerer
    # to hallucinate. Weak rows still follow so callers can inspect.
    def _row_evidence(r):
        txt = (str(r.get("predicate", "")) + " " +
               str(r.get("object") if r.get("object") is not None else "")).lower()
        hits = {t for t in terms if t in txt}
        hits |= {s for s in {_FACT_SYNONYMS[t] for t in terms if t in _FACT_SYNONYMS}
                 if s in txt}
        return len(hits) >= 2
    def _row_evidence(r):
        txt = (str(r.get("predicate", "")) + " " +
               str(r.get("object") if r.get("object") is not None else "")).lower()
        hits = {t for t in terms if t in txt}
        hits |= {s for s in {_FACT_SYNONYMS[t] for t in terms if t in _FACT_SYNONYMS}
                 if s in txt}
        subj = str(r.get("subject", "")).lower()
        ql = query.lower()
        want = "phillip" if ({"i", "my", "me", "mine", "myself"}
                             & {t.strip("?.,!") for t in ql.split()}) else None
        if want is None:
            want = next((p.lower() for p in probes if p.lower() in ql), "")
        # >=2 distinct term hits, or >=1 NON-GENERIC hit on a row whose
        # subject matches the queried entity. Generic tokens (number,
        # contact, name...) appear in boilerplate rows about anyone and
        # must never anchor an answer alone ("frequent flyer number with
        # Qantas" must not be answered with our contract number).
        _GENERIC = {"number", "contact", "name", "email", "phone", "date",
                    "location", "title", "status", "details"}
        if want and subj == want and (hits - _GENERIC):
            return True
        return len(hits) >= 2
    abstain = (span is None and not sem_strong
               and not any(_row_evidence(r) for r in out))
    if abstain:
        out.insert(0, {
            "id": None, "subject": "recall", "predicate": "NO_CONFIDENT_MATCH",
            "object": "no stored fact scored above the confidence floor for "
                      "this query; verify before answering, prefer abstaining",
            "pds": None, "activation": 0.0, "abstention": True,
        })
    conn.close()
    return out


def recall(query, top_k=5):
    """
    Combined recall: search sessions + memories + active facts.
    Returns a structured context bundle for pre-answer injection.
    Uses hybrid search when available.
    """
    result = hybrid_search(query, top_k=top_k, use_plan=False)

    # Also search facts by entity name extraction (structured view)
    conn = get_db()
    # Log real usage so EvolveMem evaluates on actual recalls
    try:
        n_results = len(result.get("sessions", [])) + len(result.get("memories", [])) + len(result.get("facts", []))
        conn.execute(
            """INSERT INTO query_logs (query, intent, views_used, results_count, results_quality, kind)
               VALUES (?, 'simple', '["semantic","lexical","structured"]', ?, ?, 'real')""",
            (query, n_results, json.dumps({"quality_score": None, "logged_by": "recall"}))
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass  # pre-migration DB without kind column

    entities = conn.execute("SELECT id, name FROM entities").fetchall()
    relevant_facts = []
    for e in entities:
        if e["name"].lower() in query.lower():
            facts = conn.execute(
                """SELECT f.*, e.name as subject_name, eo.name as object_name
                   FROM facts f
                   JOIN entities e ON f.subject_entity_id = e.id
                   LEFT JOIN entities eo ON f.object_entity_id = eo.id
                   WHERE f.subject_entity_id = ? AND f.valid_until IS NULL
                   ORDER BY f.activation DESC LIMIT 10""",
                (e["id"],)
            ).fetchall()
            for f in facts:
                relevant_facts.append({
                    "subject": f["subject_name"],
                    "predicate": f["predicate"],
                    "object": f["object_name"] or f["object_value"],
                    "pds": f["pds_decimal"],
                    "activation": round(f["activation"], 2),
                    "tier": "hot" if f["activation"] >= HOT_THRESHOLD else "warm" if f["activation"] >= WARM_THRESHOLD else "cool"
                })
                touch_fact(conn, f["id"])
    conn.commit()
    conn.close()

    # Merge hybrid search facts with entity-matched facts
    hybrid_facts = result.get("facts", [])
    seen_ids = {f.get("id") for f in hybrid_facts if f.get("id")}
    for f in relevant_facts:
        if f.get("id") not in seen_ids:
            hybrid_facts.append(f)

    return {
        "sessions": result.get("sessions", []),
        "memories": result.get("memories", []),
        "facts": hybrid_facts,
        "summary": f"Found {len(result.get('sessions', []))} sessions, {len(result.get('memories', []))} memories, {len(hybrid_facts)} facts"
    }


def pre_answer_recall(query, top_k=5):
    """
    Pre-answer recall hook for OpenClaw's memory pipeline.
    Runs the full hybrid retrieval pipeline and returns a compact context bundle.

    This is the function that should be called before answering a user's question
    to inject relevant memories from Muninn into the conversation context.
    """
    conn = get_db()
    context_budget = get_config(conn, "context_budget", 4000)
    plan_enabled = get_config(conn, "plan_enabled", 1)
    # Log real usage so EvolveMem evaluates on actual pre-answer recalls
    try:
        conn.execute(
            """INSERT INTO query_logs (query, intent, views_used, results_count, results_quality, kind)
               VALUES (?, 'pre_answer', '["planned"]', 0, NULL, 'real')""",
            (query,)
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass  # pre-migration DB without kind column

    conn.close()

    # Generate retrieval plan if enabled
    plan = None
    if plan_enabled:
        plan = plan_retrieval(query)
        if plan:
            top_k = plan.get("top_k", top_k)

    # Run hybrid search
    result = hybrid_search(query, top_k=top_k, use_plan=False)

    # Build compact context bundle
    context_parts = []
    token_estimate = 0

    # Sessions (most relevant first)
    for s in result.get("sessions", [])[:top_k]:
        snippet = s.get("content", "")[:300]
        context_parts.append(f"[Session {s.get('session_date', '?')}] {snippet}")
        token_estimate += len(snippet) // 4
        if token_estimate >= context_budget:
            break

    # Facts
    for f in result.get("facts", [])[:top_k]:
        fact_str = f"• {f.get('subject', '?')} {f.get('predicate', '?')} {f.get('object', '?')}"
        context_parts.append(fact_str)
        token_estimate += len(fact_str) // 4
        if token_estimate >= context_budget:
            break

    # Memories
    for m in result.get("memories", [])[:top_k // 2]:
        snippet = m.get("content", "")[:200]
        context_parts.append(f"[Memory] {snippet}")
        token_estimate += len(snippet) // 4
        if token_estimate >= context_budget:
            break

    # Entity-matched facts (structured view)
    conn = get_db()
    entities = conn.execute("SELECT id, name FROM entities").fetchall()
    for e in entities:
        if e["name"].lower() in query.lower():
            facts = conn.execute(
                """SELECT f.*, e.name as subject_name, eo.name as object_name
                   FROM facts f
                   JOIN entities e ON f.subject_entity_id = e.id
                   LEFT JOIN entities eo ON f.object_entity_id = eo.id
                   WHERE f.subject_entity_id = ? AND f.valid_until IS NULL
                   ORDER BY f.activation DESC LIMIT 5""",
                (e["id"],)
            ).fetchall()
            for f in facts:
                fact_str = f"• {f['subject_name']} {f['predicate']} {f['object_name'] or f['object_value']}"
                context_parts.append(fact_str)
                touch_fact(conn, f["id"])
    conn.commit()
    conn.close()

    return {
        "query": query,
        "plan": plan,
        "context": "\n".join(context_parts),
        "context_parts": context_parts,
        "stats": {
            "sessions": len(result.get("sessions", [])),
            "facts": len(result.get("facts", [])),
            "memories": len(result.get("memories", [])),
            "token_estimate": token_estimate
        }
    }


# ─── Sleep Cycle (with MEMORY.md sync) ───────────────────────────────────────

def run_sleep_cycle():
    conn = get_db()
    unconsolidated = conn.execute(
        "SELECT * FROM raw_sessions WHERE consolidated_at IS NULL ORDER BY session_date"
    ).fetchall()

    if not unconsolidated:
        print("✓ No sessions to consolidate")
        conn.close()
        return

    print(f"→ Consolidating {len(unconsolidated)} sessions...")

    # Group by entity clusters
    clusters = []
    remaining = list(unconsolidated)

    # Sessions with no extracted facts (cron/heartbeat chatter) would otherwise
    # become singleton clusters (empty entity sets never intersect) and burn one
    # LLM call each. Batch them by date instead: one consolidation call per day.
    def _has_facts(s):
        return conn.execute(
            "SELECT 1 FROM facts WHERE source_session_id = ? LIMIT 1", (s["id"],)
        ).fetchone() is not None

    no_fact_groups = {}
    still_remaining = []
    for s in remaining:
        if _has_facts(s):
            still_remaining.append(s)
        else:
            no_fact_groups.setdefault(s["session_date"][:10], []).append(s)
    for group in no_fact_groups.values():
        clusters.append(group)
    remaining = still_remaining

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        seed_entities = set(
            conn.execute(
                "SELECT DISTINCT e.name FROM facts f JOIN entities e ON f.subject_entity_id = e.id WHERE f.source_session_id = ?",
                (seed["id"],)
            ).fetchall()
        )
        seed_entities = {e["name"] for e in seed_entities}

        i = 0
        while i < len(remaining):
            s = remaining[i]
            s_entities = set(
                conn.execute(
                    "SELECT DISTINCT e.name FROM facts f JOIN entities e ON f.subject_entity_id = e.id WHERE f.source_session_id = ?",
                    (s["id"],)
                ).fetchall()
            )
            s_entities = {e["name"] for e in s_entities}
            if seed_entities & s_entities:
                cluster.append(remaining.pop(i))
            else:
                i += 1
        clusters.append(cluster)

    client = get_llm_client()

    total_memories = 0
    all_insights = []

    for cluster in clusters:
        combined = "\n\n---\n\n".join(s["content"] for s in cluster)
        session_ids = [s["id"] for s in cluster]
        dates = [s["session_date"][:10] for s in cluster]
        date_range = f"{min(dates)} to {max(dates)}" if len(dates) > 1 else dates[0]

        prompt = f"""You are a memory consolidation system. Summarise the following conversations ({date_range}) into key memories.

Create concise summaries that capture:
- Important facts and decisions
- Key topics discussed
- Action items or follow-ups
- Personal context learned

Return a JSON array of memory objects, each with:
- "content": the memory summary (2-3 sentences)
- "type": "episodic" (specific event), "semantic" (general knowledge), or "procedural" (how-to)
- "insight": a one-line distilled insight suitable for MEMORY.md (or null if none)

Return ONLY the JSON array.

Conversations ({date_range}):
---
{combined[:6000]}
---
"""

        try:
            resp = client.chat.completions.create(
                model=EXTRACTION_MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=6000,
            )
            raw_output = resp.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
            if json_match:
                raw_output = json_match.group(0)
            memories = json.loads(raw_output)
        except Exception as e:
            print(f"✗ Consolidation failed for {date_range}: {e}")
            continue

        for mem in memories:
            content = mem.get("content", "")
            mem_type = mem.get("type", "semantic")
            insight = mem.get("insight")

            conn.execute(
                "INSERT INTO consolidated_memories (content, type, source_session_ids) VALUES (?, ?, ?)",
                (content, mem_type, json.dumps(session_ids))
            )
            mem_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            try:
                emb = get_embedding(content)
                cur = conn.execute("INSERT INTO memory_embeddings (embedding) VALUES (?)", (emb.tobytes(),))
                rowid = cur.lastrowid
                conn.execute("INSERT INTO memory_embedding_map (rowid, memory_id) VALUES (?, ?)", (rowid, mem_id))
            except Exception:
                pass

            if insight:
                all_insights.append(insight)
            total_memories += 1

        for sid in session_ids:
            conn.execute("UPDATE raw_sessions SET consolidated_at = datetime('now') WHERE id = ?", (sid,))

    conn.commit()
    conn.close()

    if all_insights:
        sync_to_memory_md(all_insights)

    print(f"✓ Consolidated {total_memories} memories from {len(unconsolidated)} sessions ({len(clusters)} clusters)")
    if all_insights:
        print(f"✓ Synced {len(all_insights)} insights to MEMORY.md")


def sync_to_memory_md(insights):
    """Append distilled insights to MEMORY.md (under a managed section)."""
    marker = "<!-- MUNINN INSIGHTS -->"
    end_marker = "<!-- /MUNINN INSIGHTS -->"

    try:
        with open(MEMORY_MD, "r") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    insights_block = f"{marker}\n# Auto-promoted {timestamp}\n"
    for i, insight in enumerate(insights, 1):
        insights_block += f"- {insight}\n"
    insights_block += f"{end_marker}\n"

    if marker in content:
        pattern = re.compile(f"{marker}.*?{end_marker}\\n?", re.DOTALL)
        content = pattern.sub(insights_block, content)
    else:
        content = content.rstrip() + "\n\n" + insights_block

    with open(MEMORY_MD, "w") as f:
        f.write(content)

    print(f"✓ Synced {len(insights)} insights to MEMORY.md")


# ─── Fact Digest Sync ────────────────────────────────────────────────────────

def export_fact_digest(top_n=100):
    """
    Export top facts by activation to a memory file that OpenClaw's
    native memory_search can index. This makes Muninn's structured fact
    store visible to the file-based recall pipeline.

    Called nightly after the librarian runs, so the digest reflects
    clean, deduplicated, properly categorised facts.
    """
    conn = get_db()
    digest_path = os.path.join(DAILY_DIR, "muninn-facts-digest.md")

    # Get top facts by activation, grouped by PDS domain
    domains = conn.execute("SELECT code, name FROM pds_domains ORDER BY code").fetchall()

    sections = []
    total_facts = 0

    header = f"""# Muninn Fact Digest

> Auto-generated nightly by Muninn Librarian.
> Top {top_n} active facts by activation score, grouped by PDS domain.
> Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC

"""

    for domain in domains:
        domain_facts = conn.execute(
            """SELECT f.id, f.predicate, f.object_value, f.pds_decimal, f.activation,
                      f.confidence, f.is_composite,
                      e.name as subject_name, eo.name as object_name,
                      f.valid_from, f.valid_until
               FROM facts f
               JOIN entities e ON f.subject_entity_id = e.id
               LEFT JOIN entities eo ON f.object_entity_id = eo.id
               WHERE f.pds_domain = ? AND f.valid_until IS NULL
               AND f.activation >= ?
               ORDER BY f.activation DESC, f.confidence DESC
               LIMIT ?""",
            (domain["code"], COOL_THRESHOLD, top_n // len(domains) + 10)
        ).fetchall()

        if not domain_facts:
            continue

        section = f"\n## {domain['code']} — {domain['name']}\n\n"
        current_subdomain = None
        for f in domain_facts:
            # Group by subdomain within each domain
            subdomain = f["pds_decimal"] or "0000"
            if subdomain != current_subdomain:
                # Look up subdomain name
                sub = conn.execute(
                    "SELECT name FROM pds_subdomains WHERE code = ?", (subdomain,)
                ).fetchone()
                sub_name = sub["name"] if sub else "Other"
                section += f"### {subdomain} — {sub_name}\n\n"
                current_subdomain = subdomain

            obj = f["object_name"] or f["object_value"] or "?"
            act_bar = "●" * int(f["activation"] * 5) + "○" * (5 - int(f["activation"] * 5))
            comp_tag = " [composite]" if f["is_composite"] else ""
            conf_tag = f" (conf: {f['confidence']:.1f})" if f["confidence"] < 0.8 else ""
            section += f"- {f['subject_name']} **{f['predicate']}** {obj}{comp_tag}{conf_tag} `{act_bar}`\n"
            total_facts += 1

        sections.append(section.rstrip())

    # Get superseded facts summary for context
    superseded_summary = conn.execute(
        """SELECT COUNT(*) as c FROM facts WHERE superseded_by IS NOT NULL"""
    ).fetchone()["c"]

    # Get entity count
    entity_count = conn.execute("SELECT COUNT(*) as c FROM entities").fetchone()["c"]

    # v3.1: EvolveMem self-tuning summary for the digest footer
    try:
        evo = conn.execute(
            """SELECT generation, applied, reverted, metric_before, metric_after, created_at
               FROM evolution_history ORDER BY id DESC LIMIT 1"""
        ).fetchone()
        evo_count = conn.execute("SELECT COUNT(*) as c FROM evolution_history").fetchone()["c"]
    except sqlite3.OperationalError:
        evo, evo_count = None, 0
    if evo:
        if evo["metric_after"] is not None:
            delta = f"quality {evo['metric_before']:.3f} → {evo['metric_after']:.3f}"
        else:
            delta = f"quality {evo['metric_before']:.3f}"
        evo_line = (f"*EvolveMem: {evo_count} generations run · last gen {evo['generation']} on "
                    f"{(evo['created_at'] or '')[:10]} · {delta} · applied {evo['applied']}, reverted {evo['reverted']}*")
    else:
        evo_line = "*EvolveMem: enabled, no generations logged yet*"

    footer = f"""

---

*{total_facts} facts across {len(sections)} domains · {entity_count} entities · {superseded_summary} superseded facts archived*
{evo_line}
*Generated by Muninn Local v3 — `./muninn.sh digest`*
"""

    content = header + "\n".join(sections) + footer

    # Write to memory directory so OpenClaw indexes it
    os.makedirs(DAILY_DIR, exist_ok=True)
    with open(digest_path, "w") as f:
        f.write(content)

    conn.close()
    print(f"✓ Fact digest written to {digest_path} ({total_facts} facts)")
    return total_facts


# ─── Status ───────────────────────────────────────────────────────────────────

def status():
    conn = get_db()

    sessions = conn.execute("SELECT COUNT(*) as c FROM raw_sessions").fetchone()["c"]
    unprocessed = conn.execute("SELECT COUNT(*) as c FROM raw_sessions WHERE extracted_at IS NULL").fetchone()["c"]
    unconsolidated = conn.execute("SELECT COUNT(*) as c FROM raw_sessions WHERE consolidated_at IS NULL").fetchone()["c"]
    entities = conn.execute("SELECT COUNT(*) as c FROM entities").fetchone()["c"]
    facts = conn.execute("SELECT COUNT(*) as c FROM facts").fetchone()["c"]
    active_facts = conn.execute("SELECT COUNT(*) as c FROM facts WHERE valid_until IS NULL").fetchone()["c"]
    superseded = conn.execute("SELECT COUNT(*) as c FROM facts WHERE superseded_by IS NOT NULL").fetchone()["c"]
    composite_facts = conn.execute("SELECT COUNT(*) as c FROM facts WHERE is_composite = 1").fetchone()["c"]
    memories = conn.execute("SELECT COUNT(*) as c FROM consolidated_memories").fetchone()["c"]

    # Activation tiers
    hot = conn.execute("SELECT COUNT(*) as c FROM facts WHERE activation >= ? AND valid_until IS NULL", (HOT_THRESHOLD,)).fetchone()["c"]
    warm = conn.execute("SELECT COUNT(*) as c FROM facts WHERE activation >= ? AND activation < ? AND valid_until IS NULL", (WARM_THRESHOLD, HOT_THRESHOLD)).fetchone()["c"]
    cool = conn.execute("SELECT COUNT(*) as c FROM facts WHERE activation < ? AND activation >= ? AND valid_until IS NULL", (WARM_THRESHOLD, COOL_THRESHOLD)).fetchone()["c"]
    cold = conn.execute("SELECT COUNT(*) as c FROM facts WHERE activation < ? AND valid_until IS NULL", (COOL_THRESHOLD,)).fetchone()["c"]

    # v3 stats
    fts_sessions = conn.execute("SELECT COUNT(*) as c FROM sessions_fts").fetchone()["c"]
    fts_facts = conn.execute("SELECT COUNT(*) as c FROM facts_fts").fetchone()["c"]
    query_log_count = conn.execute("SELECT COUNT(*) as c FROM query_logs").fetchone()["c"]
    evo_count = conn.execute("SELECT COUNT(*) as c FROM evolution_history").fetchone()["c"]
    synth_count = conn.execute("SELECT COUNT(*) as c FROM synthesis_log").fetchone()["c"]

    # Config
    config_rows = conn.execute("SELECT key, value, updated_by FROM retrieval_config ORDER BY key").fetchall()

    domains = conn.execute("SELECT code, name FROM pds_domains ORDER BY code").fetchall()
    conn.close()

    print("╔══════════════════════════════════════════════╗")
    print("║       Muninn Local v3 — Memory Status        ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║ Sessions:          {sessions:>5}                    ║")
    print(f"║ Unprocessed:       {unprocessed:>5}                    ║")
    print(f"║ Unconsolidated:    {unconsolidated:>5}                    ║")
    print(f"║ Entities:          {entities:>5}                    ║")
    print(f"║ Facts (total):     {facts:>5}                    ║")
    print(f"║ Facts (active):    {active_facts:>5}                    ║")
    print(f"║ Facts (composite): {composite_facts:>5}                    ║")
    print(f"║ Superseded:        {superseded:>5}                    ║")
    print(f"║ Consolidated:      {memories:>5}                    ║")
    print("╠══════════════════════════════════════════════╣")
    print("║ Activation Tiers:                              ║")
    print(f"║   🔥 Hot (≥{HOT_THRESHOLD}):       {hot:>5}                    ║")
    print(f"║   🌡️ Warm ({WARM_THRESHOLD}-{HOT_THRESHOLD}):    {warm:>5}                    ║")
    print(f"║   ❄️ Cool ({COOL_THRESHOLD}-{WARM_THRESHOLD}):     {cool:>5}                    ║")
    print(f"║   🧊 Cold (<{COOL_THRESHOLD}):      {cold:>5}                    ║")
    print("╠══════════════════════════════════════════════╣")
    print("║ v3 Hybrid Retrieval:                          ║")
    print(f"║   FTS5 Sessions:   {fts_sessions:>5}                    ║")
    print(f"║   FTS5 Facts:       {fts_facts:>5}                    ║")
    print(f"║   Query Logs:       {query_log_count:>5}                    ║")
    print(f"║   Evolution Runs:   {evo_count:>5}                    ║")
    print(f"║   Synthesis Logs:   {synth_count:>5}                    ║")
    print("╠══════════════════════════════════════════════╣")
    print("║ Retrieval Config:                             ║")
    for c in config_rows:
        print(f"║   {c['key']:<20} = {c['value']:<10} ({c['updated_by']})     ║")
    print("╠══════════════════════════════════════════════╣")
    print("║ PDS Domains:                                   ║")
    for d in domains:
        print(f"║   {d['code']} — {d['name']:<30}   ║")
    print("╚══════════════════════════════════════════════╝")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Muninn Local v3 Memory System")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialise database")
    sub.add_parser("decay", help="Apply activation decay to all facts")
    sub.add_parser("rebuild-fts", help="Rebuild FTS5 indexes from existing data")

    ing = sub.add_parser("ingest", help="Ingest a conversation session")
    ing.add_argument("--content", required=True)
    ing.add_argument("--source", default="openclaw")
    ing.add_argument("--speakers", default="[]")
    ing.add_argument("--channel", default=None)
    ing.add_argument("--date", default=None)

    ingt = sub.add_parser("ingest-transcript", help="Ingest a JSON session transcript")
    ingt.add_argument("--file", required=True, help="Path to JSON transcript file")

    sea = sub.add_parser("search", help="Semantic search over sessions")
    sea.add_argument("query")
    sea.add_argument("--top-k", type=int, default=10)

    # v3: Hybrid search
    hsea = sub.add_parser("hybrid-search", help="Hybrid retrieval (semantic + lexical + structured)")
    hsea.add_argument("query")
    hsea.add_argument("--top-k", type=int, default=10)
    hsea.add_argument("--plan", action="store_true", help="Use intent-aware retrieval planning")
    hsea.add_argument("--pds", default=None, help="PDS domain filter (e.g. 2100)")

    # v3: Plan-based search
    psea = sub.add_parser("plan-search", help="Intent-aware retrieval planning + search")
    psea.add_argument("query")
    psea.add_argument("--top-k", type=int, default=10)

    # v3: Online semantic synthesis
    syn = sub.add_parser("synthesize", help="Run online semantic synthesis on facts")
    syn.add_argument("--entity", default=None, help="Synthesize facts for a specific entity")

    ext = sub.add_parser("extract", help="Extract facts from sessions")
    ext.add_argument("--session-id", default=None)
    ext.add_argument("--all-unprocessed", action="store_true")

    fac = sub.add_parser("facts", help="Query facts for an entity")
    fac.add_argument("--entity", required=True)
    fac.add_argument("--pds", default=None)

    rec = sub.add_parser("recall", help="Combined recall for pre-answer context")
    rec.add_argument("query")
    rec.add_argument("--top-k", type=int, default=5)

    # v3: Pre-answer recall context (JSON output for hook integration)
    rctx = sub.add_parser("recall-context", help="Pre-answer recall context (JSON for OpenClaw hook)")
    rctx.add_argument("query")
    rctx.add_argument("--top-k", type=int, default=5)

    sub.add_parser("sleep-cycle", help="Run consolidation sleep cycle")
    sub.add_parser("digest", help="Export fact digest to memory/muninn-facts-digest.md")
    sub.add_parser("status", help="Show system status")

    # v3: EvolveMem
    sub.add_parser("evolve", help="Run EvolveMem self-tuning loop")
    sub.add_parser("evaluate", help="Evaluate retrieval quality")

    # v3: Config management
    cfg = sub.add_parser("config", help="Get/set retrieval config")
    cfg.add_argument("--get", default=None, help="Get config value")
    cfg.add_argument("--set", default=None, help="Set config value")
    cfg.add_argument("--value", default=None, help="Value to set")
    cfg.add_argument("--list", action="store_true", help="List all config")

    args = parser.parse_args()

    if args.command == "init":
        init_db()
    elif args.command == "decay":
        apply_decay()
    elif args.command == "rebuild-fts":
        rebuild_fts_indexes()
    elif args.command == "ingest":
        sid = ingest_session(args.content, args.source, json.loads(args.speakers), args.channel, args.date)
        print(f"Session ID: {sid}")
    elif args.command == "ingest-transcript":
        sid = ingest_transcript(args.file)
        print(f"Session ID: {sid}")
    elif args.command == "search":
        results = search_sessions(args.query, args.top_k)
        if not results:
            print("No results found")
        for r in results:
            print(f"\n[{r['similarity']:.3f}] {r['id']} ({r['session_date']})")
            print(f"  {r['content'][:200]}...")
    elif args.command == "hybrid-search":
        result = hybrid_search(args.query, args.top_k, args.plan, args.pds)
        print(f"\n{result['summary']}")
        if result.get("plan"):
            print(f"\n📋 Plan: {json.dumps(result['plan'], indent=2)}")
        print("\n📋 Sessions:")
        for s in result["sessions"]:
            score = s.get("rrf_score", s.get("similarity", 0))
            print(f"  [{score:.4f}] {s.get('session_date', '?')} — {s.get('content', '')[:100]}...")
        print("\n📊 Facts:")
        for f in result["facts"]:
            score = f.get("rrf_score", f.get("bm25_score", 0))
            print(f"  [{score:.4f}] {f.get('subject', '?')} {f.get('predicate', '?')} {f.get('object', '?')}")
        print("\n🧠 Memories:")
        for m in result["memories"]:
            print(f"  [{m.get('similarity', 0):.3f}] [{m.get('type', '?')}] {m.get('content', '')[:100]}...")
    elif args.command == "plan-search":
        plan = plan_retrieval(args.query)
        if plan:
            print(f"📋 Retrieval Plan:")
            print(json.dumps(plan, indent=2))
            print(f"\n→ Executing plan...")
            result = hybrid_search(args.query, plan.get("top_k", args.top_k), use_plan=False,
                                   pds_filter=plan.get("pds_filter"))
            print(f"\n{result['summary']}")
            print("\n📋 Sessions:")
            for s in result["sessions"]:
                score = s.get("rrf_score", s.get("similarity", 0))
                print(f"  [{score:.4f}] {s.get('session_date', '?')} — {s.get('content', '')[:100]}...")
            print("\n📊 Facts:")
            for f in result["facts"]:
                score = f.get("rrf_score", f.get("bm25_score", 0))
                print(f"  [{score:.4f}] {f.get('subject', '?')} {f.get('predicate', '?')} {f.get('object', '?')}")
            print("\n🧠 Memories:")
            for m in result["memories"]:
                print(f"  [{m.get('similarity', 0):.3f}] [{m.get('type', '?')}] {m.get('content', '')[:100]}...")
        else:
            print("⚠ No plan generated, falling back to hybrid search...")
            result = hybrid_search(args.query, args.top_k)
            print(f"\n{result['summary']}")
    elif args.command == "synthesize":
        count = synthesize_facts(args.entity)
        print(f"✓ Synthesized {count} composite facts")
    elif args.command == "extract":
        if args.session_id:
            extract_facts_from_session(args.session_id)
        elif args.all_unprocessed:
            extract_all_unprocessed()
        else:
            print("Specify --session-id or --all-unprocessed")
    elif args.command == "facts":
        facts = get_facts_for_entity(args.entity, args.pds)
        if not facts:
            print(f"No facts found for '{args.entity}'")
        for f in facts:
            obj = f.get("object_name") or f.get("object_value") or "?"
            tier = "🔥" if f.get("activation", 0) >= HOT_THRESHOLD else "🌡️" if f.get("activation", 0) >= WARM_THRESHOLD else "❄️"
            comp = " [composite]" if f.get("is_composite") else ""
            print(f"  {tier} [{f['pds_decimal']}] {f['subject_name']} {f['predicate']} {obj} (act: {f.get('activation', 1.0):.2f}){comp}")
    elif args.command == "recall":
        result = recall(args.query, args.top_k)
        print(f"\n{result['summary']}")
        print("\n📋 Sessions:")
        for s in result["sessions"]:
            print(f"  [{s.get('similarity', s.get('rrf_score', 0)):.3f}] {s.get('content', '')[:100]}...")
        print("\n🧠 Memories:")
        for m in result["memories"]:
            print(f"  [{m.get('similarity', 0):.3f}] [{m.get('type', '?')}] {m.get('content', '')[:100]}...")
        print("\n📊 Facts:")
        for f in result["facts"]:
            tier = f.get("tier", "")
            print(f"  {tier} [{f.get('pds', '?')}] {f.get('subject', '?')} {f.get('predicate', '?')} {f.get('object', '?')}")
    elif args.command == "recall-context":
        # JSON output for pre-answer hook integration
        result = pre_answer_recall(args.query, args.top_k)
        print(json.dumps(result, indent=2))
    elif args.command == "sleep-cycle":
        apply_decay()
        run_sleep_cycle()
        # Also refresh the fact digest after sleep cycle
        export_fact_digest()
        # v3.1: nightly self-tuning — one EvolveMem generation if enabled
        try:
            _c = get_db()
            _evolve_on = get_config(_c, "evolve_enabled", 0)
            _c.close()
        except sqlite3.OperationalError:
            _evolve_on = 0
        if _evolve_on:
            print("\n── EvolveMem nightly generation ──")
            try:
                run_evolution(generations=1)
            except Exception as e:
                print(f"✗ EvolveMem failed (non-fatal): {e}")
    elif args.command == "digest":
        export_fact_digest()
    elif args.command == "evolve":
        run_evolution()
    elif args.command == "evaluate":
        score = evaluate_retrieval()
        print(f"Average quality score: {score:.3f}")
    elif args.command == "config":
        conn = get_db()
        if args.list:
            rows = conn.execute("SELECT key, value, type, description, updated_by FROM retrieval_config ORDER BY key").fetchall()
            for r in rows:
                print(f"  {r['key']:<25} = {r['value']:<10} ({r['type']}, by {r['updated_by']})")
                if r["description"]:
                    print(f"    {r['description']}")
        elif args.get:
            val = get_config(conn, args.get)
            print(f"{args.get} = {val}")
        elif args.set:
            set_config(conn, args.set, args.value, updated_by="manual")
            conn.commit()
            print(f"✓ Set {args.set} = {args.value}")
        else:
            print("Use --list, --get KEY, or --set KEY --value VALUE")
        conn.close()
    elif args.command == "status":
        status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# ---------- graph traversal recall (LOCOMO A16 work, additive; added 2026-09-29) ----------

def recall_graph(query, top_k=10, hops=2, per_hop=8):
    """Multi-hop recall: seed facts via existing recall_facts, then walk
    subject→object entity links HOPS levels, scoring by activation.
    Read-only; does not modify the store."""
    conn = get_db()
    seeds = recall_facts(query, top_k=per_hop)
    results = {}
    frontier = []
    for f in seeds:
        if isinstance(f, dict):
            fid = f.get("id")
        else:
            fid = f["id"] if hasattr(f, "keys") else None
        if fid:
            results[fid] = f
            frontier.append(fid)

    for hop in range(hops):
        if not frontier:
            break
        qm = ",".join("?" * len(frontier))
        # expand: entities reachable from current fact set (as subject or object)
        rows = conn.execute(
            f"""SELECT f.* FROM facts f
                WHERE f.valid_until IS NULL
                  AND (f.subject_entity_id IN (
                        SELECT subject_entity_id FROM facts WHERE id IN ({qm})
                        UNION SELECT object_entity_id FROM facts WHERE id IN ({qm}))
                   OR f.object_entity_id IN (
                        SELECT subject_entity_id FROM facts WHERE id IN ({qm})
                        UNION SELECT object_entity_id FROM facts WHERE id IN ({qm})))
                ORDER BY f.activation DESC LIMIT ?""",
            (*frontier, *frontier, *frontier, *frontier, per_hop * (hop + 3) * 4),
        ).fetchall()
        new_frontier = []
        for r in rows:
            if r["id"] not in results:
                results[r["id"]] = dict(r)
                new_frontier.append(r["id"])
        frontier = new_frontier

    conn.close()
    out = list(results.values())
    out.sort(key=lambda f: f.get("activation", 0) or 0, reverse=True)
    return out[:top_k]


# ---------- fact-value chain recall (added 2026-09-30, v6 groundwork) ----------
def recall_chains(entity_name, max_chains=60):
    """Fact-value chain recall. Rule (verified on LOCOMO): a fact's VALUE
    tokens token-overlap another sibling fact's COMPOUND PREDICATE tokens
    (Caroline moved_from 'home country' + Caroline home_country 'Sweden').
    Returns facts annotated with their chain continuations."""
    conn = get_db()
    conn.row_factory = sqlite3.Row
    facts = conn.execute(
        """SELECT f.id, f.predicate, COALESCE(e2.name, f.object_value) obj,
                  f.activation
           FROM facts f JOIN entities e ON f.subject_entity_id=e.id
           LEFT JOIN entities e2 ON f.object_entity_id=e2.id
           WHERE e.name=? AND f.valid_until IS NULL AND f.superseded_by IS NULL
           ORDER BY f.activation DESC LIMIT ?""",
        (entity_name, max_chains)).fetchall()
    # value-index: value tokens -> (fact_id) for chain targets (compound
    # predicates only, snake_case or space forms)
    compound = []
    for r in facts:
        if "_" in r["predicate"] or " " in r["predicate"]:
            toks = set(re.findall(r"[a-z]{3,}", r["predicate"].replace("_", " ")))
            compound.append((r["id"], r["predicate"], toks, r["obj"]))
    annotated = []
    for r in facts:
        objv = str(r["obj"])
        vtoks = set(re.findall(r"[a-z]{3,}", objv.lower()))
        # distinctive-token rule (false-chain fix): overlap must include the
        # LONGEST value token ("country" anchors moved_from->home_country;
        # "home" alone created a false provides_home_for chain)
        longest = max(vtoks, key=len) if vtoks else None
        conts = []
        for (fid2, pred2, ptoks, pobj) in compound:
            if fid2 == r["id"]:
                continue
            if (vtoks & ptoks) and longest and longest in ptoks:
                conts.append((len(vtoks & ptoks), f"{pred2}: {str(pobj)[:80]}"))
        conts.sort(key=lambda x: -x[0])
        conts = [c for _, c in conts[:2]]
        if conts:
            annotated.append(f"{entity_name} {r['predicate']}: {objv} "
                             f"(chain: {'; '.join(conts[:2])})")
        else:
            annotated.append(f"{entity_name} {r['predicate']}: {objv}")
    conn.close()
    return annotated
