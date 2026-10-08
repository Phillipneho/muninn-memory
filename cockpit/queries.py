"""All cockpit SQL lives here — one function per view need. Read-only."""
from .db import connect_ro


def health_vitals() -> dict:
    conn = connect_ro()
    try:
        counts = {}
        for t in ("facts", "entities", "entity_state", "raw_sessions",
                  "query_logs"):
            counts[t] = conn.execute(
                f"SELECT COUNT(*) FROM {t}").fetchone()[0]

        active_facts = counts["facts"] - conn.execute(
            "SELECT COUNT(*) FROM facts WHERE superseded_by IS NOT NULL "
            "OR valid_until IS NOT NULL").fetchone()[0]

        orphan_subject = conn.execute(
            """SELECT COUNT(DISTINCT f.subject_entity_id), COUNT(*)
               FROM facts f LEFT JOIN entities e ON e.id = f.subject_entity_id
               WHERE f.subject_entity_id IS NOT NULL AND e.id IS NULL"""
        ).fetchone()
        orphan_object = conn.execute(
            """SELECT COUNT(DISTINCT f.object_entity_id), COUNT(*)
               FROM facts f LEFT JOIN entities e ON e.id = f.object_entity_id
               WHERE f.object_entity_id IS NOT NULL AND e.id IS NULL"""
        ).fetchone()

        last_compiles = conn.execute(
            """SELECT attribute, COUNT(*) AS slots, MAX(compiled_at) AS at
               FROM entity_state GROUP BY attribute ORDER BY at DESC LIMIT 8"""
        ).fetchall()

        config_rows = conn.execute(
            "SELECT key, value FROM retrieval_config ORDER BY key").fetchall()

        # integrity digests (content digest over ordered rows)
        import hashlib
        def digest(t):
            h = hashlib.md5()
            for row in conn.execute(f"SELECT * FROM {t} ORDER BY rowid"):
                h.update(str(tuple(row)).encode())
            return h.hexdigest()[:12]

        return {
            "counts": counts,
            "active_facts": active_facts,
            "orphan_subject": {"ids": orphan_subject[0], "rows": orphan_subject[1]},
            "orphan_object": {"ids": orphan_object[0], "rows": orphan_object[1]},
            "last_compiles": [dict(r) for r in last_compiles],
            "config": [dict(r) for r in config_rows],
            "digests": {t: digest(t) for t in ("facts", "entities")},
        }
    finally:
        conn.close()


def entity_list(q: str = "") -> list:
    conn = connect_ro()
    try:
        rows = conn.execute(
            """SELECT e.id, e.name, e.type,
                      (SELECT COUNT(*) FROM facts f WHERE f.subject_entity_id = e.id) AS fact_count,
                      (SELECT COUNT(*) FROM entity_state s WHERE s.entity_id = e.id) AS slot_count
               FROM entities e
               WHERE e.name LIKE ? OR e.aliases LIKE ?
               ORDER BY fact_count DESC
               LIMIT 200""",
            (f"%{q}%", f"%{q}%")).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def entity_detail(entity_id: int) -> dict:
    conn = connect_ro()
    try:
        ent = conn.execute(
            "SELECT id, name, type, aliases, created_at FROM entities WHERE id = ?",
            (entity_id,)).fetchone()
        if not ent:
            return None
        slots = conn.execute(
            "SELECT attribute, state_json, evidence_json, compiled_at, source_span_count "
            "FROM entity_state WHERE entity_id = ? ORDER BY attribute", (entity_id,)
        ).fetchall()
        facts = conn.execute(
            """SELECT id, predicate, object_value, object_entity_id, pds_domain,
                      activation, valid_from, valid_until, superseded_by, created_at
               FROM facts WHERE subject_entity_id = ?
                 AND superseded_by IS NULL AND valid_until IS NULL
               ORDER BY activation DESC LIMIT 100""", (entity_id,)).fetchall()
        return {
            "entity": dict(ent),
            "slots": [dict(s) for s in slots],
            "facts": [dict(f) for f in facts],
        }
    finally:
        conn.close()


def query_log_tail(limit: int = 100) -> list:
    conn = connect_ro()
    try:
        rows = conn.execute(
            "SELECT id, query, intent, views_used, results_count, results_quality, "
            "config_snapshot, created_at, kind FROM query_logs "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def timeline_events(entity_id=None, days=90, limit=200) -> list:
    conn = connect_ro()
    try:
        ev = []
        if entity_id:
            facts = conn.execute(
                """SELECT id, predicate, object_value, created_at, valid_from,
                          valid_until, superseded_by
                   FROM facts WHERE subject_entity_id = ?
                   ORDER BY created_at DESC LIMIT ?""", (entity_id, limit)).fetchall()
            ev += [{"kind": "fact", "at": r["created_at"], "text": r["predicate"],
                    "detail": r["object_value"], "ref": r["id"],
                    "superseded": r["superseded_by"]} for r in (dict(x) for x in facts)]
            sess = conn.execute(
                """SELECT DISTINCT f.source_session_id, MIN(f.created_at) AS at
                   FROM facts f WHERE f.subject_entity_id = ?
                   GROUP BY f.source_session_id ORDER BY at DESC LIMIT ?""",
                (entity_id, limit // 4)).fetchall()
            ev += [{"kind": "session", "at": r["at"], "text": "source session",
                    "detail": r["source_session_id"], "ref": None,
                    "superseded": None} for r in (dict(x) for x in sess)]
        else:
            sess = conn.execute(
                "SELECT id, source, speakers, channel, created_at FROM raw_sessions "
                "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            ev += [{"kind": "session", "at": r["created_at"],
                    "text": f"{r['source']}/{r['channel'] or 'direct'}",
                    "detail": (r["speakers"] or "")[:80], "ref": r["id"],
                    "superseded": None} for r in (dict(x) for x in sess)]
            comp = conn.execute(
                "SELECT entity_id, attribute, compiled_at FROM entity_state "
                "ORDER BY compiled_at DESC LIMIT ?", (limit // 4,)).fetchall()
            ev += [{"kind": "compile", "at": r["compiled_at"],
                    "text": f"compiled slot {r['attribute']}",
                    "detail": f"entity #{r['entity_id']}", "ref": r["entity_id"],
                    "superseded": None} for r in (dict(x) for x in comp)]
        ev.sort(key=lambda x: x["at"] or "", reverse=True)
        return ev[:limit]
    finally:
        conn.close()

def slot_provenance(entity_id: int, attribute: str, limit=20) -> list:
    conn = connect_ro()
    try:
        row = conn.execute(
            """SELECT s.evidence_json FROM entity_state s
               WHERE s.entity_id = ? AND s.attribute = ?""",
            (entity_id, attribute)).fetchone()
        if not row or not row["evidence_json"]:
            return []
        import json as _json
        try:
            ev = _json.loads(row["evidence_json"])
        except Exception:
            return []
        out = []
        for e in ev[:limit]:
            fid = e.get("fact_id")
            snippet = ""
            if e.get("source_session_id"):
                r = conn.execute(
                    "SELECT content FROM raw_sessions WHERE id = ?",
                    (e["source_session_id"],)).fetchone()
                if r and r["content"]:
                    c = r["content"]
                    key = (e.get("value") or "")[:20]
                    idx = c.find(key) if key else -1
                    if idx >= 0:
                        snippet = c[max(0, idx - 120):idx + 180]
                    else:
                        snippet = c[:280]
            out.append({"fact_id": fid, "value": e.get("value"),
                        "pred": e.get("pred"), "ts": e.get("ts"),
                        "session_id": e.get("source_session_id"),
                        "snippet": snippet.strip()})
        return out
    finally:
        conn.close()


def entity_facts_filtered(entity_id: int, show: str = "active",
                          limit=25, offset=0) -> dict:
    conn = connect_ro()
    try:
        base = "FROM facts WHERE subject_entity_id = ?"
        if show == "superseded":
            cond = " AND (superseded_by IS NOT NULL OR valid_until IS NOT NULL)"
        else:
            cond = " AND superseded_by IS NULL AND valid_until IS NULL"
        total = conn.execute(
            f"SELECT COUNT(*) {base}{cond}", (entity_id,)).fetchone()[0]
        rows = conn.execute(
            f"""SELECT id, predicate, object_value, object_entity_id, pds_domain,
                       activation, valid_from, valid_until, superseded_by, created_at
                {base}{cond} ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (entity_id, limit, offset)).fetchall()
        sup_map = {}
        sup_ids = [r["superseded_by"] for r in rows if r["superseded_by"]]
        if sup_ids:
            marks = ",".join("?" for _ in sup_ids)
            for r in conn.execute(
                f"SELECT id, predicate, object_value FROM facts "
                f"WHERE id IN ({marks})", sup_ids).fetchall():
                sup_map[r["id"]] = dict(r)
        return {"total": total, "rows": [dict(r) for r in rows],
                "sup_map": sup_map}
    finally:
        conn.close()


def query_detail(qid: int) -> dict:
    conn = connect_ro()
    try:
        r = conn.execute(
            "SELECT id, query, intent, views_used, results_count, results_quality, "
            "config_snapshot, created_at, kind FROM query_logs WHERE id = ?",
            (qid,)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()
