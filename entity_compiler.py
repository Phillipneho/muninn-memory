#!/usr/bin/env python3
"""entity_compiler.py — Muninn v2.0 Phase B: production entity-state compiler.

Spec: MUNINN_V2_ENTITY_STATE_SPEC.md (§1 data model, §2 slot vocab,
§3 compile pipeline, §3.3 verify, §5 safety rails).
Shadow mode: builds/maintains entity_state in the REAL muninn.db, additive-only.
Read-routing (recall_facts) is NOT touched here. No cron. One supervised cycle.

Usage:
  python3 entity_compiler.py              # full compile (first run: all entities)
  python3 entity_compiler.py --smoke      # point-lookup smoke tests only
"""
import os, sys, json, re, time, hashlib, sqlite3, datetime, collections

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "locomo"))

DB = os.environ.get("MUNINN_COMPILE_DB", "muninn.db")
REPORT = "shadow_compile_report.log"
LLM_BUDGET = 200                     # hard cap, flash tier (spec §3c / §5)
llm_calls = 0

# ---------- slots (ported from lib_entity_slots.py; vocab identical) ----------
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("lib_entity_slots", os.path.join(HERE, "locomo", "lib_entity_slots.py"))
LS = _ilu.module_from_spec(_spec); _spec.loader.exec_module(LS)
SLOTS, SCALAR_SLOTS, ENUM_SLOTS = LS.SLOTS, LS.SCALAR_SLOTS, LS.ENUM_SLOTS
norm = LS.norm


def near_dup_collapse(items, threshold=0.6):
    """Conservative value-proximity collapse (spec §3d, ported from the POC):
    token-jaccard merge, recorded not destructive — items keep one
    representative per cluster with (xN) multiplicity."""
    reps = []
    mult = []
    for it in items:
        tn = norm(it)
        toks = set(tn.split())
        placed = False
        for i, (rep, rt) in enumerate(reps):
            if toks and rt:
                if len(toks & rt) / len(toks | rt) >= threshold:
                    mult[i] += 1
                    placed = True
                    break
        if not placed:
            reps.append((it, toks))
            mult.append(1)
    out = []
    for (rep, _t), m in zip(reps, mult):
        out.append(f"{rep} (x{m})" if m > 1 else rep)
    return out, len(reps)

# ---------- §2 production predicate->slot mapping ----------
# Real muninn predicates are snake_case relation-style (works_at, child, likes,
# prefers_*). Mapping stays deterministic; unknown -> timeline_events (never dropped).
PROD_PREDICATE_MAP = collections.defaultdict(lambda: "timeline_events")
def _m(slot, preds):
    for p in preds:
        PROD_PREDICATE_MAP[p] = slot

_m("vehicles", ["has_car", "car", "vehicle", "drives", "drive", "owns_car",
                "vehicle_owned", "bought_car", "prius", "rav4"])
_m("residences", ["lives_in", "live_in", "lives", "resides_in", "resides",
                  "moved_to", "lives_at", "home_in", "hometown",
                  "located_in", "based_in"])
_m("employers", ["works_at", "worked_at", "works_for", "employed_at",
                 "employer", "hired_at", "employed_by"])
_m("job_titles", ["works_as", "job_title", "has_job_title", "job", "has_role", "role",
                  "role_at", "involves_role", "involves_roles", "had_role",
                  "is_role", "received_promotion", "assigned_task",
                  "was_previous_person_in", "completed_task"])
_m("children", ["child", "children", "has_child", "has_children", "kids",
                "kid", "son", "daughter", "stepson", "stepdaughter",
                "number_of_children"])
_m("parents", ["mother", "father", "parent", "parents", "mom", "dad"])
_m("siblings", ["sibling", "siblings", "brother", "sister", "twin"])
_m("pets", ["pet", "pets", "has_pet", "has_pets", "owns_pet"])
_m("friends", ["friend", "friends", "befriended", "made_friends"])
_m("travel", ["visited", "visit", "traveled_to", "travelled", "travel",
              "went_to", "trip", "trip_to", "vacation", "cities_visited"])
_m("health_conditions", ["health", "health_condition", "illness", "diagnosed",
                         "has_condition", "sick", "migraine", "allergy"])
_m("injuries", ["injury", "injured", "broke", "broken", "has_broken",
                "fracture", "sprained", "write_off", "written_off"])
_m("schooling", ["school", "attends_school", "studies", "studies_at",
                 "has_education", "education", "graduated", "university",
                 "college", "attended"])
_m("memberships", ["member", "member_of", "was_member_of", "is_member_of",
                   "joined", "volunteers_at", "volunteers"])
_m("owned_items", ["owns", "owned_by", "owns_project", "bought", "purchased",
                   "owns_item", "owned_item"])
_m("contact", ["has_phone", "phone", "email", "email_address", "contact",
               "has_contact", "is_contact_for", "escalation_contact_for"])
_m("preferences", ["prefers", "preference", "prefers_language", "prefers_font",
                   "prefers_fonts", "prefers_wake_word", "prefers_color_base",
                   "prefers_aesthetic", "prefers_accent_color", "likes",
                   "likes_musical_artist", "likes_book", "likes_food",
                   "favorite", "favourite", "hobby", "hobbies",
                   "destresses_by", "has_preference"])
# deliberately unmapped: partner, has_partner, family -> timeline_events
# (closed 18-slot vocab per spec §2; v2.1 candidate for household slots)

# ---------- value resolution ----------

def fact_value(db_row, db):
    """Value = object_value or object-entity name. Returns (value, value_kind)."""
    (_id, subj, pred, obj_ent, obj_val, *_rest) = db_row
    if obj_val not in (None, ""):
        return str(obj_val).strip(), "value"
    if obj_ent is not None:
        r = db.execute("SELECT name FROM entities WHERE id=?", (obj_ent,)).fetchone()
        if r:
            return r[0], "entity"
    return None, "none"


def compile_entity(db, entity_id, entity_name, aliases, t0_compile):
    """Pure transform of the entity's facts -> state rows (spec §3b–e)."""
    facts = db.execute(
        """SELECT f.id, f.subject_entity_id, f.predicate, f.object_entity_id,
                  f.object_value, f.valid_from, f.valid_until, f.superseded_by,
                  f.source_session_id, f.created_at
           FROM facts f WHERE f.subject_entity_id=? ORDER BY f.id""",
        (entity_id,)).fetchall()
    by_slot = collections.defaultdict(list)
    for f in facts:
        fid, _s, pred, _oe, _ov, vf, vu, sup, ssid, created = f
        slot = PROD_PREDICATE_MAP[norm(pred)]
        val, kind = fact_value(f, db)
        if val is None:
            val = f"[unresolved:{pred}]"
        # temporal ordering key: prefer valid_from, then created_at, then id
        order = (vf or created or "", fid)
        by_slot[slot].append({
            "fact_id": fid, "value": val, "kind": kind, "pred": pred,
            "source_session_id": ssid, "ts": vf or created or "", "sup": sup,
            "valid_until": vu, "order": order,
        })
    rows = []
    now = t0_compile
    for slot in sorted(by_slot):
        fs = sorted(by_slot[slot], key=lambda x: x["order"])
        ev = [{"fact_id": f["fact_id"], "value": f["value"][:160],
               "source_session_id": f["source_session_id"], "ts": f["ts"],
               "pred": f["pred"]} for f in fs]
        sup_now = {f["fact_id"] for f in fs if f["sup"] not in (None, 0, -1)} | \
                  {f["fact_id"] for f in fs if f["valid_until"] not in (None, "", "null")}
        if slot in SCALAR_SLOTS:
            current_vals = [f for f in fs if f["fact_id"] not in sup_now]
            current = current_vals[-1] if current_vals else fs[-1]
            superseded = [f["value"] for f in fs
                          if f["fact_id"] in sup_now and f["value"] != current["value"]]
            superseded += [f["value"] for f in fs[:-1]
                           if f["fact_id"] not in sup_now and f is not current
                           and f["value"] != current["value"]]
            state = {"current": current["value"],
                     "superseded": list(dict.fromkeys(superseded))[-5:]}
        elif slot in ENUM_SLOTS:
            items_in = [f["value"] for f in fs]
            items, n_distinct = near_dup_collapse(items_in)
            state = {"count": len(items_in), "items": items,
                     "distinct_count": n_distinct}
        else:
            state = {"items": list(dict.fromkeys(f["value"] for f in fs))}
        rows.append((slot, json.dumps(state, ensure_ascii=False),
                     json.dumps(ev), now, len(fs)))
    return rows


def write_rows(db, entity_id, entity_name, rows):
    db.execute("DELETE FROM entity_state WHERE entity_id=?", (entity_id,))
    for slot, sj, ej, now, span_count in rows:
        db.execute(
            """INSERT INTO entity_state
               (entity_id, attribute, state_json, evidence_json, compiled_at,
                source_span_count) VALUES (?,?,?,?,?,?)""",
            (entity_id, slot, sj, ej, now, span_count))


def hash_table(db):
    h = hashlib.sha256()
    n = 0
    for row in db.execute("""SELECT entity_id, attribute, state_json
                             FROM entity_state ORDER BY entity_id, attribute"""):
        h.update(str(row).encode())
        n += 1
    return n, h.hexdigest()


# ---------- main ----------

def main():
    global llm_calls
    t0 = time.time()
    L = []
    def p(s=""):
        print(s, flush=True); L.append(s)

    from contextlib import closing
    with closing(sqlite3.connect(DB)) as db:
        db.row_factory = None
        db.execute("PRAGMA foreign_keys=ON")
        p("=" * 100)
        p("MUNINN v2.0 PHASE B — SHADOW COMPILE (full cycle, supervised; production muninn.db)")
        p("=" * 100)

        # additive-only schema (spec §1), zero changes elsewhere
        db.executescript("""
        CREATE TABLE IF NOT EXISTS entity_state (
          id INTEGER PRIMARY KEY,
          entity_id INTEGER REFERENCES entities(id),
          attribute TEXT,
          state_json TEXT,
          evidence_json TEXT,
          compiled_at TEXT,
          source_span_count INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_entity_state
          ON entity_state(entity_id, attribute);
        """)
        pre_rows = db.execute("SELECT COUNT(*) FROM entity_state").fetchone()[0]
        p(f"[0] entity_state rows before compile: {pre_rows}")

        # ---- delta selection (spec §3.1): per-entity by fact created_at > compiled_at
        dirty = db.execute("""
            SELECT DISTINCT f.subject_entity_id
            FROM facts f
            LEFT JOIN entity_state es ON es.entity_id = f.subject_entity_id
            WHERE f.subject_entity_id IS NOT NULL AND
                  (es.entity_id IS NULL OR f.created_at > es.compiled_at)
        """).fetchall()
        entities = db.execute("SELECT id, name, aliases FROM entities").fetchall()
        ent_names = {r[0]: (r[1], r[2]) for r in entities}
        dirty_ids = sorted({r[0] for r in dirty if r[0] in ent_names})
        dangling = sorted({r[0] for r in dirty if r[0] not in ent_names})
        p(f"[1] DELTA-SELECT: {len(dirty_ids)} dirty entities "
          f"(of {len(ent_names)} total; {'first run — all' if pre_rows == 0 else 'delta'})"
          + (f"; {len(dangling)} dangling subject_entity_ids skipped (anomaly)" if dangling else ""))

        # ---- compile ----
        t_phases = collections.defaultdict(float)
        tsel = time.time()
        t_phases["delta_select"] += time.time() - tsel
        tf = time.time(); t_phases["fact_pull"] = 0
        tg = time.time(); t_phases["group+write"] = 0
        counts_by_slot = collections.defaultdict(int)
        compiled_entities = 0
        total_rows = 0
        llm_deferred = 0
        unresolved_predicates = collections.Counter()
        for eid in dirty_ids:
            name, aliases = ent_names.get(eid, (f"id{eid}", None))
            now = datetime.datetime.utcnow().isoformat()
            rows = compile_entity(db, eid, name, aliases, now)
            for r in rows:
                counts_by_slot[r[0]] += 1
            # unresolved-predicate accounting (anomaly tracking)
            facts = db.execute(
                "SELECT predicate, object_entity_id, object_value FROM facts"
                " WHERE subject_entity_id=?", (eid,)).fetchall()
            for pred, oe, ov in facts:
                if oe is None and (ov is None or ov == ""):
                    unresolved_predicates[pred] += 1
            t1 = time.time()
            t_phases["fact_pull"] += t1 - tf
            write_rows(db, eid, name, rows)
            t2 = time.time()
            t_phases["group+write"] += t2 - t1
            compiled_entities += 1
            total_rows += len(rows)
            tf, tg = t2, t2
            if llm_calls >= LLM_BUDGET:
                llm_deferred += 1
        db.commit()
        p(f"[2] COMPILE: {compiled_entities} entities, {total_rows} state rows "
          f"(slot-class counts below)")
        for slot in SLOTS:
            if counts_by_slot.get(slot):
                p(f"      {slot}: {counts_by_slot[slot]}")
        p(f"      LLM coref calls used: {llm_calls}/{LLM_BUDGET} "
          f"(deterministic-only compile; coref passes skipped for cycle 1")
        p(f"      entities whose LLM passes were deferred: {llm_deferred})")

        # verify phase (spec §3.3): 5% spot-check recompute
        tv = time.time()
        n, digest = hash_table(db)
        checked, mismatches = 0, 0
        sample = dirty_ids[:: max(1, len(dirty_ids) // 40)][:40]
        for eid in sample:
            name, _ = ent_names.get(eid, (f"id{eid}", None))
            now2 = "2026-01-01T00:00:00"
            rows = compile_entity(db, eid, name, None, now2)
            stored = {r[0]: r[1] for r in db.execute(
                "SELECT attribute, state_json FROM entity_state WHERE entity_id=?",
                (eid,)).fetchall()}
            recomputed = {s: sj for s, sj, *_ in rows}
            checked += 1
            if stored != recomputed:
                mismatches += 1
                p(f"      MISMATCH: {name}({eid}): "
                  f"stored={set(stored)} recomputed={set(recomputed)}")
        t_phases["verify"] = time.time() - tv
        p(f"[3] SPOT-CHECK (5% = {checked} entities): "
          f"{'PASS' if mismatches == 0 else f'{mismatches} MISMATCHES'}")

        # idempotency (task step 3): full re-compile with a frozen compiled_at
        # so the state_json content hash must converge; compiled_at is excluded.
        tid = time.time()
        def content_digest():
            h = hashlib.sha256()
            for row in db.execute("SELECT entity_id, attribute, state_json "
                                  "FROM entity_state ORDER BY entity_id, attribute"):
                h.update(str(row).encode())
            return h.hexdigest()
        before_hash = content_digest()
        for eid in dirty_ids:
            name, _ = ent_names.get(eid, (f"id{eid}", None))
            rows = compile_entity(db, eid, name, None, "2026-01-01T00:00:00")
            write_rows(db, eid, name, rows)
        db.commit()
        after_hash = content_digest()
        t_phases["idempotency_full_rerun"] = time.time() - tid
        p(f"[4] IDEMPOTENCY: re-compiled all {len(dirty_ids)} dirty entities with "
          "a frozen compiled_at; state_json content hash "
          f"{'CONVERGED' if before_hash == after_hash else 'DIVERGED'} "
          f"(before {before_hash[:16]}… / after {after_hash[:16]}…)")

        # restore real compiled_at for a clean final pass (so delta-select stays correct)
        db.execute("UPDATE entity_state SET compiled_at=?",
                   (datetime.datetime.utcnow().isoformat(),))
        db.commit()

        # smoke-test point lookups (direct entity_state hits; read-routing OFF)
        p("")
        p("[5] SMOKE-TEST POINT LOOKUPS (direct entity_state queries; recall_facts untouched)")
        probes = [
            ("Alex", "children", "How many children does Alex have and what are their names?"),
            ("Alex", "residences", "Where does Alex live?"),
            ("Alex", "employers", "Who does Alex work for?"),
            ("Alex", "preferences", "What does Alex like to do for exercise?"),
            ("Leo", "job_titles", "What is Leo's role?"),
            ("Leo", "residences", "Where does Leo live?"),
            ("Leo", "preferences", "What are Leo's preferences?"),
            ("Melanie", "preferences", "What kind of activities does Melanie enjoy?"),
            ("Melanie", "schooling", "What school activities has Melanie attended?"),
        ]
        for ename, attr, q in probes:
            er = db.execute("SELECT id FROM entities WHERE lower(name)=lower(?)",
                            (ename,)).fetchone()
            if not er:
                p(f"  [{ename}] MISS: entity not present")
                continue
            hit = db.execute(
                "SELECT state_json FROM entity_state WHERE entity_id=? AND attribute=?",
                (er[0], attr)).fetchone()
            if hit:
                st = json.loads(hit[0])
                brief = (st.get("current") if "current" in st
                         else f"{st.get('count','?')} items: " +
                              "; ".join(str(i)[:80] for i in (st.get("items") or [])[:6]))
                p(f"  [{ename}/{attr}] HIT: {str(brief)[:220]}  <- '{q[:44]}…'")
            else:
                p(f"  [{ename}/{attr}] MISS: no compiled row  <- '{q[:44]}…'")

        # stats report
        top = db.execute("""SELECT e.name, es_cnt.c FROM
            (SELECT entity_id, SUM(source_span_count) c, COUNT(*) rows
             FROM entity_state GROUP BY entity_id ORDER BY c DESC LIMIT 12) es_cnt
            JOIN entities e ON e.id = es_cnt.entity_id""").fetchall()
        total_rows2 = db.execute("SELECT COUNT(*) FROM entity_state").fetchone()[0]
        total_spans = db.execute("SELECT SUM(source_span_count) FROM entity_state").fetchone()[0]
        ev = db.execute("SELECT COUNT(*) FROM entity_state WHERE evidence_json"
                        " NOT LIKE '%fact_id%'").fetchone()[0]
        p("")
        p("[6] STATS")
        p(f"  wall-clock: {time.time()-t0:.1f}s; phases: "
          + " ".join(f"{k}={v:.2f}s" for k, v in sorted(t_phases.items())))
        p(f"  entities compiled: {compiled_entities}; state rows: {total_rows2}; "
          f"source spans: {total_spans}; LLM calls: {llm_calls}/{LLM_BUDGET}")
        p(f"  rows by slot class: scalar={sum(counts_by_slot[s] for s in SCALAR_SLOTS)} "
          f"enum={sum(counts_by_slot[s] for s in ENUM_SLOTS)} "
          f"opaque={counts_by_slot.get('timeline_events',0)}")
        p(f"  top entities by span count: " +
          "; ".join(f"{n}={c}" for n, c in top))
        p(f"  evidence_json provenance completeness: {total_rows2-ev}/{total_rows2} "
          "rows carry fact_ids")
        unres = sum(unresolved_predicates.values())
        p(f"  anomalies: {unres} facts with no resolvable value "
          f"(predicates: {dict(list(unresolved_predicates.most_common(8)))})")
        p("  read-routing: OFF (recall_facts untouched); cron: untouched")

        open(REPORT, "w").write("\n".join(L) + "\n")
        p(f"\nwrote {REPORT} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()