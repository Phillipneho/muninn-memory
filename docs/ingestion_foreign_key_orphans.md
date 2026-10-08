# Technical Debt: Ingestion Foreign-Key Orphans in `facts`

**Filed:** 2026-10-07 (Phase C read-path canary work order, item 1)
**Status:** DOCUMENTED — no code fix applied this phase, per Alex's directive.
**Severity:** Medium (silent data loss at compile time; not data corruption)
**Discovered during:** MUNINN v2.0 Phase B shadow compile (`entity_compiler.py`, 2026-10-07)

## Summary

The ingestion pipeline writes fact rows whose `subject_entity_id` / `object_entity_id` reference entity ids that do **not exist** in the `entities` table. The compiler detects these and skips them safely rather than crashing on the FK violation, meaning those facts are never compiled into `entity_state`.

These facts are **skipped-safe, not corrupted**: the rows are intact in `facts` and remain visible to the v1.0 lexical/semantic retrieval lanes. Only compiler coverage is affected.

## Evidence (verified 2026-10-07 against live production DB)

From `shadow_compile_report.log`:

```
[1] DELTA-SELECT: 452 dirty entities (of 1830 total); 341 dangling subject_entity_ids skipped (anomaly)
[7] ... anomaly detail: 1221 fact rows across 341 subject_entity_ids reference entity ids
    that do not exist in entities (pre-existing ingestion integrity gap, documented 2026-10-07;
    compiler skipped them safely rather than crashing on the FK)
```

Independent FK-check reproduction (direct SQL on a read-only copy of production `muninn.db`):

| Metric | Value |
|---|---|
| facts: total rows | 6,046 |
| facts with dangling `subject_entity_id` | 1,221 rows / 341 distinct ids |
| facts with dangling `object_entity_id` | 1,127 rows / 704 distinct ids |
| entities total | 1,830 |
| entity_state rows (compiled) | 534 |
| `PRAGMA foreign_key_check` violations on facts→entities | matches the above (mass violation, all pre-existing) |

Note: Phase B report scoped the anomaly to the subject side (1,221/341); the object side is also affected (1,127/704) and was included in the compiler's safe skip.

## Provenance

- Backup taken before the Phase B compile: `backups/muninn-pre-entity-state-20261007-025844.db` (49.8 MB, sqlite3 `.backup` API, 2026-10-07T02:58 UTC) — the orphans are present in this backup, confirming they are pre-existing, not introduced by the compile.
- Post-run integrity check confirms compiler did not mutate immutable tables: facts / entities / raw_sessions UNCHANGED.
- Idempotency hash of entity_state content: `3a50f9e72a75ae68…` (converged).

## Impact

1. Facts referencing dangling entity ids are invisible to entity-centric compilation → those entities have no `entity_state` rows and slot-based read-path lookups miss them.
2. The dangling subject set is concentrated in the two legacy entity ids at the heart of the count (entity ids `2` and `3` are among the missing rows), so the damage skews toward early-session memories predating current entity normalisation.
3. No impact on v1.0 read correctness: the 3-lane retrieval queries `facts` directly and does not require FK integrity.
4. Risk of compounding: `DELTA-SELECT` reports each cycle as "dirty entities," but skipped subjects never join the delta, so orphan facts are permanently outside compiled state until repaired.

## Draft Repair Plan (for later write-path patching — DO NOT execute now)

Sketch only; to be spec'd properly before implementation:

1. **Triage the orphans.** For each dangling `subject_entity_id`/`object_entity_id`, pull the fact's `subject`/`object` text fields. Most likely the referenced entities were deleted or never normalised (the text names usually say who).
2. **Repair strategy — remap, don't invent.**
   - a. Fuzzy-match the text name against existing `entities` rows (exact → case-insensitive → token overlap); auto-link only unambiguous matches above a conservative similarity threshold.
   - b. Ambiguous/unknown references: create a new canonical entity row in a normalisation pass (or route to manual review queue with a `pn-review`-style tag).
   - c. Facts whose references are genuinely garbage (deleted test rows, resolved `[unresolved:*]` values): mark with `valid_until` rather than deleting — keep the audit trail.
3. **Then recompile:** after repair, run the Phase-B compiler in shadow mode again so previously skipped facts enter `entity_state`, and verify row count grows and spot-check passes.
4. **Prevent recurrence at the write path:** before `INSERT INTO facts`, resolve entity references to existing ids (or create entities atomically in the same transaction). Add a startup assertion / pragma: `PRAGMA foreign_keys=ON` for sessions that write to `facts`, plus a periodic integrity cron that alerts on new dangling rows.
5. **Rollback:** repairs touch `facts` only; recompile output is fully regenerable from `entity_state` (additive table), and `DROP TABLE entity_state; DROP INDEX idx_entity_state` remains the read-path rollback.

## Rollback for the read path (context)

`DROP TABLE entity_state; DROP INDEX idx_entity_state` — additive-only; immutable tables untouched.