#!/usr/bin/env python3
"""Backfill fact embeddings into fact_embeddings/fact_embedding_map.

Resumable: skips facts already in fact_embedding_map. Batched commits.
Run: nohup python3 backfill_embeddings.py > /tmp/embed_backfill.log 2>&1 &
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muninn  # noqa: E402


def main():
    t0 = time.time()
    while True:
        conn = muninn.get_db()
        rows = conn.execute(
            """SELECT f.id, e.name subj, f.predicate,
                      COALESCE(eo.name, f.object_value) obj
               FROM facts f JOIN entities e ON e.id=f.subject_entity_id
               LEFT JOIN entities eo ON eo.id=f.object_entity_id
               WHERE f.superseded_by IS NULL AND f.valid_until IS NULL
               AND f.id NOT IN (SELECT fact_id FROM fact_embedding_map)
               LIMIT 300""").fetchall()
        conn.close()
        if not rows:
            break
        bc = muninn.get_db()
        for r in rows:
            try:
                # natural-language rendering aligns better with question
                # phrasing than triple syntax (cos 0.78 vs 0.66 measured)
                emb = muninn.get_embedding(
                    f"{r['subj']}'s {r['predicate'].replace('_', ' ')} is {r['obj']}")
                cur = bc.execute(
                    "INSERT INTO fact_embeddings(embedding) VALUES (?)",
                    (emb.tobytes(),))
                bc.execute(
                    "INSERT INTO fact_embedding_map(rowid, fact_id) "
                    "VALUES (?, ?)", (cur.lastrowid, r["id"]))
            except Exception as e:
                print(f"fail id={r['id']}: {str(e)[:60]}", flush=True)
        bc.commit()
        bc.close()
        n = muninn.get_db().execute(
            "SELECT COUNT(*) FROM fact_embedding_map").fetchone()[0]
        print(f"{n} embedded ({time.time()-t0:.0f}s)", flush=True)
    print("BACKFILL COMPLETE", flush=True)


if __name__ == "__main__":
    main()