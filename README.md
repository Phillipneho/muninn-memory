# Muninn 🧠

_Bi-temporal persistent memory for AI agents. Memory to [Huginn](https://github.com/Phillipneho/huginn)'s Thought. Local-first: your agent's memory never leaves your machine._

Muninn ingests raw agent sessions, distills them into facts bound to entities, and curates itself — nightly decay, consolidation, and supersession, like a memory that sleeps. Where typical memory systems append and bloat, Muninn's store stays small, current, and auditable.

**This repository is a curated public snapshot of the Muninn engine.** The production deployment (private repo, personal data) runs on a homelab powering the Huginn agent daily. The project has two completed eras:

---

## 🏛 Muninn 1.0 — the retrieval engine (frozen)

The audited, finalised v1.0 state lives as the [`v1.0`](https://github.com/Phillipneho/muninn-local/tree/v1.0) branch/tag of the development repo. Highlights:

- **Bi-temporal facts:** every fact has a validity window (`valid_from`/`valid_until`) plus a supersession chain (`superseded_by`) — history is rewritten *and retained*
- **Hybrid retrieval:** FTS5 lexical + sqlite-vec dense (nomic-embed-text) + recency, with synonym expansion and never-empty degradation
- **Public write path:** `learn_fact()` handles entity resolution, same-subject+predicate supersession, and instant index updates — clients never hand-roll SQL
- **Self-curating pipeline:** ingestion → fact extraction → nightly sleep cycle (activation decay, consolidation, supersession, sync)
- **External gate:** the agent's self-modification proposals apply only below a size threshold; everything else queues for human approval

### Benchmark — scored BOTH ways, deliberately

We don't grade ourselves only on the community's rubric. Every eval runs two scorers side by side — the *universal* judge everyone benchmarks with, and our own *strict conveyance audit*:

**① Standard LoCoMo J-score** (Mem0/Zep unified judge protocol — directly comparable to the public pool):
- Official scope (cat1–4): **85.2%** — mid-80s band, where Mem0/Zep-class systems report
- Judge-model caveat documented: glm-5.3-flash, not GPT-4/5-class — so this number is *at a discount*

**② Muninn strict audit** (surface-form + fair-strict scorers — read-only audit fixtures):
- Surface-form: **58.3%** (cat1 46.1 / cat2 76.6 / cat3 21.9 / cat4 71.0 / cat5 36.8 abstention-correct)
- Fair strict: 43.6%

**Why publish both:** the standard judge grants partial credit for touching any gold item, accepts extra detail, and scores many declined adversarial answers as correct — the lenient track beats strict by +37.2pt overall and +42.6pt on adversarial cat5 alone. J-score measures whether the right fact was *touched*; strict audit measures whether the answer *conveys* it cleanly — no wrong extras, no confident guesses on unanswerable questions. Most systems publish only one. **Show your strict score next to your J-score.**

---

## 🚀 Muninn 2.0 — the compiled-state era (current)

2.0 adds a read path that never touches the 1.0 engine: facts compiled into deterministic entity-state slots, served by point lookup.

- **18-slot entity vocabulary** (employers, residences, children, pets, schooling, preferences, vehicles, …)
- **Supervised deterministic compiler:** facts → `entity_state` slots; scalar-with-history slots carry full supersession chains. Zero LLM calls; spot-check + idempotency-hash gates on every run; delta-select recompiles only entities whose facts changed
- **Feature-flagged read path:** `entity_state_read_enabled` gates a deterministic point-lookup ahead of the unchanged 3-lane retrieval. Hit → pinned, provenance-tagged compiled span. Miss → byte-identical fallthrough
- **Recall hardening:** abstention floor (`NO_CONFIDENT_MATCH` before guessing), first-person routing, word-token slot matching, synonym bridges — 19/19 regression matrix
- **Cockpit:** FastAPI/Jinja2/HTMX operations UI — entity cards, fact ledger with active/superseded chips, provenance drilldown to raw session snippets, timeline, query inspector

### Dual evaluation (rails frozen)

Validated against full LoCoMo with the scoring rails untouched (same judge, fixtures, prompts; only the feature flag differs), v2.0-flag-ON vs the v1.0-era baseline:

| Suite | n | Compiled-hit rate | Score delta (strict) | Point-lookup vs 3-lane retrieve |
|---|---|---|---|---|
| Gate2 | 60 | 7/60 | Cat1 +1, Cat2 flat | **3.6ms** vs 1,832ms |
| SliceC | 603 | 98/603 (pets 34, children 21, prefs 21) | net +1 | **2.8ms** (p90 4ms) vs 2,042ms |

**Verdict:** zero attributable regressions across 663 evaluated questions; flag isolation verified per-query; ~700× retrieval-side win on hits; aggregate score-neutral — the compiled layer covers state territory dense retrieval already solves. It's a cost/determinism/architecture win, banked honestly.

---

## Getting started

```bash
# initialise the store
./muninn.sh init

# ingest a session (any conversational content)
./muninn.sh ingest --content "..." --source discord --speakers '["Alex","Leo"]'

# hybrid recall (compiled front door when the flag is on)
./muninn.sh recall "query"

# nightly curation
./muninn.sh sleep-cycle

# supervised entity-state compile (delta)
python3 entity_compiler.py

# operations UI (read-only)
uvicorn cockpit.app:app --port 8337

# tests (10-case regression suite on a temp DB)
python3 tests/test_muninn.py
```

Storage: SQLite + FTS5 + sqlite-vec, WAL mode. Embeddings via local Ollama (`nomic-embed-text`) — no cloud dependency, no data exfiltration path.

## Architecture map

| File | Role |
|---|---|
| `muninn.py` | Core: ingestion, retrieval, fact extraction, consolidation, recall |
| `entity_compiler.py` | Phase B deterministic slot compiler (gated, idempotent) |
| `librarian.py` / `auto_ingest.py` | Scheduled curation pipelines |
| `cockpit/` | Read-only operations UI |
| `score_sf.py` | Strict surface-form scorer (audit fixture) |
| `tests/test_muninn.py` | Engine regression suite |
| `docs/` | Engineering findings (incl. the FK-orphan repair postmortem) |

## Status

- 1.0: **frozen** — benchmarks audited, scorers locked
- 2.0: **live in production** — dual-eval checkpoint banked; roadmap: temporal-qualifier read path ("what was my role *before* X"), prompt-capture for the query inspector, slice-C slot expansion

## License

Private project made public for benchmark transparency. All rights reserved by the author; code provided as-is for evaluation and reuse under the repository license.