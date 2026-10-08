"""Muninn Cockpit — FastAPI app. Read-only. Localhost only."""
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import queries

TEMPLATES = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent / "templates"))

app = FastAPI(title="Muninn Cockpit", docs_url=None, redoc_url=None)


def render(request: Request, name: str, **ctx):
    ctx["request"] = request
    return TEMPLATES.TemplateResponse(request, name, ctx)


@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse("/entities")


@app.get("/health", response_class=HTMLResponse)
def health(request: Request):
    v = queries.health_vitals()
    flag = next((c["value"] for c in v["config"]
                 if c["key"] == "entity_state_read_enabled"), "0")
    return render(request, "health.html", v=v, flag=flag,
                  title="Health & Integrity")


@app.get("/entities", response_class=HTMLResponse)
def entities(request: Request, q: str = ""):
    return render(request, "entities.html", rows=queries.entity_list(q),
                  q=q, title="Entity Explorer")


@app.get("/entities/{entity_id}", response_class=HTMLResponse)
def entity_detail(request: Request, entity_id: int):
    d = queries.entity_detail(entity_id)
    if not d:
        return render(request, "base.html", title="Not found")
    # parse slot state_json for display
    slots = []
    for s in d["slots"]:
        try:
            state = json.loads(s["state_json"])
        except Exception:
            state = {}
        slot = {"attribute": s["attribute"], "compiled_at": s["compiled_at"],
                "source_span_count": s["source_span_count"],
                "evidence_json": s["evidence_json"]}
        if state.get("current") not in (None, ""):
            slot["kind"] = "SCALAR"
            slot["current"] = state["current"]
            slot["superseded"] = state.get("superseded", [])
        else:
            slot["kind"] = "ENUM"
            slot["slot_items"] = state.get("items", [])
            slot["count"] = state.get("distinct_count", len(slot["slot_items"]))
        slots.append(slot)
    eid = d["entity"]["id"]
    tl = queries.timeline_events(entity_id=eid, limit=30)
    return render(request, "entity_detail.html", e=d["entity"], slots=slots,
                  facts=d["facts"], timeline=tl, title=d["entity"]["name"])


@app.get("/frag/provenance/{attribute}/{entity_id}", response_class=HTMLResponse)
def frag_provenance(request: Request, attribute: str, entity_id: int):
    facts = queries.slot_provenance(entity_id, attribute)
    return TEMPLATES.TemplateResponse(request, "_provenance.html",
                                      {"request": request, "facts": facts})


@app.get("/frag/entity-facts/{entity_id}", response_class=HTMLResponse)
def frag_entity_facts(request: Request, entity_id: int, show: str = "active",
                      offset: int = 0):
    d = queries.entity_facts_filtered(entity_id, show=show, offset=offset)
    return TEMPLATES.TemplateResponse(request, "_fact_rows.html",
                                      {"request": request,
                                       "facts": d["rows"],
                                       "sup_map": d["sup_map"],
                                       "total": d["total"],
                                       "show": show, "offset": offset,
                                       "entity_id": entity_id})


@app.get("/timeline", response_class=HTMLResponse)
def timeline(request: Request, entity: int = None, days: int = 90):
    ev = queries.timeline_events(entity_id=entity, days=days)
    entity_name = None
    if entity:
        rows = queries.entity_list("")
        entity_name = next((r["name"] for r in rows if r["id"] == entity), None)
    return render(request, "timeline.html", ev=ev, entity=entity,
                  entity_name=entity_name, title="Timeline Stream")


@app.get("/queries", response_class=HTMLResponse)
def queries_view(request: Request):
    rows = queries.query_log_tail(100)
    from .db import connect_ro, get_config
    flag = get_config(connect_ro(), "entity_state_read_enabled", "0")
    return render(request, "queries.html", rows=rows, flag=flag,
                  title="Query & Audit Log")


@app.get("/queries/{qid}", response_class=HTMLResponse)
def query_inspector(request: Request, qid: int):
    d = queries.query_detail(qid)
    if not d:
        return TEMPLATES.TemplateResponse(request, "_query_detail.html",
                                          {"request": request, "row": None,
                                           "snapshot": {}})
    import json as _json
    try:
        snap = _json.loads(d["config_snapshot"]) if d["config_snapshot"] else {}
    except Exception:
        snap = {}
    if not isinstance(snap, dict):
        snap = {"raw": str(snap)[:200]}
    return TEMPLATES.TemplateResponse(request, "_query_detail.html",
                                      {"request": request, "row": d,
                                       "snapshot": snap})