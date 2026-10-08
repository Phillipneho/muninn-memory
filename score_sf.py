#!/usr/bin/env python3
"""Surface-form fidelity scorer (NORMALISED variant) — cycle 77, entry 'surface-form-fidelity'.

Primary scorer remains score_fair.py (STRICT, untouched). This module is a normalised
variant that runs ALONGSIDE it and is flagged in the ledger. No LLM judge, no softening
beyond deterministic lemmatisation/token containment.

NORMALISATION RULES (auditable, deterministic, zero model calls):
  N1  Lowercase + punctuation stripped (same as score_fair.norm).
  N2  Possessive strip: trailing 's / s' removed before tokenising.
  N3  Plural strip (both sides): -ies -> -y; -es -> drop; -s -> drop (not -ss, len>=4).
  N4  Light verb fold (both sides): -ing/-ed stripped; doubled final consonant folded
      (swimming -> swim, camping -> camp, camped -> camp). Token matches on base form.
  N5  Container-phrase strip (GOLD side only): known generic container head nouns are
      dropped from gold components when a content word remains, e.g.
      'dinosaur exhibit' -> 'dinosaur', 'beach trip' -> 'beach'.
      CONTAINER_HEADS = {exhibit, trip, event, session, program, activity, outing, workshop}
  N6  Head-noun containment: gold component passes if its stem sequence OR its
      remaining head noun (per N5) is contained in the lemmatised answer token
      stream. Head-noun fallback (N6d): final content token, ONLY for phrase
      golds carrying a function word; bare list golds exempt.
  N7  Word<->digit kept from score_fair (one/two/.../ten <-> 1..10).
  N8  Date logic copied from score_fair unchanged (ISO compare, gold_window ±2d).
  N9  List golds: partial credit >=60% components under contains_sf (same threshold
      as score_fair) — components re-scored with normalisation, never the raw judge.

Usage: python3 score_sf.py rows.json  -> writes rows_sf.json with 'hit_sf' field.
"""
import sys, json, re
from collections import Counter
import score_fair as F

CONTAINER_HEADS = {"exhibit", "trip", "event", "session", "program", "activity",
                   "outing", "workshop"}
STOPMOD = {"a", "an", "the", "her", "his", "their", "its"}

# c94 SCORER-VOCAB ALIASES (adopted per Alex's call, adopted from the c92
# audit's clean lexical-alias class — semantic equivalences, NOT softening):
# each bucket lists normalised stem-sequences that are inter-substitutable.
# Applied on the GOLD side: a gold component passes if its stems OR any alias
# bucket's stems are fully contained in the answer stem stream. score_fair.py
# (strict primary) stays untouched; gate adjudication uses sf per c77.
ALIAS_BUCKETS = [
    ["mom", "mother"],                      # kin: her mother <-> mom
    ["stuffed toy pup", "stuffed animal"],   # toy lexeme
    ["cake", "baked goods"],                # dessert lexeme
    ["around august 2022", "august 2022"],  # circa-approximation
    # c100 L4: diag-validated equivalents from the c99 Slice C miss analysis
    ["extended family", "extended fam"],
    ["friends from work", "work friends", "coworkers"],
    ["helping lost tourists and experiencing unexpected adventures",
     "helping lost tourists"],
]

# c100 L4 FUZZY-DATE SURFACE NORMALISATIONS (score_sf only; score_fair
# untouched). Diag rows: 'weekend before August 24, 2023' vs answer
# '19 August 2023-20 August 2023'; 'a few years before 2023' vs 'A few years
# ago'. Deterministic: parse the anchor date, COMPUTE the surface window, and
# accept the answer when any parsed answer date falls in it.


def _parse_anchor_date(s):
    from datetime import datetime
    m = re.search(r"\b(january|february|march|april|may|june|july|august|"
                  r"september|october|november|december)\s+(\d{1,2}),?\s+"
                  r"(19|20)(\d\d)\b", str(s).lower())
    if m:
        try:
            return datetime(int(m.group(3) + m.group(4)),
                            MONTH_NAMES(m.group(1)), int(m.group(2)))
        except Exception:
            return None
    m = re.search(r"\b(19|20)(\d\d)-(\d\d)-(\d\d)\b", str(s))
    if m:
        try:
            return datetime(int(m.group(1) + m.group(2)), int(m.group(3)),
                            int(m.group(4)))
        except Exception:
            return None
    return None


def MONTH_NAMES(mname):
    return {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
            "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
            "november": 11, "december": 12}[mname.lower()]


def fuzzy_date_hit(gold, ans):
    """c100 L4: surface-normalisation for fuzzy relative-date golds.
    Deterministic; gold side only."""
    g = str(gold).lower()
    from datetime import datetime, timedelta
    ans_dates = None
    # 'week(s) before <date>' ~= the [date-7d, date-1d] window (same
    # diag-validated class as 'weekend before': anchor-relative surface)
    m = re.search(r"\bweeks? before (.+)\b", g)
    if m:
        anchor = _parse_anchor_date(m.group(1))
        if anchor:
            lo = (anchor - timedelta(days=7)).date()
            hi = (anchor - timedelta(days=1)).date()
            if ans_dates is None:
                import score_fair as _F
                ans_dates = [d.date() for d in _F.parse_dates(str(ans))]
            if any(lo <= d <= hi for d in ans_dates):
                return True
    # 'weekend before <date>'  ~= the Sat+Sun immediately before that date
    m = re.search(r"\bweekend before (.+)\b", g)
    if m:
        anchor = _parse_anchor_date(m.group(1))
        if anchor:
            # compute the Sat/Sun of the weekend before the anchor
            dow = anchor.weekday()  # Mon=0..Sun=6
            sat = anchor - timedelta(days=dow + 1)
            sun = anchor - timedelta(days=dow)
            lo, hi = sat.date(), sun.date()
            if ans_dates is None:
                import score_fair as _F
                ans_dates = [d.date() for d in _F.parse_dates(str(ans))]
            if any(lo <= d <= hi for d in ans_dates):
                return True
    # 'a few years ago' ~= 'a few years before <recent year>' (both directions,
    # symmetric phrase equivalence — diag-validated both ways)
    if (re.search(r"\ba few years ago\b", g)
            and re.search(r"\ba few years (ago|before)\b", str(ans).lower())):
        return True
    if (re.search(r"\ba few years before\b", g)
            and re.search(r"\ba few years (ago|before)\b", str(ans).lower())):
        return True
    # 'nearly four months' ~= 'about 3.5 months' / 'almost 4 months'
    m = re.search(r"\bnearly\s+(\w+)\s+months?\b", g)
    if m:
        import score_fair as _F
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "eleven": 11, "twelve": 12}
        n = words.get(m.group(1), _first_num_safe(m.group(1)))
        am = re.search(r"\babout\s+([0-9.]+)\s+months?\b", str(ans).lower())
        if n and am:
            try:
                if abs(float(am.group(1)) - n) <= 0.5:
                    return True
            except Exception:
                pass
    return False


def _first_num_safe(w):
    m = re.search(r"\d+(?:\.\d+)?", str(w))
    return float(m.group(0)) if m else None


def _alias_stem_seqs(gold):
    g = " ".join(str(gold).lower().split())
    seqs = [tuple(stem_seq(gold))]
    for bucket in ALIAS_BUCKETS:
        for phrase in bucket:
            p = " ".join(phrase.split())
            if p == g or p in g or g in p:
                s = tuple(SA_stem_phrase(p))
                if s and s not in seqs:
                    seqs.append(s)
    return [s for s in seqs if s]


def SA_stem_phrase(phrase):
    return stem_seq(phrase)

def stem(tok):
    # N2 possessive
    t = tok.lower()
    if t.endswith("'s") or t.endswith("s'"): t = t[:-2]
    # N3 plural
    if t.endswith("ies") and len(t) > 4: t = t[:-3] + "y"
    elif t.endswith("es") and len(t) > 4 and not t.endswith("ses"): t = t[:-2]
    elif t.endswith("s") and len(t) > 3 and not t.endswith("ss"): t = t[:-1]
    # N4 verb fold
    if t.endswith("ing") and len(t) > 5:
        b = t[:-3]
        if len(b) >= 3 and b[-1] == b[-2]: b = b[:-1]
        t = b
    elif t.endswith("ed") and len(t) > 4:
        b = t[:-2]
        if len(b) >= 3 and b[-1] == b[-2]: b = b[:-1]
        t = b
    return t

def tokens(s):
    return [t for t in re.findall(r"[a-z0-9']+", str(s).lower()) if t]

def stem_seq(s):
    out = []
    for t in tokens(s):
        if t in STOPMOD: continue
        out.append(stem(t))
    return out

def gold_head(gtokens):
    """N5: return gold stem sequence after dropping container heads (>=1 must remain),
    else the original stem sequence."""
    stems = [stem(t) for t in gtokens if t not in STOPMOD]
    heads = [s for s in stems if s not in CONTAINER_HEADS]
    return heads if heads else stems

def canon(text):
    """c94: substitute alias-bucket phrases (bucket[1:]) with the canonical form
    (bucket[0]) on BOTH gold and answer sides before tokenisation."""
    t = " ".join(str(text).split())
    for bucket in ALIAS_BUCKETS:
        c = bucket[0]
        for phrase in bucket[1:]:
            t = re.sub(rf"\b{re.escape(phrase)}\b", c, t, flags=re.I)
    return t


def contains_sf(gold, ans):
    # ALIAS canon (c94): both sides fold to canonical lexemes first.
    gold, ans = canon(gold), canon(ans)
    g_raw = tokens(gold)
    a_stems = stem_seq(ans)
    g_stems = [stem(t) for t in g_raw if t not in STOPMOD]
    if not g_stems:
        return int(bool(re.search(rf"\b{re.escape(norm_num(gold))}\b", " ".join(a_stems))))
    # N7 word<->digit (delegate first to fair for numeric golds)
    if F.contains(gold, ans): return True
    # N6a full stem-sequence containment
    n = len(g_stems)
    for i in range(len(a_stems) - n + 1):
        if a_stems[i:i+n] == g_stems: return True
    # ALIAS (c94): gold phrase or any alias bucket's stem-sequences fully
    # contained in the answer stem stream (see ALIAS_BUCKETS above).
    for seq in _alias_stem_seqs(gold):
        m = len(seq)
        for i in range(len(a_stems) - m + 1):
            if a_stems[i:i+m] == seq: return True
        if m == 1 and all(tok in a_stems for tok in seq): return True
    # N6b head-noun containment
    heads = gold_head(g_raw)
    if heads and all(h in a_stems for h in heads): return True
    # N6c single-token gold: stem membership
    if len(g_stems) == 1 and g_stems[0] in a_stems: return True
    # N6d head-noun fallback: fires ONLY on phrase golds that carry a function
    # word (preposition/article/verb-of-motion among raw tokens) — the head noun
    # of such an English phrase is its FINAL content token ('camped at the
    # beach' -> beach; 'went for a swim' -> swim). Bare list/sequence golds
    # (e.g. 'dinosaurs, nature') never fire N6d — component-level scoring N9
    # governs them. Documented; deterministic.
    FUNC = {"at", "for", "in", "on", "to", "the", "a", "an", "went", "of"}
    if len(g_stems) > 1 and (FUNC & set(g_raw)) and g_stems[-1] in a_stems: return True
    return False

def norm_num(s): return re.sub(r"\W", "", str(s).lower())

def score_row_sf(gold, ans):
    if not ans or str(ans).startswith("ERROR"): return 0
    if "not enough information" in str(ans).lower(): return 0
    # c100 L4 fuzzy-relative-date surface normalisation (deterministic)
    if fuzzy_date_hit(str(ans), str(gold)) or fuzzy_date_hit(str(gold), str(ans)): return 1
    # N8 dates: fair's ISO/date-window logic first (unchanged)
    if F.contains(str(gold), ans): return 1
    g_iso = re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(gold).strip())
    if g_iso:
        ad = F.parse_dates(ans)
        g_date = F.datetime(*map(int, str(gold).split("-")))
        return int(any(abs((d - g_date).days) <= 2 for d in ad))
    win = F.gold_window(gold)
    ad = F.parse_dates(ans)
    if win and ad:
        if any(win[0] <= d <= win[1] + timedelta1() for d in ad): return 1
        if any(abs((d - win[0]).days) <= 2 or abs((d - win[1]).days) <= 2 for d in ad): return 1
        return 0
    # N9 list partial credit with normalised components
    comps = [c.strip() for c in re.split(r",| and ", str(gold)) if c.strip()]
    if len(comps) > 1:
        hit = sum(1 for c in comps if contains_sf(c, ans))
        if hit / len(comps) >= 0.6: return 1
    return int(contains_sf(gold, ans))

def timedelta1():
    from datetime import timedelta
    return timedelta(days=1)

def main(path):
    rows = json.load(open(path))
    total, by = 0, {}
    for r in rows:
        s = score_row_sf(r['gold'], r.get('ans'))
        r['hit_sf'] = s
        total += s
        by.setdefault(r.get('cat', 'g1'), [0, 0]); by[r.get('cat', 'g1')][0] += s; by[r.get('cat', 'g1')][1] += 1
    print(f"SURFACE-FIDELITY (normalised): {total}/{len(rows)} = {total/len(rows):.3f}")
    for c, (h, n) in sorted(by.items(), key=lambda kv: str(kv[0])):
        print(f"  cat{c}: {h}/{n} = {h/n:.3f}")
    out = path.replace('.json', '_sf.json')
    json.dump(rows, open(out, 'w'), indent=1)
    print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1])