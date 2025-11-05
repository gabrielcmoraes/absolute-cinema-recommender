import os, json, csv, re, traceback
from math import exp

import numpy as np
import pandas as pd
import boto3

# Settings
REGION = "us-east-2"
ENDPOINT_NAME = "recomendador-api"  # change if needed
SAGEMAKER_RT = boto3.client("sagemaker-runtime", region_name=REGION)

# Base hyperparameters (from sweep)
# (MMR fixed; adaptive blend according to the number of seeds)
LAMBDA_MMR   = 0.8   # diversity vs relevance
GENRE_WEIGHT = 0.6   # weight for genre vs item-item similarity inside MMR

BETA_GEN     = 0.15
GAMMA_YEAR   = 0.05
DELTA_POP    = 0.0
TAU_YEAR     = 8.0

# Light guardrails
PENALTY_NO_GENRE_OVERLAP = 0.08  # penalize candidates with 0 overlap to seeds
PENALTY_DOC_IF_NO_DOC    = 0.06  # penalize 'documentary' if no seeds include documentary

def get_adaptive_blend_weights(n_seeds: int, seed_has_doc: bool, seed_has_animation: bool, top_seed_genres: set[str]):
    """
    Adaptive weights:
      - 1 seed: strengthen genre/year (avoid off-tone results)
      - 2-3 seeds: moderate boost
      - 4+ seeds: use baseline sweep weights

    If seeds are mostly animation, increase genre weight, enable anchor bonus.
    Documentary guardrails are disabled if any seed is a documentary.
    """
    if n_seeds <= 1:
        beta_gen = max(BETA_GEN, 0.25)
        gamma_y  = max(GAMMA_YEAR, 0.10)
        w_anchor = 0.08
        penalty_no_overlap = 0.06
    elif n_seeds <= 3:
        beta_gen = max(BETA_GEN, 0.20)
        gamma_y  = max(GAMMA_YEAR, 0.08)
        w_anchor = 0.10
        penalty_no_overlap = 0.08
    else:
        beta_gen = BETA_GEN
        gamma_y  = GAMMA_YEAR
        w_anchor = 0.10
        penalty_no_overlap = 0.06

    if seed_has_animation:
        beta_gen = max(beta_gen, 0.28)
        gamma_y  = max(gamma_y,  0.10)
        w_anchor = max(w_anchor, 0.15)
        penalty_no_overlap = max(penalty_no_overlap, 0.10)

    return {
        "alpha_cf": 1.00,
        "beta_gen": beta_gen,
        "gamma_year": gamma_y,
        "delta_pop": DELTA_POP,
        "tau_year": TAU_YEAR,
        "penalty_no_overlap": 0.0 if seed_has_doc else penalty_no_overlap,
        "penalty_doc_if_no_doc": 0.0 if seed_has_doc else PENALTY_DOC_IF_NO_DOC,
        "w_anchor": w_anchor,
        "anchor_genres": set(top_seed_genres),
    }

# Layer data roots
_DATA_ROOTS = ["/opt/python", "/opt"]

def _find_in_opt(fname: str):
    for root in _DATA_ROOTS:
        p = os.path.join(root, fname)
        if os.path.exists(p):
            return p
    return None

# Artifacts in Lambda layer
RUNTIME_NPZ = _find_in_opt("svd_runtime_data.npz")
LINK_CSV    = _find_in_opt("link.csv")
MOVIES_CSV  = _find_in_opt("movie.csv")
print(f"[BOOT] NPZ path: {RUNTIME_NPZ} | LINK path: {LINK_CSV} | MOVIES path: {MOVIES_CSV}")

# CORS helpers
_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
}
def _resp(code, body): return {"statusCode": code, "headers": _CORS, "body": json.dumps(body)}
def _ok(data):          return _resp(200, {"recommendations": data})
def _bad(msg):          return _resp(400, {"error": msg})
def _err(msg):          return _resp(500, {"error": msg})

def _sanitize_recs(recs):
    out = []
    for r in (recs or []):
        title  = r.get("title", "")
        genres = r.get("genres", "")
        title  = title.strip() if isinstance(title, str) else str(title or "")
        genres = genres.strip() if isinstance(genres, str) else str(genres or "")
        s = r.get("affinity_score", 0.0)
        try:
            s = float(s)
            if not np.isfinite(s): s = 0.0
        except Exception:
            s = 0.0
        s = max(0.0, min(1.0, s))
        try:
            mid = int(r.get("movie_id", -1))
        except Exception:
            mid = -1
        out.append({"movie_id": mid, "title": title, "genres": genres, "affinity_score": s})
    return out

def _ok_sanitized(recs): return _ok(_sanitize_recs(recs))

# Artifacts loaded on cold start
Q = None                  # (n_movies, k)
Q_NORM = None             # (n_movies, k) normalized for cosine similarity
MOVIE_IDS = None          # array idx -> movieId
MOVIE_INDEX = {}          # dict movieId -> idx
POP = None                # optional popularity in [0..1]
TMDB_TO_ML = {}           # dict "tmdbId_str" -> movieId
MOVIE_META = {}           # movieId -> {"title": str, "genres": str}

def _safe_int_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    try:    return str(int(float(x)))
    except: return str(x).strip()

try:
    _npz = np.load(RUNTIME_NPZ, allow_pickle=True)
    Q = _npz["Q"]
    MOVIE_INDEX = _npz["movie_index"].item()
    if "movie_ids" in _npz.files:
        MOVIE_IDS = _npz["movie_ids"]
    else:
        MOVIE_IDS = np.full(Q.shape[0], -1, dtype=np.int64)
        for mid, idx in MOVIE_INDEX.items():
            if 0 <= idx < Q.shape[0]:
                MOVIE_IDS[idx] = int(mid)
    POP = _npz["pop"] if "pop" in _npz.files else None
    Q_NORM = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    print(f"[BOOT] Q={Q.shape}, movie_index={len(MOVIE_INDEX)}, movie_ids={len(MOVIE_IDS)}, pop={'ok' if POP is not None else 'absent'}")

    if LINK_CSV:
        with open(LINK_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                tmdb = _safe_int_str(r.get("tmdbId"))
                mid  = r.get("movieId")
                if tmdb and mid:
                    TMDB_TO_ML[tmdb] = int(mid)
        print(f"[BOOT] link.csv ok. {len(TMDB_TO_ML)} TMDB→MovieLens pairs.")
    else:
        print("[BOOT] link.csv not found (ok if the frontend already sends MovieLens IDs).")

except Exception as e:
    print(f"[BOOT] ERROR loading runtime npz/link.csv: {e}")
    traceback.print_exc()
    Q, Q_NORM, MOVIE_IDS, MOVIE_INDEX, POP, TMDB_TO_ML = None, None, None, {}, None, {}

# Movie metadata for titles/genres
if MOVIES_CSV and os.path.exists(MOVIES_CSV):
    try:
        mdf = pd.read_csv(MOVIES_CSV, usecols=["movieId","title","genres"])
        for _, row in mdf.iterrows():
            try:
                mid = int(row["movieId"])
                MOVIE_META[mid] = {
                    "title":  str(row.get("title", "") or ""),
                    "genres": str(row.get("genres", "") or "")
                }
            except Exception:
                pass
        print(f"[BOOT] movie.csv ok. {len(MOVIE_META)} titles/genres loaded.")
    except Exception as e:
        print(f"[BOOT] movie.csv error: {e}")
else:
    print("[BOOT] movie.csv not found (fallback without local metadata).")

# Blend and content helpers
GENRE_SPLIT_RE = re.compile(r"\s*\|\s*")
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

def _top_seed_genres(seed_genres: set[str], top_k: int = 2) -> set[str]:
    """
    Pick up to top_k 'anchor' genres from seeds.
    Current policy: prioritize animation/anime/fantasy if present.
    """
    priority = [g.lower() for g in seed_genres]
    priority = sorted(priority, key=lambda x: (x not in {"animation", "anime", "fantasy"}, x))
    return set(priority[:top_k])

def _parse_genres(s: str):
    if not s: return set()
    return set(
        g.strip().lower()
        for g in GENRE_SPLIT_RE.split(s)
        if g.strip() and g.strip().lower() != "(no genres listed)"
    )

def _parse_year_from_title(title: str):
    if not title: return None
    m = YEAR_RE.search(title.strip())
    if not m: return None
    try:
        y = int(m.group(1))
        return y if 1900 <= y <= 2100 else None
    except: return None

def _jaccard(a: set, b: set):
    if not a or not b: return 0.0
    inter = len(a & b)
    if inter == 0: return 0.0
    return inter / float(len(a | b))

def _seed_meta_profile(selected_ml):
    all_genres = set(); years = []
    for mid in selected_ml:
        m = MOVIE_META.get(mid, {})
        gs = _parse_genres(m.get("genres"))
        if gs: all_genres |= gs
        y = _parse_year_from_title(m.get("title"))
        if y: years.append(y)
    mean_year = (sum(years)/len(years)) if years else None
    return all_genres, mean_year

def _blend_score(cos_score, genre_sim, year_c, year_seed_mean, pop_score, W):
    """
    Combine: CF (cosine) + genre + year + popularity, with adaptive weights W
    """
    a   = float(W["alpha_cf"])
    b   = float(W["beta_gen"])
    g   = float(W["gamma_year"])
    d   = float(W["delta_pop"])
    tau = float(W["tau_year"])
    term_cf = a * float(cos_score)
    term_g  = b * float(genre_sim)
    term_y  = g * (exp(-abs(year_c - year_seed_mean)/tau) if (year_c is not None and year_seed_mean is not None) else 0.0)
    term_p  = d * float(pop_score or 0.0)
    return term_cf + term_g + term_y + term_p

# MMR for diversity
def _mmr_select(cand_info, top_n, lambda_mmr=LAMBDA_MMR, genre_weight=GENRE_WEIGHT):
    """
    cand_info: list of dicts with fields:
      - 'mid': int
      - 'idx': int (index in Q / Q_NORM)
      - 'title': str
      - 'genres': str
      - 'genres_set': set[str]
      - 'cos_score': float
      - 'blended': float
    """
    lam = float(lambda_mmr)
    gw  = float(genre_weight)

    selected = []
    pool = cand_info[:]

    for c in pool:
        if not np.isfinite(c["blended"]): c["blended"] = -1.0

    while pool and len(selected) < top_n:
        if not selected:
            best = max(pool, key=lambda x: x["blended"])
            selected.append(best)
            pool.remove(best)
            continue

        best_item = None
        best_mmr  = -1e9

        for c in pool:
            max_sim = 0.0
            for s in selected:
                gsim = _jaccard(c["genres_set"], s["genres_set"])
                cos_items = float(Q_NORM[c["idx"]] @ Q_NORM[s["idx"]])
                cos_items = max(0.0, cos_items)
                sim = gw * gsim + (1.0 - gw) * cos_items
                if sim > max_sim:
                    max_sim = sim

            mmr_val = lam * c["blended"] - (1.0 - lam) * max_sim
            if mmr_val > best_mmr:
                best_mmr = mmr_val
                best_item = c

        selected.append(best_item)
        pool.remove(best_item)

    return selected

# SageMaker retrieval
def _call_twotower(movie_ids, top_k=200):
    payload = {"movie_ids": list(map(int, movie_ids)), "top_n": int(top_k)}
    resp = SAGEMAKER_RT.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    data = json.loads(resp["Body"].read().decode("utf-8"))
    return data.get("recommendations", [])

# Fallbacks
def _fallback_svd(selected_ml, top_n=5):
    if Q is None or Q_NORM is None or not MOVIE_INDEX or MOVIE_IDS is None:
        return []

    sel_idx = [MOVIE_INDEX[m] for m in selected_ml if m in MOVIE_INDEX]
    if not sel_idx:
        return []

    u = Q[sel_idx].mean(axis=0); u = u / (np.linalg.norm(u) + 1e-12)
    scores = Q_NORM @ u

    seed_genres, seed_year_mean = _seed_meta_profile(selected_ml)
    seed_g_lower = {g.lower() for g in seed_genres}
    seed_has_doc = ("documentary" in seed_g_lower)
    seed_has_animation = any(g in seed_g_lower for g in {"animation", "anime"})
    anchor_genres = _top_seed_genres(seed_g_lower, top_k=2)

    W = get_adaptive_blend_weights(
        n_seeds=len(selected_ml),
        seed_has_doc=seed_has_doc,
        seed_has_animation=seed_has_animation,
        top_seed_genres=anchor_genres
    )

    seen = set(selected_ml)
    take = min(max(top_n * 50, 500), len(scores))
    top_idx_pool = np.argpartition(-scores, kth=take-1)[:take]

    items = []
    for i in top_idx_pool:
        mid = int(MOVIE_IDS[i])
        if mid in seen:
            continue
        meta = MOVIE_META.get(mid, {})
        gset_lower = _parse_genres(meta.get("genres"))
        gsim = _jaccard(seed_g_lower, gset_lower)
        y = _parse_year_from_title(meta.get("title"))
        pop_score = float(POP[i]) if POP is not None else 0.0

        blended = _blend_score(scores[i], gsim, y, seed_year_mean, pop_score, W)

        if W["w_anchor"] > 0 and W["anchor_genres"]:
            hits = len(gset_lower & W["anchor_genres"])
            anchor_hit_ratio = hits / max(1, len(W["anchor_genres"]))
            blended += float(W["w_anchor"]) * anchor_hit_ratio

        if seed_g_lower and gsim == 0.0:
            blended -= float(W["penalty_no_overlap"])
        if ("documentary" in gset_lower) and not seed_has_doc:
            blended -= float(W["penalty_doc_if_no_doc"])

        items.append({
            "mid": mid,
            "title": meta.get("title", ""),
            "genres": meta.get("genres", ""),
            "blended": float(blended)
        })

    items.sort(key=lambda x: x["blended"], reverse=True)
    recs = []
    for it in items[:top_n]:
        aff = max(0.0, min(1.0, (it["blended"] + 1.0) / 2.0))
        recs.append({
            "movie_id": it["mid"],
            "title": it["title"],
            "genres": it["genres"],
            "affinity_score": float(aff),
        })
    return recs

def _fallback_popularity(top_n=5):
    if POP is None or MOVIE_IDS is None:
        return []
    take = min(top_n, len(POP))
    idx = np.argpartition(-POP, kth=take-1)[:take]
    idx = idx[np.argsort(-POP[idx])]
    recs = []
    for i in idx:
        meta = MOVIE_META.get(int(MOVIE_IDS[i]), {})
        recs.append({
            "movie_id": int(MOVIE_IDS[i]),
            "title": meta.get("title",""),
            "genres": meta.get("genres",""),
            "affinity_score": float(max(0.0, min(1.0, POP[i]))),
        })
    return recs

# Main handler
def lambda_handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return _resp(200, {"ok": True})

    if Q is None or Q_NORM is None or MOVIE_IDS is None or not MOVIE_INDEX:
        return _err("Internal error: re-ranking artifacts not loaded.")

    try:
        raw_body = event.get("body")
        if isinstance(raw_body, str):
            try: body = json.loads(raw_body or "{}")
            except: body = {}
        elif isinstance(raw_body, dict):
            body = raw_body
        elif isinstance(event, dict) and ("tmdb_ids" in event or "top_n" in event):
            body = event
        else:
            body = {}

        tmdb_ids = body.get("tmdb_ids") or []
        top_n = max(1, min(int(body.get("top_n", 5)), 50))

        # Map TMDB -> MovieLens (or accept ML IDs directly)
        selected_ml = []
        for t in tmdb_ids:
            key = str(t)
            if TMDB_TO_ML:
                m = TMDB_TO_ML.get(key)
                if m is not None:
                    selected_ml.append(int(m))
            else:
                try: selected_ml.append(int(key))
                except: pass

        # deduplicate, keep order
        seen = set()
        selected_ml = [x for x in selected_ml if not (x in seen or seen.add(x))]
        if not selected_ml:
            return _bad("None of the provided IDs could be mapped to MovieLens.")

        # Retrieval
        try:
            cands = _call_twotower(selected_ml, top_k=200)
        except Exception as e:
            print(f"[WARN] SageMaker unavailable: {e}")
            cands = []

        if not cands:
            recs = _fallback_svd(selected_ml, top_n=top_n)
            return _ok_sanitized(recs)

        # Normalize candidate metadata
        cand_ml_ids, meta = [], {}
        for c in cands:
            mid = c.get("movie_id") or c.get("movieId")
            if mid is None: continue
            mid = int(mid)
            cand_ml_ids.append(mid)
            meta[mid] = {
                "title":  c.get("title")  or MOVIE_META.get(mid, {}).get("title", ""),
                "genres": c.get("genres") or MOVIE_META.get(mid, {}).get("genres",""),
            }

        if not cand_ml_ids:
            recs = _fallback_svd(selected_ml, top_n=top_n)
            return _ok_sanitized(recs)

        sel_idx = [MOVIE_INDEX[m] for m in selected_ml if m in MOVIE_INDEX]
        if not sel_idx:
            pop_recs = _fallback_popularity(top_n=top_n)
            return _ok_sanitized(pop_recs) if pop_recs else _bad("No valid IDs for re-ranking.")

        u = Q[sel_idx].mean(axis=0); u = u / (np.linalg.norm(u) + 1e-12)
        cand_idx = [MOVIE_INDEX[m] for m in cand_ml_ids if m in MOVIE_INDEX]
        if not cand_idx:
            recs = _fallback_svd(selected_ml, top_n=top_n)
            return _ok_sanitized(recs)

        base_cos = Q_NORM[cand_idx] @ u
        seed_genres, seed_year_mean = _seed_meta_profile(selected_ml)

        seed_g_lower = {g.lower() for g in seed_genres}
        seed_has_doc = ("documentary" in seed_g_lower)
        seed_has_animation = any(g in seed_g_lower for g in {"animation", "anime"})
        anchor_genres = _top_seed_genres(seed_g_lower, top_k=2)

        W = get_adaptive_blend_weights(
            n_seeds=len(selected_ml),
            seed_has_doc=seed_has_doc,
            seed_has_animation=seed_has_animation,
            top_seed_genres=anchor_genres
        )

        items = []
        for local_i, idx in enumerate(cand_idx):
            mid = int(MOVIE_IDS[idx])
            m = meta.get(mid, {})
            gset = _parse_genres(m.get("genres"))
            gset_lower = {g.lower() for g in gset}
            gsim = _jaccard(seed_g_lower, gset_lower)
            y    = _parse_year_from_title(m.get("title"))
            pop_score = float(POP[idx]) if POP is not None else 0.0

            blended = _blend_score(base_cos[local_i], gsim, y, seed_year_mean, pop_score, W)

            if W["w_anchor"] > 0 and W["anchor_genres"]:
                hits = len(gset_lower & W["anchor_genres"])
                anchor_hit_ratio = hits / max(1, len(W["anchor_genres"]))
                blended += float(W["w_anchor"]) * anchor_hit_ratio

            if seed_g_lower and gsim == 0.0:
                blended -= float(W["penalty_no_overlap"])
            if ("documentary" in gset_lower) and not seed_has_doc:
                blended -= float(W["penalty_doc_if_no_doc"])

            items.append({
                "mid": mid,
                "idx": idx,
                "title": (m.get("title") or "").strip(),
                "genres": (m.get("genres") or "").strip(),
                "genres_set": gset_lower,
                "cos_score": float(base_cos[local_i]),
                "blended": float(blended),
            })

        chosen = _mmr_select(
            items, top_n=top_n,
            lambda_mmr=LAMBDA_MMR,
            genre_weight=GENRE_WEIGHT
        )

        recs = []
        for c in chosen:
            aff = max(0.0, min(1.0, (c["blended"] + 1.0) / 2.0))
            recs.append({
                "movie_id": c["mid"],
                "title": c["title"],
                "genres": c["genres"],
                "affinity_score": float(aff),
            })

        return _ok_sanitized(recs)

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return _err("Internal server error.")
