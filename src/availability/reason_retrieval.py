"""
Semantic retrieval over injury-report reason text.

The single strongest fact in the availability context is how often this player
played when previously listed with the *same* reason, and it is missing on most
rows because exact string matching within one player's history is sparse. There
are only ~1,400 distinct normalized reason strings across five seasons, and many
are near-duplicates: "Right Ankle; Sprain" and "Left Ankle; Sprain" describe the
same situation and "Injury/Illness-RightAnkle;Sprain" is the same string with
the spacing eaten by PDF extraction.

This embeds each distinct reason once, then for a query retrieves the most
similar reasons and aggregates the outcomes of *other players'* historical
listings under them. Two features come out:

- `reason_nbr_play_rate`: among similar-reason listings that were Questionable
  or Doubtful, the share where the player actually played. The dense version of
  the sparse same-reason fact.
- `reason_nbr_out_share`: among ALL similar-reason listings, the share that were
  Out. An empirical severity measure, replacing the hand-written keyword buckets
  in formula_scorer.classify_severity.

Point-in-time: a neighbour only contributes if its report was published strictly
before the query's report. Embeddings are of text alone and carry no outcome, so
they are computed once over the whole vocabulary; the outcomes attached to them
are what the date bound restricts.
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.availability.db import get_conn
from src.availability.retrieval import normalize_reason

logger = logging.getLogger(__name__)

EMBED_MODEL = "gemini-embedding-001"
FEATURE_COLUMNS = ["reason_nbr_play_rate", "reason_nbr_out_share", "reason_nbr_support", "reason_nbr_top_sim"]
SELF_FEATURE_COLUMNS = ["self_reason_nbr_play_rate", "self_reason_nbr_support"]

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS reason_embeddings (
    text_hash  TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    text       TEXT NOT NULL,
    vector     TEXT NOT NULL
);
"""


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()


class GeminiEmbedder:
    """Batch embedding with an on-disk cache, so a re-run costs nothing."""

    def __init__(self, model: str = EMBED_MODEL, db_path: str | None = None, batch: int = 64):
        import os

        from google import genai

        self._client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        self.model = model
        self.batch = batch
        self.db_path = db_path or "data/raw/availability.sqlite"

    def _conn(self) -> sqlite3.Connection:
        conn = get_conn(self.db_path)
        conn.executescript(_CACHE_SCHEMA)
        return conn

    def embed(self, texts: list[str]) -> dict[str, np.ndarray]:
        uniq = sorted(set(t for t in texts if t))
        conn = self._conn()
        keys = {t: _key(self.model, t) for t in uniq}
        out: dict[str, np.ndarray] = {}
        klist = list(keys.values())
        for i in range(0, len(klist), 500):
            chunk = klist[i : i + 500]
            q = f"SELECT text, vector FROM reason_embeddings WHERE text_hash IN ({','.join('?' * len(chunk))})"
            for t, v in conn.execute(q, chunk):
                out[t] = np.asarray(json.loads(v), dtype=np.float32)
        todo = [t for t in uniq if t not in out]
        logger.info(f"embeddings: {len(uniq)} distinct texts, {len(out)} cached, {len(todo)} to compute")

        for i in range(0, len(todo), self.batch):
            chunk = todo[i : i + self.batch]
            resp = self._client.models.embed_content(model=self.model, contents=chunk)
            for t, e in zip(chunk, resp.embeddings):
                vec = np.asarray(e.values, dtype=np.float32)
                out[t] = vec
                conn.execute(
                    "INSERT OR REPLACE INTO reason_embeddings VALUES (?,?,?,?)",
                    (keys[t], self.model, t, json.dumps([float(x) for x in vec])),
                )
            conn.commit()
            if (i // self.batch) % 5 == 0:
                logger.info(f"embedded {min(i + self.batch, len(todo))}/{len(todo)}")
        conn.close()
        return out


@dataclass
class RetrievalParams:
    """The knobs. Chosen on the tuning seasons only, never on the sealed test."""

    k: int = 25  # neighbours considered, by similarity rank
    min_sim: float = 0.80  # similarity floor; below this a neighbour is noise
    power: float = 3.0  # similarity weight exponent; higher trusts close matches more
    prior_weight: float = 10.0  # pseudo-counts of the global rate, shrinking thin evidence

    def tag(self) -> str:
        return f"k{self.k}_s{self.min_sim:g}_p{self.power:g}_w{self.prior_weight:g}"


class ReasonIndex:
    """Distinct reason vocabulary plus its pairwise cosine similarity."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.texts = sorted(vectors)
        self.pos = {t: i for i, t in enumerate(self.texts)}
        m = np.stack([vectors[t] for t in self.texts]).astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        self.sim = m @ m.T  # (V, V); V is ~1.4k so this is a few megabytes

    def ids(self, reasons: pd.Series) -> np.ndarray:
        return reasons.map(normalize_reason).map(lambda t: self.pos.get(t, -1)).to_numpy()


def build_reason_features(
    queries: pd.DataFrame,
    history: pd.DataFrame,
    index: ReasonIndex,
    params: RetrievalParams,
) -> pd.DataFrame:
    """Neighbour-aggregated features for every query row.

    `history` is every listing (Out included) with `report_date`, `status` and
    `played`. A row contributes to a query only when its report predates the
    query's report, which is enforced by walking both frames in date order and
    accumulating; nothing is recomputed per query.
    """
    v = len(index.texts)
    q_ids = index.ids(queries["reason"])
    h_ids = index.ids(history["reason"])

    h = history.assign(_rid=h_ids, _date=history["report_date"].to_numpy())
    h = h[h["_rid"] >= 0].sort_values("_date")
    h_uncertain = h["status"].isin(("Questionable", "Doubtful")).to_numpy()
    h_out = (h["status"] == "Out").to_numpy()
    h_played = h["played"].to_numpy()
    h_rid = h["_rid"].to_numpy()
    h_date = h["_date"].to_numpy()

    # Per-reason running totals, advanced as the query date moves forward.
    n_unc = np.zeros(v, dtype=np.float64)  # uncertain listings seen
    n_played = np.zeros(v, dtype=np.float64)  # ... that played
    n_all = np.zeros(v, dtype=np.float64)  # all listings seen
    n_out = np.zeros(v, dtype=np.float64)  # ... that were Out

    order = np.argsort(queries["report_date"].to_numpy(), kind="stable")
    rows = np.full((len(queries), 4), np.nan)
    cursor = 0
    for qi in order:
        qdate = queries["report_date"].iat[qi]
        while cursor < len(h_date) and h_date[cursor] < qdate:
            r = h_rid[cursor]
            n_all[r] += 1
            if h_out[cursor]:
                n_out[r] += 1
            if h_uncertain[cursor]:
                n_unc[r] += 1
                n_played[r] += h_played[cursor]
            cursor += 1

        rid = q_ids[qi]
        if rid < 0:
            continue
        sims = index.sim[rid]
        cand = np.where(sims >= params.min_sim)[0]
        if params.k and len(cand) > params.k:
            cand = cand[np.argsort(-sims[cand])[: params.k]]
        if len(cand) == 0:
            continue
        w = np.power(np.clip(sims[cand], 0.0, 1.0), params.power)

        unc_support = float((w * n_unc[cand]).sum())
        all_support = float((w * n_all[cand]).sum())
        global_play = n_played.sum() / n_unc.sum() if n_unc.sum() else np.nan
        global_out = n_out.sum() / n_all.sum() if n_all.sum() else np.nan

        pw = params.prior_weight
        if unc_support > 0 and not np.isnan(global_play):
            rows[qi, 0] = (float((w * n_played[cand]).sum()) + pw * global_play) / (unc_support + pw)
        if all_support > 0 and not np.isnan(global_out):
            rows[qi, 1] = (float((w * n_out[cand]).sum()) + pw * global_out) / (all_support + pw)
        rows[qi, 2] = unc_support
        rows[qi, 3] = float(sims[cand].max())

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, index=queries.index)


def build_self_reason_features(
    queries: pd.DataFrame,
    history: pd.DataFrame,
    index: ReasonIndex,
    params: RetrievalParams,
) -> pd.DataFrame:
    """Same retrieval, restricted to the player's OWN prior listings.

    Motivated by the diagnosis of the cross-player version: on the rows it was
    meant to rescue -- those with no exact same-reason history -- the
    cross-player rate correlates 0.009 with the outcome, against 0.127 for the
    sparse exact-match feature it was meant to densify. The predictive content
    of "same reason" turns out to be about the player (their role, their pain
    tolerance, how cautious their team is) rather than about the injury words.
    Averaging over other players discards exactly that and returns the league
    base rate.

    So match semantically but stay inside one player's history: "when THIS
    player was listed with anything like this, did he play". That widens the
    exact-match feature's coverage without crossing the boundary that carried
    the signal.
    """
    q_ids = index.ids(queries["reason"])
    h = history.assign(_rid=index.ids(history["reason"]))
    h = h[(h["_rid"] >= 0) & h["player_id"].notna()]
    h = h[h["status"].isin(("Questionable", "Doubtful"))].sort_values("report_date")

    by_player: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for pid, g in h.groupby("player_id"):
        by_player[int(pid)] = (
            g["report_date"].to_numpy(),
            g["_rid"].to_numpy(),
            g["played"].to_numpy(dtype=np.float64),
        )

    rows = np.full((len(queries), 2), np.nan)
    pids = queries["player_id"].to_numpy()
    qdates = queries["report_date"].to_numpy()
    for i in range(len(queries)):
        rid = q_ids[i]
        pid = pids[i]
        if rid < 0 or pd.isna(pid):
            continue
        rec = by_player.get(int(pid))
        if rec is None:
            continue
        dates, rids, played = rec
        prior = dates < qdates[i]
        if not prior.any():
            continue
        sims = index.sim[rid][rids[prior]]
        keep = sims >= params.min_sim
        if not keep.any():
            continue
        w = np.power(np.clip(sims[keep], 0.0, 1.0), params.power)
        support = float(w.sum())
        rows[i, 0] = float((w * played[prior][keep]).sum()) / support if support > 0 else np.nan
        rows[i, 1] = support

    return pd.DataFrame(rows, columns=SELF_FEATURE_COLUMNS, index=queries.index)
