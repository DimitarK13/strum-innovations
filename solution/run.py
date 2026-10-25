#!/usr/bin/env python
"""
Possession labeller for both halves  (H / A / D, X where not graded)

Run from the repo root (needs data/ in place and `pip install kloppy`):

    python possession_match.py                       # both halves -> evaluation/results/full_match.csv
    python possession_match.py --half 1              # first half only
    python possession_match.py --half 2              # second half only
    python possession_match.py --minutes 5           # first 5 minutes after each kick-off (quick test)
    python possession_match.py --diagnose            # compare cues / set-piece tags with the public GT

Pipeline (per half)
  1. Load ATD + Smart Tagging and put both on the video-frame index.
  2. Per-frame evidence
       team evidence : which team is nearer the ball, and how decisively
       dead evidence : ball out of bounds / ball still / players slow / set-piece window
  3. 3-state Viterbi (H, A, D) with switch penalties -> sticky possessions, no flicker.
       A frame without evidence (ball missing, ball in the air) simply keeps the last state.
  4. Second pass: the team that takes a set piece gets a short prior after the restart.
  5. Write a 199,319-row CSV. Everything outside the graded halves is X.

Second-half trap (FAQ.md): the half is graded from frame 110,808 but kicks off at 113,742.
Those 2,934 frames are all D, so they are written as D and the model only runs from the kick-off.
Nothing here depends on attacking direction, so the change of ends needs no handling.
"""
import argparse
import os
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

N_FRAMES = 199_319
FPS = 25
PITCH_L, PITCH_W = 105.3, 68.0           # metres (games.py value, measured from corrected tracking)
HALVES = {1: (11_185, 90_651), 2: (110_808, 190_197)}   # graded frames (DATA.md)
HALF2_KICKOFF = 113_742                  # FAQ.md: frames 110,808..113,741 are graded and all D
GT_PUBLIC = "evaluation/gt/BAR-ATH_possession_gt_public.csv"

H, A, D = 0, 1, 2                        # state indices
LABELS = np.array(["H", "A", "D"])


# --------------------------------------------------------------------------------------
# Tunables. These are starting values, not fitted ones. Do not tune them against the public
# 5 minutes (EVALUATION.md: 17 possessions is too few). Use --diagnose to sanity-check cues.
# --------------------------------------------------------------------------------------
@dataclass
class Cfg:
    # ball handling
    max_gap: int = 37                    # interpolate ball gaps up to 1.5 s, leave longer ones NaN
    # team evidence
    margin_scale: float = 2.0            # m: how decisive "nearer" must be (tanh scale)
    prox_scale: float = 5.0              # m: evidence fades when nobody is near the ball
    dist_cap: float = 30.0
    # dead-ball cues (thresholds are starting values: read them off --diagnose / --align)
    oob_tol: float = 0.3                 # m beyond the lines before the ball counts as out
    still_win: int = FPS                 # 1 s window
    still_m: float = 1.5                 # ball moved less than this over the window -> still
    slow_win: int = FPS
    slow_speed: float = 1.2              # m/s, median speed over tracked slots
    idle_win: int = 3 * FPS              # median player displacement over 3 s ...
    idle_m: float = 3.5                  # ... below this (m): real percentiles, dead p50 3.1 vs live p50 5.5
    w_oob: float = 1.5                   # rare but specific
    w_slow: float = 1.0                  # best single cue on real data (54% of dead vs 10% of live)
    w_idle: float = 0.6
    w_still: float = 0.6                 # weak on real data (20% vs 10%)
    w_gone: float = 0.6                  # ball lost > max_gap: 34% dead vs 6% live in the public window, but ~25%
                                         # of frames far from any tag in --align, so a value between the two
    w_sp: float = 1.0                    # set-piece window: sensitises, cannot trigger D alone
    w_core: float = 0.8                  # +-core_half around the tag start: the dip in player speed is deepest there
    core_half: int = 2 * FPS
    dead_bias: float = 1.2               # dead score must beat this (and the team evidence)
    sp_pre: int = 12 * FPS               # --align over ~60 tags (no GT): speed starts dropping ~12 s before the
    sp_post: int = 8 * FPS               # tag start and has recovered ~8 s after it
    # Viterbi penalties (in evidence units)
    # real-data lesson: 60 was far too high (6 possessions where the truth has 17).
    switch_cost: float = 20.0            # H<->A  (sweep: 16 possessions on the public window, truth 17)
    # INVARIANT: enter_dead + exit_dead must exceed switch_cost. Otherwise H->D->A is cheaper than
    # H->A and the decoder invents a short fake D at every live turnover.
    enter_dead: float = 16.0
    exit_dead: float = 16.0
    kickoff_live: int = 3 * FPS          # both halves start at the kick-off: live at the start
    # tag phases (BUILD UP, PROGRESSION, DEFENDING IN ..., ...): soft evidence about who has the ball
    phase_w: float = 0.25                # 0 disables. Per-frame weight of the phase vote (team evidence is ~0.3-0.8)
    phase_min_n: int = 8                 # a code needs this many usable intervals in the half ...
    phase_min_frames: int = 50           # ... each with this many alive frames (2 s) ...
    phase_z: float = 2.5                 # ... and a sign test over intervals with |z| >= this (false use costs more than missing it)
    phase_trim: int = 3 * FPS            # ignore the first/last 3 s of each phase: edges are blurry, and a sticky
                                         # edge makes the decoder merge real possession changes
    # restart prior (pass 2)
    restart_match: int = 15 * FPS        # tag start must be this close to a detected dead run
    min_dead_for_prior: int = 2 * FPS
    prior_len: int = 3 * FPS
    prior_w: float = 0.5


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def team_slot_ids(meta):
    """Slot ids per team from the kloppy metadata (each slot belongs to one team)."""
    home = away = None
    for team in meta.teams:
        ground = str(getattr(team.ground, "value", team.ground)).lower()
        ids = [str(p.player_id) for p in team.players]
        if "home" in ground:
            home = ids
        elif "away" in ground:
            away = ids
    if not home or not away:
        raise RuntimeError("Could not read home/away slots from metadata.teams: %r" % (meta.teams,))
    return home, away


def load_atd(ranges):
    """For each (lo, hi) frame range return ball (T,2), home (T,K,2), away (T,K,2) in metres and
    speeds (T, Kh+Ka) in m/s. The tracking file is read once."""
    from kloppy import metrica
    from games import asset_paths

    P = asset_paths()
    ds = metrica.load_tracking_epts(meta_data=P["atd_meta"], raw_data=P["atd_raw"])
    home_ids, away_ids = team_slot_ids(ds.metadata)
    df = ds.to_df()
    df["frame"] = df["frame_id"].astype(int) - 1                  # ATD frame_id is 1-based
    df = df.set_index("frame")

    def present(ids):
        return [i for i in ids if "%s_x" % i in df.columns and "%s_y" % i in df.columns]

    home_ids, away_ids = present(home_ids), present(away_ids)
    if not home_ids or not away_ids:
        raise RuntimeError("No slot columns found. Columns start with: %s" % list(df.columns[:12]))
    scols = ["%s_s" % i for i in home_ids + away_ids if "%s_s" % i in df.columns]

    def xy(sub, ids):
        return np.stack([sub[["%s_x" % i for i in ids]].to_numpy(float) * PITCH_L,
                         sub[["%s_y" % i for i in ids]].to_numpy(float) * PITCH_W], axis=-1)

    out = []
    for lo, hi in ranges:
        sub = df.reindex(np.arange(lo, hi + 1))                   # NaN rows if a frame is absent
        ball = sub[["ball_x", "ball_y"]].to_numpy(float) * np.array([PITCH_L, PITCH_W])
        out.append((ball, xy(sub, home_ids), xy(sub, away_ids), sub[scols].to_numpy(float)))
    return out


def team_to_state(x):
    """Smart Tagging 'Team' value -> H / A / NaN (N/A, missing)."""
    if not isinstance(x, str):
        return np.nan
    t = x.lower()
    if "barcel" in t:
        return H
    if "club" in t or "athl" in t or "atl" in t:
        return A
    return np.nan


def load_tags():
    """Smart Tagging as a DataFrame with video-frame intervals f0, f1 (inclusive-ish)."""
    from kloppy import sportscode
    from games import asset_paths

    ev = sportscode.load(asset_paths()["smart_tagging"]).to_df().copy()
    ev["f0"] = (ev["timestamp"].dt.total_seconds() * FPS).round().astype(int)
    ev["f1"] = (ev["end_timestamp"].dt.total_seconds() * FPS).round().astype(int)
    ev["code"] = ev["code"].astype(str).str.upper()
    for c in ("Team", "Type", "Half"):
        if c not in ev.columns:
            ev[c] = np.nan
    ev["team_state"] = ev["Team"].map(team_to_state)
    return ev


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
def smooth_ball(ball, cfg):
    """Interpolate short gaps only, then a 5-frame median. Long gaps stay NaN (= unknown)."""
    out = np.full_like(ball, np.nan)
    for k in range(2):
        s = pd.Series(ball[:, k])
        isn = s.isna()
        run_id = isn.ne(isn.shift()).cumsum()
        gap_len = isn.groupby(run_id).transform("sum")
        filled = s.interpolate(limit_area="inside").where(~isn | (gap_len <= cfg.max_gap))
        sm = filled.rolling(5, center=True, min_periods=1).median().where(filled.notna())
        out[:, k] = sm.to_numpy()
    return out


def nearest_dist(ball, players):
    d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    return np.where(np.isnan(d), np.inf, d).min(axis=1)           # inf if nobody / no ball


def team_evidence(ball, home, away, cfg):
    """>0 favours H, <0 favours A, ~0 when the ball is missing, far from everyone, or contested."""
    have = ~np.isnan(ball[:, 0])
    dH = np.minimum(nearest_dist(ball, home), cfg.dist_cap)
    dA = np.minimum(nearest_dist(ball, away), cfg.dist_cap)
    e = np.tanh((dA - dH) / cfg.margin_scale) * np.exp(-np.minimum(dH, dA) / cfg.prox_scale)
    return np.where(have, e, 0.0)


def dead_cues(ball, pos, speeds, sp_mask, core_mask, cfg):
    """Boolean cues + continuous features. pos is (T, K, 2) for all slots of both teams."""
    T = len(ball)
    x, y = ball[:, 0], ball[:, 1]
    oob = ((x < -cfg.oob_tol) | (x > PITCH_L + cfg.oob_tol) |
           (y < -cfg.oob_tol) | (y > PITCH_W + cfg.oob_tol))       # NaN compares False
    gone = np.isnan(x)                                              # lost for longer than max_gap

    spread = np.zeros(T)
    for k in range(2):
        r = pd.Series(ball[:, k]).rolling(cfg.still_win, center=True, min_periods=cfg.still_win // 2)
        spread = np.maximum(spread, (r.max() - r.min()).to_numpy())  # NaN propagates -> not still
    still = spread < cfg.still_m

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(speeds, axis=1)
    med = pd.Series(med).rolling(cfg.slow_win, center=True, min_periods=1).mean().to_numpy()
    slow = med < cfg.slow_speed

    w = cfg.idle_win                                                # net displacement over 3 s, median over slots
    disp = np.full(T, np.nan)
    if T > w:
        d = np.hypot(pos[w:, :, 0] - pos[:-w, :, 0], pos[w:, :, 1] - pos[:-w, :, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(d, axis=1)
        disp[w // 2: w // 2 + len(m)] = m
    idle = disp < cfg.idle_m

    cues = {"ball_oob": oob, "ball_still": still, "ball_gone": gone,
            "players_slow": slow, "players_idle": idle, "set_piece_win": sp_mask,
            "set_piece_core": core_mask}
    return cues, {"median_speed": med, "disp_3s": disp}


def build_scores(ball, home, away, speeds, sp_mask, core_mask, cfg):
    e = team_evidence(ball, home, away, cfg)
    cues, feats = dead_cues(ball, np.concatenate([home, away], axis=1), speeds, sp_mask, core_mask, cfg)
    dead = (cfg.w_oob * cues["ball_oob"] + cfg.w_still * cues["ball_still"] + cfg.w_gone * cues["ball_gone"] +
            cfg.w_slow * cues["players_slow"] + cfg.w_idle * cues["players_idle"] +
            cfg.w_sp * cues["set_piece_win"] + cfg.w_core * cues["set_piece_core"])
    return np.stack([e, -e, dead - cfg.dead_bias], axis=1), cues, feats


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------
def trans_cost(cfg):
    sw, en, ex = cfg.switch_cost, cfg.enter_dead, cfg.exit_dead
    if en + ex <= sw:
        raise ValueError("enter_dead + exit_dead (%g) must exceed switch_cost (%g), see Cfg" % (en + ex, sw))
    return np.array([[0, sw, en],
                     [sw, 0, en],
                     [ex, ex, 0]], dtype=float)                   # cost[i, j] = i -> j


def viterbi(S, C, init):
    T = len(S)
    delta = init + S[0]
    back = np.zeros((T, 3), dtype=np.int8)
    for t in range(1, T):
        cand = delta[:, None] - C
        back[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + S[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(S, init_team, cfg):
    S = S.copy()
    S[: cfg.kickoff_live, D] -= 100.0
    init = np.array([-50.0, -50.0, -100.0])
    if init_team is None:
        init[[H, A]] = 0.0
    else:
        init[init_team] = 0.0
    return viterbi(S, trans_cost(cfg), init)


def runs(path):
    """[(state, start, end_exclusive), ...] for a 1-D integer array."""
    cut = np.flatnonzero(np.diff(path)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(path)]
    return [(int(path[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def apply_restart_prior(S, path, restarts, cfg):
    """After each detected dead run that matches a set-piece tag with a known team,
    nudge the first seconds of live play towards the team that takes the set piece.
    `restarts` = [(tag_start_frame_relative, team_state), ...]."""
    dead_runs = [(s, e) for st, s, e in runs(path) if st == D and e - s >= cfg.min_dead_for_prior]
    used, S2 = set(), S.copy()
    for f0, team in restarts:
        best, best_gap = None, cfg.restart_match + 1
        for i, (s, e) in enumerate(dead_runs):
            if i in used:
                continue
            gap = max(s - f0, 0, f0 - e)                          # distance from f0 to the run
            if gap < best_gap:
                best, best_gap = i, gap
        if best is None or best_gap > cfg.restart_match:
            continue
        used.add(best)
        e = dead_runs[best][1]
        sl = slice(e, e + cfg.prior_len)
        S2[sl, team] += cfg.prior_w
        S2[sl, 1 - team] -= cfg.prior_w
    return S2


def prepare(ball_raw, home, away, speeds, sp_mask, cfg, core_mask=None):
    if core_mask is None:
        core_mask = np.zeros(len(ball_raw), dtype=bool)
    ball = smooth_ball(ball_raw, cfg)
    S, cues, feats = build_scores(ball, home, away, speeds, sp_mask, core_mask, cfg)
    return S, cues, feats


def decode_with_prior(S, restarts, init_team, cfg):
    path = decode(S, init_team, cfg)
    if restarts:
        path = decode(apply_restart_prior(S, path, restarts, cfg), init_team, cfg)
    return path


def run_pipeline(ball_raw, home, away, speeds, sp_mask, restarts, init_team, cfg, core_mask=None):
    S, cues, _ = prepare(ball_raw, home, away, speeds, sp_mask, cfg, core_mask)
    return decode_with_prior(S, restarts, init_team, cfg), cues


def phase_evidence(ev, start, hi, path0, cfg):
    """Per-frame vote in [-1, 1] (>0 favours H) from the Smart Tagging phases.

    The sign of each tag code is LEARNED, not assumed: for a code like PROGRESSION the Team should be
    the team with the ball (agreement with the first-pass decode well above 50%), for DEFENDING IN ...
    it should be the team without it (well below 50%). No ground truth is involved, only our own
    first-pass labels against independent automatic tags.

    Frames inside one interval are strongly correlated, so significance is judged on INTERVALS: a sign
    test over per-interval agreement. A code is used only if |z| >= phase_z; otherwise weight 0."""
    T = hi - start + 1
    alive = path0 != D
    sub = ev[(ev["code"] != "SET PIECES") & ev["team_state"].notna() & (ev["f1"] >= start) & (ev["f0"] <= hi)]
    votes, rows = np.zeros(T), []
    for code, g in sub.groupby("code"):
        spans, rates = [], []
        for r in g.itertuples():
            a = max(int(r.f0) - start, 0) + cfg.phase_trim
            b = min(int(r.f1) - start + 1, T) - cfg.phase_trim
            m = alive[a:b] if b > a else np.zeros(0, bool)
            if m.sum() < cfg.phase_min_frames:
                continue
            rates.append(float((path0[a:b][m] == int(r.team_state)).mean()))
            spans.append((a, b, int(r.team_state)))
        n = len(spans)
        if n < cfg.phase_min_n:
            rows.append((code, n, float("nan"), 0.0, None))
            continue
        rates = np.array(rates)
        decided = rates[rates != 0.5]
        p = float((decided > 0.5).mean()) if len(decided) else 0.5
        z = (p - 0.5) / (0.5 / np.sqrt(max(len(decided), 1)))
        rate = float(rates.mean())
        sign = float(np.clip((rate - 0.5) / 0.25, -1.0, 1.0)) if abs(z) >= cfg.phase_z else 0.0
        rows.append((code, n, rate, z, sign))
        for a, b, tm in spans:
            votes[a:b] += sign * (1.0 if tm == H else -1.0)
    return np.clip(votes, -1.0, 1.0), rows


def print_phase_table(rows):
    print("  tag phases vs first-pass decode (agree = decoded possessor is the tag's Team; z = interval sign test)")
    print("    %-30s %4s %6s %6s  %s" % ("code", "n", "agree", "z", "weight"))
    for code, n, rate, z, sign in sorted(rows, key=lambda r: -(r[2] if r[2] == r[2] else 0)):
        w = "unused" if sign is None else ("%+.2f" % sign if sign != 0 else "0 (not significant)")
        print("    %-30s %4d %6.2f %6.1f  %s" % (code[:30], n, rate, z, w))


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------
def to_states(lab):
    """label array of H/A/D/X -> ints 0/1/2, -1 for X."""
    out = np.full(len(lab), -1, dtype=np.int8)
    for i, c in enumerate("HAD"):
        out[np.asarray(lab) == c] = i
    return out


def possessions(labels):
    """Maximal runs of one live team. H H D D H H is two possessions."""
    return [(LABELS[st], s, e) for st, s, e in runs(to_states(labels)) if st in (H, A)]


def quick_check(pred, gt):
    """Frame-level sanity check on the public window. The official spellF1 comes from
    evaluation/run_evaluation.py, run that too."""
    m = np.isin(gt, list("HAD")) & (pred != "X")
    if not m.any():
        print("quick check: no overlap with the public ground truth")
        return
    f1s = []
    for c in "HAD":
        tp = ((pred == c) & (gt == c) & m).sum()
        fp = ((pred == c) & (gt != c) & m).sum()
        fn = ((pred != c) & (gt == c) & m).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    pg, gg = [int((pred[m] == c).sum()) for c in "HAD"], [int((gt[m] == c).sum()) for c in "HAD"]
    print("\npublic window, %d frames: accuracy %.3f | macroF1 %.3f (H %.2f A %.2f D %.2f)"
          % (m.sum(), (pred[m] == gt[m]).mean(), np.mean(f1s), *f1s))
    print("  frames pred H/A/D = %s   GT H/A/D = %s" % (pg, gg))
    print("  possessions: predicted %d, GT %d" % (len(possessions(pred[m])), len(possessions(gt[m]))))
    for name, arr in (("predicted", pred[m]), ("GT", gt[m])):
        d = [e - s for st, s, e in runs(to_states(arr)) if st == D]
        print("  dead-ball runs %-9s: %d, mean %.1f s, total %.0f s"
              % (name, len(d), np.mean(d) / FPS if d else 0, sum(d) / FPS))


def diagnose(gt, cues, feats, sp, lo):
    """How good are the dead cues, and what do the set-piece tags mean? (public window only)"""
    ok = np.isin(gt, list("HAD"))
    if not ok.any():
        print("diagnose: no public ground truth in the processed range")
        return
    dead, live = ok & (gt == "D"), ok & (gt != "D")
    print("\ndead-ball cues on the public window (fraction of frames where the cue fires)")
    print("  %-14s %10s %10s" % ("cue", "on GT dead", "on GT live"))
    for k, v in cues.items():
        print("  %-14s %10.2f %10.2f" % (k, v[dead].mean(), v[live].mean()))

    print("\ncontinuous features, percentiles 10 / 50 / 90 (use them to place thresholds)")
    for k, v in feats.items():
        f = lambda m: "/".join("%.2f" % x for x in np.nanpercentile(v[m], [10, 50, 90]))
        print("  %-14s dead: %-18s live: %s" % (k, f(dead), f(live)))

    g = to_states(gt)
    druns = [(s, e) for st, s, e in runs(g) if st == D]
    print("\nset pieces in the window: tag boundaries relative to the nearest GT dead run [s]")
    print("  %-22s %-14s %9s %9s %9s  %s" % ("type", "team", "f0-start", "f0-end", "f1-end", "GT label after"))
    for r in sp.itertuples():
        a = r.f0 - lo
        if not (0 <= a < len(g)) or not ok[a] or not druns:
            continue
        s, e = min(druns, key=lambda se: max(se[0] - a, 0, a - se[1]))
        after = "HAD"[g[e]] if e < len(g) and g[e] >= 0 else "?"
        print("  %-22s %-14s %9.1f %9.1f %9.1f  %s" % (
            str(r.Type)[:22], str(r.Team)[:14], (a - s) / FPS, (a - e) / FPS, (r.f1 - lo - e) / FPS, after))
    print("  f0-start > 0 and f0-end < 0 -> the tag starts INSIDE the stoppage, before the restart")


def align_report(sp, cues, feats, start, title):
    """Event-triggered average around the set-piece tag start, over EVERY tag in the half.
    No ground truth involved: it shows where the stoppage sits relative to f0."""
    arrs = [("median_speed", feats["median_speed"]), ("disp_3s", feats["disp_3s"]),
            ("ball_gone", cues["ball_gone"]), ("ball_still", cues["ball_still"]),
            ("players_slow", cues["players_slow"])]
    f0s = [int(f) - start for f in sp["f0"]]
    print("\nalignment around the tag start f0, %s (%d tags, 4 s bins)" % (title, len(f0s)))
    print("  %7s %4s " % ("offset", "n") + " ".join("%13s" % n for n, _ in arrs))
    for o in range(-24, 37, 4):
        row, n = [], 0
        for _, arr in arrs:
            vals = []
            for f0 in f0s:
                a, b = f0 + o * FPS, f0 + (o + 4) * FPS
                if a >= 0 and b <= len(arr):
                    vals.append(np.nanmean(arr[a:b].astype(float)))
            n = max(n, len(vals))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                row.append(np.nanmean(vals) if vals else np.nan)
        print("  %+6ds %4d " % (o, n) + " ".join("%13.2f" % v for v in row))
    print("  a dip in median_speed / disp_3s (and a rise in the other three) marks the stoppage")


def segments_report(gt, path):
    """Segment-by-segment view of the public window: what did we predict inside each true segment?"""
    ok = np.flatnonzero(np.isin(gt, list("HAD")))
    if not len(ok):
        return
    a, b = int(ok[0]), int(ok[-1]) + 1
    g, p = to_states(gt[a:b]), path[a:b]
    print("\nsegments in the public window (seconds from the first graded frame)")
    print("  %-4s %6s %6s | share of this true segment that we labelled" % ("true", "start", "dur"))
    print("  %-4s %6s %6s | %6s %6s %6s" % ("", "", "", "H", "A", "D"))
    for st, s, e in runs(g):
        seg = p[s:e]
        print("  %-4s %6.1f %6.1f | %6.2f %6.2f %6.2f" % ("HAD"[st], s / FPS, (e - s) / FPS, *[(seg == k).mean() for k in range(3)]))
    print("predicted segments:")
    for st, s, e in runs(p):
        print("  %-4s %6.1f %6.1f" % ("HAD"[st], s / FPS, (e - s) / FPS))


def structure(path, win=None):
    r = runs(path)
    pos_ = [e - s for st, s, e in r if st in (H, A)]
    dd = [e - s for st, s, e in r if st == D]
    return len(pos_), (np.mean(pos_) / FPS if pos_ else 0), len(dd), 100.0 * (path == D).mean()


def sweep(S, init_team, cfg, win):
    """Structure only. Pick a row by whether the COUNTS look like football, never by score."""
    print("\nstructure sweep (no restart prior).  Public window truth: 17 possessions, 4 dead runs, ~20% dead")
    print("  %6s %10s | %6s %7s %6s %6s | public window: %5s %6s %6s"
          % ("switch", "enter/exit", "poss", "mean_s", "Druns", "dead%", "poss", "Druns", "dead%"))
    for sc in (10, 15, 20, 25, 35, 50, 60):
        ed = max(int(round(sc * 0.8)), 8)
        path = decode(S, init_team, replace(cfg, switch_cost=sc, enter_dead=ed, exit_dead=ed))
        p, m, d, pct = structure(path)
        tail = ""
        if win is not None:
            wp, _, wd, wpct = structure(path[win[0]: win[1]])
            tail = "%19d %6d %6.0f" % (wp, wd, wpct)
        print("  %6d %10d | %6d %7.1f %6d %6.0f | %s" % (sc, ed, p, m, d, pct, tail))


# --------------------------------------------------------------------------------------
def half_bounds():
    try:
        from games import half_video_blocks
        return {h: (a, b) for h, a, b in half_video_blocks("BAR-ATH")}
    except Exception:
        return dict(HALVES)


def label_half(start, hi, data, ev, cfg, use_prior, fallback_team=None):
    """Run the pipeline on frames start..hi (start = kick-off). Returns a dict of results."""
    ball, home, away, speeds = data

    sp = ev[(ev["code"] == "SET PIECES") & (ev["f1"] >= start) & (ev["f0"] <= hi)]
    sp_mask = np.zeros(hi - start + 1, dtype=bool)
    core_mask = np.zeros_like(sp_mask)
    for f0 in sp["f0"]:
        sp_mask[max(f0 - cfg.sp_pre - start, 0): max(f0 + cfg.sp_post - start + 1, 0)] = True
        core_mask[max(f0 - cfg.core_half - start, 0): max(f0 + cfg.core_half - start + 1, 0)] = True

    restarts = [] if not use_prior else \
        [(int(r.f0) - start, int(r.team_state)) for r in sp.itertuples() if pd.notna(r.team_state)]

    is_ko = ev["Type"].astype(str).str.lower().str.contains("kick off", regex=False)
    ko = ev[is_ko & ev["Type"].astype(str).str.lower().str.contains("(start)", regex=False)
            & ev["f0"].between(start - 500, start + 500) & ev["team_state"].notna()]
    init_team = int(ko["team_state"].iloc[0]) if len(ko) else None
    source = "tag" if init_team is not None else None
    if init_team is None:
        near = ev[is_ko & ev["f0"].between(start - 3000, start + 3000)]
        print("  no kick-off team from the tags. Kick-off rows within 2 min of frame %d:" % start)
        print(near[["f0", "f1", "Type", "Team"]].to_string(index=False) if len(near) else "    (none)")
        if fallback_team is not None:
            init_team, source = fallback_team, "assumed: the other team than in the first half"
            print("  assuming the other team kicks off the second half")

    S, cues, feats = prepare(ball, home, away, speeds, sp_mask, cfg, core_mask)
    path = decode_with_prior(S, restarts, init_team, cfg)
    if cfg.phase_w > 0:
        votes, rows = phase_evidence(ev, start, hi, path, cfg)
        print_phase_table(rows)
        S = S.copy()
        S[:, H] += cfg.phase_w * votes
        S[:, A] -= cfg.phase_w * votes
        path = decode_with_prior(S, restarts, init_team, cfg)
    return dict(path=path, S=S, cues=cues, feats=feats, sp=sp, init_team=init_team, init_source=source)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half", choices=["1", "2", "both"], default="both")
    ap.add_argument("--minutes", type=float, default=None,
                    help="only label this many minutes after each kick-off (quick test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--diagnose", action="store_true", help="cue quality + set-piece tags vs the public GT")
    ap.add_argument("--align", action="store_true", help="where does the stoppage sit around each tag? (no GT)")
    ap.add_argument("--segments", action="store_true", help="true vs predicted segments in the public window")
    ap.add_argument("--sweep", action="store_true", help="possession / dead-run counts for a grid of costs")
    ap.add_argument("--no-team-prior", action="store_true", help="skip the set-piece team prior")
    ap.add_argument("--no-phase", action="store_true", help="do not use the tag-phase evidence")
    args = ap.parse_args()
    cfg = Cfg(phase_w=0.0) if args.no_phase else Cfg()

    bounds = half_bounds()
    halves = [1, 2] if args.half == "both" else [int(args.half)]
    plan = []                                                     # (half, graded_lo, kickoff, hi)
    for h in halves:
        g_lo, hi = bounds[h]
        start = g_lo if h == 1 else HALF2_KICKOFF
        if args.minutes:
            hi = min(hi, start + int(args.minutes * 60 * FPS) - 1)
        plan.append((h, g_lo, start, hi))
    out = args.out or "evaluation/results/%s.csv" % {"1": "first_half", "2": "second_half",
                                                      "both": "full_match"}[args.half]

    datas = load_atd([(start, hi) for _, _, start, hi in plan])
    ev = load_tags()
    gt = None
    if os.path.exists(GT_PUBLIC):
        gt = pd.read_csv(GT_PUBLIC)["label"].astype(str).str.strip().str.upper().to_numpy()

    labels = np.full(N_FRAMES, "X", dtype="<U1")
    kickoff_team = {}
    for (h, g_lo, start, hi), data in zip(plan, datas):
        print("\nhalf %d: labelling frames %d..%d (%.1f min), ball observed on %.1f%%"
              % (h, start, hi, (hi - start + 1) / FPS / 60, 100 * (~np.isnan(data[0][:, 0])).mean()))
        fallback = 1 - kickoff_team[1] if h == 2 and kickoff_team.get(1) is not None else None
        r = label_half(start, hi, data, ev, cfg, not args.no_team_prior, fallback)
        path = r["path"]
        kickoff_team[h] = r["init_team"] if r["init_team"] is not None else int(path[0])
        print("  kick-off team: %s (%s)" % ("HA"[kickoff_team[h]], r["init_source"] or "decoded from the first frames"))

        labels[g_lo:start] = "D"                                  # half 2: graded dead time before kick-off
        labels[start: hi + 1] = LABELS[path]
        n, m, nd, pct = structure(path)
        print("  %d possessions (mean %.1f s), %d dead runs, %.0f%% of frames dead" % (n, m, nd, pct))

        win = None
        if gt is not None:
            idx = np.flatnonzero(np.isin(gt[start: hi + 1], list("HAD")))
            if len(idx):
                win = (int(idx[0]), int(idx[-1]) + 1)
        if args.diagnose and win is not None:
            diagnose(gt[start: hi + 1], r["cues"], r["feats"], r["sp"], start)
        if args.segments and win is not None:
            segments_report(gt[start: hi + 1], path)
        if args.align and len(r["sp"]):
            align_report(r["sp"], r["cues"], r["feats"], start, "half %d, all set pieces" % h)
            thr = r["sp"][r["sp"]["Type"].astype(str).str.lower().str.contains("throw")]
            if len(thr) >= 3:
                align_report(thr, r["cues"], r["feats"], start, "half %d, throw-ins only" % h)
        if args.sweep:
            sweep(r["S"], r["init_team"], cfg, win)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(out, index=False)
    print("\nwrote %s  (%s)" % (out, ", ".join("%s=%d" % (c, (labels == c).sum()) for c in "HADX")))

    if gt is not None:
        quick_check(labels, gt)                                   # only the first 5 minutes have a GT
    if args.half != "both":
        print("\nnote: this is a partial submission, do not upload it as final")
    print("next: python evaluation/validate_submission.py %s && python evaluation/run_evaluation.py" % out)


if __name__ == "__main__":
    main()