"""STEP 1 - How good are simple start/sit rules?

Metric: CAPTURE RATE = points the rule actually scored / points a perfect-hindsight
lineup would have scored, from the same roster. Bounded and interpretable:

    100%  = you started exactly the right players every week (impossible in real life)
    random = the floor, filling slots at random from who was available

The gap between the best simple rule and 100% is the ENTIRE room an AI has to work in.
If simple rules capture 95%, there is 5% left and most of it is luck.

Rosters: 12-team snake draft off point-in-time ADP, with a target composition
(2 QB, 4 RB, 4 WR, 2 TE) so every team can actually field a legal lineup. Without that,
all rules get forced into identical choices and the comparison measures nothing.

Rules all fall back to the ADP prior when a player has no games yet (week 1 especially).

Run: uv run python checks/step1_baseline.py
"""
from __future__ import annotations
import random
import sys
from pathlib import Path
import nflreadpy as nfl
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ffeval.models.expected import fit_expected_points, project_ppg
from ffeval.scoring.league import League, snake_pick_numbers
from ffeval.scoring.lineup import best_lineup

POOL = Path(__file__).parent / "out" / "pool_full.parquet"
df = pl.read_parquet(POOL)
SEASONS = sorted(df["season"].unique().to_list())
LEAGUE = League(rounds=12)
TARGET = {"QB": 2, "RB": 4, "WR": 4, "TE": 2}          # sums to 12
RNG = random.Random(20260830)

# ---------------------------------------------------------------- realized weekly points
st = nfl.load_player_stats(seasons=SEASONS, summary_level="week").filter(
    pl.col("season_type") == "REG")
WK = {r["season"]: range(1, int(r["mx"]) + 1)
      for r in st.group_by("season").agg(pl.col("week").max().alias("mx")).to_dicts()}
PTS: dict[int, dict[tuple[str, int], float]] = {}
HIST: dict[int, dict[str, list[tuple[int, float]]]] = {}
for r in st.select("season", "player_id", "week", "fantasy_points_ppr").to_dicts():
    s, pid, w = r["season"], r["player_id"], int(r["week"])
    v = float(r["fantasy_points_ppr"] or 0.0)
    PTS.setdefault(s, {})[(pid, w)] = v
    HIST.setdefault(s, {}).setdefault(pid, []).append((w, v))
for s in HIST:
    for pid in HIST[s]:
        HIST[s][pid].sort()

# ---------------------------------------------------------------- ADP-implied PPG prior
PPG_CURVE = {}
for season in SEASONS:
    PPG_CURVE[season] = fit_expected_points(
        df.select("season", "position", "pos_rank", pl.col("ppg").alias("ppr_total")), season
    )

# ---------------------------------------------------------------- rosters
def build_rosters(season: int):
    pool = (df.filter(pl.col("season") == season).sort("adp")
              .select("gsis_id", "name", "position", "pos_rank").to_dicts())
    on_clock = {}
    for s in range(1, LEAGUE.teams + 1):
        for pn in snake_pick_numbers(s, LEAGUE.teams, LEAGUE.rounds):
            on_clock[pn] = s
    rosters = {s: [] for s in range(1, LEAGUE.teams + 1)}
    taken = set()
    for pick in range(1, LEAGUE.picks + 1):
        slot = on_clock[pick]
        have = {}
        for p in rosters[slot]:
            have[p["position"]] = have.get(p["position"], 0) + 1
        for p in pool:
            if p["gsis_id"] in taken:
                continue
            if have.get(p["position"], 0) >= TARGET[p["position"]]:
                continue
            rosters[slot].append(p)
            taken.add(p["gsis_id"])
            break
    return rosters

# ---------------------------------------------------------------- the rules
def prior_games(season, pid, week):
    return [v for w, v in HIST.get(season, {}).get(pid, []) if w < week]

def adp_ppg(season, p):
    return PPG_CURVE[season](p["position"], int(p["pos_rank"]))

def rule_adp(season, p, week):
    return adp_ppg(season, p)

def rule_last1(season, p, week):
    h = prior_games(season, p["gsis_id"], week)
    return h[-1] if h else adp_ppg(season, p)

def rule_last3(season, p, week):
    h = prior_games(season, p["gsis_id"], week)
    return sum(h[-3:]) / len(h[-3:]) if h else adp_ppg(season, p)

def rule_season(season, p, week):
    h = prior_games(season, p["gsis_id"], week)
    return sum(h) / len(h) if h else adp_ppg(season, p)

def rule_blend(season, p, week):
    return 0.5 * rule_season(season, p, week) + 0.5 * rule_last3(season, p, week)

def rule_shrunk(season, p, week):
    """Trust ADP early, trust observed production as games accumulate.

    Imported, not copied. The 91.2% this script prints has to be produced by the code
    that runs on a Sunday, or the number stops describing the thing that ships.
    """
    return project_ppg(prior_games(season, p["gsis_id"], week), adp_ppg(season, p))

RULES = {
    "adp_static": rule_adp,
    "last1_game": rule_last1,
    "last3_games": rule_last3,
    "season_avg": rule_season,
    "blend_50_50": rule_blend,
    "shrunk_to_adp": rule_shrunk,
}

# ---------------------------------------------------------------- scoring
def score(roster, season, projector=None, randomize=False):
    tot = 0.0
    for wk in WK[season]:
        avail = [(p["gsis_id"], p["position"], PTS[season][(p["gsis_id"], wk)])
                 for p in roster if (p["gsis_id"], wk) in PTS[season]]
        if not avail:
            continue
        if randomize:
            order = [a[0] for a in avail]
            RNG.shuffle(order)
            prio = {pid: i for i, pid in enumerate(order)}
        elif projector is None:
            prio = None                                  # clairvoyant
        else:
            proj = sorted(((projector(season, p, wk), p["gsis_id"])
                           for p in roster
                           if (p["gsis_id"], wk) in PTS[season]), reverse=True)
            prio = {pid: i for i, (_, pid) in enumerate(proj)}
        pts, _ = best_lineup(avail, LEAGUE, prio)
        tot += pts
    return tot

rows = []
for season in SEASONS:
    rosters = build_rosters(season)
    for slot, roster in rosters.items():
        ceil = score(roster, season, projector=None)
        rnd = sum(score(roster, season, randomize=True) for _ in range(5)) / 5
        rec = {"season": season, "slot": slot, "ceiling": ceil, "random": rnd}
        for name, fn in RULES.items():
            rec[name] = score(roster, season, projector=fn)
        rows.append(rec)

res = pl.DataFrame(rows)
res.write_parquet(Path(__file__).parent / "out" / "step1_results.parquet")

print(f"{res.height} roster-seasons  ({len(SEASONS)} seasons x {LEAGUE.teams} slots)")
print(f"roster = {TARGET}, {LEAGUE.rounds} rounds, {LEAGUE.n_starters} starters/week\n")
print("CAPTURE RATE = rule points / perfect-hindsight points\n")
print(f"{'rule':<16}{'points':>9}{'capture':>10}{'vs random':>11}{'gap to perfect':>16}")
print("-" * 63)
ceil_mean = res["ceiling"].mean()
rnd_mean = res["random"].mean()
print(f"{'PERFECT (ceiling)':<16}{ceil_mean:>9.0f}{1.0:>9.1%}{'':>11}{0.0:>15.0f}")
ranked = sorted(RULES, key=lambda n: -res[n].mean())
for name in ranked:
    m = res[name].mean()
    cap = (res[name] / res["ceiling"]).mean()
    print(f"{name:<16}{m:>9.0f}{cap:>9.1%}{m - rnd_mean:>+11.0f}{ceil_mean - m:>15.0f}")
print(f"{'RANDOM (floor)':<16}{rnd_mean:>9.0f}{(res['random']/res['ceiling']).mean():>9.1%}"
      f"{0:>+11.0f}{ceil_mean - rnd_mean:>15.0f}")

best = ranked[0]
print(f"\nBest simple rule: {best}")
print(f"  captures {(res[best]/res['ceiling']).mean():.1%} of achievable points")
print(f"  beats random by {res[best].mean() - rnd_mean:.0f} pts/season")
print(f"  leaves {ceil_mean - res[best].mean():.0f} pts/season on the table")
sk = (res[best].mean() - rnd_mean) / (ceil_mean - rnd_mean)
print(f"  = {sk:.1%} of the way from random to perfect")

print("\nHow much do the rules differ from each other?")
print(f"  best - worst = {res[ranked[0]].mean() - res[ranked[-1]].mean():.0f} pts/season")
print(f"  sd of roster outcomes = {res['ceiling'].std():.0f} pts (the noise to beat)")
