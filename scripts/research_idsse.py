"""Quality, label-definition and method checks on the public IDSSE matches.

    python -X utf8 scripts/research_idsse.py [--match J03WMX ...]

Writes outputs/idsse/report.json and prints tables. No competition data or GT is read. Scores here
describe other matches with a related label definition; they are not competition performance.
"""

import argparse
import json
from pathlib import Path
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "official"), str(ROOT / "official" / "evaluation"), str(ROOT / "official" / "starter")]

import possession_metrics as pm  # noqa: E402
from baseline import smooth  # noqa: E402
from scripts import data_quality, idsse  # noqa: E402
from solution.decoder import DecoderConfig, decode  # noqa: E402
from solution.possession import fill_abstentions, predict  # noqa: E402


FPS = idsse.FPS
OUT = ROOT / "outputs" / "idsse"
RESTARTS = {"ThrowIn", "FreeKick", "CornerKick", "GoalKick", "KickOff", "Penalty"}
MERGE_SECONDS = (1, 2, 3, 5)
CONTROLLED_METHODS = ("baseline_smooth50", "step1_filled", "decoder", "decoder_switch2")


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return edges[::2], edges[1::2]


def segments(labels, blocks):
    out = []
    for first, last in blocks:
        window = labels[first:last + 1]
        bounds = np.r_[0, np.flatnonzero(window[1:] != window[:-1]) + 1, len(window)]
        out.extend((first + int(a), first + int(b), int(window[a])) for a, b in zip(bounds, bounds[1:]))
    return out


def share(mask):
    return round(100 * float(np.mean(mask)), 2) if np.size(mask) else None


def quantiles(values, points=(50, 90)):
    return [round(float(v), 2) for v in np.percentile(values, points)] if len(values) else [None] * len(points)


def label_statistics(match):
    segs = segments(match.labels, match.blocks)
    minutes = len(match.labels) / FPS / 60
    live = np.array([b - a for a, b, c in segs if c in (idsse.H, idsse.A)]) / FPS
    dead = np.array([b - a for a, b, c in segs if c == idsse.D]) / FPS
    handovers = via_dead_same = via_dead_other = quick_returns = 0
    for one, two, three in zip(segs, segs[1:], segs[2:] + [(0, 0, -1)]):
        if one[2] != idsse.D and two[2] != idsse.D and one[1] == two[0]:
            handovers += 1
            if three[2] == one[2] and two[1] - two[0] < 3 * FPS:
                quick_returns += 1
    for one, two, three in zip(segs, segs[1:], segs[2:]):
        if two[2] == idsse.D and one[2] != idsse.D and three[2] != idsse.D:
            if one[2] == three[2]:
                via_dead_same += 1
            else:
                via_dead_other += 1
    # Does the team named during a stoppage match the team that restarts?
    named, total = 0, 0
    for (a, b, code), following in zip(segs, segs[1:]):
        if code == idsse.D and following[2] != idsse.D and a != b:
            total += 1
            named += int(match.possession[b - 1] == following[2] + 1)
    changes_inside = sum(int((np.diff(match.possession[a:b]) != 0).sum()) for a, b, c in segs if c == idsse.D)
    return {
        "minutes": round(minutes, 1), "dead_pct": share(match.labels == idsse.D),
        "home_share_of_alive_pct": share(match.labels[match.labels != idsse.D] == idsse.H),
        "possessions": len(live), "possessions_per_minute": round(len(live) / minutes, 2),
        "possession_median_p90_s": quantiles(live), "possession_mean_s": round(float(live.mean()), 2),
        "possession_under_s_pct": {str(s): share(live < s) for s in (0.5, 1, 2, 3, 5)},
        "stoppages": len(dead), "stoppage_median_p90_max_s": quantiles(dead) + [round(float(dead.max()), 2)],
        "direct_handovers": handovers, "returns_within_3s": quick_returns,
        "stoppage_same_team_after": via_dead_same, "stoppage_other_team_after": via_dead_other,
        "team_named_in_stoppage_restarts_pct": round(100 * named / max(total, 1), 2),
        "possession_changes_inside_stoppages": changes_inside,
    }


def nearest_by_team(match):
    scale = np.asarray(match.pitch, dtype=np.float32)
    distance = np.linalg.norm((match.players - match.ball[:, None, :]) * scale, axis=2)
    distance = np.where(np.isnan(distance), np.inf, distance)
    return np.stack([distance[:, match.teams == team].min(axis=1) for team in (0, 1)], axis=1)


def definition_tests(match, nearest):
    labels = match.labels
    alive = labels != idsse.D
    closest = nearest.min(axis=1)
    team = (nearest[:, 1] < nearest[:, 0]).astype(np.int8)
    results = {
        "nearest_team_agrees_alive_pct": share(team[alive] == labels[alive]),
        "nearest_team_agrees_within_2m_pct": share(team[alive & (closest <= 2)] == labels[alive & (closest <= 2)]),
        "nearest_team_agrees_beyond_5m_pct": share(team[alive & (closest > 5)] == labels[alive & (closest > 5)]),
        "ball_within_2m_alive_pct": share(closest[alive] <= 2),
    }
    # When does the new owner reach the ball, relative to the labelled change?
    lags, segs = [], segments(labels, match.blocks)
    window = 3 * FPS
    for one, two in zip(segs, segs[1:]):
        if one[2] == idsse.D or two[2] == idsse.D or one[1] != two[0]:
            continue
        change = two[0]
        low, high = max(change - window, one[0]), min(change + window, len(labels))
        close = np.flatnonzero(nearest[low:high, two[2]] <= 2)
        if len(close):
            lags.append((low + close[np.argmin(np.abs(low + close - change))] - change) / FPS)
    lags = np.array(lags)
    results["new_owner_within_2m_of_change_pct"] = round(100 * len(lags) / max(sum(
        1 for a, b in zip(segs, segs[1:]) if a[2] != idsse.D and b[2] != idsse.D and a[1] == b[0]), 1), 2)
    results["new_owner_contact_minus_change_median_p10_p90_s"] = quantiles(lags, (50, 10, 90))
    # Opponent contacts that did not change possession: "a simple touch" in the competition wording.
    opponent = np.where(labels == idsse.H, nearest[:, 1], np.where(labels == idsse.A, nearest[:, 0], np.inf))
    contacts = runs(opponent <= 1.0)
    changed = np.zeros(len(labels), dtype=bool)
    for one, two in zip(segs, segs[1:]):
        if one[2] != idsse.D and two[2] != idsse.D and one[1] == two[0]:
            changed[max(0, two[0] - 2 * FPS):two[0] + 2 * FPS] = True
    no_change = sum(1 for a, b in zip(*contacts) if b - a >= 3 and not changed[a:b].any())
    results["opponent_contacts_over_0_1s"] = int(sum(1 for a, b in zip(*contacts) if b - a >= 3))
    results["opponent_contacts_without_change_pct"] = round(100 * no_change / max(results["opponent_contacts_over_0_1s"], 1), 2)
    return results


def event_frames(match):
    """Events mapped to frame indexes with the position file's own clock."""
    clock = idsse.frame_times(match.match_id)
    first, second = match.blocks
    mapped = []
    for time, kind, team, inner in idsse.events(match.match_id):
        for section, (start, _) in ((("firstHalf"), first), (("secondHalf"), second)):
            t0 = clock[section][0]
            index = start + int(round((time - t0) * FPS))
            block = first if section == "firstHalf" else second
            if block[0] <= index <= block[1]:
                mapped.append((index, kind, team, inner))
                break
    return mapped


def event_tests(match):
    home, guest = idsse.team_roles(match.match_id)
    code = {home: idsse.H, guest: idsse.A}
    labels, n = match.labels, len(match.labels)
    mapped = event_frames(match)
    play = [(i, code[t]) for i, kind, t, _ in mapped if kind == "Play" and t in code]
    agree = [int((labels[max(i - FPS, 0):min(i + FPS, n)] == team).any()) for i, team in play]
    exact = [int(labels[i] == team) for i, team in play]
    restart_lag, before_dead, restart_team = [], [], []
    d_to_live = [b[0] for a, b in zip(segments(labels, match.blocks), segments(labels, match.blocks)[1:])
                 if a[2] == idsse.D and b[2] != idsse.D]
    d_to_live = np.array(d_to_live)
    for i, kind, team, _ in mapped:
        if kind not in RESTARTS:
            continue
        before_dead.append(int((labels[max(i - 2 * FPS, 0):i + 1] == idsse.D).any()))
        if len(d_to_live):
            nearest_start = d_to_live[np.argmin(np.abs(d_to_live - i))]
            if abs(nearest_start - i) <= 5 * FPS:
                restart_lag.append((nearest_start - i) / FPS)
                if team in code and nearest_start < n:
                    restart_team.append(int(labels[nearest_start] == code[team]))
    # A live stoppage cause: how does the ball sit when the status drops to dead?
    segs = segments(labels, match.blocks)
    dead_starts = [b[0] for a, b in zip(segs, segs[1:]) if b[2] == idsse.D and a[1] == b[0]]
    outside = (match.ball < 0) | (match.ball > 1)
    outside = outside.any(axis=1)
    out_nearby = [int(outside[max(s - 2 * FPS, 0):s + FPS].any()) for s in dead_starts]
    leave_lag = []
    for s in dead_starts:
        hits = np.flatnonzero(outside[max(s - 3 * FPS, 0):s + 3 * FPS])
        if len(hits):
            leave_lag.append((hits[0] + max(s - 3 * FPS, 0) - s) / FPS)
    return {
        "play_events": len(play), "play_team_agrees_exact_pct": round(100 * np.mean(exact), 2),
        "play_team_agrees_within_1s_pct": round(100 * np.mean(agree), 2),
        "restart_events": len(before_dead), "restart_dead_in_prior_2s_pct": round(100 * np.mean(before_dead), 2),
        "restart_team_is_next_label_pct": round(100 * np.mean(restart_team), 2) if restart_team else None,
        "live_start_minus_restart_event_median_p10_p90_s": quantiles(restart_lag, (50, 10, 90)),
        "stoppages_with_ball_outside_nearby_pct": round(100 * np.mean(out_nearby), 2) if out_nearby else None,
        "ball_outside_minus_dead_start_median_p10_p90_s": quantiles(leave_lag, (50, 10, 90)),
    }


def activity_test(match):
    """How well does team movement alone separate stoppages, in the decoder's own terms?"""
    scale = np.asarray(match.pitch, dtype=np.float32)
    speed = np.full(match.players.shape[:2], np.nan, dtype=np.float32)
    for start in range(1, len(match.players), 20000):
        stop = min(start + 20000, len(match.players))
        speed[start:stop] = np.linalg.norm((match.players[start:stop] - match.players[start - 1:stop - 1]) * scale, axis=2) * FPS
    median = np.nanmedian(speed[1:], axis=1)
    window = np.ones(FPS * 2 + 1) / (FPS * 2 + 1)
    smooth_speed = np.convolve(median, window, mode="same")
    dead = match.labels[1:] == idsse.D
    order = np.argsort(smooth_speed)
    ranks = np.empty(len(order))
    ranks[order] = np.arange(len(order))
    auc = (ranks[~dead].sum() - (~dead).sum() * ((~dead).sum() - 1) / 2) / max((~dead).sum() * dead.sum(), 1)
    ball_speed = np.linalg.norm(np.diff(match.ball, axis=0) * scale, axis=1) * FPS
    outside = ((match.ball < 0) | (match.ball > 1)).any(axis=1)[1:]
    return {
        "ball_frozen_dead_pct": share(ball_speed[dead] == 0),
        "ball_frozen_alive_pct": share(ball_speed[~dead] == 0),
        "ball_outside_dead_pct": share(outside[dead]),
        "ball_outside_alive_pct": share(outside[~dead]),
        "median_speed_dead_p10_p50_p90": quantiles(smooth_speed[dead], (10, 50, 90)),
        "median_speed_alive_p10_p50_p90": quantiles(smooth_speed[~dead], (10, 50, 90)),
        "activity_auc_alive_vs_dead": round(float(auc), 3),
        "ball_speed_dead_p50_p90": quantiles(ball_speed[dead]),
        "ball_speed_alive_p50_p90": quantiles(ball_speed[~dead]),
    }


def quality(match):
    signals = data_quality.frame_signals(match.players, match.ball, match.teams, FPS, match.pitch)
    return data_quality.scope_metrics(signals, match.blocks, FPS)


def hide_ball(match, rng, dead_biased):
    """Hide about 40% of ball frames in long gaps. The gap sizes are assumptions, not measurements."""
    ball = match.ball.copy()
    n = len(ball)
    t = 0
    if dead_biased:
        for a, b, code in segments(match.labels, match.blocks):
            if code == idsse.D and rng.random() < 0.8:
                ball[a:b] = np.nan
        visible_median, hidden_median = 40, 6
    else:
        visible_median, hidden_median = 14, 10
    while t < n:
        t += int(rng.lognormal(np.log(visible_median * FPS), 0.9))
        stop = t + int(rng.lognormal(np.log(hidden_median * FPS), 0.9))
        ball[t:stop] = np.nan
        t = stop
    return ball


def unfreeze_dead_ball(match, ball, rng):
    """Remove the freeze leak: DFL's ball never moves in a stoppage, an automatic tracker's would."""
    out = ball.copy()
    dead = match.labels == idsse.D
    noise = rng.normal(0, 0.05, (int(dead.sum()), 2)) / np.asarray(match.pitch)  # about 5 cm per frame
    out[dead] += noise.astype(out.dtype)
    return out


def controlled_statistics(match):
    """Possession counts if brief spells are not treated as possession changes."""
    minutes = len(match.labels) / FPS / 60
    out = {}
    for seconds in MERGE_SECONDS:
        merged = idsse.merge_brief_spells(match.labels, match.blocks, seconds)
        live = np.array([b - a for a, b, c in segments(merged, match.blocks) if c != idsse.D]) / FPS
        out[str(seconds)] = {"per_minute": round(len(live) / minutes, 2), "mean_s": round(float(live.mean()), 2),
                             "median_s": round(float(np.median(live)), 2)}
    return out


def score(match, prediction, labels=None):
    """Official scorer. One X row separates the halves, as halftime does in the competition."""
    split = match.blocks[1][0]
    gt = np.insert(match.labels if labels is None else labels, split, pm.X)
    pred = np.insert(prediction, split, pm.X)
    result = pm.evaluate(gt, pred)
    return {"spell": result["spell_f1_mean"], "macro": result["macro_f1"], "D_f1": result["per_class"]["D"]["f1"],
            "team_acc": result["team_acc_both_alive"], "n_pred": result["n_pred_spells"], "n_gt": result["n_gt_spells"]}


def to_codes(labels):
    return pm.label_codes(np.asarray(labels))


DECODERS = {
    "decoder": DecoderConfig(),
    "decoder_min1s": DecoderConfig(min_live_seconds=1.0),
    "decoder_min3s": DecoderConfig(min_live_seconds=3.0),
    "decoder_switch0.5": DecoderConfig(switch_cost_seconds=0.5),
    "decoder_switch2": DecoderConfig(switch_cost_seconds=2.0),
}


def methods(match, ball):
    distance = np.linalg.norm((match.players - ball[:, None, :]) * np.asarray(match.pitch, dtype=np.float32), axis=2)
    distance = np.where(np.isnan(distance), np.inf, distance)
    per_team = np.stack([distance[:, match.teams == team].min(axis=1) for team in (0, 1)], axis=1)
    codes = np.where(np.isfinite(ball).all(axis=1), (per_team[:, 1] < per_team[:, 0]).astype(np.int8), -1).astype(np.int8)
    out = {}
    for name, window in (("baseline_raw", 0), ("baseline_smooth25", 25), ("baseline_smooth50", 50), ("baseline_smooth125", 125)):
        labels = np.empty(len(codes), dtype="<U1")
        for first, last in match.blocks:
            labels[first:last + 1] = smooth(codes[first:last + 1], window)
        out[name] = to_codes(labels)
    step1, _ = predict(match.players, ball, match.teams, fps=FPS, pitch=match.pitch)
    out["step1_memory_dead"] = to_codes(step1)
    out["step1_filled"] = to_codes(fill_abstentions(step1, match.blocks))
    for name, config in DECODERS.items():
        out[name] = to_codes(decode(match.players, ball, match.teams, fps=FPS, pitch=match.pitch,
                                    blocks=match.blocks, config=config))
    return out


def analyse(match_id):
    match = idsse.load_match(match_id)
    nearest = nearest_by_team(match)
    report = {"match": match_id, "quality": quality(match), "labels": label_statistics(match),
              "definition": definition_tests(match, nearest), "events": event_tests(match),
              "activity": activity_test(match), "controlled": controlled_statistics(match), "methods": {}}
    merged = {seconds: idsse.merge_brief_spells(match.labels, match.blocks, seconds) for seconds in MERGE_SECONDS}
    seed = zlib.crc32(match_id.encode("ascii"))
    inputs = {"clean": match.ball,
              "hidden_independent": hide_ball(match, np.random.default_rng(seed), False),
              "hidden_dead_biased": hide_ball(match, np.random.default_rng(seed + 1), True)}
    inputs["hidden_independent_unfrozen"] = unfreeze_dead_ball(match, inputs["hidden_independent"], np.random.default_rng(seed + 2))
    inputs["hidden_dead_biased_unfrozen"] = unfreeze_dead_ball(match, inputs["hidden_dead_biased"], np.random.default_rng(seed + 3))
    for name, ball in inputs.items():
        predictions = methods(match, ball)
        report["methods"][name] = {
            "ball_visible_pct": share(np.isfinite(ball).all(axis=1)),
            "dead_visible_pct": share(np.isfinite(ball[match.labels == idsse.D]).all(axis=1)),
            "scores": {method: score(match, prediction) for method, prediction in predictions.items()},
            "scores_controlled": {str(seconds): {method: score(match, predictions[method], labels)
                                                 for method in CONTROLLED_METHODS}
                                  for seconds, labels in merged.items()},
        }
    return report


def show(reports):
    for section in ("labels", "definition", "events", "activity"):
        print(f"\n--- {section} ---")
        ids = [r["match"] for r in reports]
        print(f"{'':<52}" + "".join(f"{i:>22}" for i in ids))
        for key in reports[0][section]:
            print(f"{key:<52}" + "".join(f"{str(r[section][key]):>22}" for r in reports))
    print("\n--- quality (same thresholds as scripts/data_quality.py) ---")
    picks = ["ball.jump_pct", "ball.out_of_pitch_pct", "nearest.median_m", "nearest.p90_m", "nearest.within_control_pct",
             "crowd.visible_mean", "tracks.mean", "segments.per_minute", "jumps.per_1000", "duplicates.frames_pct"]
    for key in picks:
        print(f"{key:<36}" + "".join(f"{str(data_quality.lookup(r['quality'], key)):>10}" for r in reports))
    for scenario in reports[0]["methods"]:
        print(f"\n--- methods, mean over {len(reports)} match(es): {scenario} ---")
        vis = np.mean([r["methods"][scenario]["ball_visible_pct"] for r in reports])
        dead = np.mean([r["methods"][scenario]["dead_visible_pct"] for r in reports])
        print(f"ball visible {vis:.1f}% of frames, {dead:.1f}% of dead frames")
        print(f"{'method':<22}{'spell':>8}{'sd':>7}{'macro':>8}{'D F1':>8}{'team':>7}{'pred/gt':>9}")
        for method in reports[0]["methods"][scenario]["scores"]:
            rows = [r["methods"][scenario]["scores"][method] for r in reports]
            spells = [row["spell"] for row in rows]
            print(f"{method:<22}{np.mean(spells):>8.4f}{np.std(spells):>7.3f}{np.mean([row['macro'] for row in rows]):>8.4f}"
                  f"{np.mean([row['D_f1'] for row in rows]):>8.4f}{np.mean([row['team_acc'] for row in rows]):>7.3f}"
                  f"{np.mean([row['n_pred'] / row['n_gt'] for row in rows]):>9.2f}")
        print(f"spellF1 against labels with brief spells merged, seconds {MERGE_SECONDS}:")
        for method in CONTROLLED_METHODS:
            means = [np.mean([r["methods"][scenario]["scores_controlled"][str(s)][method]["spell"] for r in reports])
                     for s in MERGE_SECONDS]
            print(f"  {method:<20}" + "".join(f"{value:>8.4f}" for value in means))
    print("\n--- possessions if brief spells are merged (mean over matches) ---")
    print(f"{'seconds':<10}{'per minute':>12}{'mean s':>9}{'median s':>10}")
    for seconds in MERGE_SECONDS:
        rows = [r["controlled"][str(seconds)] for r in reports]
        print(f"{seconds:<10}{np.mean([row['per_minute'] for row in rows]):>12.2f}"
              f"{np.mean([row['mean_s'] for row in rows]):>9.2f}{np.mean([row['median_s'] for row in rows]):>10.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", nargs="*", help="Match ids; default all downloaded matches")
    args = parser.parse_args()
    ids = args.match or idsse.match_ids()
    reports = []
    for match_id in ids:
        print("analysing " + match_id, flush=True)
        reports.append(analyse(match_id))
    show(reports)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("\nSaved outputs/idsse/report.json. Not a competition score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
