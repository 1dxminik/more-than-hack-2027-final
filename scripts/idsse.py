"""Read one public IDSSE match into arrays shaped like the project's tracking input.

Positions are normalised to 0-1 like the competition ATD. Labels come from the per-frame ball
fields: BallStatus 0 is D, otherwise BallPossession 1 is H and 2 is A. Arrays are cached next to
the data in the gitignored datasets/idsse/cache/.
"""

from collections import namedtuple
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "idsse"
FPS = 25
H, A, D, X = 0, 1, 2, 3  # the official scorer's codes
Match = namedtuple("Match", "match_id players ball teams labels possession status frame_ids blocks pitch")


def match_ids(data_dir=DATA):
    return sorted(path.stem[-6:] for path in Path(data_dir).glob("DFL_04_03_positions_raw_observed_*.xml"))


def source_path(kind, match_id, data_dir=DATA):
    paths = list(Path(data_dir).glob(f"DFL_{kind}_DFL-COM-*_DFL-MAT-{match_id}.xml"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {kind} file for {match_id}")
    return paths[0]


def team_roles(match_id, data_dir=DATA):
    """(home team id, guest team id) from the match information file."""
    roles = {team.get("Role"): team.get("TeamId")
             for team in ET.parse(source_path("02_01_matchinformation", match_id, data_dir)).getroot().iter("Team")}
    return roles["home"], roles["guest"]


def parse_positions(match_id, data_dir=DATA):
    """Stream the large XML once. Frame numbers leave a gap between the two halves."""
    home, guest = team_roles(match_id, data_dir)
    tracks, ball, pitch = {}, {}, (105.0, 68.0)
    current, team, person, section = None, None, None, None
    path = source_path("04_03_positions_raw_observed", match_id, data_dir)
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            if element.tag == "FrameSet":
                section, team, person = element.get("GameSection"), element.get("TeamId"), element.get("PersonId")
                current = [] if team in (home, guest, "BALL") else None
            continue
        if element.tag == "PitchSize":
            pitch = (float(element.get("X")), float(element.get("Y")))
        elif element.tag == "Frame":
            if current is not None:
                row = (int(element.get("N")), float(element.get("X")), float(element.get("Y")))
                if team == "BALL":
                    row += (int(element.get("BallPossession", 0)), int(element.get("BallStatus", -1)))
                current.append(row)
            element.clear()
        elif element.tag == "FrameSet":
            if current:
                if team == "BALL":
                    ball[section] = np.asarray(current, dtype=np.float64)
                else:
                    tracks.setdefault((team, person), {})[section] = np.asarray(current, dtype=np.float64)
            current = None
            element.clear()
    if set(ball) != {"firstHalf", "secondHalf"}:
        raise ValueError(f"{match_id}: expected exactly two halves with ball data")
    return home, tracks, ball, pitch


def build(match_id, data_dir=DATA):
    home, tracks, ball, pitch = parse_positions(match_id, data_dir)
    halves = [ball["firstHalf"], ball["secondHalf"]]
    for half in halves:
        if (np.diff(half[:, 0]) != 1).any():
            raise ValueError(f"{match_id}: ball frames are not contiguous inside a half")
    n = sum(len(half) for half in halves)
    offsets = {"firstHalf": 0, "secondHalf": len(halves[0])}
    starts = {"firstHalf": int(halves[0][0, 0]), "secondHalf": int(halves[1][0, 0])}
    keys = sorted(tracks)
    players = np.full((n, len(keys), 2), np.nan, dtype=np.float32)
    teams = np.array([0 if team == home else 1 for team, _ in keys], dtype=np.int8)
    for slot, key in enumerate(keys):
        for section, rows in tracks[key].items():
            index = rows[:, 0].astype(np.int64) - starts[section]
            keep = (index >= 0) & (index < len(ball[section]))
            players[index[keep] + offsets[section], slot] = rows[keep, 1:3]
    joined = np.concatenate(halves)
    size = np.array(pitch)
    # DFL coordinates are metres from the centre spot; the project uses 0-1 from a corner.
    players = players / size.astype(np.float32) + 0.5
    ball_xy = joined[:, 1:3] / size + 0.5
    possession = joined[:, 3].astype(np.int8)
    status = joined[:, 4].astype(np.int8)
    labels = np.where(status == 1, np.where(possession == 1, H, np.where(possession == 2, A, X)), D).astype(np.int8)
    labels[status < 0] = X
    blocks = np.array([[0, len(halves[0]) - 1], [len(halves[0]), n - 1]], dtype=np.int64)
    return dict(players=players, ball=ball_xy.astype(np.float32), teams=teams, labels=labels,
                possession=possession, status=status, frame_ids=joined[:, 0].astype(np.int64),
                blocks=blocks, pitch=size)


def merge_brief_spells(labels, blocks, seconds, fps=FPS):
    """Approximate "controlled" possession: a live spell shorter than `seconds` right after the
    other team's spell goes back to that team. Spells after a stoppage are kept: a restart has a
    clear possessor. D frames never change. This is a hypothesis about the definition, not DFL's.
    """
    merged = labels.copy()
    limit = seconds * fps
    for first, last in blocks:
        window = merged[first:last + 1]
        bounds = np.r_[0, np.flatnonzero(window[1:] != window[:-1]) + 1, len(window)]
        for start, stop in zip(bounds[1:], bounds[2:]):
            if (stop - start < limit and window[start] in (H, A) and window[start - 1] in (H, A)):
                window[start:stop] = window[start - 1]
    return merged


def load_match(match_id, data_dir=DATA):
    cache = Path(data_dir) / "cache" / f"{match_id}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
    else:
        arrays = build(match_id, data_dir)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **arrays)
    return Match(match_id=match_id, players=arrays["players"], ball=arrays["ball"], teams=arrays["teams"],
                 labels=arrays["labels"], possession=arrays["possession"], status=arrays["status"],
                 frame_ids=arrays["frame_ids"], blocks=[tuple(map(int, b)) for b in arrays["blocks"]],
                 pitch=tuple(map(float, arrays["pitch"])))


def events(match_id, data_dir=DATA):
    """[(utc seconds, kind, team id)] for match events; the event file's clock is ISO 8601."""
    from datetime import datetime
    out = []
    for item in ET.parse(source_path("03_02_events_raw", match_id, data_dir)).getroot().iter("Event"):
        children = list(item)
        if not children or not item.get("EventTime"):
            continue
        child = children[0]
        kind = child.tag
        team = child.get("Team") or child.get("WinnerTeam") or child.get("TeamFouled")
        inner = next(iter(child), None)
        if team is None and inner is not None:
            team = inner.get("Team")
        out.append((datetime.fromisoformat(item.get("EventTime")).timestamp(), kind, team,
                    inner.tag if inner is not None else None))
    return out


def frame_times(match_id, data_dir=DATA):
    """UTC seconds of the first ball frame in each half, for aligning events with frames."""
    from datetime import datetime
    path = source_path("04_03_positions_raw_observed", match_id, data_dir)
    out, ball_set = {}, None
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start" and element.tag == "FrameSet":
            ball_set = element.get("GameSection") if element.get("TeamId") == "BALL" else None
        elif event == "end":
            if element.tag == "Frame" and ball_set and ball_set not in out:
                out[ball_set] = (datetime.fromisoformat(element.get("T")).timestamp(), int(element.get("N")))
            element.clear()
            if len(out) == 2:
                break
    return out
