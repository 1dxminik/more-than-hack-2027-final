"""Turn one public IDSSE match into a file shaped like the competition ATD, with automatic-tracking noise.

The output has the same layout as the tactical ATD described in official/DATA.md: a FIFA/EPTS
metadata XML and a raw text file "frame:x,y,s;...(46 slots):ball_x,ball_y,ball_s" with
coordinates normalised to 0-1 from the top-left corner (Metrica convention), speeds in m/s and
NaN for missing observations. The frame count is 199,319 with the same padding around the
halves as the competition video. Labels go to a separate frame,label CSV, never into the input.

The degradation is driven only by public figures from official/ (ball missing on 40.7% of graded
frames, mean 20.5 tracked objects, 46 anonymous slots) and by stated assumptions in Config. It
never reads competition data. The result is a training/test aid, not competition performance.

    python -m scripts.degrade_idsse --match J03WMX --seed 0
    python -m scripts.degrade_idsse --match J03WMX --seed 0 --evaluate
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from scripts import idsse

ROOT = Path(__file__).resolve().parent.parent
FPS = 25
GENERATOR_VERSION = 1
# Public competition facts: official/DATA.md and official/CHALLENGE.md.
COMPETITION_FRAMES = 199319
COMPETITION_BEFORE = 11185      # video frames before the first graded frame
COMPETITION_HALFTIME = 20156    # ungraded frames between the two graded blocks
BALL_MISSING_GRADED = 1 - 94214 / 158857
MEAN_OBJECTS_GRADED = 20.5
N_SLOTS = 46
NAN_GROUP = "NaN,NaN,NaN"


@dataclass(frozen=True)
class Config:
    """Every knob of the degradation. Values not taken from official/ are assumptions."""
    seed: int = 0
    total_frames: int = COMPETITION_FRAMES
    frames_before: int = COMPETITION_BEFORE
    frames_halftime: int = COMPETITION_HALFTIME
    # Calibration targets (public, official/DATA.md).
    ball_missing_graded: float = BALL_MISSING_GRADED
    mean_objects_graded: float = MEAN_OBJECTS_GRADED
    # Assumption: the tactical camera sits on one long side, above the halfway line (DFL metres).
    camera_y: float = -64.0
    camera_height: float = 25.0
    # Dead ball: IDSSE freezes the ball, so its position there is removed. On average this share
    # of each stoppage (its end, where the ball is placed for the restart) is shown instead.
    dead_visible_share: float = 0.5
    # Live ball: occlusion score weights (assumptions) and the noise around them.
    ball_crowd_radius: float = 2.0
    ball_weight_crowd: float = 0.6
    ball_weight_air: float = 1.2
    ball_weight_fast: float = 0.6
    ball_weight_contact: float = 0.4
    ball_noise_frames: float = 12.0
    ball_min_visible_run: int = 3
    ball_noise_m: float = 0.25
    ball_false_per_minute: float = 0.3
    # Players.
    player_crowd_radius: float = 1.5
    player_weight_crowd: float = 0.8
    player_weight_far: float = 0.8
    player_noise_frames: float = 50.0
    player_min_visible_run: int = 10
    player_noise_near_m: float = 0.10
    player_noise_far_m: float = 0.35
    breaks_per_player_minute: float = 0.5
    swap_distance_m: float = 1.2
    swap_same_team: float = 0.3
    swap_other_team: float = 0.05
    duplicates_per_player_minute: float = 0.05
    duplicate_seconds: tuple = (0.5, 3.0)
    duplicate_offset_m: tuple = (0.4, 1.5)
    # Labels: 0 keeps IDSSE possession as is; >0 merges shorter live spells (idsse.merge_brief_spells).
    merge_seconds: float = 0.0


# ---------------------------------------------------------------- reading IDSSE

def parse_full(match_id, data_dir=idsse.DATA):
    """Positions in DFL metres (centre origin, y up), ball height kept for the occlusion model."""
    home, guest = idsse.team_roles(match_id, data_dir)
    names = {team.get("TeamId"): team.get("TeamName") for team in
             ET.parse(idsse.source_path("02_01_matchinformation", match_id, data_dir)).getroot().iter("Team")}
    tracks, ball, pitch = {}, {}, (105.0, 68.0)
    current, team, person, section = None, None, None, None
    path = idsse.source_path("04_03_positions_raw_observed", match_id, data_dir)
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
                    row += (float(element.get("Z", 0)), int(element.get("BallPossession", 0)),
                            int(element.get("BallStatus", -1)))
                current.append(row)
            element.clear()
        elif element.tag == "FrameSet":
            if current:
                target = ball if team == "BALL" else tracks.setdefault((team, person), {})
                target[section] = np.asarray(current, dtype=np.float64)
            current = None
            element.clear()
    if set(ball) != {"firstHalf", "secondHalf"}:
        raise ValueError(f"{match_id}: expected exactly two halves with ball data")
    halves = [ball["firstHalf"], ball["secondHalf"]]
    for half in halves:
        if (np.diff(half[:, 0]) != 1).any():
            raise ValueError(f"{match_id}: ball frames are not contiguous inside a half")
    n = sum(len(half) for half in halves)
    offsets = {"firstHalf": 0, "secondHalf": len(halves[0])}
    starts = {"firstHalf": int(halves[0][0, 0]), "secondHalf": int(halves[1][0, 0])}
    keys = sorted(tracks)
    players = np.full((n, len(keys), 2), np.nan)
    for slot, key in enumerate(keys):
        for section, rows in tracks[key].items():
            index = rows[:, 0].astype(np.int64) - starts[section]
            keep = (index >= 0) & (index < len(ball[section]))
            players[index[keep] + offsets[section], slot] = rows[keep, 1:3]
    joined = np.concatenate(halves)
    return dict(players=players, teams=np.array([0 if t == home else 1 for t, _ in keys], dtype=np.int8),
                ball=joined[:, 1:4], possession=joined[:, 4].astype(np.int8),
                status=joined[:, 5].astype(np.int8), half_lengths=np.array([len(h) for h in halves]),
                pitch=np.array(pitch), team_names=np.array([names.get(home, "Home"), names.get(guest, "Away")]))


def load_full(match_id, data_dir=idsse.DATA):
    cache = Path(data_dir) / "cache" / f"{match_id}_full.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            return {key: stored[key] for key in stored.files}
    arrays = parse_full(match_id, data_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **arrays)
    return arrays


# ---------------------------------------------------------------- helpers

def runs(mask):
    """[(start, stop)) of True runs in a 1-D boolean array."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def ar_noise(rng, shape, tau_frames):
    """Unit-variance AR(1) noise along axis 0 with correlation time `tau_frames`."""
    a = math.exp(-1.0 / max(tau_frames, 1e-9))
    shocks = rng.standard_normal(shape) * math.sqrt(1 - a * a)
    out = np.empty(shape)
    out[0] = rng.standard_normal(shape[1:])
    for t in range(1, shape[0]):
        out[t] = a * out[t - 1] + shocks[t]
    return out


def drop_short(mask, min_run):
    """Clear True runs shorter than min_run (a tracker needs a few frames to confirm an object)."""
    out = mask.copy()
    for start, stop in runs(mask):
        if stop - start < min_run:
            out[start:stop] = False
    return out


def bisect(function, target, low, high, steps=40):
    """Threshold where a decreasing function crosses target."""
    for _ in range(steps):
        middle = (low + high) / 2
        if function(middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def half_masks(n, half_lengths):
    """[(start, stop)) of the two halves in IDSSE frame order."""
    return [(0, int(half_lengths[0])), (int(half_lengths[0]), n)]


# ---------------------------------------------------------------- ball

def ball_speed_true(ball, blocks):
    speed = np.full(len(ball), np.nan)
    for start, stop in blocks:
        xy = ball[start:stop, :2]
        speed[start:stop] = np.linalg.norm(np.gradient(xy, axis=0), axis=1) * FPS
    return speed


def project_to_ground(xy, z, config):
    """Where a single elevated camera sees a ball at height z when it assumes the ball is on the grass."""
    camera = np.array([0.0, config.camera_y])
    scale = config.camera_height / np.maximum(config.camera_height - np.clip(z, 0, None), 1.0)
    return camera + (xy - camera) * scale[:, None]


def degrade_ball(data, blocks, rng, config):
    ball, status, players = data["ball"], data["status"], data["players"]
    length, width = data["pitch"]
    n = len(ball)
    live = status == 1
    out = np.full((n, 2), np.nan)
    visible = np.zeros(n, dtype=bool)

    # Dead ball: IDSSE's frozen position is dropped; the end of a stoppage shows the restart spot.
    dead_visible = np.zeros(n, dtype=bool)
    for start, stop in runs(status == 0):
        if stop >= n or status[stop] != 1:
            continue
        share = min(1.0, rng.uniform(0, 2 * config.dead_visible_share))
        shown = int(round(share * (stop - start)))
        if shown:
            dead_visible[stop - shown:stop] = True
            out[stop - shown:stop] = ball[stop, :2]

    # Live ball: occlusion score from crowd, height, speed and contact, plus correlated noise.
    distance = np.linalg.norm(players - ball[:, None, :2], axis=2)
    with np.errstate(invalid="ignore"):
        crowd = np.sum(distance < config.ball_crowd_radius, axis=1)
    nearest = np.nanmin(np.where(np.isnan(distance), np.inf, distance), axis=1)
    speed = ball_speed_true(ball, blocks)
    air = np.clip((ball[:, 2] - 1.0) / 2.0, 0, 1.5)
    fast = np.clip((speed - 12.0) / 10.0, 0, 2.0)
    score = (config.ball_weight_crowd * crowd + config.ball_weight_air * air
             + config.ball_weight_fast * fast + config.ball_weight_contact * np.exp(-nearest))
    latent = score + ar_noise(rng, (n,), config.ball_noise_frames)
    on_pitch = (np.abs(ball[:, 0]) <= length / 2 + 0.5) & (np.abs(ball[:, 1]) <= width / 2 + 0.5)
    candidates = live & on_pitch
    target_missing = config.ball_missing_graded * n - (np.sum(~live) - np.sum(dead_visible))

    def missing_for(threshold):
        seen = candidates & (latent <= threshold)
        seen = np.concatenate([drop_short(seen[a:b], config.ball_min_visible_run) for a, b in blocks])
        return np.sum(live) - np.sum(seen), seen

    threshold = bisect(lambda t: missing_for(t)[0], target_missing, latent.min() - 1, latent.max() + 1)
    live_visible = missing_for(threshold)[1]
    visible = live_visible | dead_visible
    ground = project_to_ground(ball[:, :2], ball[:, 2], config)
    out[live_visible] = ground[live_visible]
    noise = ar_noise(rng, (n, 2), 5.0) * config.ball_noise_m
    out[visible] += noise[visible]

    # Rare false detections: a short burst at a random player's position (head or feet).
    false_frames = 0
    expected = config.ball_false_per_minute * n / (60 * FPS)
    for _ in range(rng.poisson(expected)):
        t = int(rng.integers(0, n))
        duration = int(rng.integers(1, 6))
        span = slice(t, min(n, t + duration))
        candidates_now = np.flatnonzero(~np.isnan(players[t, :, 0]))
        if len(candidates_now) == 0:
            continue
        who = rng.choice(candidates_now)
        out[span] = players[span, who] + rng.normal(0, 0.3, 2)
        visible[span] = ~np.isnan(out[span, 0])
        false_frames += int(np.sum(visible[span]))
    out[~visible] = np.nan
    stats = {"live_missing_share": float(1 - np.sum(live_visible) / max(np.sum(live), 1)),
             "dead_visible_share": float(np.sum(dead_visible) / max(np.sum(~live), 1)),
             "false_detection_frames": false_frames, "threshold": float(threshold)}
    return out, stats


# ---------------------------------------------------------------- players

def far_side(y, pitch_width, config):
    """0 at the camera's touchline side of the pitch region, 1 at the far touchline."""
    return np.nan_to_num(np.clip((y - config.camera_y) / (pitch_width / 2 - config.camera_y), 0, 1))


def player_noise(data, rng, config):
    players = data["players"]
    far = far_side(players[:, :, 1], data["pitch"][1], config)
    sigma = config.player_noise_near_m + (config.player_noise_far_m - config.player_noise_near_m) * far
    return players + ar_noise(rng, players.shape, 8.0) * sigma[:, :, None]


def calibrated_visibility(data, blocks, rng, config, extra_objects):
    """Occlusion mask per person, calibrated so frames average the published object count."""
    players = data["players"]
    n, people = players.shape[:2]
    present = ~np.isnan(players[:, :, 0])
    crowd = np.zeros((n, people))
    for start in range(0, n, 20000):
        chunk = players[start:start + 20000]
        gap = np.linalg.norm(chunk[:, :, None] - chunk[:, None], axis=3)
        with np.errstate(invalid="ignore"):
            crowd[start:start + 20000] = np.sum(gap < config.player_crowd_radius, axis=2) - 1
    far = far_side(players[:, :, 1], data["pitch"][1], config)
    score = config.player_weight_crowd * np.clip(crowd, 0, None) + config.player_weight_far * far
    latent = score + ar_noise(rng, (n, people), config.player_noise_frames)
    target = config.mean_objects_graded * n - extra_objects

    def visible_for(threshold):
        seen = present & (latent <= threshold)
        for person in range(people):
            seen[:, person] = np.concatenate([drop_short(seen[a:b, person], config.player_min_visible_run)
                                              for a, b in blocks])
        return seen

    low, high = float(latent[present].min()) - 1, float(latent[present].max()) + 1
    for _ in range(30):
        middle = (low + high) / 2
        if visible_for(middle).sum() < target:
            low = middle
        else:
            high = middle
    return visible_for((low + high) / 2)


def build_segments(data, visible, blocks, rng, config):
    """Cut visible runs into tracklets: random breaks, identity swaps on close encounters, duplicates.

    Returns segments (tracklet, person, start, stop, dx, dy) and the tracklet team (from its first
    person). A tracklet can carry several people after swaps, exactly as a slot does in the ATD.
    """
    players, teams = data["players"], data["teams"]
    n, people = visible.shape
    events = []  # (frame, order, kind, a, b)
    for person in range(people):
        for start, stop in runs(visible[:, person]):
            events.append((start, 1, "start", person, None))
            events.append((stop, 0, "stop", person, None))
            length = stop - start
            for _ in range(rng.poisson(config.breaks_per_player_minute * length / (60 * FPS))):
                cut = int(rng.integers(start + 1, stop)) if length > 1 else start
                if cut > start:
                    events.append((cut, 2, "break", person, None))
    for a in range(people):
        for b in range(a + 1, people):
            both = visible[:, a] & visible[:, b]
            if not both.any():
                continue
            with np.errstate(invalid="ignore"):
                close = both & (np.linalg.norm(players[:, a] - players[:, b], axis=1) < config.swap_distance_m)
            probability = config.swap_same_team if teams[a] == teams[b] else config.swap_other_team
            for start, stop in runs(close):
                if stop < n and both[stop] and rng.random() < probability:
                    events.append((stop, 3, "swap", a, b))
    events.sort(key=lambda event: (event[0], event[1]))

    segments, tracklet_team = [], []
    current, opened = {}, {}

    def open_tracklet(person, frame):
        tracklet_team.append(int(teams[person]))
        current[person] = len(tracklet_team) - 1
        opened[person] = frame

    def close(person, frame):
        if person in current and frame > opened[person]:
            segments.append((current[person], person, opened[person], frame, 0.0, 0.0))

    for frame, _, kind, a, b in events:
        if kind == "start":
            open_tracklet(a, frame)
        elif kind == "stop":
            close(a, frame)
            current.pop(a, None)
        elif kind == "break" and a in current:
            close(a, frame)
            open_tracklet(a, frame)
        elif kind == "swap" and a in current and b in current:
            close(a, frame)
            close(b, frame)
            current[a], current[b] = current[b], current[a]
            opened[a] = opened[b] = frame

    # Duplicates: a ghost of a visible player, slightly offset, as its own short tracklet.
    minutes = visible.sum() / (60 * FPS)
    duplicates = 0
    for _ in range(rng.poisson(config.duplicates_per_player_minute * minutes)):
        t = int(rng.integers(0, n))
        options = np.flatnonzero(visible[t])
        if len(options) == 0:
            continue
        person = int(rng.choice(options))
        stop = t + int(rng.uniform(*config.duplicate_seconds) * FPS)
        stop = min(stop, t + int(np.argmin(visible[t:, person])) if not visible[t:, person].all() else n)
        if stop - t < 3:
            continue
        angle = rng.uniform(0, 2 * math.pi)
        radius = rng.uniform(*config.duplicate_offset_m)
        tracklet_team.append(int(teams[person]))
        segments.append((len(tracklet_team) - 1, person, t, stop, radius * math.cos(angle), radius * math.sin(angle)))
        duplicates += stop - t
    return segments, np.array(tracklet_team, dtype=np.int8), duplicates


def assign_slots(segments, tracklet_team, n_slots, rng):
    """Give every tracklet a slot of its team; a freed slot is reused least-recently-first."""
    per_team = n_slots // 2
    slot_team = np.array([0] * per_team + [1] * (n_slots - per_team), dtype=np.int8)
    spans = {}
    for tracklet, _, start, stop, _, _ in segments:
        first, last = spans.get(tracklet, (start, stop))
        spans[tracklet] = (min(first, start), max(last, stop))
    free = {team: list(rng.permutation(np.flatnonzero(slot_team == team))) for team in (0, 1)}
    busy, slot_of, dropped = [], {}, 0
    for tracklet, (start, stop) in sorted(spans.items(), key=lambda item: item[1][0]):
        busy.sort()
        while busy and busy[0][0] <= start:
            _, released = busy.pop(0)
            free[int(slot_team[released])].append(released)
        team = int(tracklet_team[tracklet])
        if not free[team]:
            dropped += 1
            continue
        slot = int(free[team].pop(0))
        slot_of[tracklet] = slot
        busy.append((stop, slot))
    return slot_of, slot_team, dropped


def speeds(xy, ids):
    """Central-difference speed in m/s, only where the same tracklet covers t-2..t+2."""
    n = len(xy)
    speed = np.full(xy.shape[:-1], np.nan)
    if n < 5:
        return speed
    step = np.linalg.norm(xy[4:] - xy[:-4], axis=-1) / (4 / FPS)
    same = (ids[4:] == ids[:-4]) & (ids[4:] == ids[2:-2]) & (ids[2:-2] >= 0)
    speed[2:-2] = np.where(same, step, np.nan)
    return speed


# ---------------------------------------------------------------- whole match

def degrade(data, config):
    rng = np.random.default_rng(config.seed)
    n = len(data["ball"])
    blocks = half_masks(n, data["half_lengths"])
    length, width = data["pitch"]
    ball, ball_stats = degrade_ball(data, blocks, rng, config)
    noisy = player_noise(data, rng, config)

    expected_duplicates = config.duplicates_per_player_minute * 22 * n / (60 * FPS) * np.mean(config.duplicate_seconds) * FPS
    visible = calibrated_visibility(data, blocks, rng, config, expected_duplicates)
    segments, tracklet_team, duplicate_frames = build_segments(data, visible, blocks, rng, config)
    slot_of, slot_team, dropped = assign_slots(segments, tracklet_team, N_SLOTS, rng)

    slots = np.full((n, N_SLOTS, 2), np.nan)
    ids = np.full((n, N_SLOTS), -1, dtype=np.int64)
    for tracklet, person, start, stop, dx, dy in segments:
        slot = slot_of.get(tracklet)
        if slot is None:
            continue
        slots[start:stop, slot] = noisy[start:stop, person] + (dx, dy)
        ids[start:stop, slot] = tracklet

    # Normalise: Metrica's 0-1 from the top-left corner, y pointing down; DFL is metres, y up.
    def normalise(xy):
        return np.stack([xy[..., 0] / length + 0.5, 0.5 - xy[..., 1] / width], axis=-1)

    slot_speed = speeds(slots, ids)
    ball_ids = np.where(np.isnan(ball[:, 0]), -1, 0)
    ball_speed = speeds(ball[:, None], ball_ids[:, None])[:, 0]
    objects = np.sum(~np.isnan(slots[:, :, 0]), axis=1)
    teams_in = {}
    for tracklet, person, *_ in segments:
        teams_in.setdefault(tracklet, set()).add(int(data["teams"][person]))
    swaps_cross = sum(len(seen) > 1 for seen in teams_in.values())
    stats = {"ball": ball_stats, "tracklets": int(len(tracklet_team)), "tracklets_dropped": int(dropped),
             "duplicate_frames": int(duplicate_frames), "mean_objects": float(objects.mean()),
             "frames_over_22": int(np.sum(objects > 22)), "mixed_team_tracklets": swaps_cross}
    return dict(slots=normalise(slots), slot_speed=slot_speed, slot_team=slot_team,
                ball=normalise(ball), ball_speed=ball_speed, stats=stats)


def labels_for(data, config):
    status, possession = data["status"], data["possession"]
    codes = np.where(status == 1, np.where(possession == 1, idsse.H, np.where(possession == 2, idsse.A, idsse.X)),
                     idsse.D).astype(np.int8)
    codes[status < 0] = idsse.X
    if config.merge_seconds > 0:
        n = len(codes)
        blocks = [(a, b - 1) for a, b in half_masks(n, data["half_lengths"])]
        codes = idsse.merge_brief_spells(codes, blocks, config.merge_seconds)
    return np.array(list("HADX"))[codes]


def layout(data, config):
    """Video-frame index of every IDSSE frame, padded like the competition video."""
    first, second = (int(x) for x in data["half_lengths"])
    after = config.total_frames - config.frames_before - first - config.frames_halftime - second
    if after < 0:
        raise ValueError("Match is longer than total_frames; lower frames_before or frames_halftime")
    start2 = config.frames_before + first + config.frames_halftime
    index = np.r_[np.arange(config.frames_before, config.frames_before + first), np.arange(start2, start2 + second)]
    graded = [(config.frames_before, config.frames_before + first - 1), (start2, start2 + second - 1)]
    return index, graded


# ---------------------------------------------------------------- writing

def parameter(parent, name, value):
    item = ET.SubElement(parent, "ProviderParameter")
    ET.SubElement(item, "Name").text = name
    ET.SubElement(item, "Value").text = str(value)


def metadata_xml(team_names, slot_team, graded, total, home_attacks_right, source_note):
    root = ET.Element("main")
    root.append(ET.Comment(source_note))
    meta = ET.SubElement(root, "Metadata")
    config = ET.SubElement(meta, "GlobalConfig")
    ET.SubElement(config, "TrackingType").text = "Optical"
    ET.SubElement(config, "ProviderName").text = "Metrica Sports"  # kloppy picks its coordinate system from this
    ET.SubElement(config, "FrameRate").text = str(FPS)
    params = ET.SubElement(config, "ProviderGlobalParameters")
    for name, (first, last) in zip(("first_half", "second_half"), graded):
        parameter(params, name + "_start", first + 1)
        parameter(params, name + "_end", last + 1)
    session = ET.SubElement(ET.SubElement(meta, "Sessions"), "Session", id="1")
    field = ET.SubElement(session, "FieldSize")  # kloppy reads Session.FieldSize directly
    ET.SubElement(field, "Width").text = "105"
    ET.SubElement(field, "Height").text = "68"
    teams = ET.SubElement(meta, "Teams")
    team_ids = ("TEAM_HOME", "TEAM_AWAY")
    for index, (team_id, name) in enumerate(zip(team_ids, team_names)):
        team = ET.SubElement(teams, "Team", id=team_id)
        ET.SubElement(team, "Name").text = str(name)
        right = home_attacks_right if index == 0 else not home_attacks_right
        parameter(ET.SubElement(team, "ProviderTeamsParameters"), "attack_direction_first_half",
                  "left_to_right" if right else "right_to_left")
    players = ET.SubElement(meta, "Players")
    for slot, team in enumerate(slot_team, start=1):
        player = ET.SubElement(players, "Player", id=str(slot), teamId=team_ids[int(team)])
        ET.SubElement(player, "Name").text = f"Track {slot}"
        ET.SubElement(player, "ShirtNumber").text = str(slot)
    sensors = ET.SubElement(ET.SubElement(ET.SubElement(meta, "Devices"), "Device", id="device1"), "Sensors")
    for sensor_id, channels in (("position", (("x", "normalized"), ("y", "normalized"))),
                                ("speed", (("s", "meters per second"),))):
        sensor = ET.SubElement(sensors, "Sensor", id=sensor_id)
        ET.SubElement(sensor, "Name").text = sensor_id
        group = ET.SubElement(sensor, "Channels")
        for channel_id, unit in channels:
            channel = ET.SubElement(group, "Channel", id=channel_id)
            ET.SubElement(channel, "Name").text = channel_id
            ET.SubElement(channel, "Unit").text = unit
    player_channels = ET.SubElement(meta, "PlayerChannels")
    for slot in range(1, len(slot_team) + 1):
        for channel in "xys":
            ET.SubElement(player_channels, "PlayerChannel", id=f"player{slot}_{channel}",
                          playerId=str(slot), channelId=channel)
    spec = ET.SubElement(ET.SubElement(root, "DataFormatSpecifications"), "DataFormatSpecification",
                         startFrame="1", endFrame=str(total), separator=":")
    ET.SubElement(spec, "StringRegister", name="frameCount")
    groups = ET.SubElement(spec, "SplitRegister", separator=";")
    for slot in range(1, len(slot_team) + 1):
        group = ET.SubElement(groups, "SplitRegister", separator=",")
        for channel in "xys":
            ET.SubElement(group, "PlayerChannelRef", playerChannelId=f"player{slot}_{channel}")
    ball = ET.SubElement(spec, "SplitRegister", separator=",")
    for channel in "xys":
        ET.SubElement(ball, "BallChannelRef", channelId=channel)
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def group_text(x, y, s):
    if math.isnan(x) or math.isnan(y):
        return NAN_GROUP
    return f"{x:.5f},{y:.5f}," + ("NaN" if math.isnan(s) else f"{s:.2f}")


def write_raw(path, out, index, total):
    slots, speed, ball, ball_speed = out["slots"], out["slot_speed"], out["ball"], out["ball_speed"]
    empty = ";".join([NAN_GROUP] * slots.shape[1]) + ":" + NAN_GROUP
    row_of = np.full(total, -1, dtype=np.int64)
    row_of[index] = np.arange(len(index))
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for frame in range(total):
            row = row_of[frame]
            if row < 0:
                stream.write(f"{frame + 1}:{empty}\n")
                continue
            groups = ";".join(group_text(*slots[row, slot], speed[row, slot]) for slot in range(slots.shape[1]))
            stream.write(f"{frame + 1}:{groups}:{group_text(*ball[row], ball_speed[row])}\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(match_id, out_dir, config, data_dir=idsse.DATA):
    data = load_full(match_id, data_dir)
    out = degrade(data, config)
    index, graded = layout(data, config)
    letters = labels_for(data, config)
    labels = np.full(config.total_frames, "X")
    labels[index] = letters

    out_dir = Path(out_dir)
    tactical = out_dir / "tactical"
    tactical.mkdir(parents=True, exist_ok=True)
    stem = f"{match_id}_tactical"
    first_half = data["players"][: int(data["half_lengths"][0])]
    home_mean_x = np.nanmean(first_half[:, data["teams"] == 0, 0])
    note = (f" Derived from IDSSE {match_id} (DFL/Sportec, CC BY 4.0, doi:10.1038/s41597-025-04505-y)"
            f" by scripts/degrade_idsse.py v{GENERATOR_VERSION}, seed {config.seed}. Not competition data. ")
    files = {"atd_meta": tactical / f"{stem}_atd.xml", "atd_raw": tactical / f"{stem}_atd.txt",
             "gt": out_dir / f"{match_id}_gt.csv"}
    files["atd_meta"].write_text(metadata_xml(data["team_names"], out["slot_team"], graded, config.total_frames,
                                              home_mean_x < 0, note), encoding="utf-8", newline="\n")
    write_raw(files["atd_raw"], out, index, config.total_frames)
    with files["gt"].open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("frame,label\n")
        stream.writelines(f"{i},{label}\n" for i, label in enumerate(labels))

    graded_mask = np.zeros(config.total_frames, dtype=bool)
    for first, last in graded:
        graded_mask[first:last + 1] = True
    ball_present = np.zeros(config.total_frames, dtype=bool)
    ball_present[index] = ~np.isnan(out["ball"][:, 0])
    objects = np.zeros(config.total_frames)
    objects[index] = np.sum(~np.isnan(out["slots"][:, :, 0]), axis=1)
    spells = np.sum((letters[1:] != letters[:-1]) & np.isin(letters[1:], ["H", "A"])) + int(letters[0] in "HA")
    manifest = {
        "kind": "idsse_degraded", "generator_version": GENERATOR_VERSION,
        "source": {"dataset": "IDSSE", "match_id": match_id, "license": "CC BY 4.0",
                   "doi": "10.1038/s41597-025-04505-y", "data_owner": "Deutsche Fussball Liga (DFL)"},
        "warning": "Derived from a public Bundesliga match. Not competition data or performance.",
        "config": asdict(config),
        "graded_blocks": [list(block) for block in graded],
        "labels": {label: int(np.sum(labels == label)) for label in "HADX"},
        "n_spells_approx": int(spells),
        "checks": {"ball_missing_graded": float(1 - ball_present[graded_mask].mean()),
                   "ball_present_all": float(ball_present.mean()),
                   "mean_objects_graded": float(objects[graded_mask].mean()),
                   "targets": {"ball_missing_graded": config.ball_missing_graded,
                               "mean_objects_graded": config.mean_objects_graded}},
        "degradation": out["stats"],
        "files": {name: path.relative_to(out_dir).as_posix() for name, path in files.items()},
        "sha256": {path.relative_to(out_dir).as_posix(): sha256(path) for path in files.values()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8", newline="\n")
    return manifest, files


# ---------------------------------------------------------------- optional checks

def evaluate(out_dir, manifest, files, pitch=(105.0, 68.0)):
    """Official baseline and scorer on the derived match; data_quality report next to it."""
    import subprocess
    sys.path.insert(0, str(ROOT / "official"))
    sys.path.insert(0, str(ROOT / "official" / "starter"))
    import baseline  # noqa: E402  (official, unchanged)
    out_dir = Path(out_dir)
    results = out_dir / "baseline"
    results.mkdir(exist_ok=True)
    n = manifest["config"]["total_frames"]
    codes = baseline.nearest_player_labels(str(files["atd_raw"]), baseline.team_of_each_track(str(files["atd_meta"])),
                                           n, pitch)
    for name, window in baseline.VARIANTS:
        baseline.write_submission(str(results / f"{name}.csv"), baseline.smooth(codes, window))
    subprocess.run([sys.executable, str(ROOT / "official/evaluation/run_evaluation.py"), "--gt", str(files["gt"]),
                    "--results", str(results), "--out", str(out_dir / "evaluation"), "--bootstrap", "0",
                    "--no-detail"], check=True)
    blocks = ",".join(f"{a}-{b}" for a, b in manifest["graded_blocks"])
    subprocess.run([sys.executable, str(ROOT / "scripts/data_quality.py"), "analyze", "--atd-xml", str(files["atd_meta"]),
                    "--atd-raw", str(files["atd_raw"]), "--graded", blocks, "--name", "IDSSE-degraded",
                    "--out", str(out_dir / "quality")], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match", default="J03WMX", help="IDSSE match id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--merge-seconds", type=float, default=0.0,
                        help="Merge live spells shorter than this into the previous team (0 = raw IDSSE labels)")
    parser.add_argument("--dead-visible-share", type=float, default=Config.dead_visible_share)
    parser.add_argument("--out", type=Path, help="Default: datasets/idsse/degraded/<match>/seed<seed>")
    parser.add_argument("--evaluate", action="store_true", help="Run the official baseline, scorer and data_quality")
    args = parser.parse_args(argv)
    config = Config(seed=args.seed, merge_seconds=args.merge_seconds, dead_visible_share=args.dead_visible_share)
    out_dir = args.out or idsse.DATA / "degraded" / args.match / f"seed{args.seed}"
    manifest, files = generate(args.match, out_dir, config)
    print(json.dumps({"out": str(out_dir), "checks": manifest["checks"], "labels": manifest["labels"],
                      "degradation": manifest["degradation"]}, indent=2))
    if args.evaluate:
        evaluate(out_dir, manifest, files)


if __name__ == "__main__":
    main()
