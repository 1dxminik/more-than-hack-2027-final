"""Read the explicit synthetic ATD file; never search for a dataset."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def load_tracking(metadata_path, raw_path, n_frames):
    """Preserve XML slot order and normalised coordinates, including NaN.

    As in the pinned official baseline, the first Team is home (0), the second
    away (1). A slot is not a player identity. ATD frame 1 becomes array row 0.
    This is the baseline's minimal XML/raw contract, not a full EPTS parser.
    """
    if isinstance(n_frames, bool) or not isinstance(n_frames, (int, np.integer)) or n_frames <= 0:
        raise ValueError("Expected a positive integer frame count")
    try:
        root = ET.parse(metadata_path).getroot()
    except ET.ParseError as exc:
        raise ValueError("Invalid tracking metadata XML") from exc
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2 or any(not team or not team.strip() for team in team_ids) or len(set(team_ids)) != 2:
        raise ValueError("Expected two distinct teams")
    slots = list(root.iter("Player"))
    slot_ids = [player.get("id") for player in slots]
    if any(not slot or not slot.strip() for slot in slot_ids) or len(set(slot_ids)) != len(slot_ids):
        raise ValueError("Expected distinct non-empty track IDs")
    slot_teams = [player.get("teamId") for player in slots]
    if not slot_teams or any(team not in team_ids for team in slot_teams):
        raise ValueError("Invalid track-to-team mapping")
    teams = np.array([team_ids.index(team) for team in slot_teams])
    players = np.full((n_frames, len(teams), 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    count = 0
    with Path(raw_path).open(encoding="utf-8") as stream:
        for count, line in enumerate(stream, start=1):
            parts = line.strip().split(":")
            if len(parts) != 3:
                raise ValueError(f"ATD record {count}: expected frame:players:ball")
            frame, player_text, ball_text = parts
            if count > n_frames or int(frame) != count:
                raise ValueError("ATD frames must run from 1 to n_frames in order")
            groups = player_text.split(";")
            if len(groups) != len(teams):
                raise ValueError("ATD track count differs from metadata")
            try:
                values = [list(map(float, group.split(","))) for group in groups]
                ball_values = list(map(float, ball_text.split(",")))
            except ValueError as exc:
                raise ValueError(f"ATD record {count}: expected numeric values or NaN") from exc
            if any(len(group) != 3 for group in values) or len(ball_values) != 3:
                raise ValueError("Expected x,y,speed for every object")
            if np.isinf(values).any() or np.isinf(ball_values).any():
                raise ValueError(f"ATD record {count}: infinity is not an observation or NaN")
            players[count - 1] = np.asarray(values)[:, :2]
            ball[count - 1] = ball_values[:2]
    if count != n_frames:
        raise ValueError("ATD row count differs from expected frame count")
    return players, ball, teams
