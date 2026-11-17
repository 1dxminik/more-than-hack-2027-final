"""Read the explicit synthetic ATD file; never search for a dataset."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def load_tracking(metadata_path, raw_path, n_frames):
    root = ET.parse(metadata_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2 or len(set(team_ids)) != 2:
        raise ValueError("Expected two distinct teams")
    slot_teams = [player.get("teamId") for player in root.iter("Player")]
    if not slot_teams or any(team not in team_ids for team in slot_teams):
        raise ValueError("Invalid track-to-team mapping")
    teams = np.array([team_ids.index(team) for team in slot_teams])
    players = np.full((n_frames, len(teams), 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    count = 0
    with Path(raw_path).open(encoding="utf-8") as stream:
        for count, line in enumerate(stream, start=1):
            frame, player_text, ball_text = line.strip().split(":")
            if count > n_frames or int(frame) != count:
                raise ValueError("ATD frames must run from 1 to n_frames in order")
            groups = player_text.split(";")
            if len(groups) != len(teams):
                raise ValueError("ATD track count differs from metadata")
            values = [list(map(float, group.split(","))) for group in groups]
            ball_values = list(map(float, ball_text.split(",")))
            if any(len(group) != 3 for group in values) or len(ball_values) != 3:
                raise ValueError("Expected x,y,speed for every object")
            players[count - 1] = np.asarray(values)[:, :2]
            ball[count - 1] = ball_values[:2]
    if count != n_frames:
        raise ValueError("ATD row count differs from expected frame count")
    return players, ball, teams
