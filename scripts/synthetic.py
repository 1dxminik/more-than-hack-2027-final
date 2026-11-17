"""Independent invented 46-slot tracking, without any real-match input."""
from pathlib import Path
import json
import numpy as np

from solution.assets import digest


def observations(frames=6000):
    if type(frames) is not int or frames < 300:
        raise ValueError("Use at least 300 invented frames")
    teams = np.repeat([0, 1], 23)
    time = np.arange(frames) / 25.0
    index = np.arange(frames)
    phase = (index // 300) % 7
    motion = np.cumsum(np.where(phase == 4, 0.0, 0.04))
    xy = np.empty((frames, 22, 2))
    for i in range(22):
        side, local = divmod(i, 11)
        xy[:, i, 0] = .20 + .36 * side + .014 * local + .018 * np.sin(1.9 * motion + i)
        xy[:, i, 1] = .20 + .05 * local + .012 * np.cos(1.3 * motion + i)
    players = np.full((frames, 46, 2), np.nan)
    rotation = (index // 100) % 23
    for i in range(22):
        side, local = divmod(i, 11)
        slots = side * 23 + (local + rotation) % 23
        players[index, slots] = xy[:, i]
    players[index[index % 97 == 0], rotation[index % 97 == 0]] = np.nan
    carrier = np.where(phase == 2, 11, 0)
    carrier[(phase == 0) & (index % 233 < 3)] = 11
    ball = xy[index, carrier] + [.003, .001]
    flight = phase == 1
    t = (index[flight] % 300) / 299.0
    ball[flight] = xy[flight, 0] * (1 - t[:, None]) + xy[flight, 11] * t[:, None]
    ball[phase == 4] = [.95, .95]
    ball[phase == 5] = [.001, .50]
    ball[(phase == 3) | (phase == 6) | ((index % 181 >= 120) & (index % 181 < 145))] = np.nan
    players[phase == 6] = np.nan
    return players, ball, teams


def generate(folder, frames):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    tactical = folder / "tactical"
    tactical.mkdir()
    xml, raw = tactical / "synthetic_atd.xml", tactical / "synthetic_atd.txt"
    players, ball, teams = observations(frames)
    lines = ['<?xml version="1.0" encoding="utf-8"?>', '<SyntheticC5Fixture>',
             '<Team id="synthetic-home"/>', '<Team id="synthetic-away"/>']
    lines.extend(f'<Player id="{i + 1}" teamId="synthetic-{"home" if side == 0 else "away"}"/>' for i, side in enumerate(teams))
    lines.append('</SyntheticC5Fixture>')
    xml.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    def group(xy):
        return "NaN,NaN,NaN" if not np.isfinite(xy).all() else f"{xy[0]:.7f},{xy[1]:.7f},NaN"
    with raw.open("w", encoding="utf-8", newline="\n") as stream:
        for i in range(frames):
            stream.write(f"{i + 1}:" + ";".join(group(xy) for xy in players[i]) + ":" + group(ball[i]) + "\n")
    metadata = dict(kind="wholly_synthetic", frames=frames, slots=46, max_visible=22,
                    source_sha256=digest(Path(__file__)),
                    files={p.relative_to(folder).as_posix(): digest(p) for p in (xml, raw)})
    (folder / "synthetic_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def verify(folder):
    folder = Path(folder)
    meta = json.loads((folder / "synthetic_manifest.json").read_text(encoding="utf-8"))
    if meta.get("kind") != "wholly_synthetic" or meta.get("source_sha256") != digest(Path(__file__)):
        raise ValueError("Expected the unchanged independent C5 synthetic fixture")
    expected = {"tactical/synthetic_atd.xml", "tactical/synthetic_atd.txt"}
    if set(meta["files"]) != expected:
        raise ValueError("Unexpected synthetic input manifest")
    for name, expected_hash in meta["files"].items():
        path = folder / name
        if path.is_symlink() or digest(path) != expected_hash:
            raise ValueError("Synthetic input changed")
    return meta
