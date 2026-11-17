"""Aggregate quality report for one ATD recording; two reports can be compared side by side.

Standalone on purpose (standard library and numpy): copy this one file out of the project and
run it next to the data, without an assistant. Reports hold counts and percentages only, never
positions, frame numbers, team names, file paths or file content. Thresholds are engineering
assumptions and describe the tracking; they say nothing about possession or expected score.

    python data_quality.py analyze --atd-xml M.xml --atd-raw M.txt --preset bar-ath --out DIR
    python data_quality.py compare DIR_A/quality_report.json DIR_B/quality_report.json --out DIR
"""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


VERSION = 1
BALL_MAX_SPEED_MPS = 45.0
PLAYER_MAX_SPEED_MPS = 12.0
DUPLICATE_RADIUS_M = 0.5
DUPLICATE_MIN_SECONDS = 1.0
CONTROL_RADIUS_M = 2.0
CROWD_RADIUS_M = 5.0
LONG_GAP_SECONDS = 1.0
BUCKET_MINUTES = 5
CHUNK_FRAMES = 20000
GAP_BINS = (("under_1s", 0, 1), ("from_1_to_5s", 1, 5), ("from_5_to_30s", 5, 30), ("over_30s", 30, math.inf))

# Published by the organisers in official/DATA.md and official/FAQ.md. Used only to prove that
# this script read the file correctly; a mismatch means the script or the file is wrong.
PRESETS = {
    "bar-ath": {
        "blocks": ((11185, 90651), (110808, 190197)),
        "pitch": (105.3, 68.0),
        "expected": (
            ("load.records", 199319),
            ("all.ball.present_frames", 113488),
            ("graded.frames", 158857),
            ("graded.ball.present_frames", 94214),
            ("graded.tracks.mean_rounded", 20.5),
        ),
    },
}

# Dotted key in a scope, label in the report. "" starts a new section; its label explains it.
ROWS = (
    ("", "Piłka: czy jest widoczna. Długie luki są trudniejsze niż wiele krótkich."),
    ("frames", "Klatki w zakresie"),
    ("ball.present_pct", "Piłka widoczna (%)"),
    ("ball.gap_count", "Liczba luk w piłce"),
    ("ball.gap_median_s", "Luka: mediana (s)"),
    ("ball.gap_p90_s", "Luka: 90. percentyl (s)"),
    ("ball.gap_max_s", "Luka: najdłuższa (s)"),
    ("ball.missing_share.under_1s", "Braki w lukach do 1 s (% braków)"),
    ("ball.missing_share.from_1_to_5s", "Braki w lukach 1-5 s (% braków)"),
    ("ball.missing_share.from_5_to_30s", "Braki w lukach 5-30 s (% braków)"),
    ("ball.missing_share.over_30s", "Braki w lukach ponad 30 s (% braków)"),
    ("", "Piłka: czy pozycja jest wiarygodna. Skok to fizycznie niemożliwa prędkość, zwykle fałszywe wykrycie."),
    ("ball.jump_pct", f"Skoki powyżej {BALL_MAX_SPEED_MPS:.0f} m/s (% przejść między klatkami)"),
    ("ball.out_of_pitch_pct", "Piłka poza boiskiem (% klatek z piłką)"),
    ("", "Piłka a zawodnicy. Pokazuje, jak często wskazanie najbliższego zawodnika ma sens i czy piłka znika w tłoku."),
    ("nearest.median_m", "Najbliższy zawodnik: mediana (m)"),
    ("nearest.p90_m", "Najbliższy zawodnik: 90. percentyl (m)"),
    ("nearest.within_control_pct", f"Zawodnik do {CONTROL_RADIUS_M:.0f} m od piłki (% klatek z piłką)"),
    ("crowd.visible_mean", f"Zawodnicy do {CROWD_RADIUS_M:.0f} m od widocznej piłki (średnio)"),
    ("crowd.long_gap_start_mean", f"Zawodnicy do {CROWD_RADIUS_M:.0f} m tuż przed luką od {LONG_GAP_SECONDS:.0f} s (średnio)"),
    ("", "Zawodnicy: kompletność. Na boisku jest 22 zawodników, więc mniej to braki, a więcej to duplikaty."),
    ("tracks.mean", "Torów w klatce: średnio"),
    ("tracks.p5", "Torów w klatce: 5. percentyl"),
    ("tracks.p95", "Torów w klatce: 95. percentyl"),
    ("tracks.home_mean", "Pierwsza drużyna: średnio"),
    ("tracks.away_mean", "Druga drużyna: średnio"),
    ("tracks.below_18_pct", "Mniej niż 18 torów (% klatek)"),
    ("tracks.above_22_pct", "Więcej niż 22 tory (% klatek)"),
    ("tracks.team_above_11_pct", "Drużyna ma ponad 11 torów (% klatek)"),
    ("tracks.none_pct", "Brak jakiegokolwiek toru (% klatek)"),
    ("", "Zawodnicy: ciągłość. Krótkie odcinki toru to częste gubienie osoby; skok to zwykle zamiana torów."),
    ("segments.slots_used", "Używane sloty"),
    ("segments.per_minute", "Nowe odcinki toru na minutę"),
    ("segments.median_s", "Odcinek toru: mediana (s)"),
    ("segments.p90_s", "Odcinek toru: 90. percentyl (s)"),
    ("segments.time_under_10s_pct", "Czas w odcinkach do 10 s (% czasu torów)"),
    ("segments.time_over_60s_pct", "Czas w odcinkach ponad 60 s (% czasu torów)"),
    ("jumps.per_1000", f"Skoki powyżej {PLAYER_MAX_SPEED_MPS:.0f} m/s (na 1000 przejść)"),
    ("duplicates.frames_pct", f"Duplikaty: tory tej drużyny do {DUPLICATE_RADIUS_M} m przez {DUPLICATE_MIN_SECONDS:.0f} s lub dłużej (% klatek)"),
    ("duplicates.events", "Duplikaty: liczba epizodów"),
)


def runs(mask):
    """Start and stop (exclusive) of every run of True values."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return edges[::2], edges[1::2]


def pct(part, whole):
    return round(100.0 * float(part) / float(whole), 2) if whole else None


def stat(function, values, *args):
    return round(float(function(values, *args)), 2) if len(values) else None


def number(text):
    value = float(text) if text else math.nan
    return value if math.isfinite(value) else math.nan


def parse_line(line, n_slots):
    """(frame_id, players, ball), or None when malformed. Errors never repeat file content."""
    parts = line.strip().split(":")
    if len(parts) != 3:
        return None
    groups = parts[1].rstrip(";").split(";")
    ball = parts[2].split(",")
    if len(groups) != n_slots or len(ball) < 2:
        return None
    try:
        players = [number(value) for group in groups for value in group.split(",")[:2]]
        if len(players) != 2 * n_slots:
            return None
        return int(parts[0]), players, (number(ball[0]), number(ball[1]))
    except ValueError:
        return None


def load_atd(meta_path, raw_path):
    """Read an EPTS pair. The first team in the metadata is home, as in official/starter/baseline.py."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if len(team_ids) != 2 or not slots or any(slot not in team_ids for slot in slots):
        raise ValueError("Metadata must list two teams and assign every track slot to one of them")
    teams = np.array([team_ids.index(slot) for slot in slots], dtype=np.int8)
    with Path(raw_path).open(encoding="utf-8") as stream:
        n_records = sum(1 for _ in stream)
    if not n_records:
        raise ValueError("Tracking file is empty")
    players = np.full((n_records, len(teams), 2), np.nan, dtype=np.float32)
    ball = np.full((n_records, 2), np.nan, dtype=np.float32)
    frame_ids = np.zeros(n_records, dtype=np.int64)
    malformed = 0
    with Path(raw_path).open(encoding="utf-8") as stream:
        for row, line in enumerate(stream):
            parsed = parse_line(line, len(teams))
            if parsed is None:
                malformed += 1
                frame_ids[row] = frame_ids[row - 1] + 1 if row else 1
                continue
            frame_ids[row] = parsed[0]
            players[row] = np.reshape(parsed[1], (-1, 2))
            ball[row] = parsed[2]
    load = {"records": n_records, "malformed_lines": malformed, "first_frame_id": int(frame_ids[0]),
            "contiguous": bool((np.diff(frame_ids) == 1).all())}
    return players, ball, teams, load


def frame_signals(players, ball, teams, fps, pitch):
    """Per-frame arrays used by every scope. They stay in memory and never reach a report."""
    scale = np.asarray(pitch, dtype=np.float32)
    present = np.isfinite(players).all(axis=2)
    ball_present = np.isfinite(ball).all(axis=1)
    track_speed = np.full(present.shape, np.nan, dtype=np.float32)
    track_speed[1:] = np.linalg.norm(np.diff(players, axis=0) * scale, axis=2) * fps
    ball_speed = np.full(len(ball), np.nan, dtype=np.float32)
    ball_speed[1:] = np.linalg.norm(np.diff(ball, axis=0) * scale, axis=1) * fps

    nearest = np.full(len(ball), np.nan, dtype=np.float32)
    crowd = np.zeros(len(ball), dtype=np.int16)
    for start in range(0, len(ball), CHUNK_FRAMES):
        stop = start + CHUNK_FRAMES
        distances = np.linalg.norm((players[start:stop] - ball[start:stop, None, :]) * scale, axis=2)
        crowd[start:stop] = (distances <= CROWD_RADIUS_M).sum(axis=1)
        distances = np.where(np.isnan(distances), np.inf, distances).min(axis=1)
        nearest[start:stop] = np.where(np.isinf(distances), np.nan, distances)

    duplicate = np.zeros(len(ball), dtype=bool)
    min_frames = math.ceil(DUPLICATE_MIN_SECONDS * fps)
    for team in (0, 1):
        for a, b in combinations(np.flatnonzero(teams == team), 2):
            both = present[:, a] & present[:, b]
            if not both.any():
                continue
            close = both.copy()
            close[both] = np.linalg.norm((players[both, a] - players[both, b]) * scale, axis=1) <= DUPLICATE_RADIUS_M
            for start, stop in zip(*runs(close)):
                if stop - start >= min_frames:
                    duplicate[start:stop] = True
    return {
        "present": present,
        "ball_present": ball_present,
        "ball_out": ball_present & ((ball < 0) | (ball > 1)).any(axis=1),
        "ball_transition": np.isfinite(ball_speed),
        "ball_jump": ball_speed > BALL_MAX_SPEED_MPS,
        "nearest": nearest,
        "crowd": crowd,
        "track_transitions": np.isfinite(track_speed).sum(axis=1),
        "track_jumps": (track_speed > PLAYER_MAX_SPEED_MPS).sum(axis=1),
        "n_home": present[:, teams == 0].sum(axis=1),
        "n_away": present[:, teams == 1].sum(axis=1),
        "duplicate": duplicate,
    }


def scope_metrics(signals, blocks, fps):
    """Summarize inclusive frame blocks. Gaps and track segments never cross a block boundary."""
    mask = np.zeros(len(signals["ball_present"]), dtype=bool)
    for first, last in blocks:
        mask[first:last + 1] = True
    frames = int(mask.sum())
    gap_lengths, long_gap_crowd, segment_lengths, duplicate_events = [], [], [], 0
    for first, last in blocks:
        for start, stop in zip(*runs(~signals["ball_present"][first:last + 1])):
            gap_lengths.append(stop - start)
            if stop - start >= LONG_GAP_SECONDS * fps and start > 0:
                long_gap_crowd.append(signals["crowd"][first + start - 1])
        for slot in range(signals["present"].shape[1]):
            starts, stops = runs(signals["present"][first:last + 1, slot])
            segment_lengths.extend((stops - starts).tolist())
        duplicate_events += len(runs(signals["duplicate"][first:last + 1])[0])
    gaps = np.asarray(gap_lengths, dtype=float) / fps
    segments = np.asarray(segment_lengths, dtype=float) / fps
    missing_seconds = gaps.sum()
    visible = mask & signals["ball_present"]
    nearest = signals["nearest"][visible]
    nearest = nearest[np.isfinite(nearest)]
    n_tracks = signals["n_home"][mask] + signals["n_away"][mask]
    track_seconds = segments.sum()
    return {
        "frames": frames,
        "ball": {
            "present_frames": int(visible.sum()),
            "present_pct": pct(visible.sum(), frames),
            "gap_count": len(gaps),
            "gap_median_s": stat(np.median, gaps),
            "gap_p90_s": stat(np.percentile, gaps, 90),
            "gap_max_s": stat(np.max, gaps),
            "missing_share": {name: pct(gaps[(gaps >= low) & (gaps < high)].sum(), missing_seconds)
                              for name, low, high in GAP_BINS},
            "jump_count": int((signals["ball_jump"] & mask).sum()),
            "jump_pct": pct((signals["ball_jump"] & mask).sum(), (signals["ball_transition"] & mask).sum()),
            "out_of_pitch_pct": pct((signals["ball_out"] & mask).sum(), visible.sum()),
        },
        "nearest": {
            "median_m": stat(np.median, nearest),
            "p90_m": stat(np.percentile, nearest, 90),
            "within_control_pct": pct((nearest <= CONTROL_RADIUS_M).sum(), visible.sum()),
        },
        "crowd": {
            "visible_mean": stat(np.mean, signals["crowd"][visible]),
            "long_gap_start_mean": stat(np.mean, long_gap_crowd),
        },
        "tracks": {
            "mean": stat(np.mean, n_tracks),
            "mean_rounded": round(float(n_tracks.mean()), 1) if frames else None,
            "p5": stat(np.percentile, n_tracks, 5),
            "p95": stat(np.percentile, n_tracks, 95),
            "home_mean": stat(np.mean, signals["n_home"][mask]),
            "away_mean": stat(np.mean, signals["n_away"][mask]),
            "below_18_pct": pct((n_tracks < 18).sum(), frames),
            "above_22_pct": pct((n_tracks > 22).sum(), frames),
            "team_above_11_pct": pct(((signals["n_home"][mask] > 11) | (signals["n_away"][mask] > 11)).sum(), frames),
            "none_pct": pct((n_tracks == 0).sum(), frames),
        },
        "segments": {
            "slots_used": int(signals["present"][mask].any(axis=0).sum()),
            "count": len(segments),
            "per_minute": round(len(segments) / (frames / fps / 60), 2) if frames else None,
            "median_s": stat(np.median, segments),
            "p90_s": stat(np.percentile, segments, 90),
            "time_under_10s_pct": pct(segments[segments < 10].sum(), track_seconds),
            "time_over_60s_pct": pct(segments[segments > 60].sum(), track_seconds),
        },
        "jumps": {
            "count": int(signals["track_jumps"][mask].sum()),
            "per_1000": round(1000.0 * signals["track_jumps"][mask].sum() / max(signals["track_transitions"][mask].sum(), 1), 3),
        },
        "duplicates": {
            "frames_pct": pct((signals["duplicate"] & mask).sum(), frames),
            "events": duplicate_events,
        },
    }


def time_profile(signals, blocks, fps):
    """Quality in coarse buckets, relative to each block: enough to spot a bad stretch of play."""
    size = BUCKET_MINUTES * 60 * fps
    rows = []
    for number_, (first, last) in enumerate(blocks, start=1):
        for start in range(first, last + 1, size):
            stop = min(start + size, last + 1)
            transitions = signals["track_transitions"][start:stop].sum()
            rows.append({
                "block": number_,
                "from_minute": (start - first) // (60 * fps),
                "ball_present_pct": pct(signals["ball_present"][start:stop].sum(), stop - start),
                "tracks_mean": round(float((signals["n_home"][start:stop] + signals["n_away"][start:stop]).mean()), 2),
                "jumps_per_1000": round(1000.0 * signals["track_jumps"][start:stop].sum() / max(transitions, 1), 3),
                "duplicate_frames_pct": pct(signals["duplicate"][start:stop].sum(), stop - start),
            })
    return rows


def lookup(tree, dotted):
    for key in dotted.split("."):
        tree = tree[key] if isinstance(tree, dict) and key in tree else None
    return tree


def analyze(meta_path, raw_path, name, blocks, pitch, fps, expected=()):
    players, ball, teams, load = load_atd(meta_path, raw_path)
    if blocks:
        if load["first_frame_id"] != 1 or not load["contiguous"]:
            raise ValueError("Graded blocks require frame ids numbered 1..n without gaps")
        if any(first < 0 or first > last or last >= load["records"] for first, last in blocks):
            raise ValueError("Graded blocks are outside the tracking file")
    signals = frame_signals(players, ball, teams, fps, pitch)
    whole = ((0, load["records"] - 1),)
    report = {
        "version": VERSION, "name": name, "fps": fps, "pitch_m": list(pitch), "slots": len(teams),
        "load": load, "all": scope_metrics(signals, whole, fps),
    }
    if blocks:
        report["graded"] = scope_metrics(signals, blocks, fps)
    report["time_profile"] = time_profile(signals, blocks or whole, fps)
    report["reference_checks"] = [
        {"key": key, "expected": value, "actual": lookup(report, key), "ok": lookup(report, key) == value}
        for key, value in expected]
    return report


def show(value):
    if value is None:
        return "-"
    return str(value).replace(".", ",") if isinstance(value, float) else str(value)


def metric_table(headers, columns):
    """Sections from ROWS with one value column per scope or per dataset."""
    lines = []
    for key, label in ROWS:
        if not key:
            lines += ["", label, "", "| Miara | " + " | ".join(headers) + " |",
                      "|---|" + "---:|" * len(headers)]
        else:
            lines.append(f"| {label} | " + " | ".join(show(lookup(column, key)) for column in columns) + " |")
    return lines


def render_report(report):
    lines = [f"# Jakość danych: {report['name']}", "",
             "Raport zawiera wyłącznie liczby zbiorcze. Progi są wstępnymi założeniami.",
             "Nie sprawdza prawdziwości pozycji, tylko kompletność i fizyczną wiarygodność.", "",
             f"Rekordy: {report['load']['records']}. Błędne linie: {report['load']['malformed_lines']}. "
             f"Numeracja ciągła: {'tak' if report['load']['contiguous'] else 'NIE'}. "
             f"Sloty: {report['slots']}. Boisko: {report['pitch_m'][0]} x {report['pitch_m'][1]} m."]
    if report["reference_checks"]:
        lines += ["", "## Kontrola wczytania (liczby opublikowane przez organizatorów)", "",
                  "| Wartość | Oczekiwana | Otrzymana | Wynik |", "|---|---:|---:|---|"]
        lines += [f"| {check['key']} | {show(check['expected'])} | {show(check['actual'])} | "
                  f"{'ZGODNE' if check['ok'] else 'NIEZGODNE'} |" for check in report["reference_checks"]]
    scopes = [("Cały plik", report["all"])]
    if "graded" in report:
        scopes.append(("Oceniane klatki", report["graded"]))
    lines += ["", "## Wyniki"] + metric_table([title for title, _ in scopes], [scope for _, scope in scopes])
    lines += ["", f"## Jakość w czasie (co {BUCKET_MINUTES} min)", "",
              "| Blok | Od minuty | Piłka widoczna (%) | Torów średnio | Skoki na 1000 | Duplikaty (% klatek) |",
              "|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['block']} | {row['from_minute']} | {show(row['ball_present_pct'])} | {show(row['tracks_mean'])} | "
              f"{show(row['jumps_per_1000'])} | {show(row['duplicate_frames_pct'])} |" for row in report["time_profile"]]
    return "\n".join(lines) + "\n"


def write_report(report, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "quality_report.md").write_text(render_report(report), encoding="utf-8", newline="\n")


def compare(report_paths, out_dir):
    """Side-by-side table. Each dataset uses its graded scope when it has one, else the whole file."""
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths]
    if any(report.get("version") != VERSION for report in reports):
        raise ValueError("Reports come from a different version of this script")
    columns = [report.get("graded", report["all"]) for report in reports]
    scopes = ", ".join(f"{report['name']}: {'oceniane klatki' if 'graded' in report else 'cały plik'}" for report in reports)
    lines = ["# Porównanie jakości danych", "", f"Zakres: {scopes}.",
             "Porównuj te same miary. Różne kamery i metody pomiaru mogą tłumaczyć część różnic."]
    lines += metric_table([report["name"] for report in reports], columns)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def parse_blocks(text):
    try:
        return tuple(tuple(int(value) for value in block.split("-", 1)) for block in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("Use first-last,first-last, for example 100-900,1000-1900") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("analyze", help="Write quality_report.md and quality_report.json")
    one.add_argument("--atd-xml", type=Path, required=True, help="Tracking metadata XML")
    one.add_argument("--atd-raw", type=Path, required=True, help="Tracking raw text file")
    one.add_argument("--out", type=Path, required=True, help="Output directory")
    one.add_argument("--name", default="zbiór", help="Label shown in reports; do not use private details")
    one.add_argument("--preset", choices=sorted(PRESETS), help="Competition windows, pitch and published counts")
    one.add_argument("--graded", type=parse_blocks, help="Inclusive video-frame blocks, e.g. 100-900,1000-1900")
    one.add_argument("--pitch", type=float, nargs=2, default=(105.0, 68.0), metavar=("LENGTH", "WIDTH"))
    one.add_argument("--fps", type=int, default=25)
    two = commands.add_parser("compare", help="Write quality_comparison.md from two or more JSON reports")
    two.add_argument("reports", type=Path, nargs="+")
    two.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            if len(args.reports) < 2:
                raise ValueError("Compare needs at least two reports")
            compare(args.reports, args.out)
            print("Saved quality_comparison.md")
            return 0
        preset = PRESETS.get(args.preset, {})
        report = analyze(args.atd_xml, args.atd_raw, args.name, preset.get("blocks", args.graded),
                         preset.get("pitch", tuple(args.pitch)), args.fps, preset.get("expected", ()))
        write_report(report, args.out)
    except (ValueError, OSError, ET.ParseError, json.JSONDecodeError, KeyError) as exc:
        # Only the exception type: a message could quote a line of the input file.
        print(f"ERROR: {type(exc).__name__}. No report was written.", file=sys.stderr)
        return 1
    failed = [check["key"] for check in report["reference_checks"] if not check["ok"]]
    print("Saved quality_report.md and quality_report.json")
    if failed:
        print("WARNING: differs from published counts; do not trust this report: " + ", ".join(failed))
    elif report["reference_checks"]:
        print("OK: all published counts match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
