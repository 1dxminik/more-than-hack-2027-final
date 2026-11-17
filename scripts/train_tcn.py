"""Leave-one-match-out training of the TCN + Viterbi possession model on degraded IDSSE matches.

Reads datasets/idsse/degraded/<match>/seed<k>/ (scripts/degrade_idsse.py), caches 5 fps features in
datasets/idsse/features/tcn_v1/ and writes every result to outputs/tcn/<run>/ (WYNIKI.md first).

    python -X utf8 -m scripts.train_tcn features --jobs 4
    python -X utf8 -m scripts.train_tcn cv --run smoke --folds 1 --epochs 2
    python -X utf8 -m scripts.train_tcn cv --run v1
    python -X utf8 -m scripts.train_tcn final --run v1      # once, on the locked match, at the end

Scores are on public IDSSE with simulated tracking errors. They are not competition performance.
"""

import argparse
import copy
import csv
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "official" / "evaluation"))
sys.path.insert(0, str(ROOT / "official" / "starter"))
import possession_metrics as pm  # noqa: E402  (official, unchanged)
import baseline  # noqa: E402  (official, unchanged)

from solution import tcn_features as tf  # noqa: E402
from solution.tracking import load_tracking  # noqa: E402

DEGRADED = ROOT / "datasets" / "idsse" / "degraded"
FEATURE_VERSION = 1
CACHE = ROOT / "datasets" / "idsse" / "features" / f"tcn_v{FEATURE_VERSION}"
OUTPUTS = ROOT / "outputs" / "tcn"
# Locked final test match: random.Random(2027).choice over the six matches other than J03WMX
# (J03WMX had already been looked at). Never used for training, tuning or model selection.
LOCKED_TEST = "J03WN1"
CV_MATCHES = ("J03WMX", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SEEDS = (0, 1, 2, 3, 4)
EVAL_SEEDS = (0, 1)
PENALTIES = (0, 1, 2, 4, 8, 16, 32, 64)
METRICS = ("spell_f1_mean", "macro_f1", "f1_H", "f1_A", "f1_D", "accuracy", "team_acc_gt_alive",
           "alive_dead_f1", "n_gt_spells", "n_pred_spells", "fragmentation")
TRAIN_KEYS = ("epochs", "eval_every", "channels", "layers", "stages", "dropout", "lr", "weight_decay",
              "window", "batch", "crop_factor", "smooth", "seed")


# ---------------------------------------------------------------- features

def cache_path(match, seed):
    return CACHE / f"{match}_seed{seed}.npz"


def pitch_of(xml_path):
    root = ET.parse(xml_path).getroot()
    width, height = root.find(".//FieldSize/Width"), root.find(".//FieldSize/Height")
    return (float(width.text) if width is not None else 105.0,
            float(height.text) if height is not None else 68.0)


def build_one(job):
    match, seed = job
    out = cache_path(match, seed)
    if out.exists():
        return f"{match} seed{seed}: jest w cache"
    start = time.time()
    folder = DEGRADED / match / f"seed{seed}"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    n = manifest["config"]["total_frames"]
    xml = folder / "tactical" / f"{match}_tactical_atd.xml"
    players, ball, teams = load_tracking(xml, folder / "tactical" / f"{match}_tactical_atd.txt", n)
    pitch = pitch_of(xml)
    x = tf.bin_features(tf.frame_features(players, ball, teams, pitch))
    gt = pm.label_codes(pm.read_label_csv(str(folder / f"{match}_gt.csv")))
    if len(gt) != n:
        raise ValueError(f"{match} seed{seed}: GT has {len(gt)} rows, expected {n}")
    base50 = pm.label_codes(baseline.smooth(tf.nearest_player_codes(players, ball, teams, pitch), 50))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, x=x, y=tf.bin_labels(gt), gt=gt, base50=base50,
                        version=FEATURE_VERSION, names=np.array(tf.NAMES))
    os.replace(tmp, out)
    return f"{match} seed{seed}: {time.time() - start:.0f} s"


def ensure_features(matches, jobs):
    todo = [(m, s) for m in matches for s in SEEDS if not cache_path(m, s).exists()]
    missing = [f"{m}/seed{s}" for m, s in todo if not (DEGRADED / m / f"seed{s}" / "manifest.json").exists()]
    if missing:
        raise SystemExit("Brak zniekształconych danych: " + ", ".join(missing))
    if not todo:
        return
    print(f"Liczę cechy dla {len(todo)} zestawów ({jobs} procesy). Robi się to raz.", flush=True)
    start = time.time()
    with Pool(max(1, jobs)) as pool:
        for line in pool.imap_unordered(build_one, todo):
            print("  " + line, flush=True)
    print(f"Cechy gotowe w {(time.time() - start) / 60:.1f} min.", flush=True)


def load_sets(matches):
    data = {}
    for m in matches:
        for s in SEEDS:
            with np.load(cache_path(m, s)) as z:
                if int(z["version"]) != FEATURE_VERSION or list(z["names"]) != tf.NAMES:
                    raise SystemExit(f"Nieaktualny cache {cache_path(m, s)}; usuń katalog {CACHE}.")
                data[m, s] = {k: z[k] for k in ("x", "y", "gt", "base50")}
    return data


# ---------------------------------------------------------------- logging

class Run:
    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def log(self, message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.folder / "log.txt").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def append_csv(self, name, row):
        path = self.folder / name
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except OSError:
        return {"commit": None}


# ---------------------------------------------------------------- evaluation

def scores(gt, pred):
    result = pm.evaluate(gt, pred, bootstrap_n=0)
    flat = {k: result[k] for k in METRICS if k in result}
    for name in ("H", "A", "D"):
        flat[f"f1_{name}"] = result["per_class"][name]["f1"]
    return {k: round(float(flat[k]), 6) for k in METRICS}


def tune_penalty(log_probs_by_seed, gts):
    """Mean spellF1 over the evaluation seeds for every penalty; returns (best, {penalty: score})."""
    table = {p: [] for p in PENALTIES}
    for logp, gt in zip(log_probs_by_seed, gts):
        runs = pm.valid_runs(gt)
        for p, path in zip(PENALTIES, tf.viterbi_many(logp, PENALTIES)):
            table[p].append(pm.spell_scores(gt, tf.upsample(path, len(gt)), runs)["mean_f1"])
    table = {p: float(np.mean(v)) for p, v in table.items()}
    return max(table, key=lambda p: (table[p], -p)), table


# ---------------------------------------------------------------- training

def train_fold(args, run, data, name, train, val, test):
    import torch
    from solution import tcn_model

    index = (CV_MATCHES + (LOCKED_TEST,)).index(test)
    torch.manual_seed(args.seed + index)
    rng = np.random.default_rng(args.seed + index)
    stacked = np.concatenate([data[m, s]["x"] for m in train for s in SEEDS])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    del stacked
    labels = np.concatenate([data[m, 0]["y"] for m in train])
    counts = np.array([(labels == c).sum() for c in range(3)], float)
    weight = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)

    model = tcn_model.MSTCN(len(tf.NAMES), args.channels, args.layers, args.stages, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    n_params = sum(p.numel() for p in model.parameters())
    run.log(f"{name}: trening {train}, walidacja {val}, test {test}; {n_params} parametrów; "
            f"wagi klas H/A/D {np.round(weight.numpy(), 2).tolist()}")

    lengths = {m: len(data[m, 0]["y"]) for m in train}
    best = {"score": -1.0}
    val_gts = [data[val, s]["gt"] for s in EVAL_SEEDS]
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        crops = [(m, int(rng.choice(SEEDS)), int(rng.integers(0, lengths[m] - args.window + 1)))
                 for m in train for _ in range(math.ceil(lengths[m] / args.window) * args.crop_factor)]
        rng.shuffle(crops)
        losses = []
        for b in range(0, len(crops), args.batch):
            xs, ys = [], []
            for m, s, at in crops[b:b + args.batch]:
                x, y = tf.flip(data[m, s]["x"][at:at + args.window], data[m, s]["y"][at:at + args.window],
                               swap_teams=rng.random() < .5, mirror_x=rng.random() < .5, mirror_y=rng.random() < .5)
                xs.append(((x - mean) / std).T)
                ys.append(y)
            x = torch.from_numpy(np.stack(xs)).float()
            y = torch.from_numpy(np.stack(ys).astype(np.int64))
            loss = tcn_model.loss(model(x), y, weight, smooth=args.smooth)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {"fold": name, "test": test, "val": val, "epoch": epoch, "train_loss": round(float(np.mean(losses)), 5),
               "lr": round(scheduler.get_last_lr()[0], 7), "train_seconds": round(time.time() - start, 1),
               "val_spell_f1": "", "val_penalty": "", "val_spell_f1_by_penalty": ""}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logps = [tcn_model.predict_log_probs(model, data[val, s]["x"], mean, std) for s in EVAL_SEEDS]
            penalty, table = tune_penalty(logps, val_gts)
            row.update(val_spell_f1=round(table[penalty], 5), val_penalty=penalty,
                       val_spell_f1_by_penalty=json.dumps({str(k): round(v, 4) for k, v in table.items()}))
            if table[penalty] > best["score"]:
                best = {"score": table[penalty], "epoch": epoch, "penalty": penalty,
                        "state": copy.deepcopy(model.state_dict())}
        row["seconds_total"] = round(time.time() - start, 1)
        run.append_csv("epochs.csv", row)
        extra = f", walidacja spellF1 {row['val_spell_f1']} (kara {row['val_penalty']})" if row["val_spell_f1"] != "" else ""
        run.log(f"{name} epoka {epoch}/{args.epochs}: strata {row['train_loss']}, {row['seconds_total']} s{extra}")

    model.load_state_dict(best["state"])
    torch.save({"state": best["state"], "mean": mean, "std": std, "penalty": best["penalty"],
                "names": tf.NAMES, "config": {k: getattr(args, k) for k in TRAIN_KEYS},
                "train": train, "val": val, "test": test, "epoch": best["epoch"]},
               run.folder / f"model_{name}.pt")
    rows = []
    for s in EVAL_SEEDS:
        gt = data[test, s]["gt"]
        logp = tcn_model.predict_log_probs(model, data[test, s]["x"], mean, std)
        variants = {"tcn_viterbi": tf.upsample(tf.viterbi(logp, best["penalty"]), len(gt)),
                    "tcn_argmax": tf.upsample(logp.argmax(axis=1).astype(np.int8), len(gt)),
                    "baseline_smooth50": data[test, s]["base50"]}
        for method, pred in variants.items():
            row = {"fold": name, "test": test, "val": val, "seed": s, "method": method,
                   "best_epoch": best["epoch"], "penalty": best["penalty"] if method == "tcn_viterbi" else "",
                   **scores(gt, pred)}
            rows.append(row)
            run.append_csv("wyniki.csv", row)
    summary = {"fold": name, "train": train, "val": val, "test": test, "best_epoch": best["epoch"],
               "penalty": best["penalty"], "val_spell_f1": round(best["score"], 5), "rows": rows}
    (run.folder / f"fold_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tcn = np.mean([r["spell_f1_mean"] for r in rows if r["method"] == "tcn_viterbi"])
    base = np.mean([r["spell_f1_mean"] for r in rows if r["method"] == "baseline_smooth50"])
    run.log(f"{name} GOTOWE: test {test} spellF1 TCN+Viterbi {tcn:.4f}, baseline smooth50 {base:.4f}")
    return summary


# ---------------------------------------------------------------- report

def write_report(run, title, prefix):
    folds = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run.folder.glob(f"{prefix}*.json"))]
    if not folds:
        return
    methods = ("tcn_viterbi", "tcn_argmax", "baseline_smooth50")
    keys = ("spell_f1_mean", "macro_f1", "f1_H", "f1_A", "f1_D", "n_pred_spells", "n_gt_spells")

    def mean_of(fold, method, key):
        return float(np.mean([r[key] for r in fold["rows"] if r["method"] == method]))

    lines = [f"# {title}", "",
             "Dane: zniekształcone IDSSE (publiczne mecze Bundesligi, CC BY 4.0). "
             "**To nie jest wynik konkursowy.** Na pełnym meczu różnice spellF1 poniżej ~0,04 to szum.",
             f"Każdy wynik to średnia z ziaren {list(EVAL_SEEDS)} meczu testowego. "
             "Szczegóły: wyniki.csv (każde ziarno), epochs.csv (każda epoka), log.txt.", "",
             "| fold | test | walidacja | epoka | kara | spellF1 TCN+Viterbi | spellF1 TCN bez Viterbiego | "
             "spellF1 baseline50 | macroF1 TCN | F1 D TCN | posiadania pred/GT |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in folds:
        lines.append(f"| {f['fold']} | {f['test']} | {f['val']} | {f['best_epoch']} | {f['penalty']} | "
                     f"{mean_of(f, 'tcn_viterbi', 'spell_f1_mean'):.4f} | {mean_of(f, 'tcn_argmax', 'spell_f1_mean'):.4f} | "
                     f"{mean_of(f, 'baseline_smooth50', 'spell_f1_mean'):.4f} | {mean_of(f, 'tcn_viterbi', 'macro_f1'):.4f} | "
                     f"{mean_of(f, 'tcn_viterbi', 'f1_D'):.4f} | "
                     f"{mean_of(f, 'tcn_viterbi', 'n_pred_spells'):.0f}/{mean_of(f, 'tcn_viterbi', 'n_gt_spells'):.0f} |")
    lines += ["", f"## Średnio po {len(folds)} foldach (± odchylenie standardowe między foldami)", "",
              "| metoda | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for method in methods:
        cells = []
        for key in keys:
            values = [mean_of(f, method, key) for f in folds]
            cells.append(f"{np.mean(values):.1f}" if key.startswith("n_") else f"{np.mean(values):.4f} ± {np.std(values):.4f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    (run.folder / ("WYNIKI.md" if prefix == "fold_cv" else "WYNIK_KONCOWY.md")).write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- commands

def configure(args, run, mode):
    import torch
    torch.set_num_threads(args.threads)
    config = {"mode": mode, "train": {k: getattr(args, k) for k in TRAIN_KEYS}, "threads": args.threads,
              "feature_version": FEATURE_VERSION, "features": tf.NAMES, "fps_model": tf.FPS / tf.STEP,
              "locked_test": LOCKED_TEST, "cv_matches": CV_MATCHES, "train_seeds": SEEDS,
              "eval_seeds": EVAL_SEEDS, "penalties": PENALTIES, "git": git_state(),
              "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
              "platform": platform.platform(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = run.folder / "config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["train"] != config["train"]:
            raise SystemExit(f"Przebieg {run.folder.name} istnieje z innymi ustawieniami. Użyj innej nazwy --run.")
        run.log("Wznawiam istniejący przebieg; ukończone foldy są pomijane.")
    else:
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def command_cv(args):
    run = Run(OUTPUTS / args.run)
    ensure_features(CV_MATCHES, args.jobs)
    configure(args, run, "cv")
    data = load_sets(CV_MATCHES)
    plan = [(f"cv{i + 1}_{test}", [m for m in CV_MATCHES if m not in (test, CV_MATCHES[(i + 1) % 6])],
             CV_MATCHES[(i + 1) % 6], test) for i, test in enumerate(CV_MATCHES)][:args.folds]
    start = time.time()
    for name, train, val, test in plan:
        if (run.folder / f"fold_{name}.json").exists():
            run.log(f"{name}: już ukończony, pomijam")
            continue
        train_fold(args, run, data, name, train, val, test)
        write_report(run, f"Leave-one-match-out: {args.run}", "fold_cv")
    write_report(run, f"Leave-one-match-out: {args.run}", "fold_cv")
    run.log(f"Koniec po {(time.time() - start) / 60:.1f} min. Wyniki: {run.folder / 'WYNIKI.md'}")


def command_final(args):
    run = Run(OUTPUTS / args.run)
    cv_config = run.folder / "config.json"
    if not cv_config.exists():
        raise SystemExit("Najpierw uruchom cv z tą samą nazwą --run; final używa jego ustawień.")
    for key, value in json.loads(cv_config.read_text(encoding="utf-8"))["train"].items():
        setattr(args, key, value)
    if (run.folder / "fold_final.json").exists():
        raise SystemExit("Test końcowy już był uruchomiony dla tego przebiegu. Nie powtarzaj go.")
    ensure_features(CV_MATCHES + (LOCKED_TEST,), args.jobs)
    import torch
    torch.set_num_threads(args.threads)
    data = load_sets(CV_MATCHES + (LOCKED_TEST,))
    val = CV_MATCHES[0]
    train = [m for m in CV_MATCHES if m != val]
    train_fold(args, run, data, "final", train, val, LOCKED_TEST)
    write_report(run, f"Test końcowy na zamkniętym meczu {LOCKED_TEST}: {args.run}", "fold_final")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    features = sub.add_parser("features", help="Policz cechy dla wszystkich meczów i ziaren (raz)")
    features.add_argument("--jobs", type=int, default=4)
    for name in ("cv", "final"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True, help="Nazwa przebiegu: outputs/tcn/<run>/")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--jobs", type=int, default=4, help="Procesy do liczenia brakujących cech")
        if name == "cv":
            p.add_argument("--folds", type=int, default=len(CV_MATCHES))
            p.add_argument("--epochs", type=int, default=20)
            p.add_argument("--eval-every", type=int, default=3)
            p.add_argument("--channels", type=int, default=64)
            p.add_argument("--layers", type=int, default=10)
            p.add_argument("--stages", type=int, default=2)
            p.add_argument("--dropout", type=float, default=0.3)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--weight-decay", type=float, default=1e-4)
            p.add_argument("--window", type=int, default=2048, help="Długość wycinka w krokach 5 fps")
            p.add_argument("--batch", type=int, default=4)
            p.add_argument("--crop-factor", type=int, default=2)
            p.add_argument("--smooth", type=float, default=0.15)
            p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "features":
        ensure_features(CV_MATCHES + (LOCKED_TEST,), args.jobs)
    elif args.command == "cv":
        command_cv(args)
    else:
        command_final(args)


if __name__ == "__main__":
    main()
