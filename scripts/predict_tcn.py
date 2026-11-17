"""Write a frame,label CSV from one ATD file with trained TCN models (ensemble) + Viterbi.

Reads only the two files given on the command line; never searches for data. Prints no
per-frame content, only the output path and the official validation result.

    python -X utf8 -m scripts.predict_tcn --atd-xml X_tactical_atd.xml --atd-raw X_tactical_atd.txt --out C:\\poza\\projektem\\team.csv

Without --models it uses every weights/tcn_v1/model_*.pt (published, hashes in SHA256SUMS) and averages
their probabilities.
"""

import argparse
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phase1 import N_FRAMES, validate_csv  # noqa: E402
from scripts.train_tcn import pitch_of  # noqa: E402
from solution import tcn_features as tf  # noqa: E402
from solution.tracking import load_tracking  # noqa: E402

LABELS = np.array(["H", "A", "D"])


def load_models(paths):
    import torch
    from solution import tcn_model

    models = []
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if list(saved["names"]) != tf.NAMES:
            raise SystemExit(f"{path}: model trenowano na innych cechach")
        c = saved["config"]
        model = tcn_model.MSTCN(len(tf.NAMES), c["channels"], c["layers"], c["stages"], c["dropout"])
        model.load_state_dict(saved["state"])
        models.append((model, saved["mean"], saved["std"], saved["penalty"]))
    return models


def predict(models, x, penalty=None):
    """Mean of the models' probabilities per 5 fps step, then Viterbi; returns codes 0/1/2 per step."""
    from solution import tcn_model

    probs = np.mean([np.exp(tcn_model.predict_log_probs(m, x, mean, std)) for m, mean, std, _ in models], axis=0)
    if penalty is None:
        penalty = statistics.median(p for *_, p in models)
    return tf.viterbi(np.log(np.maximum(probs, 1e-12)), penalty), penalty


def write_csv(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{label}\n" for i, label in enumerate(labels))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--atd-xml", type=Path, required=True, help="Metadane ATD (…_tactical_atd.xml)")
    parser.add_argument("--atd-raw", type=Path, required=True, help="Surowe ATD (…_tactical_atd.txt)")
    parser.add_argument("--out", type=Path, required=True, help="Wynikowy CSV frame,label")
    parser.add_argument("--models", type=Path, nargs="+",
                        default=sorted((ROOT / "weights" / "tcn_v1").glob("model_*.pt")))
    parser.add_argument("--frames", type=int, default=N_FRAMES, help="Liczba klatek ATD (konkurs: 199319)")
    parser.add_argument("--penalty", type=float, default=None, help="Kara Viterbiego; domyślnie mediana z modeli")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not args.models:
        raise SystemExit("Brak modeli: podaj --models albo najpierw uruchom scripts.train_tcn cv")

    import torch
    torch.set_num_threads(args.threads)
    players, ball, teams = load_tracking(args.atd_xml, args.atd_raw, args.frames)
    x = tf.bin_features(tf.frame_features(players, ball, teams, pitch_of(args.atd_xml)))
    models = load_models(args.models)
    path, penalty = predict(models, x, args.penalty)
    write_csv(args.out, LABELS[tf.upsample(path, args.frames)])
    validate_csv(args.out, args.frames)
    print(f"OK: {args.out} ({args.frames} wierszy, {len(models)} modeli, kara {penalty}), walidacja oficjalna przeszła")


if __name__ == "__main__":
    main()
