# Sources and licenses

The `official/` directory contains selected, unchanged public files from
[More Than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027).

- Author: Metrica Sports; copyright 2026–2027.
- License for these files: [MIT](official/LICENSE).
- Version: `f8f2629ff1e0985f81d7e59d8413678514ee1530`.
- Source and hash of each file: [official/UPSTREAM.json](official/UPSTREAM.json).
- Integrity check: `./run.ps1 check`.

This is a selection of files, not a full repository clone. Links in the original
documentation to notebooks, images, ground truth and results may refer to files
intentionally omitted here. The main README.md documents this repository's commands.

The organiser repository's license does not cover the private match bundle.
No organiser ground truth or prediction files were downloaded. All demonstration
input is generated from invented coordinates by our own generator.

Public IDSSE dataset: Bassek et al., *Scientific Data* 2025,
[DOI 10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y), data owner DFL,
CC BY 4.0. `scripts/fetch_idsse.py` downloads it locally from the
[pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data) mirror.
The dataset is not included in this repository. Local processing produces arrays,
H/A/D labels and degraded tracking variants for training and evaluation.
The network weights in `weights/tcn_v1/` were trained on degraded IDSSE matches
(CC BY 4.0; attribution as above). The checkpoints contain network parameters,
feature means and standard deviations, configuration and training metadata
(training, validation and test match IDs, selected epoch and decoder penalty).
They do not contain frame-level tracking or label arrays.
PyTorch (BSD-3-Clause) is a model dependency (`requirements-nn.txt`).

No code was copied from BallRadar, PathCRF, ligafooty or football-cv-analytics.
Licenses for any future components must be checked separately.

Environment dependencies: NumPy (BSD-3-Clause), pandas (BSD-3-Clause),
python-dateutil (Apache-2.0/BSD-3-Clause), six (MIT), tzdata (Apache-2.0;
the IANA database material has its own rights notices). These packages retain
their respective licenses. This notice does not assign a license to the team's
own code or model weights.
