# Local C5-alpha-0.1 model assets

Exactly seven files listed in `configs/assets.json` are required:
five `tcn_*.npz` checkpoints, `linear.json` for full N14 and `meta.npz`
for alpha=0.1. The manifest records their roles, sizes, SHA-256 hashes and
member names in the source package. The runtime verifies the complete set
before loading it.

Download [C5-alpha-0.1 weights v1](https://github.com/1dxminik/more-than-hack-2027-final/releases/tag/c5-alpha-0.1-weights-v1)
and follow the download and SHA-256 verification commands in the root README.
Then install with `python -m scripts.install_assets --from-package PACKAGE_PATH`.
Only the seven required model assets are copied, without data, outputs or CSVs.
The installer also accepts the original local C5 code-and-weights package.

Published archive: `c5-alpha-0.1-weights-v1.zip`, 726,575 bytes.
SHA-256: `9794366fae14fc32065975d147efb17816dbd23ee696a51d0460ebda5c2c23e7`.
The release includes a manifest, per-file checksums and a model card with provenance.
Installed weights remain ignored by Git; public distribution is through the release asset.
