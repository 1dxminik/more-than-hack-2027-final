# Local C5-alpha-0.1 model assets

Exactly seven files listed in `configs/assets.json` are required:
five `tcn_*.npz` checkpoints, `linear.json` for full N14 and `meta.npz`
for alpha=0.1. The manifest records their roles, sizes, SHA-256 hashes and
member names in the source package. The runtime verifies the complete set
before loading it.

Install with `python -m scripts.install_assets --from-package PACKAGE_PATH`.
Supply the local `code-and-weights.zip` or its extracted code directory.
Only the required model assets are copied, without data, outputs or CSVs.

The models are available locally for testing this branch and are ignored by Git.
Cloning the branch requires the package to be supplied separately.
The weights have not been published.
