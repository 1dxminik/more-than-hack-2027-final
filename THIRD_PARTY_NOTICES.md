# Provenance and licences

C5/N14 originates from the local MoreThanHack project, commit
`d99253c24e739735510f76be40f1adaaafe39344`.
Source hashes and migration details are in `SOURCE_MANIFEST.json`.
The seven model assets are publicly available in the
[versioned weights release](https://github.com/1dxminik/more-than-hack-2027-final/releases/tag/c5-alpha-0.1-weights-v1),
outside Git history. The release makes these assets accessible for running and
inspecting C5; it does not assign a new licence to the team-authored code or weights.

`official/` is an unchanged selection of Metrica Sports' public More Than a Hack
2027 code, version `f8f2629ff1e0985f81d7e59d8413678514ee1530`.
The [MIT licence](official/LICENSE) and [UPSTREAM.json](official/UPSTREAM.json)
are preserved. That public-code licence does not cover competition match assets.

Historical TCN and meta-model training used IDSSE: Bassek et al., Scientific Data
(2025), [DOI 10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y).
Data owner: Deutsche Fussball Liga (DFL). Public mirror:
[pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data), CC BY 4.0.
The original data were transformed into features and used for training; the
dataset itself is not included or redistributed. This attribution does not imply
that the dataset licence has been assigned to the weights. Linear N14 was trained
on independent synthetic data. The release model card records asset provenance.

Dependencies: PyTorch, NumPy and pandas (BSD-3-Clause), python-dateutil
(Apache-2.0/BSD-3-Clause), six (MIT), tzdata (Apache-2.0 and IANA database notices).
Other packages retain their own licences and are pinned in `requirements-nn.txt`.
