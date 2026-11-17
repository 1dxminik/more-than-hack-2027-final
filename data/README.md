# Local inputs

`phase1.py demo` and `improve` create independent synthetic data in
`synthetic-c5/<run>/`. The manifest records the data type and hashes.
These commands do not read the competition match or an external directory
specified through the environment.

The operator supplies explicit XML/TXT paths outside the repository.
Inputs must remain outside Git and AI services.
