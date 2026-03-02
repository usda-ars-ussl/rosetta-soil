# v0.3.2 (2026-03-02)
- changed named option for `estimate_type`. The default option "linear"
  is now "arith".
- networks are no longer validated by default

# v0.3.1 (2026-03-02)
- vectorize nn_model.py (thanks to @brownag for the contribution)

# v0.3 (2026-02-27)
- Major re-write with breaking changes.
- Add UsaturatedK class for predictions of K0 and L.
- rosetta() and rosesoil() functions have new call signatures and returns.
- Rosetta and UnsaturatedK classes provide access to bootstrap resamples.
- New implementation of FFNN's.
- New .npz data store of weights, biases, and metadata.
- Switch build system to hatchling.


