# Mechanism benchmark, archived run (2026-09-15, 49-increment signature path)

Superseded on 2026-09-20. This run used the earlier signature path (W points
starting at (0, r_1, |r_1|), so the first return of every window never
reached the raw signature; no `signature_path` entry in `config.json`). The
rerun under the origin-anchored path is in `results/beyond_marginal_benchmark/`.
Feature blocks B1/B2/B3 are unaffected by the path convention; the signature,
B3 + signature and shuffled-signature cells are not.
