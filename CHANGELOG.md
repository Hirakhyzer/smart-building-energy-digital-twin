# Changelog

## v0.1.1 — 2026-11-24

- Reject non-finite, nonpositive, or malformed two-state RC thermal parameters and controller inputs.
- Require finite, nonempty and shape-consistent weather, HVAC, and gain trajectories for thermal simulation.
- Verify finite timestamps and split fractions before chronological training, validation, and test partitioning.
- Keep records sharing a timestamp in the same partition and reject empty temporal periods.
- Reject invalid or negative electricity-price proxies, non-finite scenario targets, and reversed comfort bounds.
- Add regression coverage for parameter validation, chronological data integrity, and comfort-cost scenario selection.
- Preserve the existing synthetic models, example datasets, power assumptions, and research limitations.

The changes protect the integrity of numerical research inputs. Passing tests does not establish calibrated building performance or deployed control safety.

## v0.1.0

- Initial simulation-first smart-building energy digital-twin research infrastructure.
