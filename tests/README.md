# Test Suite Documentation

This repository uses three nested test suites:

- `short` (~2 minutes): fast confidence checks
- `medium` (~5 minutes): `short` + broad fast integration coverage
- `long` (~11 minutes): comprehensive validation

The suites are cumulative:

- `short` is a subset of `medium`
- `medium` is a subset of `long`

## Run Suites

From repository root:

```bash
# Runner
bash tests/run_suites.sh short
bash tests/run_suites.sh medium
bash tests/run_suites.sh long
```

## Suite Composition

### `short` (~2m)

- Unit tests (`tests/unit/`)
- Golden oracles full block
- Cross-path equivalence fast subset (`matrix and not slow and not nightly`)
- Differential fast subset (`differential and not slow and not nightly`)

### `medium` (~5m)

Includes everything in `short`, plus:

- Core integration full block (explicit path list, disjoint from dedicated
  matrix/property/differential/golden blocks)

### `long` (~11m)

Includes all checks needed for full coverage, with each category run once:

- Core integration full block
- Full property suite (`property`)
- Full equivalence matrix suite (`matrix`)
- Full differential suite (`differential`)

## Current Local Timing Snapshot

Measured/estimated from local suite timing runs on this machine:

- `short`: ~2:08
- `medium`: ~5:21
- `long`: ~11:03

## Marker Reference

Markers are configured in `tests/pytest.ini`:

- `matrix`
- `golden`
- `property`
- `differential`
- `regression`
- `slow`
- `nightly`
- `unit`
- `integration`

## CI

No GitHub workflow is configured currently. Run suites locally with `tests/run_suites.sh`.

## Notes

- Keep new expensive tests marked as `slow` and `nightly` where relevant.
- If suite runtimes drift, rebalance tests between tiers to keep approximate targets.
