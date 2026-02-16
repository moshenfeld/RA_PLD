# Testing Guide

Test suite lives under `tests/` and uses `pytest`.

## Layout

```text
tests/
├── unit/
│   ├── core/
│   └── convolution/
├── integration/
│   ├── convolution/
│   ├── core/
│   ├── edge_cases/
│   ├── preamble/
│   └── subsampling/
├── system/
├── conftest.py
└── pytest.ini
```

## Markers

Configured in `tests/pytest.ini`:
- `slow`
- `unit`
- `integration`
- `system`

## Run Commands

From repo root:

```bash
pytest tests/
pytest tests/ -m "not slow"
pytest tests/unit/
pytest tests/integration/
pytest tests/system/
```

From `tests/` directory:

```bash
cd tests
pytest
pytest -m "not slow"
```

## Useful Options

```bash
pytest tests/ -v
pytest tests/ -x
pytest tests/ --lf
pytest tests/ -s
```

## Coverage

If `pytest-cov` is installed:

```bash
pytest tests/ --cov=. --cov-report=term --cov-report=html
```

A helper script exists at `tests/run_tests.sh`:

```bash
bash tests/run_tests.sh --fast
bash tests/run_tests.sh --coverage
```

## Notes

- Subsampling comparison tests may require optional external packages for baseline checks.
- Numerical assertions should use tolerant comparisons (`np.isclose` / `np.allclose`) due to floating-point effects.
