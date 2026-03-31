# Loopstack Release

## Preflight

1. Update `version` in `pyproject.toml`.
2. Review `README.md` and `LICENSE`.
3. Run the local quality gate:

```bash
uv run ruff check .
uv run mypy .
uv run pytest -q
```

## Build

Create fresh source and wheel distributions:

```bash
rm -rf dist
uv run python -m build
```

Validate package metadata and README rendering:

```bash
uv run twine check dist/*
```

## Smoke Test

Install the built wheel into a clean environment and verify the CLI:

```bash
python3 -m venv .release-venv
source .release-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install dist/*.whl
loopstack --help
deactivate
rm -rf .release-venv
```

## Upload

Upload after the build and smoke checks pass:

```bash
uv run twine upload dist/*
```
