# Contributing

## Setup
```bash
uv sync
```

## Development Commands
```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run pyright
```

## Monorepo Notes
- Use `packages/*` for reusable library code.
- Keep CLI glue in `scripts/`.
- Register plugin entry points in package `pyproject.toml` files.
