#!/usr/bin/env python3
"""Deprecated deepanal launcher.

The explorer now lives in Grafana backed by:
- Prometheus for live operational metrics
- TimescaleDB for the durable trade archive and OHLCV history

Use the Grafana dashboard instead of Streamlit.
"""

raise SystemExit(
    "deepanal has been retired. Open the Grafana dashboard instead and run "
    "`uv run python scripts/grafana_archive.py` to refresh the archive."
)
