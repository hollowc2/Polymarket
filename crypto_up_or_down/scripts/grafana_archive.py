#!/usr/bin/env python3
"""Archive Polymarket trade history and OHLCV into the Grafana datastore.

This script is the centralized data layer for the explorer workflow:

- raw trade history JSON is preserved in `polymarket_trade_archive`
- settled trades continue to populate `polymarket_trades`
- OHLCV parquet files are loaded into `polymarket_ohlcv`

Grafana can then read the database for durable history while Prometheus keeps
handling live operational metrics from the exporter.

Usage:
    uv run python scripts/grafana_archive.py
    uv run python scripts/grafana_archive.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.environ.get("STATE_DIR", "/opt/polymarket/state"))
if not STATE_DIR.exists():
    STATE_DIR = ROOT / "state"
DATA_DIR = ROOT / "data"

DB_HOST = os.environ.get("DB_HOST", os.environ.get("DATABASE_HOST", "127.0.0.1"))
DB_PORT = int(os.environ.get("DB_PORT", os.environ.get("DATABASE_PORT", "5432")))
DB_NAME = os.environ.get("DB_NAME", os.environ.get("DATABASE_NAME", "butterfly_guy"))
DB_USER = os.environ.get("DB_USER", os.environ.get("DATABASE_USER", "butterfly"))
DB_PASS = os.environ.get("DB_PASS", os.environ.get("DATABASE_PASSWORD", "butterfly_dev"))
ARCHIVE_LOCK_ID = 786761254

_TF_RE = re.compile(r"-(5m|15m|1h|4h|1d)(?:-|$)")
_ASSET_RE = re.compile(r"^(eth|btc)-updown", re.IGNORECASE)
_OHLCV_RE = re.compile(
    r"^(?P<symbol>[a-z0-9]+)_(?P<timeframe>5m|15m|1h|4h)(?:_(?P<market>perp|spot))?$",
    re.IGNORECASE,
)


def parse_market_slug(slug: str) -> tuple[str | None, str | None]:
    """Extract `(asset, timeframe)` from a market slug."""
    asset = None
    timeframe = None
    if slug:
        match = _ASSET_RE.match(slug)
        if match:
            asset = match.group(1).lower()
        match = _TF_RE.search(slug)
        if match:
            timeframe = match.group(1)
    return asset, timeframe


def _trade_ts(record: dict[str, Any]) -> dt.datetime | None:
    market = record.get("market") or {}
    execution = record.get("execution") or {}

    ts_raw = market.get("timestamp")
    if ts_raw:
        return dt.datetime.fromtimestamp(float(ts_raw), tz=dt.UTC)

    exec_ts = execution.get("timestamp")
    if exec_ts:
        return dt.datetime.fromtimestamp(float(exec_ts) / 1000.0, tz=dt.UTC)

    return None


def _normalise_trade_record(
    record: dict[str, Any],
    *,
    strategy: str,
    source_file: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return archive + settled rows for a single raw trade record."""
    if not isinstance(record, dict):
        return None, None

    market = record.get("market") or {}
    position = record.get("position") or {}
    execution = record.get("execution") or {}
    settlement = record.get("settlement") or {}
    context = record.get("context") or {}
    session = record.get("session") or {}
    timing = record.get("timing") or {}
    gate = record.get("gate") or {}

    ts = _trade_ts(record)
    if ts is None:
        return None, None

    trade_id = str(record.get("id") or record.get("trade_id") or "")
    if not trade_id:
        trade_id = f"{strategy}:{source_file}:{int(ts.timestamp())}"

    asset, timeframe = parse_market_slug(str(market.get("slug", "")))
    paper = context.get("mode", "paper") == "paper"

    archive_row = {
        "id": trade_id,
        "ts": ts,
        "strategy": strategy,
        "asset": asset,
        "timeframe": timeframe,
        "direction": position.get("direction"),
        "amount": position.get("amount"),
        "entry_price": execution.get("entry_price"),
        "fill_price": execution.get("fill_price"),
        "confidence": context.get("confidence"),
        "outcome": settlement.get("outcome"),
        "pnl": settlement.get("net_profit"),
        "won": settlement.get("won"),
        "paper": paper,
        "market_slug": market.get("slug"),
        "gate_name": gate.get("name"),
        "gate_boosted": gate.get("boosted"),
        "slippage_pct": execution.get("slippage_pct"),
        "spread": execution.get("spread"),
        "fill_pct": execution.get("fill_pct"),
        "best_bid": execution.get("best_bid"),
        "best_ask": execution.get("best_ask"),
        "price_movement_pct": execution.get("price_movement_pct"),
        "session_trade_n": session.get("trade_number"),
        "hour_utc": timing.get("hour_utc"),
        "day_of_week": timing.get("day_of_week"),
        "consecutive_wins": session.get("consecutive_wins"),
        "consecutive_losses": session.get("consecutive_losses"),
        "bankroll_before": session.get("bankroll_before"),
        "settlement_status": settlement.get("status"),
        "raw_json": psycopg2.extras.Json(record),
        "source_file": source_file,
    }

    settled_row: dict[str, Any] | None = None
    if settlement.get("won") is not None:
        settled_row = {
            "id": trade_id,
            "ts": ts,
            "strategy": strategy,
            "asset": asset,
            "timeframe": timeframe,
            "direction": position.get("direction"),
            "amount": position.get("amount"),
            "entry_price": execution.get("entry_price"),
            "confidence": context.get("confidence"),
            "outcome": settlement.get("outcome"),
            "pnl": settlement.get("net_profit"),
            "won": settlement.get("won"),
            "paper": paper,
        }

    return archive_row, settled_row


def build_rows_from_history_records(
    records: list[dict[str, Any]],
    *,
    strategy: str,
    source_file: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build archive and settled rows from a raw JSON history list."""
    archive_rows: list[dict[str, Any]] = []
    settled_rows: list[dict[str, Any]] = []
    for record in records:
        archive_row, settled_row = _normalise_trade_record(
            record,
            strategy=strategy,
            source_file=source_file,
        )
        if archive_row is not None:
            archive_rows.append(archive_row)
        if settled_row is not None:
            settled_rows.append(settled_row)
    return archive_rows, settled_rows


def load_history_file(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read one history file and return archive rows plus settled rows."""
    strategy = path.name.removesuffix("-history.json")
    try:
        with path.open() as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: could not read {path}: {exc}", file=sys.stderr)
        return [], []

    if not isinstance(raw, list):
        return [], []

    return build_rows_from_history_records(
        raw,
        strategy=strategy,
        source_file=path.name,
    )


def discover_history_files(state_dir: Path) -> list[Path]:
    """Return the canonical set of per-bot history files."""
    files = sorted(Path(p) for p in glob.glob(str(state_dir / "*-history.json")))
    if files:
        return files

    combined = state_dir / "trade_history_full.json"
    return [combined] if combined.exists() else []


def discover_ohlcv_files(data_dir: Path) -> list[Path]:
    """Return parquet files that look like OHLCV datasets."""
    return sorted(p for p in data_dir.glob("*.parquet") if _OHLCV_RE.match(p.stem))


def load_ohlcv_rows(data_dir: Path) -> list[dict[str, Any]]:
    """Load OHLCV parquet files into rows suitable for DB insertion."""
    rows: list[dict[str, Any]] = []
    for path in discover_ohlcv_files(data_dir):
        match = _OHLCV_RE.match(path.stem)
        if match is None:
            continue

        symbol = match.group("symbol").lower()
        timeframe = match.group("timeframe").lower()
        market = (match.group("market") or "spot").lower()
        asset = "btc" if symbol.startswith("btc") else "eth" if symbol.startswith("eth") else symbol

        frame = pd.read_parquet(path, columns=["open_time", "open", "high", "low", "close", "volume"])
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)

        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "asset": asset,
                    "timeframe": timeframe,
                    "market": market,
                    "ts": row.open_time.to_pydatetime() if hasattr(row.open_time, "to_pydatetime") else row.open_time,
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                    "source_file": path.name,
                }
            )

    return rows


def ensure_schema(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS strategies (
                name      TEXT PRIMARY KEY,
                is_active BOOLEAN NOT NULL DEFAULT true
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS polymarket_trade_archive (
                id                  TEXT NOT NULL,
                ts                  TIMESTAMPTZ NOT NULL,
                strategy            TEXT NOT NULL,
                asset               TEXT,
                timeframe           TEXT,
                direction           TEXT,
                amount              DOUBLE PRECISION,
                entry_price         DOUBLE PRECISION,
                fill_price          DOUBLE PRECISION,
                confidence          DOUBLE PRECISION,
                outcome             TEXT,
                pnl                 DOUBLE PRECISION,
                won                 BOOLEAN,
                paper               BOOLEAN,
                market_slug         TEXT,
                gate_name           TEXT,
                gate_boosted        BOOLEAN,
                slippage_pct        DOUBLE PRECISION,
                spread              DOUBLE PRECISION,
                fill_pct            DOUBLE PRECISION,
                best_bid            DOUBLE PRECISION,
                best_ask            DOUBLE PRECISION,
                price_movement_pct  DOUBLE PRECISION,
                session_trade_n     INTEGER,
                hour_utc            INTEGER,
                day_of_week         INTEGER,
                consecutive_wins    INTEGER,
                consecutive_losses  INTEGER,
                bankroll_before     DOUBLE PRECISION,
                settlement_status   TEXT,
                raw_json            JSONB NOT NULL,
                source_file         TEXT NOT NULL,
                inserted_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (id, ts)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS polymarket_ohlcv (
                asset       TEXT NOT NULL,
                timeframe   TEXT NOT NULL,
                market      TEXT NOT NULL,
                ts          TIMESTAMPTZ NOT NULL,
                open        DOUBLE PRECISION NOT NULL,
                high        DOUBLE PRECISION NOT NULL,
                low         DOUBLE PRECISION NOT NULL,
                close       DOUBLE PRECISION NOT NULL,
                volume      DOUBLE PRECISION NOT NULL,
                source_file TEXT NOT NULL,
                inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (asset, timeframe, market, ts)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS polymarket_trade_archive_strategy_ts_idx
                ON polymarket_trade_archive (strategy, ts DESC)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS polymarket_trade_archive_time_idx
                ON polymarket_trade_archive (ts DESC)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS polymarket_ohlcv_lookup_idx
                ON polymarket_ohlcv (asset, timeframe, market, ts)
            """
        )
    conn.commit()


def sync_strategies(conn: psycopg2.extensions.connection, state_dir: Path) -> None:
    """Keep the strategies table in sync with known history files and running bots."""
    try:
        import subprocess

        proc = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        running = set(proc.stdout.strip().splitlines())
    except Exception as exc:
        print(f"WARN: docker ps failed, skipping strategy sync: {exc}", file=sys.stderr)
        return

    active_from_docker = set()
    for name in running:
        if name.startswith("polymarket-") and name.endswith("-bot"):
            active_from_docker.add(name[len("polymarket-") : -len("-bot")])

    history_files = glob.glob(str(state_dir / "*-history.json"))
    strategies = {Path(path).name.removesuffix("-history.json") for path in history_files}
    active_strategies = active_from_docker & strategies

    rows = [{"name": name, "is_active": name in active_strategies} for name in strategies]
    if not rows:
        return

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO strategies (name, is_active)
            VALUES (%(name)s, %(is_active)s)
            ON CONFLICT (name) DO UPDATE SET is_active = EXCLUDED.is_active
            """,
            rows,
        )
    conn.commit()


def try_acquire_archive_lock(conn: psycopg2.extensions.connection) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (ARCHIVE_LOCK_ID,))
        return bool(cur.fetchone()[0])


def _upsert_rows(
    conn: psycopg2.extensions.connection,
    sql: str,
    rows: list[dict[str, Any]],
    *,
    page_size: int = 500,
) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=page_size)
    conn.commit()
    return len(rows)


def upsert_archive_rows(conn: psycopg2.extensions.connection, rows: list[dict[str, Any]]) -> int:
    sql = """
        INSERT INTO polymarket_trade_archive
            (id, ts, strategy, asset, timeframe, direction, amount, entry_price,
             fill_price, confidence, outcome, pnl, won, paper, market_slug,
             gate_name, gate_boosted, slippage_pct, spread, fill_pct, best_bid,
             best_ask, price_movement_pct, session_trade_n, hour_utc,
             day_of_week, consecutive_wins, consecutive_losses,
             bankroll_before, settlement_status, raw_json, source_file)
        VALUES
            (%(id)s, %(ts)s, %(strategy)s, %(asset)s, %(timeframe)s, %(direction)s,
             %(amount)s, %(entry_price)s, %(fill_price)s, %(confidence)s,
             %(outcome)s, %(pnl)s, %(won)s, %(paper)s, %(market_slug)s,
             %(gate_name)s, %(gate_boosted)s, %(slippage_pct)s, %(spread)s,
             %(fill_pct)s, %(best_bid)s, %(best_ask)s, %(price_movement_pct)s,
             %(session_trade_n)s, %(hour_utc)s, %(day_of_week)s,
             %(consecutive_wins)s, %(consecutive_losses)s,
             %(bankroll_before)s, %(settlement_status)s, %(raw_json)s,
             %(source_file)s)
        ON CONFLICT (id, ts) DO UPDATE SET
            strategy = EXCLUDED.strategy,
            asset = EXCLUDED.asset,
            timeframe = EXCLUDED.timeframe,
            direction = EXCLUDED.direction,
            amount = EXCLUDED.amount,
            entry_price = EXCLUDED.entry_price,
            fill_price = EXCLUDED.fill_price,
            confidence = EXCLUDED.confidence,
            outcome = EXCLUDED.outcome,
            pnl = EXCLUDED.pnl,
            won = EXCLUDED.won,
            paper = EXCLUDED.paper,
            market_slug = EXCLUDED.market_slug,
            gate_name = EXCLUDED.gate_name,
            gate_boosted = EXCLUDED.gate_boosted,
            slippage_pct = EXCLUDED.slippage_pct,
            spread = EXCLUDED.spread,
            fill_pct = EXCLUDED.fill_pct,
            best_bid = EXCLUDED.best_bid,
            best_ask = EXCLUDED.best_ask,
            price_movement_pct = EXCLUDED.price_movement_pct,
            session_trade_n = EXCLUDED.session_trade_n,
            hour_utc = EXCLUDED.hour_utc,
            day_of_week = EXCLUDED.day_of_week,
            consecutive_wins = EXCLUDED.consecutive_wins,
            consecutive_losses = EXCLUDED.consecutive_losses,
            bankroll_before = EXCLUDED.bankroll_before,
            settlement_status = EXCLUDED.settlement_status,
            raw_json = EXCLUDED.raw_json,
            source_file = EXCLUDED.source_file
        WHERE (
            polymarket_trade_archive.strategy,
            polymarket_trade_archive.asset,
            polymarket_trade_archive.timeframe,
            polymarket_trade_archive.direction,
            polymarket_trade_archive.amount,
            polymarket_trade_archive.entry_price,
            polymarket_trade_archive.fill_price,
            polymarket_trade_archive.confidence,
            polymarket_trade_archive.outcome,
            polymarket_trade_archive.pnl,
            polymarket_trade_archive.won,
            polymarket_trade_archive.paper,
            polymarket_trade_archive.market_slug,
            polymarket_trade_archive.gate_name,
            polymarket_trade_archive.gate_boosted,
            polymarket_trade_archive.slippage_pct,
            polymarket_trade_archive.spread,
            polymarket_trade_archive.fill_pct,
            polymarket_trade_archive.best_bid,
            polymarket_trade_archive.best_ask,
            polymarket_trade_archive.price_movement_pct,
            polymarket_trade_archive.session_trade_n,
            polymarket_trade_archive.hour_utc,
            polymarket_trade_archive.day_of_week,
            polymarket_trade_archive.consecutive_wins,
            polymarket_trade_archive.consecutive_losses,
            polymarket_trade_archive.bankroll_before,
            polymarket_trade_archive.settlement_status,
            polymarket_trade_archive.raw_json,
            polymarket_trade_archive.source_file
        ) IS DISTINCT FROM (
            EXCLUDED.strategy,
            EXCLUDED.asset,
            EXCLUDED.timeframe,
            EXCLUDED.direction,
            EXCLUDED.amount,
            EXCLUDED.entry_price,
            EXCLUDED.fill_price,
            EXCLUDED.confidence,
            EXCLUDED.outcome,
            EXCLUDED.pnl,
            EXCLUDED.won,
            EXCLUDED.paper,
            EXCLUDED.market_slug,
            EXCLUDED.gate_name,
            EXCLUDED.gate_boosted,
            EXCLUDED.slippage_pct,
            EXCLUDED.spread,
            EXCLUDED.fill_pct,
            EXCLUDED.best_bid,
            EXCLUDED.best_ask,
            EXCLUDED.price_movement_pct,
            EXCLUDED.session_trade_n,
            EXCLUDED.hour_utc,
            EXCLUDED.day_of_week,
            EXCLUDED.consecutive_wins,
            EXCLUDED.consecutive_losses,
            EXCLUDED.bankroll_before,
            EXCLUDED.settlement_status,
            EXCLUDED.raw_json,
            EXCLUDED.source_file
        )
    """
    return _upsert_rows(conn, sql, rows)


def upsert_settled_rows(conn: psycopg2.extensions.connection, rows: list[dict[str, Any]]) -> int:
    sql = """
        INSERT INTO polymarket_trades
            (id, ts, strategy, asset, timeframe, direction, amount,
             entry_price, confidence, outcome, pnl, won, paper)
        VALUES
            (%(id)s, %(ts)s, %(strategy)s, %(asset)s, %(timeframe)s,
             %(direction)s, %(amount)s, %(entry_price)s, %(confidence)s,
             %(outcome)s, %(pnl)s, %(won)s, %(paper)s)
        ON CONFLICT (id, ts) DO NOTHING
    """
    return _upsert_rows(conn, sql, rows)


def upsert_ohlcv_rows(conn: psycopg2.extensions.connection, rows: list[dict[str, Any]]) -> int:
    sql = """
        INSERT INTO polymarket_ohlcv
            (asset, timeframe, market, ts, open, high, low, close, volume, source_file)
        VALUES
            (%(asset)s, %(timeframe)s, %(market)s, %(ts)s, %(open)s, %(high)s,
             %(low)s, %(close)s, %(volume)s, %(source_file)s)
        ON CONFLICT (asset, timeframe, market, ts) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            source_file = EXCLUDED.source_file
        WHERE (
            polymarket_ohlcv.open,
            polymarket_ohlcv.high,
            polymarket_ohlcv.low,
            polymarket_ohlcv.close,
            polymarket_ohlcv.volume,
            polymarket_ohlcv.source_file
        ) IS DISTINCT FROM (
            EXCLUDED.open,
            EXCLUDED.high,
            EXCLUDED.low,
            EXCLUDED.close,
            EXCLUDED.volume,
            EXCLUDED.source_file
        )
    """
    return _upsert_rows(conn, sql, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=str(STATE_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    data_dir = Path(args.data_dir)

    history_files = discover_history_files(state_dir)
    if not history_files:
        print(f"No history files found in {state_dir}", file=sys.stderr)
        sys.exit(1)

    conn = None
    try:
        if not args.dry_run:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASS,
            )
            if not try_acquire_archive_lock(conn):
                print("Another grafana_archive run is active; skipping this tick.")
                return

        archive_rows: list[dict[str, Any]] = []
        settled_rows: list[dict[str, Any]] = []
        for path in history_files:
            rows, settled = load_history_file(path)
            print(f"  {path.name}: {len(rows)} archive rows, {len(settled)} settled rows")
            archive_rows.extend(rows)
            settled_rows.extend(settled)

        ohlcv_rows = load_ohlcv_rows(data_dir)
        print(f"  OHLCV rows: {len(ohlcv_rows)}")
        print(f"Total archive rows: {len(archive_rows)}")
        print(f"Total settled rows: {len(settled_rows)}")

        if args.dry_run:
            print("Dry run — skipping DB write.")
            return

        ensure_schema(conn)
        sync_strategies(conn, state_dir)
        inserted_archive = upsert_archive_rows(conn, archive_rows)
        inserted_settled = upsert_settled_rows(conn, settled_rows)
        inserted_ohlcv = upsert_ohlcv_rows(conn, ohlcv_rows)
        print(f"Upserted {inserted_archive} archive rows")
        print(f"Upserted {inserted_settled} settled rows")
        print(f"Upserted {inserted_ohlcv} OHLCV rows")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
