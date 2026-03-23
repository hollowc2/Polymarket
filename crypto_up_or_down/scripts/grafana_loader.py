#!/usr/bin/env python3
"""Load polymarket crypto_up_or_down history files into TimescaleDB.

Reads all *-history.json files from /opt/polymarket/state/ and upserts
settled trades into the polymarket_trades hypertable. Fully idempotent —
safe to run repeatedly via cron.

Usage:
    python grafana_loader.py [--state-dir DIR] [--dry-run]
"""

import argparse
import datetime
import glob
import json
import os
import re
import sys

import psycopg2
import psycopg2.extras

DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "butterfly_guy")
DB_USER = os.environ.get("DB_USER", "butterfly")
DB_PASS = os.environ.get("DB_PASS", "butterfly_dev")

STATE_DIR = os.environ.get("STATE_DIR", "/opt/polymarket/state")

_TF_RE = re.compile(r"-(5m|15m|1h|4h|1d)(?:-|$)")
_ASSET_RE = re.compile(r"^(eth|btc)-updown", re.IGNORECASE)


def parse_market_slug(slug: str) -> tuple[str | None, str | None]:
    """Extract (asset, timeframe) from a market slug like 'eth-updown-5m-...'."""
    asset = None
    timeframe = None
    if slug:
        m = _ASSET_RE.match(slug)
        if m:
            asset = m.group(1).lower()
        m = _TF_RE.search(slug)
        if m:
            timeframe = m.group(1)
    return asset, timeframe


def load_history_file(path: str) -> list[dict]:
    """Return all settled trade records from a history JSON file."""
    strategy = os.path.basename(path).replace("-history.json", "")
    try:
        with open(path) as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARN: could not read {path}: {e}", file=sys.stderr)
        return []

    if not isinstance(records, list):
        return []

    rows = []
    for trade in records:
        if not isinstance(trade, dict):
            continue
        settlement = trade.get("settlement") or {}
        if settlement.get("won") is None:
            continue  # unsettled — skip

        trade_id = trade.get("id", "")
        if not trade_id:
            continue

        market = trade.get("market") or {}
        position = trade.get("position") or {}
        execution = trade.get("execution") or {}
        context = trade.get("context") or {}

        # Timestamp: prefer market.timestamp (unix seconds), fall back to
        # execution.timestamp (unix ms)
        ts_raw = market.get("timestamp")
        if ts_raw:
            ts = datetime.datetime.fromtimestamp(ts_raw, tz=datetime.timezone.utc)
        else:
            exec_ts = execution.get("timestamp")
            if exec_ts:
                ts = datetime.datetime.fromtimestamp(
                    exec_ts / 1000, tz=datetime.timezone.utc
                )
            else:
                continue

        asset, timeframe = parse_market_slug(market.get("slug", ""))
        paper = context.get("mode", "paper") == "paper"

        rows.append({
            "id": trade_id,
            "ts": ts,
            "strategy": strategy,
            "asset": asset,
            "timeframe": timeframe,
            "direction": position.get("direction"),
            "amount": position.get("amount"),
            "entry_price": execution.get("entry_price"),
            "confidence": None,  # not in history schema
            "outcome": settlement.get("outcome"),
            "pnl": settlement.get("net_profit"),
            "won": settlement.get("won"),
            "paper": paper,
        })
    return rows


def upsert_rows(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
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
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    conn.commit()
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=STATE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pattern = os.path.join(args.state_dir, "*-history.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No history files found in {args.state_dir}", file=sys.stderr)
        sys.exit(1)

    all_rows: list[dict] = []
    for path in files:
        rows = load_history_file(path)
        strategy = os.path.basename(path).replace("-history.json", "")
        print(f"  {strategy}: {len(rows)} settled trades")
        all_rows.extend(rows)

    print(f"Total: {len(all_rows)} rows to upsert")

    if args.dry_run:
        print("Dry run — skipping DB write.")
        return

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    try:
        inserted = upsert_rows(conn, all_rows)
        print(f"Upserted {inserted} rows (ON CONFLICT DO NOTHING for duplicates)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
