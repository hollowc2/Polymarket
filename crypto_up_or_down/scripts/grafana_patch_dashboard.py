#!/usr/bin/env python3
"""Add leaderboard + trade drill-down panels to the existing Grafana dashboard.

Finds the "crypto up down" dashboard via the Grafana search API, injects a
Bot Leaderboard table and a per-strategy trade list, then saves it back.

Both panels query the polymarket_trades TimescaleDB table (populated by
grafana_loader.py).  A $strategy template variable is added to the dashboard
so you can drill into any strategy from the leaderboard.

Usage:
    python scripts/grafana_patch_dashboard.py \\
        --grafana-url http://localhost:3000 \\
        --api-key <service-account-token>   \\   # preferred
        --dashboard "crypto up down"            # substring match on title

    # Basic-auth fallback (if no service account configured):
    python scripts/grafana_patch_dashboard.py \\
        --grafana-url http://localhost:3000 \\
        --user admin --password admin \\
        --dashboard "crypto up down"
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from urllib.error import HTTPError, URLError


# ── Panel definitions ─────────────────────────────────────────────────────────

LEADERBOARD_SQL = """\
WITH base AS (
  SELECT
    strategy,
    COUNT(*)                                                          AS n,
    SUM(CASE WHEN won THEN 1.0 ELSE 0.0 END)                         AS wins,
    AVG(entry_price)                                                  AS avg_fill,
    SUM(pnl)                                                          AS total_pnl,
    AVG(pnl)                                                          AS avg_pnl,
    STDDEV_SAMP(pnl)                                                  AS std_pnl
  FROM polymarket_trades
  WHERE won IS NOT NULL
  GROUP BY strategy
),
cumulative AS (
  SELECT strategy, ts,
    SUM(pnl) OVER (PARTITION BY strategy ORDER BY ts
                   ROWS UNBOUNDED PRECEDING)                          AS cum_pnl
  FROM polymarket_trades
  WHERE won IS NOT NULL
),
drawdown AS (
  SELECT strategy, cum_pnl,
    cum_pnl - MAX(cum_pnl) OVER (PARTITION BY strategy ORDER BY ts
                                 ROWS UNBOUNDED PRECEDING)            AS dd
  FROM cumulative
),
max_dd AS (
  SELECT strategy, MIN(dd) AS max_drawdown
  FROM drawdown GROUP BY strategy
),
loss_runs AS (
  SELECT strategy, won, ts,
    ROW_NUMBER() OVER (PARTITION BY strategy ORDER BY ts) -
    ROW_NUMBER() OVER (PARTITION BY strategy, won ORDER BY ts)        AS grp
  FROM polymarket_trades WHERE won IS NOT NULL
),
max_cl AS (
  SELECT strategy, COALESCE(MAX(run_len), 0) AS max_consec_losses
  FROM (
    SELECT strategy, grp, COUNT(*) AS run_len
    FROM loss_runs WHERE NOT won
    GROUP BY strategy, grp
  ) sub
  GROUP BY strategy
)
SELECT
  b.strategy                                                                          AS "Bot",
  b.n                                                                                 AS "n",
  ROUND((b.wins / NULLIF(b.n, 0) * 100)::numeric, 1)                                AS "Win%",
  ROUND((1.96 * SQRT(b.wins * (b.n - b.wins) / NULLIF(b.n, 0) + 0.9604)
           / NULLIF(b.n + 3.8416, 0) * 100)::numeric, 1)                            AS "± CI",
  ROUND((b.avg_fill / NULLIF(
           (1 - b.avg_fill) * (1 - 0.02 * b.avg_fill * (1 - b.avg_fill)) + b.avg_fill,
           0) * 100)::numeric, 1)                                                    AS "BE%",
  ROUND(((b.wins / NULLIF(b.n, 0)) - b.avg_fill / NULLIF(
           (1 - b.avg_fill) * (1 - 0.02 * b.avg_fill * (1 - b.avg_fill)) + b.avg_fill,
           0)) * 100, 1)                                                             AS "Edge%",
  ROUND(b.total_pnl::numeric, 2)                                                     AS "PnL $",
  ROUND(b.avg_pnl::numeric, 4)                                                       AS "Avg $/trade",
  ROUND((b.avg_pnl / NULLIF(b.std_pnl, 0) * SQRT(b.n))::numeric, 2)                AS "Sharpe",
  ROUND(d.max_drawdown::numeric, 2)                                                  AS "MaxDD $",
  c.max_consec_losses                                                                AS "MaxCL"
FROM base b
LEFT JOIN max_dd d ON d.strategy = b.strategy
LEFT JOIN max_cl c ON c.strategy = b.strategy
ORDER BY b.total_pnl DESC
"""

TRADES_SQL = """\
WITH ordered AS (
  SELECT
    ROW_NUMBER() OVER (ORDER BY ts DESC)                               AS rn,
    TO_CHAR(ts AT TIME ZONE 'UTC', 'MM-DD HH24:MI')                   AS dt,
    CASE WHEN won THEN 'W' ELSE 'L' END                               AS wl,
    entry_price,
    pnl,
    ROUND(SUM(pnl) OVER (ORDER BY ts
                         ROWS UNBOUNDED PRECEDING)::numeric, 2)        AS cum_pnl,
    amount,
    direction
  FROM polymarket_trades
  WHERE won IS NOT NULL
    AND strategy = '$strategy'
)
SELECT
  rn            AS "#",
  dt            AS "Date/Time",
  wl            AS "W/L",
  ROUND(entry_price::numeric, 3)  AS "Fill",
  ROUND(pnl::numeric, 2)          AS "Net P&L",
  cum_pnl                         AS "Cum P&L",
  ROUND(amount::numeric, 2)       AS "Size $",
  direction                       AS "Dir"
FROM ordered
ORDER BY rn
"""

# Shared field colour overrides for money columns (green > 0, red < 0)
def _money_override(col_name: str) -> dict:
    return {
        "matcher": {"id": "byName", "options": col_name},
        "properties": [
            {"id": "unit", "value": "currencyUSD"},
            {"id": "decimals", "value": 2},
            {"id": "custom.cellOptions", "value": {"type": "color-text"}},
            {
                "id": "thresholds",
                "value": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "green", "value": 0.01},
                    ],
                },
            },
        ],
    }


def _pct_override(col_name: str, green_above: float = 0.01) -> dict:
    return {
        "matcher": {"id": "byName", "options": col_name},
        "properties": [
            {"id": "unit", "value": "percent"},
            {"id": "decimals", "value": 1},
            {"id": "custom.cellOptions", "value": {"type": "color-text"}},
            {
                "id": "thresholds",
                "value": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "green", "value": green_above},
                    ],
                },
            },
        ],
    }


def leaderboard_panel(ds_uid: str, panel_id: int, y: int) -> dict:
    return {
        "id": panel_id,
        "title": "Bot Leaderboard",
        "type": "table",
        "gridPos": {"h": 14, "w": 24, "x": 0, "y": y},
        "datasource": {"type": "postgres", "uid": ds_uid},
        "options": {
            "cellHeight": "sm",
            "footer": {"show": False},
            "showHeader": True,
            "sortBy": [{"displayName": "PnL $", "desc": True}],
        },
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "custom": {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False},
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "text", "value": None}],
                },
            },
            "overrides": [
                # Bot column — clickable link that sets $strategy
                {
                    "matcher": {"id": "byName", "options": "Bot"},
                    "properties": [
                        {"id": "custom.width", "value": 200},
                        {
                            "id": "links",
                            "value": [
                                {
                                    "title": "View trades",
                                    "url": "?var-strategy=${__data.fields.Bot}",
                                    "targetBlank": False,
                                }
                            ],
                        },
                    ],
                },
                {"matcher": {"id": "byName", "options": "n"}, "properties": [
                    {"id": "custom.width", "value": 60},
                ]},
                # Percentage columns (already formatted as 0-100 in SQL)
                _pct_override("Win%", green_above=50.0),
                {"matcher": {"id": "byName", "options": "± CI"}, "properties": [
                    {"id": "unit", "value": "percent"},
                    {"id": "decimals", "value": 1},
                    {"id": "custom.width", "value": 75},
                ]},
                _pct_override("BE%", green_above=0),
                _pct_override("Edge%", green_above=0.01),
                _money_override("PnL $"),
                {**_money_override("Avg $/trade"), "properties": [
                    *_money_override("Avg $/trade")["properties"],
                    {"id": "custom.width", "value": 110},
                ]},
                {
                    "matcher": {"id": "byName", "options": "Sharpe"},
                    "properties": [
                        {"id": "decimals", "value": 2},
                        {"id": "custom.width", "value": 80},
                        {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                        {
                            "id": "thresholds",
                            "value": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "red", "value": None},
                                    {"color": "yellow", "value": 0},
                                    {"color": "green", "value": 1},
                                ],
                            },
                        },
                    ],
                },
                {
                    "matcher": {"id": "byName", "options": "MaxDD $"},
                    "properties": [
                        {"id": "unit", "value": "currencyUSD"},
                        {"id": "decimals", "value": 2},
                        {"id": "custom.width", "value": 100},
                        {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                        {
                            "id": "thresholds",
                            "value": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "red", "value": None},
                                    {"color": "text", "value": 0},
                                ],
                            },
                        },
                    ],
                },
                {"matcher": {"id": "byName", "options": "MaxCL"}, "properties": [
                    {"id": "custom.width", "value": 70},
                ]},
            ],
        },
        "targets": [
            {
                "refId": "A",
                "rawSql": LEADERBOARD_SQL,
                "format": "table",
            }
        ],
        "transformations": [],
    }


def trades_panel(ds_uid: str, panel_id: int, y: int) -> dict:
    return {
        "id": panel_id,
        "title": "$strategy — Trades",
        "type": "table",
        "gridPos": {"h": 20, "w": 24, "x": 0, "y": y},
        "datasource": {"type": "postgres", "uid": ds_uid},
        "options": {
            "cellHeight": "sm",
            "footer": {"show": False},
            "showHeader": True,
        },
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "custom": {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False},
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "text", "value": None}],
                },
            },
            "overrides": [
                {"matcher": {"id": "byName", "options": "#"}, "properties": [
                    {"id": "custom.width", "value": 50},
                ]},
                {"matcher": {"id": "byName", "options": "Date/Time"}, "properties": [
                    {"id": "custom.width", "value": 110},
                ]},
                {
                    "matcher": {"id": "byName", "options": "W/L"},
                    "properties": [
                        {"id": "custom.width", "value": 60},
                        {
                            "id": "mappings",
                            "value": [
                                {
                                    "type": "value",
                                    "options": {
                                        "W": {"text": "W", "color": "green", "index": 0},
                                        "L": {"text": "L", "color": "red", "index": 1},
                                    },
                                }
                            ],
                        },
                        {"id": "custom.cellOptions", "value": {"type": "color-background"}},
                    ],
                },
                {"matcher": {"id": "byName", "options": "Fill"}, "properties": [
                    {"id": "decimals", "value": 3},
                    {"id": "custom.width", "value": 70},
                ]},
                _money_override("Net P&L"),
                _money_override("Cum P&L"),
                {"matcher": {"id": "byName", "options": "Size $"}, "properties": [
                    {"id": "unit", "value": "currencyUSD"},
                    {"id": "decimals", "value": 2},
                    {"id": "custom.width", "value": 80},
                ]},
                {
                    "matcher": {"id": "byName", "options": "Dir"},
                    "properties": [
                        {"id": "custom.width", "value": 65},
                        {
                            "id": "mappings",
                            "value": [
                                {
                                    "type": "value",
                                    "options": {
                                        "UP": {"text": "UP", "color": "green", "index": 0},
                                        "DOWN": {"text": "DOWN", "color": "red", "index": 1},
                                    },
                                }
                            ],
                        },
                        {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                    ],
                },
            ],
        },
        "targets": [
            {
                "refId": "A",
                "rawSql": TRADES_SQL,
                "format": "table",
            }
        ],
        "transformations": [],
    }


def strategy_variable(ds_uid: str) -> dict:
    return {
        "current": {},
        "datasource": {"type": "postgres", "uid": ds_uid},
        "definition": "SELECT DISTINCT strategy FROM polymarket_trades WHERE won IS NOT NULL ORDER BY 1",
        "hide": 0,
        "includeAll": False,
        "label": "Strategy",
        "multi": False,
        "name": "strategy",
        "options": [],
        "query": "SELECT DISTINCT strategy FROM polymarket_trades WHERE won IS NOT NULL ORDER BY 1",
        "refresh": 2,
        "regex": "",
        "sort": 1,
        "type": "query",
    }


# ── Grafana API client ────────────────────────────────────────────────────────


class GrafanaClient:
    def __init__(self, base_url: str, api_key: str | None = None, user: str = "", password: str = ""):
        self.base_url = base_url.rstrip("/")
        self.headers: dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        elif user:
            creds = base64.b64encode(f"{user}:{password}".encode()).decode()
            self.headers["Authorization"] = f"Basic {creds}"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            print(f"HTTP {e.code} {method} {url}: {e.read().decode()}", file=sys.stderr)
            raise
        except URLError as e:
            print(f"Connection error {method} {url}: {e.reason}", file=sys.stderr)
            raise

    def search(self, query: str) -> list[dict]:
        return self._request("GET", f"/api/search?query={urllib.request.quote(query)}&type=dash-db")

    def get_dashboard(self, uid: str) -> dict:
        return self._request("GET", f"/api/dashboards/uid/{uid}")

    def save_dashboard(self, dashboard: dict, folder_id: int = 0, message: str = "") -> dict:
        return self._request("POST", "/api/dashboards/db", {
            "dashboard": dashboard,
            "folderId": folder_id,
            "message": message,
            "overwrite": True,
        })

    def get_datasources(self) -> list[dict]:
        return self._request("GET", "/api/datasources")


# ── Main ──────────────────────────────────────────────────────────────────────


def find_postgres_datasource(client: GrafanaClient) -> tuple[str, str]:
    """Return (uid, name) of the first TimescaleDB/PostgreSQL datasource."""
    pg_types = {"postgres", "grafana-postgresql-datasource"}
    for ds in client.get_datasources():
        if ds.get("type") in pg_types:
            return ds["uid"], ds["name"]
    raise RuntimeError("No PostgreSQL/TimescaleDB datasource found in Grafana.")


def next_panel_id(panels: list[dict]) -> int:
    existing_ids = {p.get("id", 0) for p in panels}
    i = max(existing_ids, default=0) + 1
    while i in existing_ids:
        i += 1
    return i


def bottom_y(panels: list[dict]) -> int:
    return max((p.get("gridPos", {}).get("y", 0) + p.get("gridPos", {}).get("h", 0) for p in panels), default=0)


def patch_dashboard(dashboard: dict, ds_uid: str) -> dict:
    panels = dashboard.get("panels", [])

    # Skip if already patched
    existing_titles = {p.get("title", "") for p in panels}
    if "Bot Leaderboard" in existing_titles:
        print("Bot Leaderboard panel already present — skipping.")
        return dashboard

    # Add $strategy variable if not already present
    templating = dashboard.setdefault("templating", {})
    variables: list[dict] = templating.setdefault("list", [])
    if not any(v.get("name") == "strategy" for v in variables):
        variables.append(strategy_variable(ds_uid))
        print("  + added $strategy variable")

    # Append panels below all existing content
    y = bottom_y(panels)
    id1 = next_panel_id(panels)
    id2 = id1 + 1

    panels.append(leaderboard_panel(ds_uid, id1, y))
    panels.append(trades_panel(ds_uid, id2, y + 14))
    print(f"  + added Bot Leaderboard (id={id1}, y={y})")
    print(f"  + added Trades drill-down (id={id2}, y={y + 14})")

    dashboard["panels"] = panels
    dashboard["version"] = dashboard.get("version", 1) + 1
    return dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Grafana dashboard with leaderboard + trades panels")
    parser.add_argument("--grafana-url", default="http://localhost:3000", metavar="URL")
    parser.add_argument("--api-key", default=None, metavar="TOKEN", help="Service account bearer token")
    parser.add_argument("--user", default="admin", metavar="USER", help="Basic auth username")
    parser.add_argument("--password", default="admin", metavar="PASS", help="Basic auth password")
    parser.add_argument("--dashboard", default="crypto up down", metavar="TITLE",
                        help="Substring of the dashboard title to patch (default: 'crypto up down')")
    parser.add_argument("--dry-run", action="store_true", help="Print patched JSON without saving")
    args = parser.parse_args()

    client = GrafanaClient(args.grafana_url, api_key=args.api_key, user=args.user, password=args.password)

    # --- Find dashboard ---
    results = client.search(args.dashboard)
    if not results:
        print(f"No dashboard found matching '{args.dashboard}'", file=sys.stderr)
        sys.exit(1)
    if len(results) > 1:
        print(f"Multiple dashboards matched '{args.dashboard}':")
        for r in results:
            print(f"  [{r['uid']}] {r['title']}")
        print("Use a more specific --dashboard value.", file=sys.stderr)
        sys.exit(1)

    uid = results[0]["uid"]
    title = results[0]["title"]
    print(f"Found: [{uid}] {title}")

    # --- Find TimescaleDB datasource ---
    ds_uid, ds_name = find_postgres_datasource(client)
    print(f"Datasource: {ds_name} (uid={ds_uid})")

    # --- Fetch full dashboard ---
    resp = client.get_dashboard(uid)
    meta = resp["meta"]
    dashboard = resp["dashboard"]

    # --- Patch ---
    patched = patch_dashboard(dashboard, ds_uid)

    if args.dry_run:
        print(json.dumps(patched, indent=2))
        return

    # --- Save ---
    save_resp = client.save_dashboard(
        patched,
        folder_id=meta.get("folderId", 0),
        message="Add bot leaderboard + trade drill-down panels",
    )
    print(f"Saved: {save_resp.get('url', '(no url)')}")
    print("Done — refresh your Grafana dashboard.")


if __name__ == "__main__":
    main()
