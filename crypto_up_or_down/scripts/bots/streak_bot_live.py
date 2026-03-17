"""streak_bot_live.py — live production bot.

Streak reversal with FOK market orders (guaranteed fill on every signal).

Limit orders were abandoned after live data showed that spread-normalized
discounts of 10-20% below ask rarely fill in 285s on the thin Polymarket
ETH CLOB, causing the live bot to miss ~83% of its winning trades. FOK
market orders sacrifice the entry-price discount but guarantee execution.

Startup:
  - Queries CLOB for any open GTC orders left from a prior crash and cancels them.
  - Then runs the standard streak reversal loop with market orders.

Usage:
    uv run python scripts/streak_bot_live.py
    uv run python scripts/streak_bot_live.py --asset eth
"""

import os
import sys

# Live defaults prepended so runtime CLI args (e.g. --asset) still override them
_LIVE_DEFAULTS = [
    "--live",
    "--no-limit-orders",
]

sys.argv = [sys.argv[0]] + _LIVE_DEFAULTS + sys.argv[1:]
sys.path.insert(0, os.path.dirname(__file__))
from streak_bot_alternative_entry import main  # noqa: E402

main()
