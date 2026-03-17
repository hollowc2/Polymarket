import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

load_dotenv()


# Timezone configuration — accepts any IANA name (e.g. UTC, Europe/Paris, America/New_York)
TIMEZONE_NAME = os.getenv("TIMEZONE", "UTC")
try:
    LOCAL_TZ = ZoneInfo(TIMEZONE_NAME)
except ZoneInfoNotFoundError:
    print(f"[config] Unknown timezone {TIMEZONE_NAME!r}, falling back to UTC")
    TIMEZONE_NAME = "UTC"
    LOCAL_TZ = ZoneInfo("UTC")


class Config:
    # Wallet
    PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
    # Proxy wallet address — the Polymarket account wallet that holds USDC.
    # Set this when PRIVATE_KEY is an operator/signing key for a proxy wallet
    # (the key's derived address won't hold funds; the proxy wallet does).
    WALLET_ADDRESS: str = os.getenv("WALLET_ADDRESS", "")

    # Polymarket APIs
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    CHAIN_ID = 137  # Polygon mainnet
    POLYGON_RPC_URL: str = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")

    # Strategy
    STREAK_TRIGGER: int = int(os.getenv("STREAK_TRIGGER", "4"))
    TIMEFRAME: str = os.getenv("TIMEFRAME", "5m")  # 5m | 15m | 1h
    BET_AMOUNT: float = float(os.getenv("BET_AMOUNT", "5"))
    MIN_BET: float = float(os.getenv("MIN_BET", "1"))
    MAX_DAILY_BETS: int = int(os.getenv("MAX_DAILY_BETS", "100"))
    MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "50"))
    INITIAL_BANKROLL: float = float(os.getenv("INITIAL_BANKROLL", "100"))

    # Timing
    ENTRY_SECONDS_BEFORE: int = int(os.getenv("ENTRY_SECONDS_BEFORE", "30"))

    # Mode
    PAPER_TRADE: bool = os.getenv("PAPER_TRADE", "true").lower() == "true"

    # Logging
    LOG_FILE: str = os.getenv("LOG_FILE", "bot.log")
    TRADES_FILE: str = os.getenv("TRADES_FILE", "trades.json")
    HISTORY_FILE: str = os.getenv("HISTORY_FILE", "trade_history_full.json")

    # Copytrade
    DATA_API = "https://data-api.polymarket.com"
    COPY_WALLETS: list[str] = [w.strip() for w in os.getenv("COPY_WALLETS", "").split(",") if w.strip()]
    COPY_POLL_INTERVAL: int = int(os.getenv("COPY_POLL_INTERVAL", "5"))

    # WebSocket settings
    WS_CLOB_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    WS_USER_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    WS_RTDS_URL = "wss://ws-live-data.polymarket.com"
    USE_WEBSOCKET: bool = os.getenv("USE_WEBSOCKET", "true").lower() == "true"

    # Fast polling mode (1-2s for copytrade)
    FAST_POLL_INTERVAL: float = float(os.getenv("FAST_POLL_INTERVAL", "1.5"))

    # REST client settings
    REST_TIMEOUT: float = float(os.getenv("REST_TIMEOUT", "3"))  # Faster timeout
    REST_RETRIES: int = int(os.getenv("REST_RETRIES", "2"))

    # Trading client settings
    SIGNATURE_TYPE: int = int(os.getenv("SIGNATURE_TYPE", "0"))  # 0=EOA/MetaMask, 1=Magic/proxy
    FUNDER_ADDRESS: str = os.getenv("FUNDER_ADDRESS", "")  # Required for proxy wallets

    # Polymarket CLOB API credentials (from polymarket.com → Profile → API Keys).
    # When set, these are used directly instead of deriving credentials from PRIVATE_KEY.
    # Required for proxy wallets (SIGNATURE_TYPE=1) where the CLOB account was created
    # via the Polymarket web UI.
    CLOB_API_KEY: str = os.getenv("CLOB_API_KEY", "")
    CLOB_API_SECRET: str = os.getenv("CLOB_API_SECRET", "")
    CLOB_API_PASSPHRASE: str = os.getenv("CLOB_API_PASSPHRASE", "")

    # Resilience settings
    CIRCUIT_BREAKER_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
    CIRCUIT_BREAKER_RECOVERY_TIME: int = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIME", "60"))
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "120"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Polygonscan API
    POLYGONSCAN_API_KEY: str = os.getenv("POLYGONSCAN_API_KEY", "")

    # Delay impact model parameters
    DELAY_MODEL_BASE_COEF: float = float(os.getenv("DELAY_MODEL_BASE_COEF", "0.8"))
    DELAY_MODEL_MAX_IMPACT: float = float(os.getenv("DELAY_MODEL_MAX_IMPACT", "10.0"))
    DELAY_MODEL_BASELINE_SPREAD: float = float(os.getenv("DELAY_MODEL_BASELINE_SPREAD", "0.02"))

    # Selective copytrade filter
    SELECTIVE_FILTER: bool = os.getenv("SELECTIVE_FILTER", "false").lower() == "true"
    SELECTIVE_MAX_DELAY_MS: int = int(os.getenv("SELECTIVE_MAX_DELAY_MS", "20000"))
    SELECTIVE_MIN_FILL_PRICE: float = float(os.getenv("SELECTIVE_MIN_FILL_PRICE", "0.55"))
    SELECTIVE_MAX_FILL_PRICE: float = float(os.getenv("SELECTIVE_MAX_FILL_PRICE", "0.80"))
    SELECTIVE_MAX_PRICE_MOVEMENT_PCT: float = float(os.getenv("SELECTIVE_MAX_PRICE_MOVEMENT_PCT", "15.0"))
    SELECTIVE_MAX_SPREAD: float = float(os.getenv("SELECTIVE_MAX_SPREAD", "0.025"))
    SELECTIVE_MAX_VOLATILITY_FACTOR: float = float(os.getenv("SELECTIVE_MAX_VOLATILITY_FACTOR", "1.25"))
    SELECTIVE_MIN_DEPTH_AT_BEST: float = float(os.getenv("SELECTIVE_MIN_DEPTH_AT_BEST", "5.0"))

    # Session filter — UTC hour ranges, e.g. "13-20" for US session, empty = no filter
    SESSION_FILTER_HOURS: str = os.getenv("SESSION_FILTER_HOURS", "")

    # Alternative entry features (streak_bot_alternative_entry.py)
    ALT_ENTRY_USE_STREAK_FILTER: bool = os.getenv("ALT_ENTRY_USE_STREAK_FILTER", "true").lower() == "true"
    ALT_ENTRY_MIN_STREAK: int = int(os.getenv("ALT_ENTRY_MIN_STREAK", "4"))
    ALT_ENTRY_USE_PRICE_FLOOR: bool = os.getenv("ALT_ENTRY_USE_PRICE_FLOOR", "false").lower() == "true"
    ALT_ENTRY_MAX_ENTRY_PRICE: float = float(os.getenv("ALT_ENTRY_MAX_ENTRY_PRICE", "0.44"))
    ALT_ENTRY_USE_LIMIT_ORDERS: bool = os.getenv("ALT_ENTRY_USE_LIMIT_ORDERS", "true").lower() == "true"
    ALT_ENTRY_FILL_WINDOW_SEC: int = int(os.getenv("ALT_ENTRY_FILL_WINDOW_SEC", "285"))
    # Per-streak-length discount off the best ask (JSON string or Python dict literal).
    # Example .env: ALT_ENTRY_DISCOUNTS='{"4": 0.05, "5": 0.07, "6": 0.10, "7": 0.13, "8": 0.15}'
    _alt_entry_discounts_raw: str = os.getenv(
        "ALT_ENTRY_DISCOUNTS", '{"4": 0.05, "5": 0.07, "6": 0.10, "7": 0.13, "8": 0.15}'
    )
    try:
        import json as _json

        ALT_ENTRY_DISCOUNTS: dict[int, float] = {
            int(k): float(v) for k, v in _json.loads(_alt_entry_discounts_raw).items()
        }
    except Exception:
        ALT_ENTRY_DISCOUNTS = {4: 0.05, 5: 0.07, 6: 0.10, 7: 0.13, 8: 0.15}

    # Spread-normalized discount: limit = ask - (spread * fraction).
    # Set ALT_ENTRY_USE_SPREAD_DISCOUNT=true to use instead of the fixed table above.
    # fraction=0.5 → mid-price targeting; fraction=0.7 → 70% of the way from ask toward bid.
    ALT_ENTRY_USE_SPREAD_DISCOUNT: bool = os.getenv("ALT_ENTRY_USE_SPREAD_DISCOUNT", "false").lower() == "true"
    ALT_ENTRY_DISCOUNT_FRACTION: float = float(os.getenv("ALT_ENTRY_DISCOUNT_FRACTION", "0.5"))

    # Dynamic fill window: adjust the expiry deadline based on live price movement.
    # Cancel early if the ask drifts more than CANCEL_DRIFT_THRESHOLD above our limit
    # (price moving against us after 30 s). Extend the window by EXTEND_BY_SEC if the
    # ask is within EXTEND_THRESHOLD of our limit (almost there — hold on).
    # Max total window is capped at 2× the original fill window.
    ALT_ENTRY_DYNAMIC_FILL_WINDOW: bool = os.getenv("ALT_ENTRY_DYNAMIC_FILL_WINDOW", "true").lower() == "true"
    ALT_ENTRY_CANCEL_DRIFT_THRESHOLD: float = float(os.getenv("ALT_ENTRY_CANCEL_DRIFT_THRESHOLD", "0.03"))
    ALT_ENTRY_EXTEND_THRESHOLD: float = float(os.getenv("ALT_ENTRY_EXTEND_THRESHOLD", "0.005"))
    ALT_ENTRY_EXTEND_BY_SEC: int = int(os.getenv("ALT_ENTRY_EXTEND_BY_SEC", "30"))
