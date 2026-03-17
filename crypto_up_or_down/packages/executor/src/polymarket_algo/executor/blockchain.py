"""Polygonscan API client for fetching on-chain transaction data.

Provides blockchain details for Polygon transactions:
- Gas costs and fees
- Block confirmation
- Transaction status
"""

import time
from dataclasses import dataclass

import requests
from polymarket_algo.core.config import Config
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class OnChainTxData:
    """On-chain transaction details from Polygonscan."""

    tx_hash: str
    block_number: int
    from_address: str
    to_address: str
    gas_limit: int
    gas_used: int
    gas_price_gwei: float
    tx_fee_matic: float
    status: str  # "success" or "failed"
    timestamp: int  # Block timestamp (unix seconds)


class PolygonscanClient:
    """Client for Etherscan v2 API (Polygon chain).

    Uses the unified Etherscan v2 API which supports multiple chains
    via the chainid parameter. Polygon mainnet = 137.
    """

    BASE_URL = "https://api.etherscan.io/v2/api"
    CHAIN_ID = 137  # Polygon mainnet

    def __init__(self, api_key: str | None = None):
        """Initialize Polygonscan client.

        Args:
            api_key: Polygonscan API key. If not provided, uses Config.POLYGONSCAN_API_KEY.
        """
        self.api_key = api_key or Config.POLYGONSCAN_API_KEY

        # Create session with connection pooling
        self.session = requests.Session()

        retry_strategy = Retry(
            total=2,
            backoff_factor=0.2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            pool_connections=5,
            pool_maxsize=5,
            max_retries=retry_strategy,
        )
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "User-Agent": "PolymarketBot/2.0",
                "Accept": "application/json",
            }
        )

        # Cache to avoid refetching same transactions
        self._cache: dict[str, OnChainTxData] = {}
        self._cache_max_size = 1000

    def get_transaction(self, tx_hash: str) -> OnChainTxData | None:
        """Fetch transaction details from Polygonscan.

        Combines data from eth_getTransactionByHash and eth_getTransactionReceipt
        to get complete transaction information.

        Args:
            tx_hash: Transaction hash (0x-prefixed)

        Returns:
            OnChainTxData with transaction details, or None if not found/error
        """
        if not self.api_key:
            return None

        # Check cache first
        if tx_hash in self._cache:
            return self._cache[tx_hash]

        try:
            # Fetch transaction details
            tx_data = self._call("proxy", "eth_getTransactionByHash", txhash=tx_hash)
            if not tx_data or tx_data == "null" or not isinstance(tx_data, dict):
                return None

            # Fetch transaction receipt for gas used and status
            receipt = self._call("proxy", "eth_getTransactionReceipt", txhash=tx_hash)
            if not receipt or receipt == "null" or not isinstance(receipt, dict):
                return None

            # Parse block number (hex to int)
            block_number = int(tx_data.get("blockNumber", "0x0"), 16)

            # Parse gas values
            gas_limit = int(tx_data.get("gas", "0x0"), 16)
            gas_used = int(receipt.get("gasUsed", "0x0"), 16)

            # Gas price in wei -> gwei (1 gwei = 10^9 wei)
            gas_price_wei = int(tx_data.get("gasPrice", "0x0"), 16)
            gas_price_gwei = gas_price_wei / 1e9

            # Effective gas price (for EIP-1559 transactions)
            effective_gas_price_wei = int(receipt.get("effectiveGasPrice", tx_data.get("gasPrice", "0x0")), 16)

            # Calculate tx fee in MATIC (wei -> MATIC = wei / 10^18)
            tx_fee_wei = gas_used * effective_gas_price_wei
            tx_fee_matic = tx_fee_wei / 1e18

            # Transaction status: 0x1 = success, 0x0 = failed
            status_hex = receipt.get("status", "0x1")
            status = "success" if status_hex == "0x1" else "failed"

            # Get block timestamp (requires additional API call)
            timestamp = self._get_block_timestamp(block_number)

            result = OnChainTxData(
                tx_hash=tx_hash,
                block_number=block_number,
                from_address=tx_data.get("from", ""),
                to_address=tx_data.get("to", ""),
                gas_limit=gas_limit,
                gas_used=gas_used,
                gas_price_gwei=gas_price_gwei,
                tx_fee_matic=tx_fee_matic,
                status=status,
                timestamp=timestamp,
            )

            # Cache the result
            self._cache_result(tx_hash, result)

            return result

        except Exception as e:
            print(f"[polygonscan] Error fetching tx {tx_hash[:10]}...: {e}")
            return None

    def _get_block_timestamp(self, block_number: int) -> int:
        """Get timestamp of a block.

        Args:
            block_number: Block number

        Returns:
            Unix timestamp of the block, or current time if not found
        """
        try:
            block_hex = hex(block_number)
            block_data = self._call("proxy", "eth_getBlockByNumber", tag=block_hex, boolean="false")
            if block_data and block_data != "null" and isinstance(block_data, dict):
                timestamp_hex = block_data.get("timestamp", "0x0")
                return int(timestamp_hex, 16)
        except Exception:
            pass
        return int(time.time())

    def _call(self, module: str, action: str, **params) -> dict | str | None:
        """Make API call to Etherscan.

        Args:
            module: API module (e.g., "proxy")
            action: API action (e.g., "eth_getTransactionByHash")
            **params: Additional parameters

        Returns:
            API result or None on error
        """
        request_params = {
            "chainid": self.CHAIN_ID,
            "module": module,
            "action": action,
            "apikey": self.api_key,
        }
        request_params.update(params)

        try:
            resp = self.session.get(
                self.BASE_URL,
                params=request_params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            # Etherscan returns {"status": "1", "result": ...} for success
            # or {"status": "0", "message": "...", "result": "..."} for error
            result = data.get("result")

            # Check for error responses
            if data.get("status") == "0" and data.get("message") != "OK":
                error_msg = data.get("message", "Unknown error")
                if "rate limit" in error_msg.lower():
                    print(f"[polygonscan] Rate limited: {error_msg}")
                return None

            return result

        except requests.exceptions.Timeout:
            return None
        except Exception as e:
            print(f"[polygonscan] API error: {e}")
            return None

    def _cache_result(self, tx_hash: str, data: OnChainTxData):
        """Cache transaction data with size limit."""
        # Evict oldest entries if cache is full
        if len(self._cache) >= self._cache_max_size:
            # Remove first 100 entries (oldest)
            keys_to_remove = list(self._cache.keys())[:100]
            for key in keys_to_remove:
                del self._cache[key]

        self._cache[tx_hash] = data

    def is_available(self) -> bool:
        """Check if Polygonscan API is available (has API key)."""
        return bool(self.api_key)
