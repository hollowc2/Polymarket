#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["web3>=6.0", "requests"]
# ///
"""Redeem all unredeemed winning positions for a raw EOA wallet.

Polymarket does NOT auto-redeem for EOA wallets — only for proxy wallets
created through their web UI.  Run this script to recover USDC.e from any
won trades whose conditionId can be resolved via the Gamma API.

Usage (on VPS):
    docker exec --env-file /opt/polymarket/app/.env polymarket-streak-live-bot \\
        uv run --script scripts/redeem_positions.py

    # To also redeem from a custom history file:
    docker exec --env-file /opt/polymarket/app/.env polymarket-streak-live-bot \\
        uv run --script scripts/redeem_positions.py --history /path/to/history.json
"""

import argparse
import json
import os
import sys
import time

import requests
from web3 import Web3

# ── Config ────────────────────────────────────────────────────────────────────
PRIVATE_KEY = os.environ.get("PRIVATE_KEY", "")
POLYGON_RPC = os.environ.get("POLYGON_RPC_URL", "https://rpc.ankr.com/polygon")
STATE_DIR = os.environ.get("STATE_DIR", "/opt/polymarket/state")
GAMMA_API = "https://gamma-api.polymarket.com"

FALLBACK_RPCS = [
    "https://rpc.ankr.com/polygon",
    "https://polygon.llamarpc.com",
    "https://polygon-bor-rpc.publicnode.com",
]

# Gnosis ConditionalTokens contract on Polygon (holds ERC-1155 shares)
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
# NegRisk CTF for multi-outcome markets
NEG_RISK_CTF = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
# USDC.e collateral on Polygon
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

REDEEM_ABI = [
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]


def get_w3() -> Web3:
    candidates = [POLYGON_RPC] + [r for r in FALLBACK_RPCS if r != POLYGON_RPC]
    for url in candidates:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 10}))
            if w3.is_connected():
                print(f"  Connected via {url}")
                return w3
        except Exception:
            continue
    print("ERROR: all RPC endpoints failed")
    sys.exit(1)


def fetch_condition_id(slug: str) -> tuple[str | None, bool]:
    """Fetch conditionId and negRisk flag from Gamma API for a market slug."""
    try:
        resp = requests.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None, False
        markets = data[0].get("markets", [])
        if not markets:
            return None, False
        m = markets[0]
        return m.get("conditionId"), bool(m.get("negRisk", False))
    except Exception as e:
        print(f"  Gamma API error for {slug}: {e}")
        return None, False


def redeem(w3: Web3, account, condition_id: str, direction: str, neg_risk: bool, slug: str) -> bool:
    """Submit a redeemPositions transaction. Returns True on success."""
    ctf_addr = NEG_RISK_CTF if neg_risk else CTF_CONTRACT
    # UP = outcome index 0 → indexSet 1; DOWN = outcome index 1 → indexSet 2
    index_set = 1 if direction == "up" else 2
    condition_bytes = bytes.fromhex(condition_id.replace("0x", ""))

    try:
        ctf = w3.eth.contract(address=Web3.to_checksum_address(ctf_addr), abi=REDEEM_ABI)
        tx = ctf.functions.redeemPositions(
            Web3.to_checksum_address(USDC_E),
            b"\x00" * 32,
            condition_bytes,
            [index_set],
        ).build_transaction(
            {
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "gas": 200_000,
                "gasPrice": w3.eth.gas_price,
            }
        )
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"  TX submitted: {tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        if receipt["status"] == 1:
            print(f"  ✓ Redeemed {slug} ({direction.upper()})")
            return True
        else:
            print(f"  ✗ TX reverted for {slug}")
            return False
    except Exception as e:
        print(f"  ERROR redeeming {slug}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Redeem unredeemed winning Polymarket positions")
    parser.add_argument("--history", help="Path to trade history JSON (default: auto-detect in STATE_DIR)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch conditionIds but don't submit TXs")
    args = parser.parse_args()

    if not PRIVATE_KEY:
        print("ERROR: PRIVATE_KEY not set")
        sys.exit(1)

    # Find history files
    if args.history:
        history_files = [args.history]
    else:
        history_files = [
            os.path.join(STATE_DIR, f)
            for f in os.listdir(STATE_DIR)
            if f.endswith("-history.json") or f == "trade_history_full.json"
        ]

    if not history_files:
        print(f"No history files found in {STATE_DIR}")
        sys.exit(0)

    # Collect all winning trades from all history files
    wins: list[dict] = []
    for path in history_files:
        try:
            with open(path) as f:
                trades = json.load(f)
            file_wins = [t for t in trades if t.get("settlement", {}).get("won")]
            print(f"{path}: {len(file_wins)} wins out of {len(trades)} trades")
            wins.extend(file_wins)
        except Exception as e:
            print(f"Could not read {path}: {e}")

    if not wins:
        print("No winning trades found — nothing to redeem.")
        return

    print(f"\nTotal winning trades to process: {len(wins)}")
    print("Fetching conditionIds from Gamma API...\n")

    # Deduplicate by slug (one redemption call covers all positions in a market)
    seen_slugs: set[str] = set()
    to_redeem: list[tuple[str, str, str, bool, float]] = []  # (slug, direction, condition_id, neg_risk, shares)

    for trade in wins:
        slug = trade.get("market", {}).get("slug", "")
        direction = trade.get("position", {}).get("direction", "")
        shares = trade.get("position", {}).get("shares", 0.0)
        if not slug or not direction:
            continue
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        condition_id, neg_risk = fetch_condition_id(slug)
        if not condition_id:
            print(f"  SKIP {slug}: conditionId not found")
            continue

        contract_label = "NegRisk CTF" if neg_risk else "CTF"
        cid_short = condition_id[:16] if condition_id else "?"
        print(f"  {slug}  dir={direction.upper()}  shares={shares:.4f}  conditionId={cid_short}...  [{contract_label}]")
        to_redeem.append((slug, direction, condition_id, neg_risk, shares))
        time.sleep(0.2)  # gentle rate limit on Gamma API

    if not to_redeem:
        print("\nNo redeemable positions found.")
        return

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Redeeming {len(to_redeem)} position(s)...\n")

    if args.dry_run:
        for slug, direction, _condition_id, _neg_risk, shares in to_redeem:
            print(f"  [dry-run] Would redeem {slug} {direction.upper()} ({shares:.4f} shares)")
        return

    w3 = get_w3()
    account = w3.eth.account.from_key(PRIVATE_KEY)
    print(f"Wallet: {account.address}\n")

    redeemed = 0
    for slug, direction, condition_id, neg_risk, shares in to_redeem:
        print(f"Redeeming {slug} ({direction.upper()}, {shares:.4f} shares)...")
        if redeem(w3, account, condition_id, direction, neg_risk, slug):
            redeemed += 1

    print(f"\nDone. {redeemed}/{len(to_redeem)} positions redeemed.")


if __name__ == "__main__":
    main()
