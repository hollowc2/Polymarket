#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["web3>=6.0", "python-dotenv"]
# ///
"""
approve_usdc.py — Grant ERC20 allowance to Polymarket exchange contracts on Polygon.

Run this when the live bot fails with "not enough balance / allowance".

Usage:
    # Local
    uv run --script scripts/approve_usdc.py

    # On VPS inside container
    docker exec --env-file /opt/polymarket/app/.env polymarket-streak-live-bot \
        uv run --script scripts/approve_usdc.py
"""

import os
import sys

from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e (bridged)
USDC_NATIVE = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"  # native USDC

EXCHANGE_CONTRACTS = [
    ("CTF Exchange", "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"),
    ("NegRisk CTF", "0xC5d563A36AE78145C45a50134d48A1215220f80a"),
    ("NegRisk Adapter", "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"),
]

ERC20_ABI = [
    {
        "name": "allowance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "approve",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "decimals",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
    },
]

MAX_UINT256 = 2**256 - 1
APPROVAL_THRESHOLD_USDC = 1000.0  # only approve if allowance < $1000


def main() -> int:
    # -----------------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------------
    private_key = os.getenv("PRIVATE_KEY", "")
    funder_address = os.getenv("FUNDER_ADDRESS", "")
    signature_type = int(os.getenv("SIGNATURE_TYPE", "0"))
    polygon_rpc = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")

    if not private_key:
        print("ERROR: PRIVATE_KEY not set in environment / .env")
        return 1

    # -----------------------------------------------------------------------
    # Connect to Polygon
    # -----------------------------------------------------------------------
    w3 = Web3(Web3.HTTPProvider(polygon_rpc))
    if not w3.is_connected():
        print(f"ERROR: Cannot connect to Polygon RPC: {polygon_rpc}")
        return 1
    print(f"Connected to Polygon (chain {w3.eth.chain_id}) via {polygon_rpc}")

    # -----------------------------------------------------------------------
    # Determine wallet address
    # -----------------------------------------------------------------------
    try:
        account = w3.eth.account.from_key(private_key)
        eoa_address = account.address
    except Exception as e:
        print(f"ERROR: Invalid PRIVATE_KEY: {e}")
        return 1

    if signature_type == 1:
        if not funder_address:
            print("ERROR: SIGNATURE_TYPE=1 but FUNDER_ADDRESS is not set.")
            return 1
        wallet = Web3.to_checksum_address(funder_address)
        # Sanity check: is FUNDER_ADDRESS the same as the EOA derived from PRIVATE_KEY?
        if wallet.lower() != eoa_address.lower():
            print(f"\nNOTE: FUNDER_ADDRESS ({wallet}) does NOT match the EOA derived from PRIVATE_KEY ({eoa_address}).")
            print("This usually means FUNDER_ADDRESS is a Gnosis Safe or smart-contract wallet.")
            print("This script cannot sign on behalf of a Safe. Please approve the exchange contracts")
            print("via the Polymarket web UI or your Safe's transaction builder instead.")
            return 1
        print(f"Wallet (SIGNATURE_TYPE=1, proxy): {wallet}")
    else:
        wallet = eoa_address
        print(f"Wallet (SIGNATURE_TYPE=0, EOA): {wallet}")

    # -----------------------------------------------------------------------
    # MATIC balance
    # -----------------------------------------------------------------------
    matic_wei = w3.eth.get_balance(wallet)
    matic = matic_wei / 1e18
    print(f"MATIC balance: {matic:.4f} MATIC")
    if matic < 0.01:
        print("WARNING: Very low MATIC — you may not have enough gas for approve() txs.")

    # -----------------------------------------------------------------------
    # Approve each token × each exchange contract
    # -----------------------------------------------------------------------
    tokens = [
        ("USDC.e", Web3.to_checksum_address(USDC_E)),
        ("USDC", Web3.to_checksum_address(USDC_NATIVE)),
    ]

    nonce = w3.eth.get_transaction_count(wallet)

    for token_name, token_addr in tokens:
        token = w3.eth.contract(address=token_addr, abi=ERC20_ABI)
        decimals = token.functions.decimals().call()
        raw_balance = token.functions.balanceOf(wallet).call()
        balance = raw_balance / (10**decimals)
        print(f"\n{token_name} ({token_addr})")
        print(f"  Balance: {balance:.6f}")

        for exchange_name, exchange_addr in EXCHANGE_CONTRACTS:
            spender = Web3.to_checksum_address(exchange_addr)
            raw_allowance = token.functions.allowance(wallet, spender).call()
            allowance = raw_allowance / (10**decimals)
            print(f"  {exchange_name}: allowance = {allowance:,.2f}", end="")

            if allowance >= APPROVAL_THRESHOLD_USDC:
                print("  ✓ (sufficient, skipping)")
                continue

            print("  → approving MAX_UINT256 ...")
            try:
                tx = token.functions.approve(spender, MAX_UINT256).build_transaction(
                    {
                        "from": wallet,
                        "nonce": nonce,
                        "gas": 100_000,
                        "gasPrice": w3.eth.gas_price,
                    }
                )
                signed = w3.eth.account.sign_transaction(tx, private_key=private_key)
                tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                print(f"     tx sent: {tx_hash.hex()}")
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                if receipt.status == 1:
                    new_raw = token.functions.allowance(wallet, spender).call()
                    new_allowance = new_raw / (10**decimals)
                    print(f"     confirmed — new allowance: {new_allowance:,.0f}")
                else:
                    print("     ERROR: tx reverted!")
                nonce += 1
            except Exception as e:
                print(f"     ERROR: {e}")
                return 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n--- Final allowances ---")
    for token_name, token_addr in tokens:
        token = w3.eth.contract(address=Web3.to_checksum_address(token_addr), abi=ERC20_ABI)
        decimals = token.functions.decimals().call()
        for exchange_name, exchange_addr in EXCHANGE_CONTRACTS:
            spender = Web3.to_checksum_address(exchange_addr)
            raw = token.functions.allowance(wallet, spender).call()
            val = raw / (10**decimals)
            status = "✓" if val >= APPROVAL_THRESHOLD_USDC else "✗ LOW"
            print(f"  {token_name} → {exchange_name}: {val:,.0f}  {status}")

    print("\nDone. Restart the live bot to pick up updated allowances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
