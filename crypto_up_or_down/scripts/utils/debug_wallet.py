#!/usr/bin/env python3
"""Debug wallet balance — run inside the live bot container to diagnose balance reads.

Tests three approaches and shows which one matches the Polymarket Cash balance.

Usage:
    docker exec --env-file /opt/polymarket/app/.env polymarket-streak-live-bot \
        uv run python scripts/debug_wallet.py
"""

import os

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
from web3 import Web3

key = os.getenv("PRIVATE_KEY", "")
if not key:
    print("ERROR: PRIVATE_KEY not set")
    raise SystemExit(1)

sig_type = int(os.getenv("SIGNATURE_TYPE", "0"))
rpc = os.getenv("POLYGON_RPC_URL", "https://rpc.ankr.com/polygon")
wallet_address_override = os.getenv("WALLET_ADDRESS", "")
funder = os.getenv("FUNDER_ADDRESS", "")

# For proxy wallets (sig_type=1), pass funder= so CLOB checks the proxy's balance/allowances
if sig_type == 1 and funder:
    client = ClobClient(host="https://clob.polymarket.com", key=key, chain_id=137, signature_type=1, funder=funder)
else:
    client = ClobClient(host="https://clob.polymarket.com", key=key, chain_id=137)
creds = client.create_or_derive_api_creds()
client.set_api_creds(creds)

signing_addr = client.get_address()
wallet = wallet_address_override or signing_addr
usdc_e = client.get_collateral_address()
native_usdc = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"

print(f"Signing addr : {signing_addr}")
print(f"WALLET_ADDRESS (override): {wallet_address_override or '(not set — using signing addr)'}")
print(f"Balance addr : {wallet}  ← this is what we query")
print(f"USDC.e addr  : {usdc_e}")
print(f"NativeUSDC   : {native_usdc}")
print(f"RPC          : {rpc}")
print(f"Sig type     : {sig_type}")
print()

# ── Approach 1: raw wallet balanceOf via web3 ─────────────────────────────────
print("── Approach 1: web3 wallet balanceOf ─────────────────────")
abi = [
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]
EXCHANGE_CONTRACTS = [
    ("CTF Exchange", "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"),
    ("NegRisk CTF", "0xC5d563A36AE78145C45a50134d48A1215220f80a"),
    ("NegRisk Adapter", "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"),
]

allowance_abi = [
    {
        "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    }
]

try:
    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
    print(f"  Connected : {w3.is_connected()}")
    wallet_cs = Web3.to_checksum_address(wallet)
    for label, addr in [("USDC.e", usdc_e), ("native USDC", native_usdc)]:
        tok = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=abi)
        raw = tok.functions.balanceOf(wallet_cs).call()
        dec = tok.functions.decimals().call()
        print(f"  {label:12s}: ${raw / 10**dec:.2f}  (raw={raw})")
    print()
    print("  On-chain allowances for USDC.e:")
    usdc_tok = w3.eth.contract(address=Web3.to_checksum_address(usdc_e), abi=abi + allowance_abi)
    for label, spender in EXCHANGE_CONTRACTS:
        raw = usdc_tok.functions.allowance(wallet_cs, Web3.to_checksum_address(spender)).call()
        print(f"  → {label:20s}: ${raw / 1_000_000:.2f}  (raw={raw})")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# ── Approach 2: CLOB get_balance_allowance (no update) ───────────────────────
print("── Approach 2: CLOB get_balance_allowance (no update) ────")
try:
    params = BalanceAllowanceParams(
        asset_type=AssetType.COLLATERAL,  # type: ignore[arg-type]
        signature_type=sig_type,
    )
    result = client.get_balance_allowance(params)
    print(f"  raw response : {result}")
    if isinstance(result, dict):
        bal = result.get("balance", "N/A")
        alw = result.get("allowance", "N/A")
        try:
            print(f"  balance  : ${float(bal) / 1_000_000:.2f}  (raw={bal})")
            print(f"  allowance: ${float(alw) / 1_000_000:.2f}  (raw={alw})")
        except Exception:
            print(f"  balance={bal}  allowance={alw}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# ── Approach 3: update first, then get_balance_allowance ─────────────────────
print("── Approach 3: update_balance_allowance then get ─────────")
try:
    params = BalanceAllowanceParams(
        asset_type=AssetType.COLLATERAL,  # type: ignore[arg-type]
        signature_type=sig_type,
    )
    update_resp = client.update_balance_allowance(params)
    print(f"  update response: {update_resp}")
    result = client.get_balance_allowance(params)
    print(f"  raw response   : {result}")
    if isinstance(result, dict):
        bal = result.get("balance", "N/A")
        alw = result.get("allowance", "N/A")
        try:
            print(f"  balance  : ${float(bal) / 1_000_000:.2f}  (raw={bal})")
            print(f"  allowance: ${float(alw) / 1_000_000:.2f}  (raw={alw})")
        except Exception:
            print(f"  balance={bal}  allowance={alw}")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("── Which approach matches your Polymarket Cash? ───────────")
