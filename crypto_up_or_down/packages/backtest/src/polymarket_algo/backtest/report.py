def format_metrics(metrics: dict) -> str:
    wr = f"win_rate={metrics.get('win_rate', 0):.2%}"
    pnl = f"pnl={metrics.get('total_pnl', 0):.2f}"
    trades = f"trades={metrics.get('trade_count', 0)}"
    return f"{wr} {pnl} {trades}"
