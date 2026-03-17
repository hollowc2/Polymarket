import pandas as pd


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    return float(drawdown.min()) if not drawdown.empty else 0.0
