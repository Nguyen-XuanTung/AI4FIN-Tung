"""Causal, one-bar trading experiment; prices are indicative, not broker fills."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradingCosts:
    spread_pips: float = 0.1
    commission_per_lot_side: float = 3.5
    slippage_pips_side: float = 0.1
    units: float = 10_000
    initial_equity: float = 10_000

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Costs and sizing must be finite and nonnegative")
        if self.units <= 0 or self.initial_equity <= 0:
            raise ValueError("Units and equity must be positive")

    @property
    def round_trip_usd(self):
        return (self.spread_pips + 2 * self.slippage_pips_side) * 0.0001 * self.units + 2 * self.commission_per_lot_side * self.units / 100_000


def backtest(frame, turbulence_state, threshold=None, costs=None):
    """Prior candle body sign -> next open entry -> same candle close exit.

    No position spans a candle boundary. A missing/partial predecessor cancels
    the next entry. Each period starts flat; no cross-split signals are used.
    """
    costs = costs or TradingCosts()
    df = frame.reset_index(drop=True).copy()
    ts = pd.to_datetime(df.timestamp, utc=True)
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError("Unique, increasing timestamps required")
    prices = df[["open", "close"]].to_numpy(dtype=float)
    probs = df[f"prob_state_{turbulence_state}"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Invalid prices")
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Invalid probabilities")
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Threshold must be in [0, 1]")
    signal = np.sign(df.close - df.open).shift(1).fillna(0)
    eligible = ts.diff().eq(pd.Timedelta(minutes=15))
    if "is_complete" in df:
        eligible &= df.is_complete.eq(True) & df.is_complete.shift(1).eq(True)
    if threshold is not None:
        eligible &= pd.Series(probs).shift(1).lt(threshold)
    position = signal.where(eligible, 0)
    traded = position.ne(0)
    gross = position * (df.close - df.open) * costs.units
    net = gross - traded.astype(int) * costs.round_trip_usd
    equity = costs.initial_equity + net.cumsum()
    peak = np.maximum.accumulate(np.r_[costs.initial_equity, equity])
    drawdown = 1 - np.r_[costs.initial_equity, equity] / peak
    returns = net / costs.initial_equity
    var = float(np.quantile(returns, 0.05)) if len(df) else 0.0
    tail = returns[returns <= var]
    metrics = dict(bars=len(df), trades=int(traded.sum()), net_profit_usd=float(net.sum()),
                   total_cost_usd=float(traded.sum() * costs.round_trip_usd),
                   return_pct=float(net.sum() / costs.initial_equity * 100),
                   max_drawdown_pct=float(drawdown.max() * 100),
                   var95_bar_pct=-var * 100,
                   es95_bar_pct=-float(tail.mean()) * 100 if len(tail) else 0.0)
    ledger = pd.DataFrame(dict(timestamp=ts, position=position, gross_usd=gross,
                               net_usd=net, equity_usd=equity))
    return metrics, ledger


def select_threshold(validation, turbulence_state, costs, candidates=(0.25, 0.5, 0.75)):
    """Select on validation only; maximize net P&L, prefer lower threshold on ties."""
    scores = [(float(t), backtest(validation, turbulence_state, t, costs)[0]) for t in candidates]
    if not scores:
        raise ValueError("At least one threshold required")
    winner = max(scores, key=lambda item: (item[1]["net_profit_usd"], -item[0]))
    return winner[0], {str(t): metrics for t, metrics in scores}
