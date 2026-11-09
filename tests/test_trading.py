import numpy as np
import pandas as pd
import pytest

from src.evaluation.trading import TradingCosts, backtest, select_threshold


def frame():
    return pd.DataFrame(dict(timestamp=pd.date_range('2023-01-02', periods=4, freq='15min', tz='UTC'),
                             open=[1., 1., 1., 1.], close=[1.001, 1.002, .999, 1.003],
                             prob_state_1=[.9, .1, .1, .9], is_complete=True))


def test_next_open_signal_and_prior_probability():
    metrics, ledger = backtest(frame(), 1, .5)
    assert ledger.position.tolist() == [0, 0, 1, -1]
    assert metrics['trades'] == 2
    assert metrics['total_cost_usd'] == pytest.approx(2)
    assert metrics['net_profit_usd'] == pytest.approx(-42)
    assert metrics['max_drawdown_pct'] == pytest.approx(.42)


def test_gap_and_partial_cancel_entries():
    df = frame()
    df.loc[2:, 'timestamp'] += pd.Timedelta(days=2)
    df.loc[2, 'is_complete'] = False
    _, ledger = backtest(df, 1)
    assert ledger.position.tolist() == [0, 1, 0, 0]


def test_future_prices_do_not_change_past_trades():
    df = frame()
    _, before = backtest(df, 1, .5)
    df.loc[3, 'close'] = 2
    _, after = backtest(df, 1, .5)
    pd.testing.assert_frame_equal(before.iloc[:3], after.iloc[:3])
    assert before.position.equals(after.position)


def test_validation_selection_and_cash_baseline():
    threshold, scores = select_threshold(frame(), 1, TradingCosts())
    assert threshold in (.25, .5, .75)
    assert scores[str(threshold)]['net_profit_usd'] == max(x['net_profit_usd'] for x in scores.values())
    metrics, _ = backtest(frame(), 1, 0)
    assert metrics['trades'] == metrics['net_profit_usd'] == metrics['max_drawdown_pct'] == 0


@pytest.mark.parametrize('kind', ['duplicate', 'nan', 'bad_price'])
def test_invalid_input_rejected(kind):
    df = frame()
    if kind == 'duplicate':
        df.loc[1, 'timestamp'] = df.loc[0, 'timestamp']
    elif kind == 'nan':
        df.loc[1, 'prob_state_1'] = np.nan
    else:
        df.loc[1, 'open'] = -1
    with pytest.raises(ValueError):
        backtest(df, 1)


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        TradingCosts(spread_pips=-1)
