from datetime import date
import pandas as pd
import pytest
from portrisk.core.option import BidAsk


def test_bidask_adding():
    # Input
    df1 = pd.DataFrame()

    # Setup
    bid_ask = BidAsk(df1)

    # Action
    bid_ask_2 = bid_ask + bid_ask

    # Assertions
    assert bid_ask_2.net_charge == 0


def test_bidask_calc_empty():
    # Input
    df1 = None

    # Setup
    bid_ask = BidAsk(df1)

    # Action
    bid_ask.calc()

    # Assertions
    assert bid_ask.net_charge == 0


def test_bidask_calc_one_value():
    # Input
    df1 = pd.DataFrame({'Expiry': ['2021-01-02'], 'PositionVega': [1.0]})
    valuation_date = date(2020, 10, 9)

    # Setup
    bid_ask = BidAsk(df1, {'Factor': 1.0}, valuation_date=valuation_date)

    # Action
    bid_ask.calc()

    # Assertions
    assert pytest.approx(bid_ask.net_charge, 0.00001) == -1.02312
