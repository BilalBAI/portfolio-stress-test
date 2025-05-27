import numpy as np
import pandas as pd
from pandas import DataFrame
from .black_scholes import calc_greeks_df

COLUMNS_PG = ['time_to_expiry_day1', 'spot_day1', 'vol_day1', 'time_to_expiry_day2',
              'spot_day2', 'vol_day2', 'strike', 'rate', 'put_call', 'cost_of_carry_rate']


def pnl_greeks_attribution(df: DataFrame, trading_days: int = 365):
    '''
        Pnl Attribution between greeks. 
        PnL ≈ Δ · ΔS
            + Vega · Δσ
            + Theta · Δt
            + (1/2) · Γ · (ΔS)²
            + Higher Order terms
    '''

    if not set(COLUMNS_PG).issubset(set(df.columns)):
        raise Exception(f'Input data should include columns: {COLUMNS_PG}')
    df['normalized_greeks'] = True
    df['trading_days'] = trading_days
    df = calc_greeks_df(df, suffix='_day1')
    df = calc_greeks_df(df, suffix='_day2')

    df['delta_pnl'] = df['delta_day1'] * (df['spot_day2'] / df['spot_day1'] - 1)
    df['gamma_pnl'] = 0.5 * df['gamma_day1'] * (df['spot_day2'] / df['spot_day1'] - 1) ^ 2 * 100
    df['vega_pnl'] = df['vega_day1'] * (df['vol_day2'] - df['vol_day1']) * 100
    df['theta_pnl'] = df['theta_day1']
    return df
