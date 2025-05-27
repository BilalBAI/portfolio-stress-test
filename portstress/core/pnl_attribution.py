import numpy as np
import pandas as pd
from pandas import DataFrame
from .black_scholes import calc_greeks_df

COLUMNS_PG = ['time_to_expiry_T1', 'spot_T1', 'vol_T1', 'time_to_expiry_T2',
              'spot_T2', 'vol_T2', 'strike', 'rate', 'put_call', 'cost_of_carry_rate']


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
    df = calc_greeks_df(df, suffix='_T1')
    df = calc_greeks_df(df, suffix='_T2')

    df['delta_pnl'] = df['delta_T1'] * (df['spot_T2'] / df['spot_T1'] - 1)
    df['gamma_pnl'] = 0.5 * df['gamma_T1'] * (df['spot_T2'] / df['spot_T1'] - 1)**2 * 100
    df['vega_pnl'] = df['vega_T1'] * (df['vol_T2'] - df['vol_T1']) * 100
    df['theta_pnl'] = df['theta_T1']
    return df
