from __future__ import annotations

import pandas as pd
import numpy as np
from pandas import DataFrame

from ..utils import CryptoParameters
from ..black_scholes import calc_delta, DELTA_PARAMETERS
from .stress_test import StressTest


class CryptoRisk:
    """
    product_level_data: DataFrame (Without Index decomposition)
    symbol_level_data: DataFrame (With Index decomposition) // not applicable for cryptos
    config: build Config object using the relevant class in utils.
    """

    def __init__(self, product_level_data, parameters: CryptoParameters):
        self.parameters = parameters
        self.total_loss = 0
        self.macro_loss = 0
        self.sector_loss = 0
        self.concentration_loss = 0
        self.rv_loss = 0
        self.delta_liq_loss = 0
        self.option_liq_loss = 0
        self.symbol_level_data_ungrouped = DataFrame()
        self.product_level_data, self.st_columns = self.apply_stress_test(
            product_level_data
        )
        self.gmv = (
            self.product_level_data['market_price'] * self.product_level_data['quantity'] *
            self.product_level_data['FX'] * self.product_level_data['multiplier']
        ).abs().sum()

    def apply_spot_vol_shocks(self, product_level_data):
        # Apply Stress Tests to product level data
        product_level_data = product_level_data.copy()
        st = StressTest()
        st_columns = ['delta']
        for i in self.parameters.crypto_shocks:
            product_level_data['spot_shock'] = i['spot_shock']
            product_level_data['vol_shock'] = i['vol_shock']
            product_level_data = st.shock_df(product_level_data, f"spot {i['spot_shock']} vol {i['vol_shock']}")
            st_columns.append(f"spot {i['spot_shock']} vol {i['vol_shock']}")
        # Calc $delta = delta * quantity * multiplier * spot
        product_level_data['delta'] = np.vectorize(
            lambda spot, **DELTA_PARAMETERS: calc_delta(spot=spot, **DELTA_PARAMETERS) if spot > 0 else 0
        )(**{col: product_level_data[col] for col in DELTA_PARAMETERS})
        product_level_data['delta'] = product_level_data['delta'] * product_level_data['quantity'] * product_level_data[
            'multiplier'] * product_level_data['spot']

        return product_level_data, st_columns

    def rv_risk(self, df_input='default'):
        """
        Return: RV details: DataFrame (sum up 'Spot RV' column to get the final number)
        """
        if df_input == 'default':
            df = self.product_level_data.copy()
        else:
            df = df_input.copy()
        for i in self.parameters.rv_scenarios.keys():
            df[i] = df[f"spot {self.parameters.rv_scenarios[i]} vol 0"]
        df['Spot RV'] = df[list(self.parameters.rv_scenarios.keys())].min(axis=1, numeric_only=True)
        # TODO # ADR/ORD offset
        return df

    @staticmethod
    def calc_delta_liq(row):
        days_to_liq = row['days_to_liq']
        delta = row['delta']
        if days_to_liq > 1:
            if delta > 0:
                return -min(1, 0.05 * (days_to_liq - 1)) * abs(delta)
            else:
                return -min(3, 0.05 * (days_to_liq - 1)) * abs(delta)
        else:
            return 0

    def delta_liquidation(self, df_input='default'):
        """
        Required fields: ['quantity', 'instrumentType', 'SECURITY_TYP2', 'VOLUME_AVG_20D', 'Market Value'],
        Return: Liq_charge: DataFrame
        df = df[(df['instrumentType'] == 'EquitySecurity') & (df['SECURITY_TYP2'] != 'Mutual Fund') &
        (df['VOLUME_AVG_20D'] != 0)]
        """
        if df_input == 'default':
            df = self.symbol_level_data_ungrouped.copy()
        else:
            df = df_input.copy()
        df = df[df['VOLUME_AVG_20D'] != 0]
        df['days_to_liq'] = df['delta'].abs() / (df['VOLUME_AVG_20D'] * df['price'])
        df['Liq Charge'] = df[['days_to_liq', 'delta']].apply(EquityRisk.calc_delta_liq, axis=1)
        return df.sort_values(by='Liq Charge', ascending=False)

    def run(self):
        """
        Attributes Created: rv_summary, .macro_summary, .sector_summary, .concentration_summary, .equity_liq_summary,
                            .equity_risk_summary
        """
        if self.product_level_data.empty or self.symbol_level_data.empty:
            self.total_loss = 0
            return 'Warning: Empty DataFrame input'
        df_rv = self.rv_risk()
        self.rv_loss = df_rv['Spot RV'].sum()

        df_liq = self.delta_liquidation()
        self.delta_liq_loss = df_liq['Liq Charge'].sum()

        self.total_loss = min(
            self.rv_loss, self.macro_loss, self.sector_loss, self.concentration_loss
        ) + self.delta_liq_loss
        self.summary_info()
        return 'Success'

    def summary_info(self):
        print('\n---------------------------------')
        print(f"GMV: {self.gmv:,.0f}")
        print(f'Equity Risk Summary: {self.total_loss:,.0f}\n')
        print('Equity Stress Tests:')
        print(f'RV Summary: {self.rv_loss:,.0f}')
        print(f'Macro Summary: {self.macro_loss:,.0f}')
        print(f'Sector Summary: {self.sector_loss:,.0f}')
        print(f'Concentration Summary: {self.concentration_loss:,.0f}\n')
        print('Liquidity Summary:')
        print(f'Delta Liq Summary: {self.delta_liq_loss:,.0f}')
        print('---------------------------------\n')
