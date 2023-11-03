from __future__ import annotations

import pandas as pd
import numpy as np
from pandas import DataFrame

from ..utils import EquityParameters
from ..black_scholes import calc_delta, DELTA_PARAMETERS
from .stress_test import EQMultilevelST, StressTest

RISK_FIELDS = ['COUNTRY_ISO', 'ISSUER_BULK', 'MKT_CAP_USD_M', 'GICS_INDUSTRY_GROUP_NAME']


class EquityRisk:
    """
    product_level_data: DataFrame (Without Index decomposition)
    symbol_level_data: DataFrame (With Index decomposition)
    config: build Config object using the relevant class in utils.
    """

    def __init__(self, product_level_data, symbol_level_data, parameters: EquityParameters):
        self.parameters = parameters
        self.total_loss = 0
        self.macro_loss = 0
        self.sector_loss = 0
        self.concentration_loss = 0
        self.rv_loss = 0
        self.delta_liq_loss = 0
        self.option_liq_loss = 0
        self.symbol_level_data_ungrouped = DataFrame()
        self.product_level_data, self.symbol_level_data, self.st_columns = self.apply_stress_test(
            product_level_data, symbol_level_data
        )
        self.gmv = (
            self.product_level_data['market_price'] * self.product_level_data['quantity'] *
            self.product_level_data['FX'] * self.product_level_data['multiplier']
        ).abs().sum()

    def apply_stress_test(self, product_level_data, symbol_level_data):
        symbol_level_data = symbol_level_data[[
            'ticker', 'ref_ticker', 'book', 'weight', 'price', 'COUNTRY_ISO', 'ISSUER_BULK', 'MKT_CAP_USD_M',
            'GICS_INDUSTRY_GROUP_NAME', 'VOLUME_AVG_20D'
        ]].copy()
        # Apply Stress Tests to product level data
        product_level_data = product_level_data.copy()
        st = StressTest()
        st_columns = ['delta']
        for i in self.parameters.equity_shocks:
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
        # Map stress test losses and delta from product level data to symbol level data
        tem = product_level_data[['ticker', 'FX', 'book'] + st_columns].rename(columns={'ticker': 'ref_ticker'})
        tem = tem.groupby(by=['ref_ticker', 'FX', 'book'], as_index=False).sum()
        symbol_level_data = pd.merge(symbol_level_data, tem, on=['ref_ticker', 'book'], how='left')
        for c in st_columns:
            symbol_level_data[c] = symbol_level_data[c] * symbol_level_data['weight'] * symbol_level_data['FX']
        self.symbol_level_data_ungrouped = symbol_level_data.copy()   # for delta liq
        # net symbol level data by RISK_FIELDS
        symbol_level_data = symbol_level_data[RISK_FIELDS + st_columns]
        symbol_level_data = symbol_level_data.groupby(by=RISK_FIELDS, as_index=False).sum()
        return product_level_data, symbol_level_data, st_columns

    def directional_risk(self, df_input: pd.DataFrame, scenario_type: str) -> pd.DataFrame:
        df = df_input.copy()
        # Country Groups classify
        df['group'] = df['COUNTRY_ISO'].map(self.parameters.dict_country_grouping)
        # Apply Equity Multilevel Stress Tests
        st_list = []
        for scenario in self.parameters.macro_scenarios:
            # pass a copy of df to it, so that won't keep adding columns to df
            st = EQMultilevelST(df.copy(), scenario, scenario_type)
            st.run()
            st_list.append(st)
        # Aggregate
        results = []
        worst_case = st_list[0]
        for s in st_list:
            results.append(
                {
                    "Scenario": s.scenario_name,
                    'Total Loss': s.total_loss,
                    'Global Loss': s.global_loss,
                    f'{scenario_type.capitalize()} Loss': s.disp1_loss,
                    'Asset Loss': s.disp2_loss
                }
            )
            if s.total_loss < worst_case.total_loss:
                worst_case = s
        return DataFrame(results), worst_case.df_grouped, worst_case.data

    def macro_risk(self):
        """
        Return: Scenarios Summary: DataFrame, Country Dispersion Summary: DataFrame, Asset Dispersion Summary: DataFrame
        """
        return self.directional_risk(self.symbol_level_data, 'macro')

    def sector_risk(self):
        """
        Return: Scenarios Summary: DataFrame, Sector Dispersion Summary: DataFrame, Asset Dispersion Summary: DataFrame
        """
        return self.directional_risk(self.symbol_level_data, 'sector')

    def rv_risk(self, df_input='default'):
        """
        Return: RV details: DataFrame (sum up 'Spot RV' column to get the final number)
        """
        if df_input == 'default':
            df = self.symbol_level_data.copy()
        else:
            df = df_input.copy()
        for i in self.parameters.rv_scenarios.keys():
            df[i] = df[f"spot {self.parameters.rv_scenarios[i]} vol 0"]
        df['Spot RV'] = df[list(self.parameters.rv_scenarios.keys())].min(axis=1, numeric_only=True)
        # TODO # ADR/ORD offset
        return df

    def concentration_risk(self, df_input='default'):
        """
        Return: single_name_max_loss: Series, seven_name_max_loss: DataFrame, concentration_details: DataFrame
        """
        if df_input == 'default':
            df = self.symbol_level_data.copy()
        else:
            df = df_input.copy()
        if df.empty:
            return df, 'NA', 'NA'
        df = df[~df['ISSUER_BULK'].isin(self.parameters.concentration_scenarios['exclude'])]
        df['Single Name Up'] = 0
        df['Single Name Down'] = 0
        df['Seven Name Up'] = 0
        df['Seven Name Down'] = 0
        # classify and map shocks
        df['MktCap Level'] = pd.cut(
            df['MKT_CAP_USD_M'],
            np.array([-float("inf"), 5, 200, 500, 1000, 5000, 150000,
                      float("inf")]),
            labels=['MktCap 1', 'MktCap 2', 'MktCap 3', 'MktCap 4', 'MktCap 5', 'MktCap 6', 'MktCap 7']
        )
        df['Group'] = df['COUNTRY_ISO'].map({k: 'core' for k in self.parameters.dict_country_core['core']}
                                           ).fillna('non_core')
        for i in df.index:
            # single name shock
            shocks1 = self.parameters.concentration_scenarios['single_name_concentration_scenarios'][df.loc[
                i, 'Group']][df.loc[i, 'MktCap Level']]
            df.loc[i, 'Single Name Up'] = df.loc[i, f"spot {shocks1[0]} vol 0"]
            df.loc[i, 'Single Name Down'] = df.loc[i, f"spot {shocks1[1]} vol 0"]
            # seven names shock
            shocks7 = self.parameters.concentration_scenarios['seven_name_concentration_scenarios'][df.loc[i, 'Group']][
                df.loc[i, 'MktCap Level']]
            df.loc[i, 'Seven Name Up'] = df.loc[i, f"spot {shocks7[0]} vol 0"]
            df.loc[i, 'Seven Name Down'] = df.loc[i, f"spot {shocks7[1]} vol 0"]

        df = df.reset_index(drop=True)
        if df['Single Name Up'].min() < df['Single Name Down'].min():
            single_name_max_loss = df.loc[df['Single Name Up'].idxmin()]
        else:
            single_name_max_loss = df.loc[df['Single Name Down'].idxmin()]

        seven_name_up_max_loss = df.sort_values(by=['Seven Name Up'])
        seven_name_up_max_loss = seven_name_up_max_loss.reset_index(drop=True)
        seven_name_up_max_loss = seven_name_up_max_loss.loc[0:6]

        seven_name_down_max_loss = df.sort_values(by=['Seven Name Down'])
        seven_name_down_max_loss = seven_name_down_max_loss.reset_index(drop=True)
        seven_name_down_max_loss = seven_name_down_max_loss.loc[0:6]

        if seven_name_up_max_loss['Seven Name Up'].sum() < seven_name_down_max_loss['Seven Name Down'].sum():
            seven_name_max_loss = seven_name_up_max_loss
            seven_name_max_loss['Seven Name Loss'] = seven_name_up_max_loss['Seven Name Up']
        else:
            seven_name_max_loss = seven_name_down_max_loss
            seven_name_max_loss['Seven Name Loss'] = seven_name_down_max_loss['Seven Name Down']

        return single_name_max_loss, seven_name_max_loss, df

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

        [df_macro_summary, *_] = self.macro_risk()
        self.macro_loss = df_macro_summary['Total Loss'].min()

        [df_sector_summary, *_] = self.sector_risk()
        self.sector_loss = df_sector_summary['Total Loss'].min()

        [single_name_max_loss, seven_name_max_loss, *_] = self.concentration_risk()
        self.concentration_loss = min(
            single_name_max_loss.loc['Single Name Up'], single_name_max_loss.loc['Single Name Down'],
            seven_name_max_loss['Seven Name Up'].sum(), seven_name_max_loss['Seven Name Down'].sum()
        )

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
