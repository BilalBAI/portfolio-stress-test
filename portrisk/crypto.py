from __future__ import annotations
import typing as ty

from datetime import datetime, date

import pandas as pd
import numpy as np

from .core.utils import CryptoParameters, VolSurfaceParameters
from .core.black_scholes import calc_delta, DELTA_PARAMETERS
from .core.stress_test import StressTest
from .core.vol_surface_shocks import Concentration, TermStructure, Skew, BidAsk


class CryptoSpotVolShocks:
    """
    data: DataFrame (Without Index decomposition)
    config: build Config object using the relevant class in utils.
    """

    def __init__(self, parameters: CryptoParameters):
        self.parameters = parameters

    def apply_spot_vol_shocks(self, data):
        # Apply Stress Tests to product level data
        data = data.copy()
        st = StressTest()
        st_columns = ['delta']
        shocks = self.parameters.crypto_shocks
        for k in shocks.keys():
            data['spot_shock'] = data['underlying'].map(shocks[k]['spot_shock'])
            data['vol_shock'] = data['underlying'].map(shocks[k]['vol_shock'])
            data = st.shock_df(data, f"{k}")
            st_columns.append(f"{k}")
        # Calc $delta = delta * quantity * multiplier * spot
        data['delta'] = np.vectorize(
            lambda spot, **DELTA_PARAMETERS: calc_delta(spot=spot, **DELTA_PARAMETERS) if spot > 0 else 0
        )(**{col: data[col] for col in DELTA_PARAMETERS})
        data['delta'] = data['delta'] * data['quantity'] * data[
            'multiplier'] * data['spot']

        return data, st_columns

    # def rv_risk(self, df_input='default'):
    #     """
    #     Return: RV details: DataFrame (sum up 'Spot RV' column to get the final number)
    #     """
    #     if df_input == 'default':
    #         df = self.data.copy()
    #     else:
    #         df = df_input.copy()
    #     for i in self.parameters.rv_scenarios.keys():
    #         df[i] = df[f"spot {self.parameters.rv_scenarios[i]} vol 0"]
    #     df['Spot RV'] = df[list(self.parameters.rv_scenarios.keys())].min(axis=1, numeric_only=True)
    #     return df

    # @staticmethod
    # def calc_delta_liq(row):
    #     days_to_liq = row['days_to_liq']
    #     delta = row['delta']
    #     if days_to_liq > 1:
    #         if delta > 0:
    #             return -min(1, 0.05 * (days_to_liq - 1)) * abs(delta)
    #         else:
    #             return -min(3, 0.05 * (days_to_liq - 1)) * abs(delta)
    #     else:
    #         return 0

    # def delta_liquidation(self, df_input='default'):
    #     """
    #     Required fields: ['quantity', 'instrumentType', 'SECURITY_TYP2', 'VOLUME_AVG_20D', 'Market Value'],
    #     Return: Liq_charge: DataFrame
    #     df = df[(df['instrumentType'] == 'EquitySecurity') & (df['SECURITY_TYP2'] != 'Mutual Fund') &
    #     (df['VOLUME_AVG_20D'] != 0)]
    #     """
    #     if df_input == 'default':
    #         df = self.symbol_level_data_ungrouped.copy()
    #     else:
    #         df = df_input.copy()
    #     df = df[df['VOLUME_AVG_20D'] != 0]
    #     df['days_to_liq'] = df['delta'].abs() / (df['VOLUME_AVG_20D'] * df['price'])
    #     df['Liq Charge'] = df[['days_to_liq', 'delta']].apply(EquityRisk.calc_delta_liq, axis=1)
    #     return df.sort_values(by='Liq Charge', ascending=False)

    # def run(self):
    #     """
    #     Attributes Created: rv_summary, .macro_summary, .sector_summary, .concentration_summary, .equity_liq_summary,
    #                         .equity_risk_summary
    #     """
    #     if self.data.empty or self.symbol_level_data.empty:
    #         self.total_loss = 0
    #         return 'Warning: Empty DataFrame input'
    #     df_rv = self.rv_risk()
    #     self.rv_loss = df_rv['Spot RV'].sum()

    #     df_liq = self.delta_liquidation()
    #     self.delta_liq_loss = df_liq['Liq Charge'].sum()

    #     self.total_loss = min(
    #         self.rv_loss, self.macro_loss, self.sector_loss, self.concentration_loss
    #     ) + self.delta_liq_loss
    #     self.summary_info()
    #     return 'Success'

    # def summary_info(self):
    #     print('\n---------------------------------')
    #     print(f"GMV: {self.gmv:,.0f}")
    #     print(f'Equity Risk Summary: {self.total_loss:,.0f}\n')
    #     print('Equity Stress Tests:')
    #     print(f'RV Summary: {self.rv_loss:,.0f}')
    #     print(f'Macro Summary: {self.macro_loss:,.0f}')
    #     print(f'Sector Summary: {self.sector_loss:,.0f}')
    #     print(f'Concentration Summary: {self.concentration_loss:,.0f}\n')
    #     print('Liquidity Summary:')
    #     print(f'Delta Liq Summary: {self.delta_liq_loss:,.0f}')
    #     print('---------------------------------\n')


class CryptoVolSurfaceShocks:
    """
        Run Crypto Vol Surface Shocks
    """

    def __init__(self, products: list, parameters: VolSurfaceParameters):
        self.products = products
        self.parameters = parameters

    def run(self, data, group, scenario, days_to_trade: int = -1, valuation_date: ty.Optional[date] = None):
        results = []
        liq_b = BidAsk()
        liq_c = Concentration()
        liq_t = TermStructure()
        liq_s = Skew()
        for p in self.products:
            re, b, c, t, s = self.calc(data=data, days_to_trade=days_to_trade, valuation_date=valuation_date, **p)
            if re != {}:
                if scenario == 'Liquidity':
                    liq_b += b
                liq_c += c
                liq_t += t
                liq_s += s
                results.extend(re)
        if results != []:
            liq_c.aggregate()
            liq_t.aggregate()
            liq_s.aggregate()
            # Results
            results.extend(
                [
                    {
                        'product': 'Sum',
                        'measure': 'Concentration',
                        'value': liq_c.final_charge
                    }, {
                        'product': 'Sum',
                        'measure': 'TermStructure',
                        'value': liq_t.final_charge
                    }, {
                        'product': 'Sum',
                        'measure': 'Skew',
                        'value': liq_s.final_charge
                    }
                ]
            )
            if scenario == 'Liquidity':
                liq_b.aggregate()
                results.extend(
                    [
                        {
                            'product': 'Sum',
                            'measure': 'BidAsk',
                            'value': liq_b.final_charge
                        }, {
                            'product': 'Sum',
                            'measure': 'Sum',
                            'value': liq_b.final_charge + liq_c.final_charge + liq_t.final_charge + liq_s.final_charge
                        }
                    ]
                )
            else:
                results.extend(
                    [
                        {
                            'product': 'Sum',
                            'measure': 'Sum',
                            'value': liq_c.final_charge + liq_t.final_charge + liq_s.final_charge
                        }
                    ]
                )
            for i in results:
                i['group'] = group
                i['scenario'] = scenario
                i['type'] = 'ix'
        return results

    def calc(
        self, data, days_to_trade, product, class_, underlying, valuation_date: ty.Optional[date] = None
    ):
        df = data[data['Underlying'].isin(underlying)].copy()
        if df.empty:
            return {}, BidAsk(), Concentration(), TermStructure(), Skew()
        df['PositionVega'] = df['PositionVega']  # * self.clients.get_fx(fx)
        exp_3m_date = date.today() if valuation_date is None else valuation_date
        exp_3m = min(
            df['Expiry'].to_list(),
            key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d').date() - exp_3m_date).days - 90)
        )
        atm_ivol_3m = df.loc[df['Expiry'] == exp_3m, 'atm_ivol'].values[0]
        # Run models
        b = BidAsk(df, self.parameters.config_bid_ask(class_), valuation_date)
        c = Concentration(
            df, atm_ivol_3m, self.parameters.config_concentration(class_), days_to_trade, valuation_date
        )
        t = TermStructure(df, self.parameters.config_term_structure(class_), days_to_trade, valuation_date)
        s = Skew(df, self.parameters.config_skew(class_), days_to_trade, valuation_date)
        # Run models
        c.calc()
        t.calc()
        s.calc()
        # Results
        re = [
            {
                'product': product,
                'measure': 'Concentration',
                'value': c.concentration_charge
            }, {
                'product': product,
                'measure': 'TermStructure',
                'value': t.term_charge
            }, {
                'product': product,
                'measure': 'Skew',
                'value': s.skew_charge
            }
        ]
        if days_to_trade == -1:
            b.calc()
            re.extend(
                [
                    {
                        'product': product,
                        'measure': 'BidAsk',
                        'value': b.bid_ask_charge
                    }, {
                        'product': product,
                        'measure': 'Sum',
                        'value': b.bid_ask_charge + c.concentration_charge + t.term_charge + s.skew_charge
                    }
                ]
            )
        else:
            re.extend(
                [
                    {
                        'product': product,
                        'measure': 'Sum',
                        'value': c.concentration_charge + t.term_charge + s.skew_charge
                    }
                ]
            )
        return re, b, c, t, s
