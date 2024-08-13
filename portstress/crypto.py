from __future__ import annotations
import typing as ty

from datetime import datetime, date

import pandas as pd
import numpy as np

from .core.utils import CryptoParameters, VolSurfaceParameters
from .core.black_scholes import calc_delta, calc_vega, BS_PARAMETERS
from .core.spot_vol_stress import StressTest
from .core.vol_surface_stress import Parallel, TermStructure, Skew, BidAsk
from .clients.deribit import DeribitClient


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
    #     Attributes Created: rv_summary, .macro_summary, .sector_summary, .parallel_summary, .equity_liq_summary,
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
    #         self.rv_loss, self.macro_loss, self.sector_loss, self.parallel_loss
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
    #     print(f'Parallel Summary: {self.parallel_loss:,.0f}\n')
    #     print('Liquidity Summary:')
    #     print(f'Delta Liq Summary: {self.delta_liq_loss:,.0f}')
    #     print('---------------------------------\n')


class CryptoVolSurfaceShocks:
    """
        Run Crypto Vol Surface Shocks
    """

    def __init__(self, parameters: VolSurfaceParameters):
        self.parameters = parameters
        self.products = parameters.parameters['products']

    def run(self, data, group: str = None, liquidity: bool = False, days_to_trade: int = None, valuation_date: ty.Optional[date] = None):
        '''
        if days_to_trade == None, days_to_trade will be auto calculated using 90vega and daily vega threshold
        otherwise, use the sepcified days_to_trade.

        '''
        # Calc delta and position_delta
        # Position Delta($) = Delta × Number of Contracts × Shares per Contract × Price of the Underlying Asset
        data['delta'] = np.vectorize(
            lambda spot, **BS_PARAMETERS: calc_delta(spot=spot, **BS_PARAMETERS) if spot > 0 else 0
        )(**{col: data[col] for col in BS_PARAMETERS})
        data['position_delta'] = data['delta'] * data['quantity'] * data[
            'multiplier'] * data['spot']

        # Calc vega and position_vega
        # Position Vega($) = Vega × ΔIV × Number of Contracts × Shares per Contract
        data['vega'] = np.vectorize(
            lambda spot, **BS_PARAMETERS: calc_vega(spot=spot, **BS_PARAMETERS) if spot > 0 else 0
        )(**{col: data[col] for col in BS_PARAMETERS})
        data['position_vega'] = data['vega'] * data['quantity'] * data[
            'multiplier']

        # Run Stress tests
        results = []
        liq_b = BidAsk()
        liq_c = Parallel()
        liq_t = TermStructure()
        liq_s = Skew()
        for p in self.products:
            re, b, c, t, s = self.calc(data=data, days_to_trade=days_to_trade, valuation_date=valuation_date, **p)
            if re != {}:
                if liquidity:
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
                        'measure': 'Parallel',
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
            if liquidity:
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
                i['liquidity'] = liquidity
                i['type'] = 'ix'
        return results

    def calc(
        self, data, days_to_trade, product, class_, underlying, valuation_date: ty.Optional[date] = None
    ):
        df = data[data['underlying'].isin(underlying)].copy()
        if df.empty:
            return {}, BidAsk(), Parallel(), TermStructure(), Skew()

        # Get atm_ivol_3m if the data include all experies.
        exp_3m_date = date.today() if valuation_date is None else valuation_date
        exp_3m = min(
            df['expiry'].to_list(),
            key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d').date() - exp_3m_date).days - 90)
        )
        atm_ivol_3m = df.loc[df['expiry'] == exp_3m, 'atm_ivol'].values[0]
        # atm_ivol_3m = df['atm_ivol_3m'].values[0]
        # Run models
        b = BidAsk(df, self.parameters.config_bid_ask(class_), valuation_date)
        c = Parallel(
            df, atm_ivol_3m, self.parameters.config_parallel(class_), days_to_trade, valuation_date
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
                'measure': 'Parallel',
                'value': c.parallel_charge
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
        if days_to_trade == None:
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
                        'value': b.bid_ask_charge + c.parallel_charge + t.term_charge + s.skew_charge
                    }
                ]
            )
        else:
            re.extend(
                [
                    {
                        'product': product,
                        'measure': 'Sum',
                        'value': c.parallel_charge + t.term_charge + s.skew_charge
                    }
                ]
            )
        return re, b, c, t, s


def fetch_market_data(df_positions: pd.DataFrame, valuation_day: str):
    """
        df_positions[['instrument','quantity']]
    """
    df = df_positions.copy()
    # Splitting the 'instrument' column by "-"
    df[['underlying', 'expiry', 'strike', 'put_call']] = df['instrument'].str.split('-', expand=True)
    df['strike'] = df['strike'].astype(float)
    df['put_call'] = df['put_call'].map({'P': 'put', 'C': 'call'})
    df['underlying-expiry'] = df['underlying'] + "-" + df['expiry']
    df['expiry'] = df['expiry'].replace('PERPETUAL', None)  # e.g. BTC-PERPETUAL
    df['expiry'] = pd.to_datetime(df['expiry'], format='%d%b%y')
    # Calculate time to expiry
    valuation_day = pd.to_datetime(valuation_day)
    df['time_to_expiry'] = (df['expiry'] - valuation_day).dt.days / 365
    # convert expiry to a string
    df['expiry'] = df['expiry'].dt.strftime('%Y-%m-%d')

    # get market data from deribit
    db_client = DeribitClient()
    deribit_res = db_client.get_order_book(instruments=df['instrument'].to_list())
    df = pd.merge(df, deribit_res, how='left', left_on='instrument', right_on='instrument_name')

    # calculate input for BSM
    df['cost_of_carry_rate'] = 'default'
    df['multiplier'] = 1
    df['rate'] = 0.03
    df['vol'] = df['mark_iv'] / 100
    df['spot'] = df['index_price'].values[0]
    # df['position_vega'] = df['vega'] * df['quantity']

    # calculate atm_ivol
    df_atm = db_client.get_atm_ivol()
    df = pd.merge(df, df_atm, on='underlying-expiry', how='left')
    return df
