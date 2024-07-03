from __future__ import annotations
import typing as ty
from datetime import datetime, date
import math
from types import MappingProxyType

import pandas as pd
import numpy as np


class Concentration:
    """
    Parallel shock
    """

    def __init__(
        self,
        data: ty.Optional[pd.DataFrame] = None,
        atm_ivol_3m=0,
        config: ty.Mapping = MappingProxyType({}),
        days_to_trade: int = -1,
        valuation_date: ty.Optional[date] = None
    ):
        self.charge_array = np.array([])
        self.data: pd.DataFrame = data if data is not None else pd.DataFrame(columns=['Expiry', 'PositionVega'])
        self.config = config
        self.atm_ivol_3m = atm_ivol_3m
        self.days_to_trade = days_to_trade
        self.valuation_date: date = date.today() if valuation_date is None else valuation_date

    def __add__(self, other):
        self.charge_array = np.append(self.charge_array, other.concentration_charge)
        return self

    def _calc_vega90days(self):
        df = self.data.copy()
        df['Today'] = self.valuation_date
        df['Today'] = pd.to_datetime(df['Today'])
        df['Expiry'] = pd.to_datetime(df['Expiry'])
        df['TTX'] = df['Expiry'] - df['Today']
        df['TTX'] = df['TTX'].astype('timedelta64[D]')
        df['vega_90days'] = (90 / df['TTX'])**0.4 * df['PositionVega']
        self.data = df
        self.vega90d = df['vega_90days'].sum()

    def _vol_shock(self, alpha_r, beta_r, alpha_a, beta_a, days_to_trade):
        if days_to_trade <= 1:
            w_r = alpha_r * days_to_trade
            w_a = alpha_a * days_to_trade
        elif (days_to_trade > 1) and (days_to_trade < 30):
            w_r = alpha_r * (days_to_trade**beta_r)
            w_a = alpha_a * (days_to_trade**beta_a)
        else:
            w_r = alpha_r * (30**beta_r)
            w_a = alpha_a * (30**beta_a)
        return w_r * self.atm_ivol_3m, w_a

    def calc(self):
        self._calc_vega90days()

        if self.days_to_trade == -1:
            days_to_trade = abs(self.vega90d / self.config['3m'])
        else:
            days_to_trade = self.days_to_trade
        alpha_r = self.config['alpha_r']
        beta_r = self.config['beta_r']
        alpha_a = self.config['alpha_a']
        beta_a = self.config['beta_a']
        min_vol = self.config['min_vol']
        max_vol = self.config['max_vol']
        w_r, w_a = self._vol_shock(alpha_r, beta_r, alpha_a, beta_a, days_to_trade)
        w = min(w_r, w_a)
        if w + self.atm_ivol_3m < min_vol:
            w = self.atm_ivol_3m - min_vol
        elif w + self.atm_ivol_3m > max_vol:
            w = max_vol - self.atm_ivol_3m
        self.concentration_charge = -abs(self.vega90d * (w / 2))
        self.charge_array = np.append(self.charge_array, self.concentration_charge)

    def aggregrate(self):
        self.aggregate()

    def aggregate(self):
        self.final_charge = -math.sqrt(sum(self.charge_array**2))


class Skew:

    def __init__(
        self,
        data: ty.Optional[pd.DataFrame] = None,
        config: ty.Mapping = MappingProxyType({}),
        days_to_trade: int = -1,
        valuation_date: ty.Optional[date] = None
    ):
        self.charge_array = np.array([])
        self.data: pd.DataFrame = data if data is not None else pd.DataFrame(columns=['Expiry', 'PositionVega'])
        self.config = config
        self.days_to_trade = days_to_trade
        self.valuation_date: date = date.today() if valuation_date is None else valuation_date

    def __add__(self, other):
        self.charge_array = np.append(self.charge_array, other.skew_charge)
        return self

    def _calc_DDT(self, df):
        df = df[['Expiry', 'PositionVega']]
        df = df[['Expiry', 'PositionVega']].groupby(by='Expiry', as_index=False).sum()
        df['Today'] = self.valuation_date
        df['Today'] = pd.to_datetime(df['Today'])
        df['Expiry'] = pd.to_datetime(df['Expiry'])
        df['TTX'] = df['Expiry'] - df['Today']
        df['TTX'] = df['TTX'].astype('timedelta64[D]')
        df['WVPEB'] = 0
        if self.days_to_trade == -1:
            for i, row in df.iterrows():
                TTX = row['TTX']
                if TTX <= 30:
                    df.loc[i, 'WVPEB'] = self.config['1m']
                elif (TTX > 30) and (TTX < 90):
                    df.loc[i, 'WVPEB'] = (TTX - 30) / 60 * self.config['3m'] + (90 - TTX) / 60 * self.config['1m']
                elif TTX == 90:
                    df.loc[i, 'WVPEB'] = self.config['3m']
                elif (TTX > 90) and (TTX < 180):
                    df.loc[i, 'WVPEB'] = (TTX - 90) / 90 * self.config['6m'] + (180 - TTX) / 90 * self.config['3m']
                elif TTX == 180:
                    df.loc[i, 'WVPEB'] = self.config['6m']
                elif (TTX > 180) and (TTX < 360):
                    df.loc[i, 'WVPEB'] = (TTX - 180) / 180 * self.config['1y'] + (360 - TTX) / 180 * self.config['6m']
                elif TTX == 360:
                    df.loc[i, 'WVPEB'] = self.config['1y']
                elif (TTX > 360) and (TTX < 720):
                    df.loc[i, 'WVPEB'] = (TTX - 360) / 360 * self.config['2y'] + (720 - TTX) / 360 * self.config['1y']
                elif TTX >= 720:
                    df.loc[i, 'WVPEB'] = self.config['2y']
            df['DTT'] = df['PositionVega'] / df['WVPEB']
            df['DTT'] = df['DTT'].abs()
        else:
            df['DTT'] = self.days_to_trade
        self.check = df
        return df[['Expiry', 'DTT', 'TTX', 'WVPEB']]

    def _calc_vol_shock(self):
        df = self.data.copy()
        df['Expiry'] = pd.to_datetime(df['Expiry'])
        tem = self._calc_DDT(self.data)
        df = pd.merge(df, tem[['Expiry', 'WVPEB', 'DTT', 'TTX']].drop_duplicates(), on='Expiry', how='left')
        df['Omega'] = 0
        df['SkewShockinVolPoints'] = 0
        for i, row in df.iterrows():
            DTT = row['DTT']
            if DTT <= 1:
                omega = self.config['alpha'] * DTT
            elif (DTT > 1) and (DTT < 30):
                omega = self.config['alpha'] * (DTT**self.config['beta'])
            else:
                omega = self.config['alpha'] * (30**self.config['beta'])
            df.loc[i, 'Omega'] = omega
        # df['SkewShockinVolPoints'] = df['atm_ivol']*df['Omega']/2*abs(df['Delta']-0.5)
        df['SkewShockinVolPoints'] = df['atm_ivol'] * df['Omega'] / 2 * (
            0.5 - df['Delta'].abs()
        )   # CREST doc is using abs(df['Delta']-0.5) but GS report seems not. When we drop abs(), results are colser.
        self.data = df

    def calc(self):
        self._calc_vol_shock()
        self.data['SkewCharge'] = self.data['SkewShockinVolPoints'] * self.data['PositionVega']
        self.skew_charge = self.data['SkewCharge'].sum()
        self.skew_charge = -abs(self.skew_charge)
        self.charge_array = np.append(self.charge_array, self.skew_charge)

    def aggregrate(self):
        self.aggregate()

    def aggregate(self):
        self.final_charge = -math.sqrt(sum(self.charge_array**2))


class TermStructure:

    def __init__(
        self,
        data: ty.Optional[pd.DataFrame] = None,
        config: ty.Mapping = MappingProxyType({}),
        days_to_trade: int = -1,
        valuation_date: ty.Optional[date] = None
    ):
        self.charge_array = np.array([])
        self.data: pd.DataFrame = data if data is not None else pd.DataFrame(columns=['Expiry', 'PositionVega'])
        self.config = config
        self.days_to_trade = days_to_trade
        self.valuation_date: date = date.today() if valuation_date is None else valuation_date

    def __add__(self, other):
        self.charge_array = np.append(self.charge_array, other.term_charge)
        return self

    def _clac_omega(self, alpha, beta, days_to_trade):
        if days_to_trade <= 1:
            w = alpha * days_to_trade
        elif (days_to_trade > 1) and (days_to_trade < 30):
            w = alpha * (days_to_trade**beta)
        else:
            w = alpha * (30**beta)
        return w

    def _calc_DDT(self, df):
        df = df[['Expiry', 'PositionVega']]
        df = df[['Expiry', 'PositionVega']].groupby(by='Expiry', as_index=False).sum()
        df['Today'] = self.valuation_date
        df['Today'] = pd.to_datetime(df['Today'])
        df['Expiry'] = pd.to_datetime(df['Expiry'])
        df['TTX'] = df['Expiry'] - df['Today']
        df['TTX'] = df['TTX'].astype('timedelta64[D]')
        df['WVPEB'] = 0
        if self.days_to_trade == -1:
            for i, row in df.iterrows():
                TTX = row['TTX']
                if TTX <= 30:
                    df.loc[i, 'WVPEB'] = self.config['1m']
                elif (TTX > 30) and (TTX < 90):
                    df.loc[i, 'WVPEB'] = (TTX - 30) / 60 * self.config['3m'] + (90 - TTX) / 60 * self.config['1m']
                elif TTX == 90:
                    df.loc[i, 'WVPEB'] = self.config['3m']
                elif (TTX > 90) and (TTX < 180):
                    df.loc[i, 'WVPEB'] = (TTX - 90) / 90 * self.config['6m'] + (180 - TTX) / 90 * self.config['3m']
                elif TTX == 180:
                    df.loc[i, 'WVPEB'] = self.config['6m']
                elif (TTX > 180) and (TTX < 360):
                    df.loc[i, 'WVPEB'] = (TTX - 180) / 180 * self.config['1y'] + (360 - TTX) / 180 * self.config['6m']
                elif TTX == 360:
                    df.loc[i, 'WVPEB'] = self.config['1y']
                elif (TTX > 360) and (TTX < 720):
                    df.loc[i, 'WVPEB'] = (TTX - 360) / 360 * self.config['2y'] + (720 - TTX) / 360 * self.config['1y']
                elif TTX >= 720:
                    df.loc[i, 'WVPEB'] = self.config['2y']
            df['DTT'] = df['PositionVega'] / df['WVPEB']
            df['DTT'] = df['DTT'].abs()
        else:
            df['DTT'] = self.days_to_trade
        return df[['Expiry', 'PositionVega', 'DTT', 'TTX', 'WVPEB']]

    def _calc_shocks(self, data_input):
        df = self._calc_DDT(data_input)
        df['shock_a'] = 0
        df['shock_r'] = 0
        for i, row in df.iterrows():
            TTX = row['TTX']
            DTT = row['DTT']

            shock_a1m = self._clac_omega(self.config['alpha_a1m'], self.config['beta_a1m'], DTT)
            shock_r1m = self._clac_omega(self.config['alpha_r1m'], self.config['beta_r1m'], DTT)
            # Apply a term structure shock at each expiry by rotating the curve up and down around the 3 month expiry
            shock_a3m = 0
            shock_r3m = 0
            shock_a6m = self._clac_omega(self.config['alpha_a6m'], self.config['beta_a6m'], DTT)
            shock_r6m = self._clac_omega(self.config['alpha_r6m'], self.config['beta_r6m'], DTT)
            shock_a1y = self._clac_omega(self.config['alpha_a1y'], self.config['beta_a1y'], DTT)
            shock_r1y = self._clac_omega(self.config['alpha_r1y'], self.config['beta_r1y'], DTT)
            shock_a2y = self._clac_omega(self.config['alpha_a2y'], self.config['beta_a2y'], DTT)
            shock_r2y = self._clac_omega(self.config['alpha_r2y'], self.config['beta_r2y'], DTT)
            if TTX <= 30:
                df.loc[i, 'shock_a'] = shock_a1m
                df.loc[i, 'shock_r'] = shock_r1m
            elif (TTX > 30) and (TTX < 90):
                df.loc[i, 'shock_a'] = (TTX - 30) / 60 * shock_a3m + (90 - TTX) / 60 * shock_a1m
                df.loc[i, 'shock_r'] = (TTX - 30) / 60 * shock_r3m + (90 - TTX) / 60 * shock_r1m
            elif TTX == 90:
                df.loc[i, 'shock_a'] = shock_a3m
                df.loc[i, 'shock_r'] = shock_r3m
            elif (TTX > 90) and (TTX < 180):
                df.loc[i, 'shock_a'] = (TTX - 90) / 90 * shock_a6m + (180 - TTX) / 90 * shock_a3m
                df.loc[i, 'shock_r'] = (TTX - 90) / 90 * shock_r6m + (180 - TTX) / 90 * shock_r3m
            elif TTX == 180:
                df.loc[i, 'shock_a'] = shock_a6m
                df.loc[i, 'shock_r'] = shock_r6m
            elif (TTX > 180) and (TTX < 360):
                df.loc[i, 'shock_a'] = (TTX - 180) / 180 * shock_a1y + (360 - TTX) / 180 * shock_a6m
                df.loc[i, 'shock_r'] = (TTX - 180) / 180 * shock_r1y + (360 - TTX) / 180 * shock_r6m
            elif TTX == 360:
                df.loc[i, 'shock_a'] = shock_a1y
                df.loc[i, 'shock_r'] = shock_r1y
            elif (TTX > 360) and (TTX < 720):
                df.loc[i, 'shock_a'] = (TTX - 360) / 360 * shock_a2y + (720 - TTX) / 360 * shock_a1y
                df.loc[i, 'shock_r'] = (TTX - 360) / 360 * shock_r2y + (720 - TTX) / 360 * shock_r1y
            elif TTX >= 720:
                df.loc[i, 'shock_a'] = shock_a2y
                df.loc[i, 'shock_r'] = shock_r2y
        self.check = df
        return df[['Expiry', 'PositionVega', 'DTT', 'TTX', 'WVPEB', 'shock_a', 'shock_r']]

    def calc(self):
        df = self.data.copy()
        df['Expiry'] = pd.to_datetime(df['Expiry'])
        tem = self._calc_shocks(df)
        df = pd.merge(
            df, tem[['Expiry', 'DTT', 'TTX', 'WVPEB', 'shock_a', 'shock_r']].drop_duplicates(), on='Expiry', how='left'
        )
        df['shock_r*vol'] = df['shock_r'] * df['atm_ivol'
                                              ]   # by rotating the curve up and down around the 3 month expiry
        for i, row in df.iterrows():
            shock = min(row['shock_r*vol'], row['shock_a'])
            # ivol=row['ivol']/100
            atm_ivol = row['atm_ivol']
            tem = min(atm_ivol - 5, abs(shock))
            if tem * shock >= 0:
                df.loc[i, 'adjusted_shock'] = tem
            else:
                df.loc[i, 'adjusted_shock'] = -tem
            df.loc[i, 'shock'] = shock
        # self.ckeck2=df
        df['TermCharge'] = df['adjusted_shock'] / 2 * df['PositionVega']
        # df['TermCharge'] = df['shock_r*vol']/2*df['PositionVega']

        self.data = df
        self.term_charge = -abs(self.data['TermCharge'].sum())
        self.charge_array = np.append(self.charge_array, self.term_charge)

    def aggregrate(self):
        self.aggregate()

    def aggregate(self):
        self.final_charge = -math.sqrt(sum(self.charge_array**2))


class IRVegaLiq:
    '''
        IR Risk Vega Liquidation: charges associated with liquidating the vega component of a portfolio as accumulated from interest rate options positions.
        config example {'Product': 'BUND', 'Edge': 30, 'VegaATM90': 38, 'Fees': 0.2, 'Tcap': 3}
        Tcap is TermCap

    '''

    def __init__(
        self,
        data: ty.Optional[pd.DataFrame] = None,
        config: ty.Mapping = MappingProxyType({}),
        valuation_date: ty.Optional[date] = None
    ):
        self.config = config
        df: pd.DataFrame = data if data is not None else pd.DataFrame(
            columns=['Expiry', 'Strike', 'Quantity', 'PositionVega']
        )
        self.data = df[['Expiry', 'Strike', 'Quantity', 'PositionVega']].fillna(0)   # 'InstrumentId'
        self.valuation_date: date = date.today() if valuation_date is None else valuation_date
        self._calc_vega90days()

    def _calc_vega90days(self):
        df = self.data.copy()
        df['Today'] = self.valuation_date
        df['Today'] = pd.to_datetime(df['Today'])
        df['Expiry'] = pd.to_datetime(df['Expiry'])
        df['TTX'] = df['Expiry'] - df['Today']
        df['TTX'] = df['TTX'].astype('timedelta64[D]')
        if any(df['TTX'] <= 0):
            raise Exception('Found expired contracts')
        # Vega90d: Sqrt of 90/t with a floor of 1, and cap of 2.
        df['Vega90d'] = df.apply(lambda row: max(1, min(2, (90 / row['TTX'])**0.5)) * row['PositionVega'], axis=1)
        self.data = df

    def calc_hedging_charge(self):
        df = self.data[['TTX', 'Strike', 'Vega90d']]
        df = df.groupby(by=['TTX', 'Strike'], as_index=False).sum()
        df['VegaCharge'] = df['Vega90d'] / self.config['VegaATM90'] * self.config['Edge']
        self.vega_charge = df['VegaCharge'].sum()
        self.Hccy = -self.vega_charge / 3
        self.Hcontract = -self.vega_charge * 2 / 3
        self.hedge_charge = self.Hccy + self.Hcontract

    def calc_fees(self):
        df = self.data[['Expiry', 'Strike', 'Quantity']]
        df = df.groupby(by=['Expiry', 'Strike'], as_index=False).sum()
        self.fees = -(self.config['Fees'] * df['Quantity']).abs().sum()

    def calc_exit_charge(self):
        df = self.data[['PositionVega', 'TTX', 'Quantity', 'Strike']]
        df = df.groupby(by=['TTX', 'Strike'], as_index=False).sum()
        df['TermAdj'] = df['TTX'].apply(lambda t: min(self.config['Tcap'], self.config['Tcap']**((t - 90) / 640)))
        df['SizeAdj'] = df['Quantity'].apply(lambda q: max(1, (abs(q) / 10000)**0.5))
        df['ExitCharge'] = 0.1 * df['Quantity'] * df['TermAdj'] * df['SizeAdj'] * self.config['Edge']
        df['ExitCharge'] = -df['ExitCharge'].abs()
        self.exit_gross_chargeA = -sum(df['ExitCharge']**2)**0.5
        self.exit_gross_chargeB = -sum(df['PositionVega'].abs()) * 0.125
        self.exit_charge = min(self.exit_gross_chargeA, self.exit_gross_chargeB)

    def calc(self):
        self.calc_hedging_charge()
        self.calc_exit_charge()
        self.calc_fees()
        return {
            'Product': self.config['Product'],
            'Hccy': self.Hccy,
            'Hcontract': self.Hcontract,
            'HedgeCharge': self.hedge_charge,
            'ExitChargeA': self.exit_gross_chargeA,
            'ExitChargeB': self.exit_gross_chargeB,
            'ExitCharge': self.exit_charge,
            'Fees': self.fees,
            'Sum': self.hedge_charge + self.exit_charge + self.fees,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


class BidAsk:

    def __init__(
        self,
        data: ty.Optional[pd.DataFrame] = None,
        config: ty.Mapping = MappingProxyType({'Factor': 0.0}),
        valuation_date: ty.Optional[date] = None
    ):
        self.net_charge = 0
        self.gross_charge = 0
        self.bid_ask_charge = 0
        self.vega90d_array = np.array([])
        self.data: pd.DataFrame = data if data is not None else pd.DataFrame(columns=['Expiry', 'PositionVega'])
        self.bid_ask_factor = config['Factor']
        self.valuation_date: date = date.today() if valuation_date is None else valuation_date
        self.vega90d: float = 0

    def __add__(self, other):
        self.net_charge = self.net_charge + other.bid_ask_charge
        self.gross_charge = self.gross_charge + abs(other.bid_ask_charge)
        self.vega90d_array = np.append(self.vega90d_array, other.vega90d)
        return self

    def _calc_vega90days(self):
        df_orig = self.data
        s_today = pd.to_datetime(pd.Series(self.valuation_date, index=df_orig.index))
        s_expiry = pd.to_datetime(df_orig['Expiry'])
        s_ttx = (s_expiry - s_today).astype('timedelta64[D]')
        s_vega_90days = (90 / s_ttx)**0.4 * df_orig['PositionVega']
        self.vega90d = s_vega_90days.sum()

    def calc(self):
        self._calc_vega90days()
        self.bid_ask_charge = -abs(self.vega90d * self.bid_ask_factor)
        self.vega90d_array = np.append(self.vega90d_array, self.vega90d)
        self.net_charge = self.bid_ask_charge
        self.gross_charge = abs(self.bid_ask_charge)

    def aggregrate(self):
        self.aggregate()

    def aggregate(self):
        g1 = sum(abs(self.vega90d_array))
        g2 = sum(self.vega90d_array)
        g3 = (g1 / g2)**2 + 6.4
        g = 2 / math.log(min(g3, 20))
        self.final_charge = -(g * abs(self.gross_charge) + (1 - g) * abs(self.net_charge))
