from __future__ import annotations

import pandas as pd

from ..utils import EquityParameters


class FXRisk:

    def __init__(self, product_level_data, symbol_level_data, config: EquityParameters):
        self.config = config
        self.product_level_data = product_level_data
        self.symbol_level_data = symbol_level_data
        self.product_level_data = self.product_level_data[~self.product_level_data['ticker'].
                                                          isin(self.config.dict_FX_scenarios['exclude'])]
        self.symbol_level_data = self.symbol_level_data[~self.symbol_level_data['ticker'].
                                                        isin(self.config.dict_FX_scenarios['exclude'])]

    def baseCCY_risk(self, df_input='default') -> float:
        if df_input == 'default':
            df = self.symbol_level_data.copy()
        else:
            df = df_input.copy()
        results_baseCCY = min(
            df.loc[df['CRNCY'] != self.config.dict_FX_scenarios['Base'], 'Market Value'].sum() *
            self.config.dict_FX_scenarios['Base Shock'][0],
            df.loc[df['CRNCY'] != self.config.dict_FX_scenarios['Base'], 'Market Value'].sum() *
            self.config.dict_FX_scenarios['Base Shock'][1]
        )
        return results_baseCCY

    def portCCY_risk(self, df_input='default') -> pd.DataFrame:
        if df_input == 'default':
            df = self.symbol_level_data.copy()
        else:
            df = df_input.copy()
        results_portfolioCCY = []
        df = df[df['CRNCY'] != self.config.dict_FX_scenarios['Base']]
        CCY = df['CRNCY'].drop_duplicates().to_list()
        if CCY == []:
            return pd.DataFrame()
        for i in CCY:
            if i in self.config.dict_FX_scenarios.keys():
                up = {}
                up['CRNCY'] = i
                up['shock'] = self.config.dict_FX_scenarios[i][0]
                up['risk'] = df.loc[df['CRNCY'] == i, 'Market Value'].sum() * self.config.dict_FX_scenarios[i][0]
                down = {}
                down['CRNCY'] = i
                down['shock'] = self.config.dict_FX_scenarios[i][1]
                down['risk'] = df.loc[df['CRNCY'] == i, 'Market Value'].sum() * self.config.dict_FX_scenarios[i][1]
                results_portfolioCCY.append(dict(up))
                results_portfolioCCY.append(dict(down))
            else:
                up = {}
                up['CRNCY'] = i
                up['shock'] = 0.25
                up['risk'] = df.loc[df['CRNCY'] == i, 'Market Value'].sum() * 0.25
                down = {}
                down['CRNCY'] = i
                down['shock'] = -0.25
                down['risk'] = df.loc[df['CRNCY'] == i, 'Market Value'].sum() * -0.25
                results_portfolioCCY.append(dict(up))
                results_portfolioCCY.append(dict(down))
        return pd.DataFrame(results_portfolioCCY)

    def peggedCCY_risk(self, df_input='default') -> pd.DataFrame:
        if df_input == 'default':
            df = self.symbol_level_data.copy()
        else:
            df = df_input.copy()
        results_peggedCCY = []
        df = df[df['CRNCY'] != self.config.dict_FX_scenarios['Base']]
        CCY = df['CRNCY'].drop_duplicates().to_list()
        if CCY == []:
            return pd.DataFrame()
        for i in CCY:
            if i in self.config.dict_FX_scenarios['Stable Pegged CCY']:
                up = {}
                up['CRNCY'] = i
                up['shock'] = self.config.dict_FX_scenarios['Stable Pegged CCY Shock'][0]
                up['risk'] = df.loc[df['CRNCY'] == i,
                                    'Market Value'].sum() * self.config.dict_FX_scenarios['Stable Pegged CCY Shock'][0]
                down = {}
                down['CRNCY'] = i
                down['shock'] = self.config.dict_FX_scenarios['Stable Pegged CCY Shock'][1]
                down['risk'] = df.loc[df['CRNCY'] == i, 'Market Value'].sum(
                ) * self.config.dict_FX_scenarios['Stable Pegged CCY Shock'][1]
                results_peggedCCY.append(dict(up))
                results_peggedCCY.append(dict(down))
            if i in self.config.dict_FX_scenarios['High Risk Pegged CCY']:
                up = {}
                up['CRNCY'] = i
                up['shock'] = self.config.dict_FX_scenarios['High Risk Pegged CCY Shock'][0]
                up['risk'] = df.loc[df['CRNCY'] == i, 'Market Value'].sum(
                ) * self.config.dict_FX_scenarios['High Risk Pegged CCY Shock'][0]
                down = {}
                down['CRNCY'] = i
                down['shock'] = self.config.dict_FX_scenarios['High Risk Pegged CCY Shock'][1]
                down['risk'] = df.loc[df['CRNCY'] == i, 'Market Value'].sum(
                ) * self.config.dict_FX_scenarios['High Risk Pegged CCY Shock'][1]
                results_peggedCCY.append(dict(up))
                results_peggedCCY.append(dict(down))
        return pd.DataFrame(results_peggedCCY)

    def risk_summary(self):
        if self.product_level_data.empty or self.symbol_level_data.empty:
            self.FX_risk_summary = 0
            return 'Warning: Empty DataFrame input'

        self.baseCCY_summary = self.baseCCY_risk()
        df_portCCY = self.portCCY_risk()
        df_peggedCCY = self.peggedCCY_risk()

        if not df_portCCY.empty:
            portfolioCCY_up = df_portCCY[df_portCCY['shock'] > 0].sort_values(by='risk').reset_index(drop=True)
            portfolioCCY_down = df_portCCY[df_portCCY['shock'] < 0].sort_values(by='risk').reset_index(drop=True)
            if (len(portfolioCCY_up.index) > 1) and (len(portfolioCCY_down.index) > 1):
                portfolioCCY_sum = min(
                    portfolioCCY_up.loc[0, 'risk'] + portfolioCCY_up.loc[1, 'risk'],
                    portfolioCCY_down.loc[0, 'risk'] + portfolioCCY_down.loc[1, 'risk']
                )
            else:
                portfolioCCY_sum = min(portfolioCCY_up.loc[0, 'risk'], portfolioCCY_down.loc[0, 'risk'])
        else:
            portfolioCCY_sum = 0

        if not df_peggedCCY.empty:
            peggedCCY = df_peggedCCY.sort_values(by='risk').reset_index(drop=True)
            peggedCCY_sum = peggedCCY.loc[0, 'risk']
        else:
            peggedCCY_sum = 0

        self.portCCY_summary = portfolioCCY_sum
        self.peggedCCY_summary = peggedCCY_sum
        self.FX_risk_summary = min(self.baseCCY_summary, self.portCCY_summary, self.peggedCCY_summary)

        return 'Success'
