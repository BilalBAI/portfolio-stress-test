from __future__ import annotations
from datetime import date, timedelta


class EquityParameters:

    def __init__(
        self, equity_shocks: list, macro_scenarios: list, sector_scenarios: list, net_pairs: dict, rv_scenarios: dict,
        dict_country_core: dict, dict_fx_scenarios: dict, dict_country_grouping: dict, concentration_scenarios: dict
    ):
        self.equity_shocks = equity_shocks
        self.macro_scenarios = macro_scenarios
        self.net_pairs = net_pairs
        self.rv_scenarios = rv_scenarios
        self.sector_scenarios = sector_scenarios
        self.dict_country_core = dict_country_core
        self.dict_fx_scenarios = dict_fx_scenarios
        self.dict_country_grouping = dict_country_grouping
        self.concentration_scenarios = concentration_scenarios


class OptParameters:
    '''
        Manage config files for options models
    '''

    def __init__(self, parameters: dict):
        self.parameters = parameters

    def _search(self, scenario, region, class_) -> dict:
        return [item for item in self.parameters[f'{region}-{scenario}'] if item["Class"] == class_][0]

    def config_ir(self, product):
        return [item for item in self.parameters['IR'] if item['Product'] == product][0]

    def config_bid_ask(self, region, class_):
        return self._search('BidAsk', region, class_)

    def config_concentration(self, region, class_):
        c1 = self._search('Concentration-Shocks', region, class_)
        c2 = self._search('Vega-Threshold', region, class_)
        return {**c1, **c2}

    def config_term_structure(self, region, class_):
        c1 = self._search('Term-Structure-Shocks', region, class_)
        c2 = self._search('Vega-Threshold', region, class_)
        return {**c1, **c2}

    def config_skew(self, region, class_):
        c1 = self._search('Skew-Shocks', region, class_)
        c2 = self._search('Vega-Threshold', region, class_)
        return {**c1, **c2}


def net(df, all_flds, group_by):
    df_net = df[all_flds].copy()
    df_net = df_net.groupby(group_by, as_index=False).sum()
    return df_net.reset_index(drop=True)


def get_last_businessdate():
    if date.today().weekday() == 0:
        last_businessdate = date.today() - timedelta(days=3)
    else:
        last_businessdate = date.today() - timedelta(days=1)
    return last_businessdate


def workday(start_date: date, days: int) -> date:
    '''
        Similar to Excel WORKDAY().
    '''
    # start_date.weekday()==0 for Monday (5,6) for weekends
    weekends = (5, 6)
    mod_days = abs(days) % 5
    if days < 0:
        while start_date.weekday() in weekends:
            start_date += timedelta(days=1)
        if start_date.weekday() - mod_days < 0:
            end_date = start_date - timedelta(days=mod_days + 2)
        else:
            end_date = start_date - timedelta(days=mod_days)
        end_date = end_date - timedelta(days=7 * (abs(days) - mod_days) / 5)

    elif days > 0:
        while start_date.weekday() in weekends:
            start_date -= timedelta(days=1)
        if start_date.weekday() + mod_days > 4:
            end_date = start_date + timedelta(days=mod_days + 2)
        else:
            end_date = start_date + timedelta(days=mod_days)
        end_date = end_date + timedelta(days=7 * (abs(days) - mod_days) / 5)

    else:
        end_date = start_date

    return end_date
