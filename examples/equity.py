import json
from pandas import DataFrame
from portrisk.utils import EquityParameters
from portrisk.clients import PortRiskClient
from portrisk.core.equity import EquityRisk

PARAMS_SUB_DIR = 'data/parameters/'
POS_DIR = 'data/positions.json'
FX_DIR = 'data/fx.json'


def get_file(filepath, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding) as filein:
        return filein.read()


def get_json(filepath, encoding='utf-8'):
    return json.loads(get_file(filepath, encoding=encoding))


def get_risk_params(sub_dir):
    return {
        'equity_shocks': get_json(sub_dir + 'equity_shocks.json'),
        'macro_scenarios': get_json(sub_dir + 'macro_scenarios.json'),
        'net_pairs': get_json(sub_dir + 'net_pairs.json'),
        'rv_scenarios': get_json(sub_dir + 'rv_scenarios.json'),
        'sector_scenarios': get_json(sub_dir + 'sector_scenarios.json'),
        'dict_country_core': get_json(sub_dir + 'country_core.json'),
        'dict_fx_scenarios': get_json(sub_dir + 'fx_scenarios.json'),
        'dict_country_grouping': get_json(sub_dir + 'country_grouping.json'),
        'concentration_scenarios': get_json(sub_dir + 'concentration_scenarios.json')
    }


def run():

    df_fx = DataFrame(get_json(FX_DIR))
    df_pos = DataFrame(get_json(POS_DIR))

    risk_client = PortRiskClient(df_fx)
    parameters = EquityParameters(**get_risk_params(sub_dir=PARAMS_SUB_DIR))
    risk_data = risk_client.get_all(df_pos)

    books = ['test']
    eq_risk = EquityRisk(risk_data.get_product_level_data(books), risk_data.get_symbol_level_data(books), parameters)
    eq_risk.run()


if __name__ == "__main__":
    run()
