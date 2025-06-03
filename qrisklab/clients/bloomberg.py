from os import getenv
import typing as ty
from datetime import datetime
import pandas as pd
import numpy as np
from pandas import DataFrame
from sqlalchemy import create_engine, text
from blp.blp import BlpQuery, BlpParser

STOCK_TYPE = ['Common Stock', 'REIT', 'Depositary Receipt', 'Preference']
FUTURE_TYPE = ['Future', 'Index']
OPTION_TYPE = ['Option']
ETF_TYPE = ['Mutual Fund']
STATIC_FIELDS = [
    'SECURITY_TYP2', 'CRNCY', 'ISSUER_BULK', 'GICS_INDUSTRY_GROUP_NAME', 'COUNTRY_ISO', 'CNTRY_ISSUE_ISO',
    'MSCI_COUNTRY_CODE', 'PRICE_MULTIPLIER', 'OPT_PUT_CALL', 'UNDERLYING_SECURITY_DES', 'UNDL_SPOT_TICKER',
    'ETF_UNDL_INDEX_TICKER'
]


class BloombergClient(BlpQuery):
    """
        Use this class to get frequently used and static Bloomberg data to avoid hitting the data limit
        WARNING:  Store Bloomberg data on a shared location may violate the licence.  Use local dir
        e.g. BBG_CACHE_DB_DIR = 'C:/risk/db/'
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8194,
        timeout: int = 9999,
        parser=BlpParser(raise_security_errors=False),
        cache_dir: ty.Optional[str] = None
    ):
        super().__init__(host, port, timeout, parser)
        self._table_name = 'bbg_ref'
        if cache_dir is None:
            cache_dir = getenv('BBG_CACHE_DB_DIR') if getenv('BBG_CACHE_DB_DIR') is not None else ''
        self.engine = create_engine(f'sqlite:///{cache_dir}bbg_cache.db', echo=False)

        sql_create_table = f""" CREATE TABLE IF NOT EXISTS {self._table_name} (
                                            ticker text,
                                            field text NOT NULL,
                                            value text,
                                            PRIMARY KEY (ticker, field)
                                        ) """
        with self.engine.connect() as con:
            con.execute(text(sql_create_table))   # create DB/TABLE if not exists
        self.data_points = 0

    def c_ref_static(self, tickers: list, fields: ty.Union[ty.List, str]):
        '''
            For Bloomberg static data without overrides
        '''
        if type(tickers) is not list:
            tickers = [tickers]
        if type(fields) is not list:
            fields = [fields]
        tickers = list(set(tickers))
        fields = list(set(fields))
        db_data = self._read_db(tickers, fields)
        existing_ticker_field = db_data[['ticker', 'field']].to_dict('records')
        existing_tickers = list(set(db_data['ticker'].to_list()))
        missing_tickers = list(np.setdiff1d(tickers, existing_tickers))

        new_data = DataFrame()
        if missing_tickers != []:
            tem = self.b_ref(missing_tickers, fields)
            new_data = pd.concat([new_data, tem], ignore_index=True)

        for t in existing_tickers:
            for f in fields:
                if {'ticker': t, 'field': f} not in existing_ticker_field:
                    tem = self.b_ref(t, f)
                    new_data = pd.concat([new_data, tem], ignore_index=True)

        self._write_db(new_data)
        final_data = pd.concat([db_data, new_data], ignore_index=True)
        return final_data

    def b_ref(self, tickers, fields, ovrds: list = None):
        if type(tickers) is not list:
            tickers = [tickers]
        if type(fields) is not list:
            fields = [fields]
        self.data_points = self.data_points + (len(tickers) * len(fields))
        print(
            f"Requesting Bloomberg Ref: {len(tickers)} tickers x {len(fields)} fields = {
                len(tickers) * len(fields)} Data Points"
        )
        print(f"Total Data Points Requested: {self.data_points}")
        re_list = []
        df_re = self.bdp(tickers, fields, ovrds).rename(columns={'security': 'ticker'})
        # convert the dataframe from bdp into ['ticker','field','value'] form which is accepted by the cache db
        for f in fields:
            tem = df_re[['ticker', f]].copy()
            tem = tem.rename(columns={f: 'value'})
            tem['field'] = f
            re_list.append(tem)
        return pd.concat(re_list, ignore_index=True).sort_values(by=['ticker'])

    def b_bulkref(self, tickers, fields, ovrds=None):
        if type(tickers) is not list:
            tickers = [tickers]
        if type(fields) is not list:
            fields = [fields]
        self.data_points = self.data_points + (len(tickers) * len(fields))
        print(
            f"Requesting Bloomberg Bulkref: {len(tickers)} tickers x {len(fields)} fields = {
                len(tickers) * len(fields)} Data Points"
        )
        print(f"Total Data Points Requested: {self.data_points}")
        try:
            re = self.bds(tickers, fields, ovrds)
        except Exception as e:
            print(e)
            re = DataFrame()
        return re

    def _read_db(self, tickers: list, fields: list):
        sql_select = f""" SELECT * from {self._table_name}
                            WHERE ticker in {str(tickers).replace('[', '(').replace(']', ')')}
                            AND field in {str(fields).replace('[', '(').replace(']', ')')} """
        return pd.read_sql(sql_select, con=self.engine)

    def _write_db(self, new_data: DataFrame):
        new_data.dropna().to_sql(f'{self._table_name}', con=self.engine, if_exists='append', index=False)


# bc = BloombergClient()
# bc.start()

# bc.bdp(['MSFT US Equity'], ['CORR_COEF'], overrides=[('BETA_OVERRIDEREL_INDEX', 'BTC Index')])
# bc.bql("get (FACTOR_EXPOSURE(FACTOR_NAME=QUALITY)) for (['MSFT US EQUITY','AAPL US EQUITY'])")
# bc.bql("get(px_last) for(['MSFT US EQUITY])")
# bc.bql("get (ID().POSITIONs) for (MEMBERS('U1234567-8),TYPE=PORT)")
