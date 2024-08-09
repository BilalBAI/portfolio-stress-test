from os import getenv
import typing as ty
from datetime import datetime
import pandas as pd
import numpy as np
from pandas import DataFrame
from sqlalchemy import create_engine
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


class RiskData:

    def __init__(self, product_level_data, symbol_level_data):
        self.product_level_data = product_level_data
        self.symbol_level_data = symbol_level_data
        self.all_books = product_level_data['book'].unique().tolist()

    def get_product_level_data(self, books: list = None):
        if books is None:
            return self.product_level_data.reset_index(drop=True)
        else:
            return self.product_level_data[self.product_level_data['book'].isin(books)].reset_index(drop=True)

    def get_symbol_level_data(self, books: list = None):
        if books is None:
            return self.symbol_level_data.reset_index(drop=True)
        else:
            return self.symbol_level_data[self.symbol_level_data['book'].isin(books)].reset_index(drop=True)


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
            con.execute(sql_create_table)   # create DB/TABLE if not exists
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
            f"Requesting Bloomberg Ref: {len(tickers)} tickers x {len(fields)} fields = {len(tickers)*len(fields)} Data Points"
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
            f"Requesting Bloomberg Bulkref: {len(tickers)} tickers x {len(fields)} fields = {len(tickers)*len(fields)} Data Points"
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
                            WHERE ticker in {str(tickers).replace('[','(').replace(']',')')} 
                            AND field in {str(fields).replace('[','(').replace(']',')')} """
        return pd.read_sql(sql_select, con=self.engine)

    def _write_db(self, new_data: DataFrame):
        new_data.dropna().to_sql(f'{self._table_name}', con=self.engine, if_exists='append', index=False)


class PortRiskClient:

    def __init__(self, df_fx: DataFrame):
        '''
        df_fx should have columns ['CRNCY','FX']
        'CRNCY' column has all the currency types in the target portfolio
        'CRNCY' should match Bloomberg 3 letters currency format
        'FX' is the fx rate to be applied to the currency
        '''
        if not set(['CRNCY', 'FX']).issubset(set(df_fx.columns)):
            raise Exception("df_fx data should include columns: ['CRNCY','FX']")
        self.df_fx = df_fx
        self.con = BloombergClient(timeout=9999)
        self.con.start()
        self.all_tickers: list = []
        self.stock_tickers: list = []
        self.option_tickers: list = []
        self.future_tickers: list = []
        self.etf_tickers: list = []
        self.decomp_exceptions: list = []
        self.missing_data: list = []

    def bcon_ref(self, tickers: list, field: str, ovrds: list = None):
        if tickers == []:
            return DataFrame({'ticker': [], 'value': []})
        if field in STATIC_FIELDS:
            df = self.con.c_ref_static(tickers, field)
        else:
            df = self.con.b_ref(tickers, field, ovrds)
        self.missing_data.extend(df[df['value'].isnull()][['ticker', 'field']].to_dict('records'))
        return df

    def get_all(self, df: DataFrame) -> RiskData:
        '''
        df['book','ticker','quantity']
        '''
        df_product = df[['book', 'ticker', 'quantity']].copy()
        df_product = self.get_type(df_product)
        df_product = self.get_fx(df_product)
        df_product = self.get_b(df_product)
        df_product = self.get_multiplier(df_product)
        df_product = self.get_put_call(df_product)
        df_product = self.get_rate(df_product)
        df_product = self.get_spot(df_product)
        df_product = self.get_strike(df_product)
        df_product = self.get_time_to_expiry(df_product)
        df_product = self.get_underlying(df_product)
        df_product = self.get_vol(df_product)
        df_symbol = self.get_symbol_level_data(df_product)
        return RiskData(df_product, df_symbol)

    def get_type(self, df: DataFrame):
        '''
        required fields: ticker
        '''
        tem = self.bcon_ref(df['ticker'].unique().tolist(), 'SECURITY_TYP2')
        df['SECURITY_TYP2'] = df['ticker'].map(dict(tem[['ticker', 'value']].values))
        df['type'] = df['SECURITY_TYP2']
        self.all_tickers = df['ticker'].unique().tolist()
        self.option_tickers = df[df['type'].isin(OPTION_TYPE)]['ticker'].unique().tolist()
        self.stock_tickers = df[df['type'].isin(STOCK_TYPE)]['ticker'].unique().tolist()
        self.future_tickers = df[df['type'].isin(FUTURE_TYPE)]['ticker'].unique().tolist()
        self.etf_tickers = df[df['type'].isin(ETF_TYPE)]['ticker'].unique().tolist()
        return df

    def get_fx(self, df: DataFrame):
        '''
        required fields: ticker
        '''
        # Get CRNCY from Bbg
        tem = self.bcon_ref(df['ticker'].unique().tolist(), 'CRNCY')
        df['CRNCY'] = df['ticker'].map(dict(tem[['ticker', 'value']].values))
        # Check if df_fx covers all CRNCY types
        set_diff = np.setdiff1d(df['CRNCY'].unique(), self.df_fx['CRNCY'].unique())
        if set_diff.size != 0:
            raise Exception(f'Missing FX for {set_diff}')
        #  Merge FX rate
        df = pd.merge(df, self.df_fx, how='left', on='CRNCY')
        return df

    def get_multiplier(self, df: DataFrame):
        '''
        required fields: ticker
        '''
        tem = self.bcon_ref(self.option_tickers + self.future_tickers, 'PRICE_MULTIPLIER')
        df['multiplier'] = df['ticker'].map(dict(tem[['ticker', 'value']].values)).fillna(1)
        df['multiplier'] = df['multiplier'].astype(float).astype(int)   # multiplier sometimes
        return df

    def get_spot(self, df: DataFrame):
        '''
        required fields: ticker
        '''
        tem1 = self.bcon_ref(self.option_tickers, 'OPT_UNDL_PX')
        tem2 = self.bcon_ref(self.option_tickers, 'PX_LAST')
        tem3 = self.bcon_ref(self.stock_tickers + self.etf_tickers + self.future_tickers, 'PX_LAST')
        # Underlying Spot Price
        tem = pd.concat([tem1, tem3], ignore_index=True)
        df['spot'] = df['ticker'].map(dict(tem[['ticker', 'value']].values)).fillna(0)
        # Market Price
        tem = pd.concat([tem2, tem3], ignore_index=True)
        df['market_price'] = df['ticker'].map(dict(tem[['ticker', 'value']].values)).fillna(0)
        return df

    def get_vol(self, df: DataFrame):
        '''
        required fields: ticker
        '''
        tem = self.bcon_ref(self.option_tickers, 'IVOL')
        df['vol'] = df['ticker'].map(dict(tem[['ticker', 'value']].values)).fillna(0)
        df['vol'] = df['vol'] / 100
        # TODO: if IVOL is 0 use IVOL_MID
        return df

    def get_strike(self, df: DataFrame):
        '''
        required fields: ticker
        '''
        tem = self.bcon_ref(self.option_tickers, 'OPT_STRIKE_PX')
        df['strike'] = df['ticker'].map(dict(tem[['ticker', 'value']].values)).fillna(0)
        return df

    def get_put_call(self, df: DataFrame):
        '''
        required fields: ticker
        '''
        tem = self.bcon_ref(self.option_tickers, 'OPT_PUT_CALL')
        df['put_call'] = df['ticker'].map(dict(tem[['ticker', 'value']].values)).fillna('not an option').str.lower()
        # Bloomberg may return put/call/p/c
        df['put_call'] = df['put_call'].replace({'p': 'put', 'c': 'call'})
        return df

    def get_time_to_expiry(self, df: DataFrame):
        '''
        required fields: ticker
        '''
        tem = self.bcon_ref(self.option_tickers, 'MATURITY')
        tem['Today'] = pd.to_datetime(datetime.today())
        tem['MATURITY'] = pd.to_datetime(tem['value'])
        tem['time_to_expiry'] = (tem['MATURITY'] - tem['Today']).astype('timedelta64[D]') / 365
        tem['time_to_expiry'] = tem['time_to_expiry'].clip(0)
        df['time_to_expiry'] = df['ticker'].map(dict(tem[['ticker', 'time_to_expiry']].values)).fillna(0)
        return df

    def get_rate(self, df: DataFrame):
        '''
        required fields: ticker
        '''
        tem = self.bcon_ref(self.option_tickers, 'OPT_FINANCE_RT')
        df['rate'] = df['ticker'].map(dict(tem[['ticker', 'value']].values)).fillna(0)
        df['rate'] = df['rate'] / 100
        return df

    def get_underlying(self, df: DataFrame):
        '''
        required fields: ticker
        '''
        # Options
        tem1 = self.bcon_ref(self.option_tickers, 'UNDERLYING_SECURITY_DES')
        tem1_1 = self.bcon_ref(tem1['value'].unique().tolist(), 'UNDL_SPOT_TICKER')
        tem1_1['value'] = tem1_1['value'] + ' Index'
        tem1['value'] = tem1['value'].map(dict(tem1_1[['ticker', 'value']].values)).fillna(tem1['value'])
        tem1_2 = self.bcon_ref(tem1['value'].unique().tolist(), 'ETF_UNDL_INDEX_TICKER')
        tem1_2['value'] = tem1_2['value'] + ' Index'
        tem1['value'] = tem1['value'].map(dict(tem1_2[['ticker', 'value']].values)).fillna(tem1['value'])
        # Futures
        tem2 = self.bcon_ref(self.future_tickers, 'UNDL_SPOT_TICKER')
        tem2['value'] = tem2['value'] + ' Index'
        # ETFs
        tem3 = self.bcon_ref(self.etf_tickers, 'ETF_UNDL_INDEX_TICKER')
        tem3['value'] = tem3['value'] + ' Index'
        # Aggregate
        tem = pd.concat([tem1, tem2, tem3], ignore_index=True)
        df['underlying_ticker'] = df['ticker'].map(dict(tem[['ticker', 'value']].values)).fillna(df['ticker'])
        return df

    def get_b(self, df: DataFrame):
        df['cost_of_carry_rate'] = 'default'
        return df

    def get_symbol_level_data(self, df):
        df_symbol = self._decomp(df)
        df_symbol = self._get_bbg_data(df_symbol)
        return df_symbol

    def _decomp(self, df):
        df_symbol = df[~df['underlying_ticker'].str.contains(' Index')].copy()
        df_symbol['ref_ticker'] = df_symbol['ticker']
        df_symbol['ticker'] = df_symbol['underlying_ticker']
        df_symbol['weight'] = 1
        df_symbol = df_symbol[['ticker', 'ref_ticker', 'book', 'weight']]
        # Indices Decomp
        df_decomp = df[df['underlying_ticker'].str.contains(' Index')].copy()
        decomp_df_list = []
        for i, row in df_decomp.iterrows():
            tem = self.con.b_bulkref(row['underlying_ticker'], 'INDX_MWEIGHT')
            tem = tem.dropna()
            if tem.empty:
                continue
            tem = tem.rename(columns={'Member Ticker and Exchange Code': 'ticker', 'Percentage Weight': 'weight'})
            tem['ticker'] = tem['ticker'] + ' Equity'
            tem['weight'] = tem['weight'] / 100
            # Mkt cap weighted decomp
            if tem['weight'].sum() <= 0:
                tem = self._mktcap_weighted_decomp(tem)
            tem['ref_ticker'] = row['ticker']
            tem['book'] = row['book']
            tem = tem[['ticker', 'ref_ticker', 'book', 'weight']]
            decomp_df_list.append(tem)
        df_symbol = pd.concat([df_symbol] + decomp_df_list, ignore_index=True)
        return df_symbol

    def _mktcap_weighted_decomp(self, df):
        tem = self.bcon_ref(df['ticker'].unique().tolist(), 'CRNCY_ADJ_MKT_CAP', [('EQY_FUND_CRNCY', 'USD')])
        df['MKT_CAP_USD_M'] = df['ticker'].map(dict(tem[['ticker', 'value']].values))
        df['weight'] = df['MKT_CAP_USD_M'] / df['MKT_CAP_USD_M'].sum()
        return df[['ticker', 'weight']].reset_index(drop=True)

    def _get_bbg_data(self, df):
        fields = [
            'SECURITY_TYP2', 'CRNCY', 'ISSUER_BULK', 'GICS_INDUSTRY_GROUP_NAME', 'COUNTRY_ISO', 'CNTRY_ISSUE_ISO',
            'MSCI_COUNTRY_CODE', 'PX_VOLUME', 'VOLUME_AVG_5D', 'VOLUME_AVG_20D', 'VOLUME_AVG_3M'
        ]
        for fld in fields:
            tem = self.bcon_ref(df['ticker'].unique().tolist(), fld)
            df[fld] = df['ticker'].map(dict(tem[['ticker', 'value']].values))
        # handle nan
        df['PX_VOLUME'] = df['PX_VOLUME'].fillna(0)
        df['VOLUME_AVG_5D'] = df['VOLUME_AVG_5D'].fillna(df['PX_VOLUME'])
        df['VOLUME_AVG_20D'] = df['VOLUME_AVG_20D'].fillna(df['VOLUME_AVG_5D'])
        df['VOLUME_AVG_3M'] = df['VOLUME_AVG_3M'].fillna(df['VOLUME_AVG_20D'])
        df['ISSUER_BULK'] = df['ISSUER_BULK'].fillna(df['ticker'])
        for fld in ['GICS_INDUSTRY_GROUP_NAME', 'COUNTRY_ISO', 'SECURITY_TYP2']:
            df[fld] = df[fld].fillna('N/A')
        for fld in ['CNTRY_ISSUE_ISO', 'MSCI_COUNTRY_CODE']:
            df[fld] = df[fld].fillna(df['COUNTRY_ISO'])
        # Mkt Cap in USD M
        mktcap = self.bcon_ref(df['ticker'].unique().tolist(), 'CRNCY_ADJ_MKT_CAP', [('EQY_FUND_CRNCY', 'USD')])
        mktcap = mktcap.rename(columns={'value': 'MKT_CAP_USD_M'})
        df = pd.merge(df, mktcap[['ticker', 'MKT_CAP_USD_M']], how='left', on='ticker')
        df['MKT_CAP_USD_M'] = df['MKT_CAP_USD_M'].fillna(0).round(0)
        # Handle CN Offshore
        df.loc[(df['CNTRY_ISSUE_ISO'] != 'CN') & (df['MSCI_COUNTRY_CODE'] == 'CN'), 'MSCI_COUNTRY_CODE'] = 'CN Offshore'
        df.loc[(df['CNTRY_ISSUE_ISO'] != 'CN') & (df['COUNTRY_ISO'] == 'CN'), 'COUNTRY_ISO'] = 'CN Offshore'
        df['currency'] = df['CRNCY']
        # USD prices
        price = self.bcon_ref(df['ticker'].unique().tolist(), 'CRNCY_ADJ_PX_LAST', [('EQY_FUND_CRNCY', 'USD')])
        df['price'] = df['ticker'].map(dict(price[['ticker', 'value']].values))
        return df.reset_index(drop=True)
