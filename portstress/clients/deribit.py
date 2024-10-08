import requests
import pandas as pd
import datetime

# Get list of instrument from deribit
ATTRIBUTES = ['underlying_price', 'timestamp', 'mark_iv', 'instrument_name',
              'index_price', 'greeks', 'bid_iv', 'ask_iv', 'best_bid_price', 'best_ask_price']


class DeribitClient:
    def __init__(
        self,
        base_url='https://deribit.com/api/v2/public'
    ):
        self.base_url = base_url

    def get_instruments(self, coins: list = ['BTC', 'ETH']):
        url = f'{self.base_url}/get_instruments'
        instruments = []
        for coin in coins:
            r = requests.get(url, params={'currency': coin,
                                          'expired': 'false',
                                          'kind': 'option'})
            for result in r.json()['result']:
                instruments.append(result['instrument_name'])
        print(f'{len(instruments)} instruments have been crawled.')
        return instruments

    # get undelying index prices

    def get_index_price(self, coins: list = ['BTC', 'ETH']):
        url = f'{self.base_url}/get_index_price'
        index_prices = {}
        for coin in coins:
            r = requests.get(url, params={'index_name': f"{coin.lower()}_usd"})
            index_prices[coin] = r.json()['result']['index_price']
        return index_prices

    # Get the order book from deribit

    def get_order_book(self, instruments, depth=1, attributes=ATTRIBUTES):
        url = f'{self.base_url}/get_order_book'
        data = {attribute: [] for attribute in attributes}
        for instrument in instruments:
            r = requests.get(url, params={'depth': depth,
                                          'instrument_name': instrument})
            try:
                c = r.json()['result']
            except:
                print(f"Instrument Name: {instrument} Not Found")
                continue
            for attribute in attributes:
                if attribute not in c.keys():
                    c[attribute] = None
                data[attribute].append(c[attribute])
        data = pd.DataFrame(data)
        return data

    def get_all_strike_expiry(self, coins: list = ['BTC', 'ETH']):
        '''
            get ivol for all strikes and expires for option chains in coins
        '''
        all_instruments = self.get_instruments(coins=coins)
        all_instruments = [i for i in all_instruments if '-P' not in i]  # call options only
        all_index_prices = self.get_index_price()
        df = self.get_order_book(all_instruments, attributes=['instrument_name', 'mark_iv'])
        df = df.rename(columns={'instrument_name': 'instrument', 'mark_iv': 'ivol'})
        df[['underlying', 'expiry', 'strike', 'put_call']] = df['instrument'].str.split('-', expand=True)
        df['strike'] = df['strike'].astype(float)
        df['put_call'] = df['put_call'].map({'P': 'put', 'C': 'call'})
        df['underlying-expiry'] = df['underlying'] + "-" + df['expiry']
        df['timestamp_ivol'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
        df['index_price'] = df['underlying'].map(all_index_prices)
        # filter out at the money ivol
        atm_strike_map = []
        for u in df['underlying'].unique().tolist():
            # atm_strike_map[u] = []
            tem = df[df['underlying'].isin([u])].copy().reset_index(drop=True)
            for e in tem['expiry'].unique().tolist():
                tem_ex = tem[tem['expiry'] == e].copy().reset_index(drop=True)
                # Find the closest value in column strike to the spot
                atm_strike = tem_ex['strike'].iloc[(tem_ex['strike'] - all_index_prices[u]).abs().argsort()[0]]
                # Retrieve the corresponding value from column vol
                atm_strike_map.append(f"{u}-{str(e)[0:10]}-{int(atm_strike)}-C")
        df_atm = df[df['instrument'].isin(atm_strike_map)].reset_index(drop=True)
        df_atm['atm_ivol'] = df_atm['ivol']
        df_atm['atm_strike'] = df_atm['strike']
        df_atm = df_atm[['underlying-expiry', 'atm_strike', 'index_price', 'atm_ivol', 'timestamp_ivol']]

        return df, df_atm

    def get_atm_ivol(self, coins: list = ['BTC', 'ETH']):
        '''
            get at the money implied vol for each expiry
        '''
        all_instruments = self.get_instruments(coins=coins)
        df_all_instruments = pd.DataFrame({'instrument': all_instruments})
        df_all_instruments[['underlying', 'expiry', 'strike', 'put_call']
            ] = df_all_instruments['instrument'].str.split('-', expand=True)
        df_all_instruments['strike'] = df_all_instruments['strike'].astype(float)
        df_all_instruments['put_call'] = df_all_instruments['put_call'].map({'P': 'put', 'C': 'call'})
        # df_all_instruments['expiry'] = pd.to_datetime(df_all_instruments['expiry'], format='%d%b%y')
        all_index_prices = self.get_index_price()

        atm_strike_map = []
        for u in df_all_instruments['underlying'].unique().tolist():
            # atm_strike_map[u] = []
            tem = df_all_instruments[df_all_instruments['underlying'].isin([u])].copy().reset_index(drop=True)
            for e in tem['expiry'].unique().tolist():
                tem_ex = tem[tem['expiry'] == e].copy().reset_index(drop=True)
                # Find the closest value in column strike to the spot
                atm_strike = tem_ex['strike'].iloc[(tem_ex['strike'] - all_index_prices[u]).abs().argsort()[0]]
                # Retrieve the corresponding value from column vol
                atm_strike_map.append(f"{u}-{str(e)[0:10]}-{int(atm_strike)}-C")

        df_atm = df_all_instruments[df_all_instruments['instrument'].isin(atm_strike_map)].reset_index(drop=True)
        df_atm = self.get_order_book(df_atm['instrument'].to_list(), attributes=['instrument_name', 'mark_iv'])
        df_atm['atm_ivol'] = df_atm['mark_iv']
        df_atm[['underlying', 'expiry', 'strike', 'put_call']] = df_atm['instrument_name'].str.split('-', expand=True)
        df_atm['underlying-expiry'] = df_atm['underlying'] + "-" + df_atm['expiry']
        df_atm['atm_strike'] = df_atm['strike']
        df_atm['timestamp_atm_ivol'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')

        return df_atm[['underlying-expiry', 'atm_strike', 'atm_ivol', 'timestamp_atm_ivol']]
