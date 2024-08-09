import requests
import pandas as pd

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
        data = {}
        for instrument in instruments:
            r = requests.get(url, params={'depth': depth,
                                          'instrument_name': instrument})
            c = r.json()['result']
            for attribute in attributes:
                if attribute == 'greeks':
                    for greek in c[attribute].keys():
                        if greek not in data.keys():
                            data[greek] = [c[attribute][greek]]
                        else:
                            data[greek].append(c[attribute][greek])
                else:
                    if attribute not in data.keys():
                        data[attribute] = [c[attribute]]
                    else:
                        data[attribute].append(c[attribute])
        data = pd.DataFrame(data)
        return data

    # get at the money implied vol for each expiry

    def get_atm_ivol(self, coins: list = ['BTC', 'ETH']):
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
        df_atm[['underlying-expiry', 'atm_strike', 'atm_ivol']]

        return df_atm[['underlying-expiry', 'atm_strike', 'atm_ivol']]
