import requests
import time
import datetime
import pandas as pd
from os.path import isfile
import pytz

# Get list of instrument from deribit

url = 'https://deribit.com/api/v2/public/get_instruments'
coins = ['BTC', 'ETH']
expiry_dates = ['4JUL24', '26JUL24', '29JUL24', '30AUG24', '27SEP24','30DEC24']

def get_instruments(url = url, coins = coins, expiry_dates = expiry_dates):
  instruments = []
  for coin in coins:
    r = requests.get(url, params = {'currency': coin,
                                    'expired': 'false',
                                    'kind': 'option'})
    for result in r.json()['result']:
      if any(x in result['instrument_name'] for x in expiry_dates):
        instruments.append(result['instrument_name'])
  print(f'{len(instruments)} instruments have been crawled.')
  return instruments

instruments = get_instruments()


# Get the order book from deribit
url = 'https://deribit.com/api/v2/public/get_order_book'
depth = 1000
filepath = 'data.csv'
attributes = ['underlying_price', 'timestamp', 'mark_iv', 'instrument_name',
              'index_price', 'greeks', 'bid_iv', 'ask_iv', 'best_bid_price', 'best_ask_price']

def get_order_book(url = url,
                   instruments = instruments, depth = depth,
                   attributes = attributes, filepath = filepath):
  while True:
    data = {}
    for instrument in instruments:
      r = requests.get(url, params = {'depth': depth,
                                      'instrument_name': instrument})
      c = r.json()['result']
      for attribute in attributes:
        if attribute == 'greeks':
          for greek in c[attribute].keys():
            if greek not in data.keys():
              data[greek]= [c[attribute][greek]]
            else:
              data[greek].append(c[attribute][greek])
        else:
          if attribute not in data.keys():
            data[attribute]= [c[attribute]]
          else:
            data[attribute].append(c[attribute])
    data = pd.DataFrame(data)
    if isfile(filepath):
      full = pd.read_csv(filepath, index_col = 0)
      full = pd.concat([full, data], ignore_index = True)
    else:
      full = data
    full.to_csv(filepath)
    print(f'{len(data)} record appends into the csv file at {datetime.datetime.now().astimezone(pytz.timezone("Asia/Hong_Kong"))}. There are {len(full)} in total.')
    print()
    time.sleep(3600)

get_order_book()