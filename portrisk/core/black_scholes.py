from __future__ import annotations

import math
from scipy.stats import norm

BS_PARAMETERS = DELTA_PARAMETERS = ['strike', 'time_to_expiry', 'spot', 'rate', 'vol', 'put_call', 'cost_of_carry_rate']
GAMMA_PARAMETERS = VEGA_PARAMETERS = ['strike', 'time_to_expiry', 'spot', 'rate', 'vol', 'cost_of_carry_rate']


def bs_pricing(strike, time_to_expiry, spot, rate, vol, put_call, cost_of_carry_rate='default'):
    '''
    Generalized Black-Scholes-Merton Option Pricing

    b=r  #default
    Black-Scholes model for European stock options without dividends
    - Black and Scholes (1973)

    b=r−q
    Merton's extension to the model for pricing European stock options with a continuous dividend yield q
    - Merton (1973)

    b=0
    Black Fischer's extension for pricing European futures options
    - Black (1976)

    b = 0 and r = 0
    The margined futures option model.
    - Asay (1982)

    b=r−rj
    Model for pricing European currency options
    - Garman and Kohlhagen (1983)

    Inputs Example:
        strike = 450
        t = date(year=2020,month=7,day=30)-date.today()
        time_to_expiry = t.days/365
        put_call = 'put'
        vol = 0.3058
        spot = 451.6
        rate = 0.00893

    '''
    r = rate
    # Price Expired Option and 0 vol with their intrinsic value
    if ((time_to_expiry <= 0) or (vol == 0)) and (put_call == 'call'):
        return min(0, spot - strike)
    elif ((time_to_expiry <= 0) or (vol == 0)) and (put_call == 'put'):
        return min(0, strike - spot)
    # Reset underlying spot to a small number if it's 0 (The formula cannot take 0 underlying spot)
    if spot == 0:
        spot = 0.0000001
    if cost_of_carry_rate == 'default':
        b = r
    else:
        b = cost_of_carry_rate
    if put_call in ['put', 'call']:
        d1 = (math.log(spot / strike) + (b + vol**2 / 2) * time_to_expiry) / (vol * time_to_expiry**0.5)
        d2 = d1 - vol * time_to_expiry**0.5
        if put_call == 'call':
            return spot * math.exp(time_to_expiry *
                                   (b - r)) * norm.cdf(d1) - strike * math.exp(-r * time_to_expiry) * norm.cdf(d2)
        else:
            return strike * math.exp(-r * time_to_expiry) * norm.cdf(-d2) - spot * math.exp(time_to_expiry *
                                                                                            (b - r)) * norm.cdf(-d1)
    else:
        raise Exception('Option type should be [put] or [call]')


def calc_delta(strike, time_to_expiry, spot, rate, vol, put_call, cost_of_carry_rate='default'):
    r = rate
    # return 1 if not an option type
    if put_call not in ['put', 'call', 'p', 'c']:
        return 1
    # Price Expired Option and 0 vol as 0
    if (time_to_expiry <= 0) or (vol == 0):
        return 0
    if cost_of_carry_rate == 'default':
        b = r
    else:
        b = cost_of_carry_rate
    d1 = (math.log(spot / strike) + (b + vol**2 / 2) * time_to_expiry) / (vol * time_to_expiry**0.5)
    discount_factor = math.exp(-(r - b) * time_to_expiry)
    if put_call in ['call', 'c']:
        return discount_factor * norm.cdf(d1)
    else:
        return discount_factor * (norm.cdf(d1) - 1)


def calc_vega(strike, time_to_expiry, spot, rate, vol, put_call, cost_of_carry_rate='default'):
    r = rate
    # return 0 if not an option type
    if put_call not in ['put', 'call', 'p', 'c']:
        return 0
    # Price Expired Option and 0 vol as 0
    if (time_to_expiry <= 0) or (vol == 0):
        return 0
    if cost_of_carry_rate == 'default':
        b = r
    else:
        b = cost_of_carry_rate
    d1 = (math.log(spot / strike) + (b + vol**2 / 2) * time_to_expiry) / (vol * time_to_expiry**0.5)
    discount_factor = math.exp(-(r - b) * time_to_expiry)
    vega = spot * time_to_expiry**0.5 * norm.pdf(d1) * discount_factor
    return vega / 100


def calc_gamma(strike, time_to_expiry, spot, rate, vol, cost_of_carry_rate='default'):
    r = rate
    if cost_of_carry_rate == 'default':
        b = r
    else:
        b = cost_of_carry_rate
    d1 = (math.log(spot / strike) + (b + vol**2 / 2) * time_to_expiry) / (vol * time_to_expiry**0.5)
    discount_factor = math.exp(-(r - b) * time_to_expiry)
    gamma = (norm.pdf(d1) * discount_factor) / (spot * vol * time_to_expiry**0.5)
    return gamma
