from __future__ import annotations

import math
from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np
import pandas as pd

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
    # Run BS pricing for put and call, if not an option (not a put nor a call), return spot
    if put_call in ['put', 'call', 'p', 'c']:
        d1 = (math.log(spot / strike) + (b + vol**2 / 2) * time_to_expiry) / (vol * time_to_expiry**0.5)
        d2 = d1 - vol * time_to_expiry**0.5
        if put_call in ['call', 'c']:
            return spot * math.exp(time_to_expiry *
                                   (b - r)) * norm.cdf(d1) - strike * math.exp(-r * time_to_expiry) * norm.cdf(d2)
        else:
            return strike * math.exp(-r * time_to_expiry) * norm.cdf(-d2) - spot * math.exp(time_to_expiry *
                                                                                            (b - r)) * norm.cdf(-d1)
    else:
        return spot


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
    # All else equal, vega for a put is equal to vega of a call
    # Here, put_call is used to decide whether the position is an option
    # return 0 if not an option type
    if put_call not in ['put', 'call', 'p', 'c']:
        return 0

    r = rate
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


def calc_gamma(strike, time_to_expiry, spot, rate, vol, put_call, cost_of_carry_rate='default'):
    # All else equal, gamma for a put is equal to gamma of a call
    # Here, put_call is used to decide whether the position is an option
    # return 0 if not an option type
    if put_call not in ['put', 'call', 'p', 'c']:
        return 0

    r = rate
    if cost_of_carry_rate == 'default':
        b = r
    else:
        b = cost_of_carry_rate
    d1 = (math.log(spot / strike) + (b + vol**2 / 2) * time_to_expiry) / (vol * time_to_expiry**0.5)
    discount_factor = math.exp(-(r - b) * time_to_expiry)
    gamma = (norm.pdf(d1) * discount_factor) / (spot * vol * time_to_expiry**0.5)
    return gamma


def calc_delta_df(df: pd.DataFrame):
    # Calc delta and position_delta for a portfolio in a DataFrame format
    # Position Delta = Delta × Number of Contracts × Shares per Contract
    # Position Delta($) = Delta × Number of Contracts × Shares per Contract × Price of the Underlying Asset
    '''
        When using np.vectorize, the return type is determined by the first output.
        If the first output is an integer, the entire output might be cast to integers, even if subsequent outputs should be floats.
        To ensure that the delta values are stored as floats, need explicitly convert the result to a float: float(calc_delta(spot=spot, **BS_PARAMETERS)
    '''
    df['delta'] = np.vectorize(
        lambda spot, **BS_PARAMETERS: float(calc_delta(spot=spot, **BS_PARAMETERS)) if spot > 0 else 0
    )(**{col: df[col] for col in BS_PARAMETERS})
    df['position_delta'] = df['delta'] * df['quantity'] * df[
        'multiplier']
    df['dollar_position_delta'] = df['delta'] * df['quantity'] * df[
        'multiplier'] * df['spot']
    return df


def calc_vega_df(df: pd.DataFrame):
    # Calc vega and position_vega for a portfolio in a DataFrame format
    # Position Vega($) = Vega × ΔIV × Number of Contracts × Shares per Contract
    '''
        When using np.vectorize, the return type is determined by the first output.
        If the first output is an integer, the entire output might be cast to integers, even if subsequent outputs should be floats.
        To ensure that the delta values are stored as floats, need explicitly convert the result to a float: float(calc_delta(spot=spot, **BS_PARAMETERS)
    '''
    df['vega'] = np.vectorize(
        lambda spot, **BS_PARAMETERS: float(calc_vega(spot=spot, **BS_PARAMETERS)) if spot > 0 else 0
    )(**{col: df[col] for col in BS_PARAMETERS})
    df['position_vega'] = df['vega'] * df['quantity'] * df[
        'multiplier']
    return df


def calc_implied_volatility(spot, strike, time_to_expiry, rate, market_price, put_call='call'):
    """
    Calculate implied volatility for a European option (call or put) given the market price.

    spot : float : spot price of the underlying asset
    strike : float : strike price of the option
    time_to_expiry : float : time to expiry in years
    rate : float : risk-free interest rate (annual)
    market_price : float : market price of the option
    put_call : str : type of option ('call' or 'put')

    Returns:
    iv : float : implied volatility (annual)
    """
    # Define the objective function (difference between market price and model price)
    def objective(vol):
        return bs_pricing(strike, time_to_expiry, spot, rate, vol, put_call, cost_of_carry_rate='default') - market_price

    # Use numerical solver to find implied volatility that results in the market price
    iv = brentq(objective, 1e-6, 5)  # Bounded between 1e-6 and 5
    return iv


def calc_prob_exe(strike, time_to_expiry, spot, rate, vol, put_call, cost_of_carry_rate='default'):
    '''
    Calculate risk neutral probability of exercise
    For calls: N(d2)
    For puts: N(-d2)

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
    # Calc N(d2) or N(-d2)
    if put_call in ['put', 'call', 'p', 'c']:
        d1 = (math.log(spot / strike) + (b + vol**2 / 2) * time_to_expiry) / (vol * time_to_expiry**0.5)
        d2 = d1 - vol * time_to_expiry**0.5
        if put_call in ['call', 'c']:
            return norm.cdf(d2)
        else:
            return norm.cdf(-d2)
    else:
        return None


def bt_pricing(strike, time_to_expiry, spot, rate, vol, put_call, N=25000):
    '''
    Binomial Tree Pricing for American Options

    spot: spot stock price
    strike: strike price
    time_to_expiry: time to maturity in years
    rate: risk free rate
    vol: diffusion coefficient or volatility

    N: number of periods or number of time steps
    put_call: put or call

    reference: https://github.com/cantaro86/Financial-Models-Numerical-Methods

    '''

    dT = float(time_to_expiry) / N  # Delta t
    u = np.exp(vol * np.sqrt(dT))  # up factor
    d = 1.0 / u  # down factor

    V = np.zeros(N + 1)  # initialize the price vector
    S_T = np.array([(spot * u**j * d ** (N - j)) for j in range(N + 1)])  # price S_T at time T

    a = np.exp(rate * dT)  # risk free compound return
    p = (a - d) / (u - d)  # risk neutral up probability
    q = 1.0 - p  # risk neutral down probability

    if put_call == "call":
        V[:] = np.maximum(S_T - strike, 0.0)
    elif put_call == "put":
        V[:] = np.maximum(strike - S_T, 0.0)

    for i in range(N - 1, -1, -1):
        V[:-1] = np.exp(-rate * dT) * (p * V[1:] + q * V[:-1])  # the price vector is overwritten at each step
        S_T = S_T * u  # it is a tricky way to obtain the price at the previous time step
        if put_call == "call":
            V = np.maximum(V, S_T - strike)
        elif put_call == "put":
            V = np.maximum(V, strike - S_T)

    return V[0]
