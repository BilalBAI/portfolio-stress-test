import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import norm
from scipy.stats import kendalltau
from scipy.linalg import cholesky


def gaussian_copula(data):
    # Ensure input is a DataFrame for correlation calculation
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # Calculate empirical CDF values
    ranks = data.rank(axis=0)
    uniform_data = (ranks - 1) / (len(data) - 1)

    # Transform to standard normal
    normal_data = norm.ppf(uniform_data)

    # Convert to DataFrame to use pandas' corr method
    normal_data = pd.DataFrame(normal_data, columns=data.columns)

    # Estimate the correlation matrix
    corr_matrix = normal_data.corr()

    # Cholesky decomposition
    L = cholesky(corr_matrix, lower=True)

    # Simulate Gaussian copula
    random_normals = norm.ppf(np.random.rand(len(data), len(data.columns)))
    copula_data = np.dot(random_normals, L.T)

    # Transform back to uniform scale
    copula_uniform = norm.cdf(copula_data)
    simulated_data = pd.DataFrame(copula_uniform, columns=data.columns)

    return simulated_data


def calculate_volume_based_metrics(data, T):
    # Liquidity Ratio
    # data['Liquidity_Ratio'] = (data['Price'] * data['Volume']).rolling(window=T).sum() / (data['Price'].diff() ** 2)

    # Hui-Heubel Ratio
    Pmax = data['Price'].rolling(window=T).max()
    Pmin = data['Price'].rolling(window=T).min()
    Pbar = data['Price'].rolling(window=T).mean()
    volume = data['Volume'].rolling(window=T).sum()
    data['Hui_Heubel_Ratio'] = ((Pmax - Pmin) / Pmin) / (volume / (Pbar * data['Outstanding_Shares']))

    # Turnover Ratio: Share Turnover = Trading Volume / Average Shares Outstanding
    data['Turnover_Ratio'] = data['Volume'].rolling(window=T).sum(
    ) / (data['Outstanding_Shares'].rolling(window=T).sum() / T)

    return data[['Liquidity_Ratio', 'Hui_Heubel_Ratio', 'Turnover_Ratio']]


# def calculate_transaction_based_metrics(data):
#     data['Bid_Ask_Spread'] = data['Ask_Price'] - data['Bid_Price']

#     high = data['High']
#     low = data['Low']
#     data['Corwin_Schultz_Spread'] = 2 * (np.log(high / low)).rolling(window=2).sum() ** 0.5 - 1

#     return data[['Bid_Ask_Spread', 'Corwin_Schultz_Spread']]


# def calculate_price_based_metrics(data):
#     data['Return'] = data['Price'].pct_change()
#     data['Amihud_Illiquidity'] = (np.abs(data['Return']) / data['Volume']).rolling(window=20).mean()

#     return data[['Amihud_Illiquidity']]


def calculate_market_based_metrics(data, market_returns):
    data['Market_Return'] = market_returns

    model = sm.OLS(data['Return'].dropna(), sm.add_constant(data['Market_Return'].dropna()))
    results = model.fit()

    data['CAPM_Residual'] = data['Return'] - \
        (results.params['const'] + results.params['Market_Return'] * data['Market_Return'])

    # Autoregression on trading volume change
    data['Volume_Change'] = data['Volume'].pct_change()
    ar_model = sm.OLS(data['CAPM_Residual'].dropna(), sm.add_constant(data['Volume_Change'].dropna()))
    ar_results = ar_model.fit()

    data['Liquidity_Level'] = ar_results.params['Volume_Change']

    return data[['Liquidity_Level']]


def dummy_data():
    # Example usage with a dummy dataframe for testing
    data = pd.DataFrame({
        'Date': pd.date_range(start='1/1/2023', periods=100),
        'Price': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(100, 1000, size=100),
        'Outstanding_Shares': np.random.randint(1e5, 1e6, size=100)
    })
    data.set_index('Date', inplace=True)
    data['High'] = data['Price'] + np.random.rand(100)
    data['Low'] = data['Price'] - np.random.rand(100)
    data['Ask_Price'] = data['Price'] + np.random.rand(100) * 0.1
    data['Bid_Price'] = data['Price'] - np.random.rand(100) * 0.1
    return data


if __name__ == "__main__":
    data = dummy_data()

    volume_metrics = calculate_volume_based_metrics(data)
    print(volume_metrics.head())

    transaction_metrics = calculate_transaction_based_metrics(data)
    print(transaction_metrics.head())

    price_metrics = calculate_price_based_metrics(data)
    print(price_metrics.head())

    market_returns = np.random.randn(100) * 0.01
    market_metrics = calculate_market_based_metrics(data, market_returns)
    print(market_metrics.head())

    copula_data = gaussian_copula(data[['Liquidity_Ratio', 'Hui_Heubel_Ratio', 'Turnover_Ratio']])
    print(copula_data.head())
