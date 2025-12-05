import pandas as pd
import numpy as np
from scipy.optimize import minimize
import datetime

# pandas settings for display of dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Lists assets and attatches corresponding indexes
asset_files = {
    'UK' : 'fred_data/NASDAQNQGBN_PC1.csv',
    'Japan' : 'fred_data/NASDAQNQJPN_PC1.csv',
    'Canada' : 'fred_data/NASDAQNQCAN_PC1.csv',
    'US' : 'fred_data/NASDAQCOM_PC1.csv'
}

# Creates dictionary to hold each dataframe
df_list = {}

# Reads the csvs for each asset and assigns columns with correct column name
for asset_name, filename in asset_files.items():
    df = pd.read_csv(
        filename,
        index_col = 0,
        parse_dates = True
    )

    df.columns = [asset_name]
    df_list[asset_name] = df

# Joins the tables for each asset into one
prices_df = pd.concat(df_list.values(), axis=1, join='outer')
prices_df = prices_df.dropna(how='all')

# prints the first and last 5 rows
print("Joined Price Data")
print(prices_df.head())
print(prices_df.tail())

# calculates the expected return for each column
mean_returns = prices_df.mean()

# calculates the covariance for each column
cov_matrix = prices_df.cov()

print("\nexpected returns for each asset:\n")
print(mean_returns)
print("\n covariance matrix:\n")
print(cov_matrix)

# defines constants
risk_free_rate = 0.0
num_assets = len(prices_df.columns)

# function that calculates: E[x]
def portfolio_return(weights, mean_returns):
    return np.sum(mean_returns * weights)

# function that calculates stdv of portfolio
def portfolio_volatility(weights, cov_matrix):
    # multiplies transposed weights by each weighted covariance
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance)

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return = portfolio_return(weights, mean_returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)

    if p_volatility == 0:
        return np.inf

    sharpe = (p_return - risk_free_rate) / p_volatility
    return -sharpe

# initial guess: equal weights for all 4 assets
initial_weights = np.array([1/num_assets] * num_assets)

constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

bounds = tuple((0,1) for asset in range(num_assets))

optimal_results = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(mean_returns, cov_matrix, risk_free_rate),
    method = 'SLSQP',
    bounds = bounds,
    constraints = constraints

)

optimal_weights = optimal_results.x.round(4)
max_sharpe_return = portfolio_return(optimal_weights, mean_returns)
max_sharpe_volatility = portfolio_volatility(optimal_weights, cov_matrix)
max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_volatility


print("\n\n--- OPTIMAL PORTFOLIO RESULTS ---")
print(f" Maximum Sharpe Ratio: {max_sharpe_ratio:.4f}")
print(f"Optimal Annualized Return: {max_sharpe_return:.2%}")
print(f"Optimal Annualized Volatility: {max_sharpe_volatility:.2%}")

optimal_portfolio = pd.DataFrame({
    'Asset': list(asset_files.keys()),
    'Optimal Weight': optimal_weights,
    'Percentage': [f"{ (w * 100):.2f}%" for w in optimal_weights]
})

print("\nOptimal Asset Allocation:")
print(optimal_portfolio.set_index('Asset'))