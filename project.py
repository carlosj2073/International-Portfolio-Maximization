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