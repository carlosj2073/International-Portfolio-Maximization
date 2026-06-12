# Portfolio Optimization Project
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

# Changes the display of dataframes in the output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2%}'.format

# Assigns respective indexes to each asset file csv as an array
asset_files = {
    'UK'    : 'fred_data/NASDAQNQGBN_PC1.csv',
    'Japan' : 'fred_data/NASDAQNQJPN_PC1.csv',
    'Canada': 'fred_data/NASDAQNQCAN_PC1.csv',
    'US'    : 'fred_data/NASDAQCOM_PC1.csv'
}

# Reads the csv files into df_list for every item in asset_files.
# This creates a list of each asseet and its data from each csv
df_list = {}
for asset_name, filename in asset_files.items():
    df = pd.read_csv(
        filename,
        index_col=0,
        parse_dates=True
    )
    df.columns = [asset_name]
    df_list[asset_name] = df

# Joins each asset's prices in df_list as columns indexed by observation date
# Drops values that do not have an observation date
prices_df = pd.concat(df_list.values(), axis=1, join='outer')
prices_df = prices_df.dropna(how='all')

# Prints data frame first 5 values
print("\nJoined Return Data (FRED PC1 series):\n")
print(prices_df.head())

# Converts prices to decimals since FRED PC1 series are percent changes
returns_df = prices_df / 100.0

# Computes expected return of each asset and covariance matrix.
# The covariance matrix includes all pairwise interactions including the asset against itself(the variance) as diagnals
mean_returns = returns_df.mean()
cov_matrix = returns_df.cov()

# Prints mean returns for each asset and covariance matrix
print("\nExpected returns for each asset:\n")
print(mean_returns)
print("\nCovariance matrix:\n")
print(cov_matrix)


# Convert index to list of asset names
assets = list(mean_returns.index)
num_assets = len(assets)

# Creates a fucntion that returns the negative Sharpe Ratio (NEGATIVE because we are minimizing the function).
# Where expected portfolio return (port_return) = weights * mean_return
# Where portfolio standard deviation (port_vol) = transposed weights * cov_matrix * weights
# (These matrix formulas are equivalent to the summation formulas we learned in class)   
def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    weights = np.array(weights)
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return - port_return / port_vol   

# Assigns a constraint to the oprimization function:
# - The sum of the weights must equal 1
constraint = ({
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1
})

# Assigns bounds to weights between 0 and 1 (no short selling)
bounds_constrained = tuple((0.0, 1.0) for _ in range(num_assets))
# Allows for shortselling
bounds_unconstrained = tuple((-1, 1) for _ in range(num_assets))

# Assigns a starting point: equal weights
initial_guess = [1.0 / num_assets] * num_assets

# Runs the optimization aimed at minimizing the the objective function(the negative sharpe ratio).
# Uses sequential least squares programming iterating through localized minimizations of the negative sharpe ratios starting at the initial guess.
# SLSQP then stops once the solver finds a local minimum indicating a minimum to the objective function
opt_constrained = minimize(
    neg_sharpe_ratio,
    initial_guess,
    args = (mean_returns, cov_matrix),
    method = 'SLSQP',
    bounds = bounds_constrained,
    constraints = constraint
)

# Runs the same optimization, but with unconstrained weights.
opt_unconstrained = minimize(
    neg_sharpe_ratio,
    initial_guess,
    args = (mean_returns, cov_matrix),
    method = 'SLSQP',
    bounds = bounds_unconstrained,
    constraints = constraint
)

# Assigns results to variables for constrained weights
results_con = opt_constrained.x
max_sharpe_con = -neg_sharpe_ratio(results_con, mean_returns, cov_matrix)

# Assigns results to variables for unconstrained weights
results_uncon = opt_unconstrained.x
max_sharpe_uncon = -neg_sharpe_ratio(results_uncon, mean_returns, cov_matrix)

# Creates side-by-side DataFrames for comparison
df_comparison = pd.DataFrame(index = assets)
df_comparison['Long Only'] = results_con
df_comparison['With Hedging'] = results_uncon

# Prints final output
print("\n" + "-" * 40)
print("     PORTFOLIO OPTIMIZATION REPORT")
print("-"*40)
print("\n   -- ASSET ALLOCATION STRATEGY -- \n")
print(f"  {df_comparison}")

print("\n    --  PERFORMANCE METRIC  --\n")
print(f"Sharpe Ratio (Long Only):  {max_sharpe_con:.4f}")
print(f"Sharpe Ratio (Hedging):   {max_sharpe_uncon: 4f}")
print(f"Improvement:              {((max_sharpe_uncon - max_sharpe_con) / max_sharpe_con) * 100: 1f}%")
print("\n" + "-" * 40)

# -- Asset Allocation Graph

# Sets up asset allocation graph using df_comparison dataframe
fig, ax = plt.subplots(figsize=(10, 6))
df_comparison.plot(kind='bar', ax=ax, width=0.6, color=['gray', '#e74c3c'])
ax.axhline(0, color = 'black', linewidth=1)

# Assigns names to graph title and axes
ax.set_title('Asset Allocation: Long Only vs. Hedging', fontsize=14, fontweight='bold')
ax.set_ylabel('Weight', fontsize=12)
ax.set_xlabel('Assets', fontsize=12)

# Formats axes and legend
plt.xticks(rotation=0)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(title='Strategy')

# Puts a lable on every bar(container) with padding, and adds margin after the loop
for container in ax.containers:
    ax.bar_label(container, fmt='{:.0%}', padding=3, fontsize=10)
ax.margins(y=0.1)

# Displays the graph
plt.tight_layout()
plt.show()

# -- Efficient Frontier Graph

# --- PART 2: THE MATH ENGINE (CALCULATING THE CURVE) ---

def get_frontier(mean_returns, cov_matrix, bounds):
    """
    Calculates the Efficient Frontier (Risk vs Return) for a given set of constraints.
    Returns two lists: volatility (x-axis) and returns (y-axis).
    """
    # Create 50 target returns between the minimum and maximum possible returns
    # We multiply max by 1.3 to see what happens if we push for higher returns
    target_returns = np.linspace(mean_returns.min(), mean_returns.max() * 1.3, 200)
    
    frontier_volatility = []
    
    for target in target_returns:
        # We need two constraints for the loop:
        # 1. Weights must sum to 1
        # 2. Portfolio Return must equal our current 'target'
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target})
        
        # Run the minimization for this specific target
        result = minimize(lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))), 
                          initial_guess, 
                          method='SLSQP', 
                          bounds=bounds, 
                          constraints=cons)
        
        if result.success:
            frontier_volatility.append(result.fun)
        else:
            # If the solver fails (e.g., return is impossible), add a dummy value
            frontier_volatility.append(np.nan) 
            
    return frontier_volatility, target_returns

# Calculate the data for the curves
# This creates the 'vol_con' and 'ret_con' variables you were missing!
print("Calculating Long-Only Curve...")
vol_con, ret_con = get_frontier(mean_returns, cov_matrix, bounds_constrained)

print("Calculating Hedged Curve...")
vol_uncon, ret_uncon = get_frontier(mean_returns, cov_matrix, bounds_unconstrained)


# --- PART 3: THE VISUALIZATION (PLOTTING THE CURVE) ---

# 1. Setup the Canvas
fig, ax = plt.subplots(figsize=(10, 6))

# 2. Plot the Lines (The Boundaries)
ax.plot(vol_con, ret_con, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.8, label='Long Only Boundary', zorder=2)
ax.plot(vol_uncon, ret_uncon, color="#281210", linewidth=2, label='Unconstrained (Hedged) Boundary', zorder=1)

# 3. Calculate Coordinates for the "Stars" (Max Sharpe Points)
# We calculate the exact x (risk) and y (return) for our optimal portfolios
opt_vol_con = np.sqrt(np.dot(results_con.T, np.dot(cov_matrix, results_con)))
opt_ret_con = np.dot(results_con, mean_returns)

opt_vol_uncon = np.sqrt(np.dot(results_uncon.T, np.dot(cov_matrix, results_uncon)))
opt_ret_uncon = np.dot(results_uncon, mean_returns)

# plotes the maximum points
# zorder=5 forces the stars to sit ON TOP of the lines
ax.scatter(opt_vol_con, opt_ret_con, color='#2c3e50', s=80, linewidth=1.5, zorder=3, label=f'Max Sharpe (Long): {max_sharpe_con:.2f}')
ax.scatter(opt_vol_uncon, opt_ret_uncon, color='#e74c3c', s=120, linewidth=1, zorder=3, label=f'Max Sharpe (Hedged): {max_sharpe_uncon:.2f}')

# 5. Professional Formatting
ax.set_title('Efficient Frontier Expansion: The Value of Hedging', fontsize=14, fontweight='bold')
ax.set_xlabel('Risk (Volatility)', fontsize=12)
ax.set_ylabel('Expected Return', fontsize=12)

# Format axes as percentages
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Add Legend
ax.legend(loc='upper left', frameon=True, framealpha=1, shadow=True)

plt.tight_layout()
plt.show()