import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------
# Display settings
# ---------------------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ---------------------------------------------------------------------
# 0. LOAD FRED DATA (Percent Change from Year Ago, Monthly, Average)
# ---------------------------------------------------------------------
asset_files = {
    'UK'    : 'fred_data/NASDAQNQGBN_PC1.csv',
    'Japan' : 'fred_data/NASDAQNQJPN_PC1.csv',
    'Canada': 'fred_data/NASDAQNQCAN_PC1.csv',
    'US'    : 'fred_data/NASDAQCOM_PC1.csv'   # market index
}

df_list = {}

for asset_name, filename in asset_files.items():
    df = pd.read_csv(
        filename,
        index_col=0,
        parse_dates=True
    )
    df.columns = [asset_name]
    df_list[asset_name] = df

# join all series on the date index
prices_df = pd.concat(df_list.values(), axis=1, join='outer')
prices_df = prices_df.dropna(how='all')

print("Joined Return Data (FRED PC1 series):")
print(prices_df.head())
print(prices_df.tail())

# FRED PC1 series are percent changes (year-over-year); convert to decimals
returns_df = prices_df / 100.0

# ---------------------------------------------------------------------
# SINGLE-INDEX MODEL WITH US AS MARKET (FOR THEORY / EXPLANATION)
# ---------------------------------------------------------------------
market_name = "US"
market_returns = returns_df[market_name]
active_assets = [a for a in returns_df.columns if a != market_name]

print("\nMarket (index) asset:", market_name)
print("Active assets:", active_assets)

alphas = {}
betas = {}
sigma2_e = {}

mu_M = market_returns.mean()
var_M = market_returns.var(ddof=1)
sigma_M = np.sqrt(var_M)

# Step 0: regress each asset on the market to get alpha, beta, residual variance
for asset in active_assets:
    Ri = returns_df[asset]

    cov_iM = np.cov(Ri, market_returns, ddof=1)[0, 1]
    beta_i = cov_iM / var_M
    alpha_i = Ri.mean() - beta_i * mu_M

    e_i = Ri - (alpha_i + beta_i * market_returns)
    sigma2_ei = e_i.var(ddof=1)

    alphas[asset] = alpha_i
    betas[asset] = beta_i
    sigma2_e[asset] = sigma2_ei

print("\nStep 0: Single-index regression results (R_i = alpha_i + beta_i R_M + e_i):")
for asset in active_assets:
    print(f"{asset:6s} | alpha = {alphas[asset]: .6f}, "
          f"beta = {betas[asset]: .4f}, sigma^2(e_i) = {sigma2_e[asset]: .6f}")

print(f"\nMarket mean E(R_M) = {mu_M:.6f},  Market variance sigma_M^2 = {var_M:.6f},  sigma_M = {sigma_M:.6f}")

# Step 1: w_i^0 = alpha_i / sigma^2(e_i)
w0_i = {asset: alphas[asset] / sigma2_e[asset] for asset in active_assets}

print("\nStep 1: Initial positions w_i^0 = alpha_i / sigma^2(e_i):")
for asset in active_assets:
    print(f"{asset:6s} | w_i^0 = {w0_i[asset]: .6f}")

# Step 2: Scale so sum_i w_i = 1 across active assets
sum_w0 = sum(w0_i.values())
w_i = {asset: w0_i[asset] / sum_w0 for asset in active_assets}

print("\nStep 2: Scaled active weights w_i (sum to 1 across active assets):")
for asset in active_assets:
    print(f"{asset:6s} | w_i = {w_i[asset]: .6f}")
print("Check sum of w_i over active assets =", sum(w_i.values()))

# Step 3: alpha_A = sum_i w_i alpha_i
alpha_A = sum(w_i[a] * alphas[a] for a in active_assets)
print(f"\nStep 3: Alpha of active portfolio A: alpha_A = {alpha_A:.6f}")

# Step 4: sigma^2(e_A) = sum_i w_i^2 sigma^2(e_i)
sigma2_eA = sum((w_i[a] ** 2) * sigma2_e[a] for a in active_assets)
sigma_eA = np.sqrt(sigma2_eA)
print(f"Step 4: Residual variance of A: sigma^2(e_A) = {sigma2_eA:.6f}, sigma(e_A) = {sigma_eA:.6f}")

# Step 5: w_A^0 = (alpha_A / sigma^2(e_A)) / (E(R_M)/sigma_M^2)
wA_0 = (alpha_A / sigma2_eA) / (mu_M / var_M)
print(f"\nStep 5: Initial position in A: w_A^0 = {wA_0:.6f}")

# Step 6: beta_A = sum_i w_i beta_i
beta_A = sum(w_i[a] * betas[a] for a in active_assets)
print(f"Step 6: Beta of active portfolio A: beta_A = {beta_A:.6f}")

# Step 7: w_A* = w_A^0 / [1 + (1 - beta_A) w_A^0]
wA_star = wA_0 / (1.0 + (1.0 - beta_A) * wA_0)
print(f"\nStep 7: Adjusted weight in active portfolio A: w_A* = {wA_star:.6f}")

# Step 8: Optimal risky portfolio weights (theoretical, allowing shorts)
wM_star = 1.0 - wA_star
final_weights_theoretical = {}
final_weights_theoretical[market_name] = wM_star
for asset in active_assets:
    final_weights_theoretical[asset] = wA_star * w_i[asset]

print("\nStep 8: Optimal risky portfolio weights (single-index model, may allow negatives):")
for asset in returns_df.columns:
    print(f"{asset:6s} | w_* = {final_weights_theoretical[asset]: .6f}")
print("Check total weight =", sum(final_weights_theoretical.values()))

# Step 9 and 10: E(R_P), sigma_P^2, Sharpe ratio S_P = E(R_P) / sigma_P
ER_P = (wM_star + wA_star * beta_A) * mu_M + wA_star * alpha_A
sigma2_P = (wM_star + wA_star * beta_A) ** 2 * var_M + (wA_star * sigma_eA) ** 2
sigma_P = np.sqrt(sigma2_P)
Sharpe_P = ER_P / sigma_P

print(f"\nStep 9: Expected return of optimal risky portfolio E(R_P) = {ER_P:.6f}")
print(f"Step 10: Variance sigma_P^2 = {sigma2_P:.6f},  sigma_P = {sigma_P:.6f}")

Sharpe_M = mu_M / sigma_M
print(f"\nSharpe ratio of optimal risky portfolio S_P = E(R_P)/sigma_P = {Sharpe_P:.6f}")
print(f"Sharpe ratio of market index S_M = {Sharpe_M:.6f}")

print(f"S_P^2 ?= S_M^2 + [alpha_A / sigma(e_A)]^2 : "
      f"{Sharpe_P**2:.6f} vs {Sharpe_M**2 + (alpha_A / sigma_eA) ** 2:.6f}")

# ---------------------------------------------------------------------
# NO-SHORT-SELLING SHARPE MAXIMIZATION (FINAL PORTFOLIO TO SUBMIT)
# ---------------------------------------------------------------------
print("\n==============================================")
print("Sharpe-maximizing portfolio with NO shorting")
print("==============================================\n")

mean_ret_opt = returns_df.mean()
cov_opt = returns_df.cov()

assets_all = list(returns_df.columns)
num_assets_all = len(assets_all)

def neg_sharpe_no_short(weights, mean_ret, cov):
    """Negative Sharpe ratio with non-negative weights and sum-to-1 constraint."""
    w = np.array(weights)
    port_ret = np.dot(w, mean_ret)
    port_vol = np.sqrt(w @ cov @ w)
    return - port_ret / port_vol

# constraint: weights sum to 1
constraints_ns = ({
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1
})

# bounds: no short-selling -> each weight between 0 and 1
bounds_ns = tuple((0.0, 1.0) for _ in range(num_assets_all))

# starting guess: equal weights
initial_guess_ns = [1.0 / num_assets_all] * num_assets_all

result_ns = minimize(
    neg_sharpe_no_short,
    initial_guess_ns,
    args=(mean_ret_opt, cov_opt),
    method="SLSQP",
    bounds=bounds_ns,
    constraints=constraints_ns
)

weights_ns = result_ns.x
max_sharpe_ns = -neg_sharpe_no_short(weights_ns, mean_ret_opt, cov_opt)

print("No-short optimal weights (sum to 1, all >= 0):")
for name, w in zip(assets_all, weights_ns):
    print(f"{name:6s} : {w:.4f}")

print("\nCheck sum of weights =", np.sum(weights_ns))
print(f"\nSharpe ratio of no-short portfolio: {max_sharpe_ns:.6f}")

