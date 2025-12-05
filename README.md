## READ ME 
# Structure 
- fred_data folder: houses all of the fred_csv files
- project.py : the python code for the project. (so far I read and joined all of the files so that they are in one dataframe)
- environment.yml : the environment settings

# Environment
- Run the following in your terminal: conda env create -f environment.yml

- make sure the environment is active: conda activate (the environment name). 
    - (This is so the libraries we use are the same so we dont run into individual problems with dependencies).

# Project overview:
1) **Data Ingestion and Consolidation:**
    * We read multiple asset price CSV files using the pandas library.
    * Each file was transformed into a data frame using a `for` loop.
    * The individual data frames were then joined into a single master data frame, indexed by the **Date** and with the respective **country's asset prices** as the columns.

2) **Return and Risk Metrics:**
    * We calculated the **mean returns** for each asset (country).
    * We calculated the **covariance matrix** ($\mathbf{\Sigma}$). The covariance matrix is essential as it accounts for the relationship (covariance) between each country, including the variance (covariance of an asset with itself) on the diagonal.

3) **Defining Constants:**
    * We created variables for constants used in the optimization: the `risk_free_rate` (set to 0 for simplicity) and the `length` of the assets (N).

---

## Objective Function Definition

4) **Sharpe Ratio Function:**
    * We created a function that returns the Sharpe Ratio ($SR$) for any given set of weights ($\mathbf{w}$), defined as: $SR = \frac{\text{Portfolio Return}}{\text{Portfolio Volatility}}$ (since the risk-free rate is 0).

    * **Portfolio Return** ($E[R_p]$) is calculated as the weighted average of the mean returns:
        $$E[R_p] = \sum_{i=1}^{N} w_i E[R_i] = \mathbf{w}^T \mathbf{\mu}$$
    * **Portfolio Volatility** ($\sigma_p$) is calculated using the covariance matrix. The portfolio variance ($\sigma_p^2$) is the sum of the cross-products of each pair of weighted covariances.
        $$\sigma_p^2 = \sum_{i=1}^{N} \sum_{j=1}^{N} w_i w_j \sigma_{ij}$$
    * **Linear Algebra Format (used in Python/NumPy):** For computational efficiency, the variance is calculated using matrix multiplication:
        $$\sigma_p^2 = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$
        Where $\mathbf{\Sigma}$ is the covariance matrix.

---

## Optimization Setup and Execution

5) **Defining the Optimization Problem:**
    We used `scipy.optimize.minimize` to find the set of optimal weights ($\mathbf{w}^*$).

* **Objective Function:** The solver minimizes the **negative Sharpe ratio** ($f(\mathbf{w}) = -SR(\mathbf{w})$) to indirectly achieve the goal of **maximizing** the positive Sharpe ratio.

* **Constraints:**
    * **Full Investment Constraint (Equality):** The sum of all asset weights must equal 1 (100%).
        $$\sum_{i=1}^{N} w_i = 1$$
    * **Non-Negativity Constraint (Boundary):** No short-selling is allowed, meaning all weights must be between 0 and 1.
        $$0 \le w_i \le 1 \quad \text{for all assets } i$$

6) **Executing the Optimization and Analyzing Results:**
    * We called the `scipy.optimize.minimize` function, supplying the objective function, initial weights (typically equal weighting), and the defined constraints, often using the **SLSQP** solver.
    * The solver returned the **optimal portfolio weights** ($\mathbf{w}^*$) that maximize the Sharpe ratio.
    * We then calculated the final metrics for the **Maximum Sharpe Ratio Portfolio**: the Sharpe Ratio, Expected Annual Return, and Expected Annual Volatility.

---