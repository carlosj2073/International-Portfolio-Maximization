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
1) First we read the csv files using the pandas library and then we transformed each file into a data frame using a for loop. We then joined the dataframes into one with date as the index and country's asset prices as the columns. 

2) We then calculated the mean returns for each country and the covariance matrix. (the covariance matrix accounts for the covariance between each country, including itself: the variance for each country).

3) We then created variables for constants: risk_free_rate = 0 and the length of the assets.

4) We then made a function that returns the sharpe ratio: portfolio return / portfolio volatility. (remember the risk free rate is 0). The function has the following constraints:
    - Portfolio return = weight * mean for each asset
    - volatility is equal to the sum of the crossproduct of each pair of weighted covariance. (Deeper explanation: There is no function or model to just find the covariance between more than one asset or variable. So, the way we find the portfolio variance is by finding the covariance between each pair. So, think of it like finding the covariance portfolio for two of the assets as a cross product and then doing that for all combinations of asssets including itself then adding them up.)
    $$\sigma_p^2 = \sum_{i=1}^{2} \sum_{j=1}^{2} w_i w_j \sigma_{ij}$$

    NOT NECESSARY FOR EXAM BUT FURTHER EXPLANATION
    Further explanation: we use the linear algebra format of the above function because of the way that python and most programs are able to store dataframes. Dataframes, and tables  can exist as matrixes and they are treated as such using numpy, a library that is used for manipulating dataframes using linear algebra. The linear algebra version is as follows: 
    $$\sigma_p^2 = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$
    Where sigma is the covariance matrix: $$\mathbf{\Sigma} = \begin{bmatrix} 
    \sigma_{1}^2 & \sigma_{12} & \dots \\ 
    \sigma_{21} & \sigma_{2}^2 & \dots \\ 
    \vdots & \vdots & \ddots 
    \end{bmatrix}$$
    and w is the weight dataframe or matrix and wT is the transposed version of the weight matrix (transposing is a rule in linear algebra when cross multiplying two matrices).

5)