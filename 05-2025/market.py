from scipy.linalg import cholesky
import pandas as pd
import numpy as np

class MarketData:
    def __init__(self, n_stocks=500, n_industries=11, n_days=252, seed=None):
        np.random.seed(seed)
        self.n_stocks = n_stocks
        self.n_industries = n_industries
        self.n_days = n_days
        self.industry_assignment = None
        self.stock_prices = None
        self.industry_returns = None
        self.industry_corr = None

    def assign_stocks_to_industries(self, industry_weights=None):
        """Assign stocks to industries with optional weights"""
        if industry_weights is None:
            industry_weights = np.ones(self.n_industries)/self.n_industries
        self.industry_assignment = np.random.choice(
            np.arange(self.n_industries),
            size=self.n_stocks,
            p=industry_weights
        )

    def set_industry_characteristics(self, expected_returns, correlation_matrix):
        """
        Set industry-level parameters
        
        Parameters:
            expected_returns: array of shape (n_industries,) 
            correlation_matrix: array of shape (n_industries, n_industries)
        """
        self.industry_returns = np.array(expected_returns)
        self.industry_corr = np.array(correlation_matrix)
        
        # Validate inputs
        assert self.industry_returns.shape == (self.n_industries,)
        assert self.industry_corr.shape == (self.n_industries, self.n_industries)
        assert np.allclose(self.industry_corr, self.industry_corr.T), "Correlation matrix must be symmetric"
        assert np.all(np.diag(self.industry_corr) == 1), "Diagonal must be 1"
        
        # Precompute Cholesky decomposition
        self.industry_cholesky = cholesky(self.industry_corr, lower=True)

    def simulate(self, market_vol=0.15, industry_vol=0.2, idio_vol=0.4):
        """Run simulation with controlled industry characteristics"""
        dt = 1/self.n_days
        
        # Generate market factor (common to all stocks)
        market_shocks = np.random.normal(0, np.sqrt(dt), self.n_days)
        
        # Generate correlated industry shocks
        industry_innovations = np.random.normal(0, np.sqrt(dt), (self.n_days, self.n_industries))
        industry_shocks = industry_innovations @ self.industry_cholesky.T
        
        # Initialize price matrix
        self.stock_prices = np.ones((self.n_days, self.n_stocks)) * 100  # Start at $100
        
        # Generate individual stock returns
        for t in range(1, self.n_days):
            # Get each stock's industry components
            industry_components = industry_shocks[t-1, self.industry_assignment]
            industry_drifts = self.industry_returns[self.industry_assignment]/self.n_days
            
            # Generate idiosyncratic shocks
            idio_shocks = np.random.normal(0, np.sqrt(dt), self.n_stocks)
            
            # Combine all factors
            returns = (
                industry_drifts - 0.5*(industry_vol**2 + idio_vol**2 + market_vol**2)*dt +
                industry_vol * industry_components +
                market_vol * market_shocks[t-1] +
                idio_vol * idio_shocks
            )
            
            # Update prices
            self.stock_prices[t] = self.stock_prices[t-1] * np.exp(returns)
        
        return self.stock_prices

    def get_industry_performance(self):
        """Calculate realized annualized returns by industry"""
        log_returns = np.log(self.stock_prices[1:]/self.stock_prices[:-1])
        annualized_returns = np.zeros(self.n_industries)
        for k in range(self.n_industries):
            mask = (self.industry_assignment == k)
            if mask.sum() > 0:  # Avoid division by zero
                annualized_returns[k] = np.mean(log_returns[:, mask]) * self.n_days
        return annualized_returns