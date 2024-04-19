import numpy as np

# from scipy.stats import skew, kurtosis

import pandas as pd
from black_scholes import american_put_option_price, american_call_option_price

# global parameters for data
T = [0, 365 * 3]
S = [40, 200]
S_MAX = 400  # very unlikely S price
K = 120
RISK_FREE = 0.03
SIGMA = 0.4


class AmericanCallData:
    """
    Generating data for American call options
    """

    def __init__(self, t_range, S_range, K, r=0.03, sigma=0.4):
        self.t_range = t_range  # time to expiration in years
        self.S_range = S_range  # stock price range
        self.K = K  # strike price
        self.r = r  # risk-free rate
        self.sigma = sigma  # volatility

    def _gs(self, x):
        return np.fmax(x - self.K, 0)  # boundary IVP condition

    def initialize_data(self, N_ivp, N_bvp, N_col, is_call=False):
        # colocation points (not boundary)
        X = np.concatenate(
            [
                np.random.uniform(
                    low=self.t_range[0], high=self.t_range[1], size=(N_col, 1)
                ),
                np.random.uniform(*self.S_range, size=(N_col, 1)),
            ],
            axis=1,
        )
        y = (
            np.ones(shape=(N_col, 1)) * 999
        )  # a trick to use F-D method only for this points
        X_col = np.concatenate([X, y], axis=1)

        # IVP points
        X = np.concatenate(
            [
                np.zeros((N_ivp, 1))
                * self.t_range[1],  # all at expiry time (time to exp = 0)
                np.random.uniform(*self.S_range, size=(N_ivp, 1)),
            ],
            axis=1,
        )
        y = self._gs(X[:, 1]).reshape(-1, 1)
        X_ivp = np.concatenate([X, y], axis=1)

        # bvp data
        T = self.t_range[-1]
        # BVP1: price at lowest, payoff lowest
        X = np.concatenate(
            [
                np.random.uniform(
                    low=self.t_range[0],
                    high=self.t_range[1],
                    size=(int(3 * N_bvp / 4), 1),
                ),
                self.S_range[0] * np.ones((int(3 * N_bvp / 4), 1)),
            ],
            axis=1,
        )
        y = np.zeros((int(3 * N_bvp / 4), 1))
        X_bvp1 = np.concatenate([X, y], axis=1)
        # BVP2: price at highest, payoff highest
        X = np.concatenate(
            [
                np.random.uniform(
                    low=self.t_range[0],
                    high=self.t_range[1],
                    size=(int(N_bvp / 4), 1),
                ),
                self.S_range[-1] * np.ones((int(N_bvp / 4), 1)),
            ],
            axis=1,
        )
        y = (self.S_range[-1] - self.K * np.exp(-self.r * X[:, 0])).reshape(-1, 1)
        X_bvp2 = np.concatenate([X, y], axis=1)
        # make one matrix for all points
        X = np.concatenate([X_col, X_ivp, X_bvp1, X_bvp2], axis=0)
        # generate greeks and OpenInterest
        OpenInterest = np.random.exponential(scale=231.5, size=(X.shape[0], 1))
        Delta = np.random.exponential(scale=0.3, size=(X.shape[0], 1))
        Gamma = np.random.exponential(scale=0.11, size=(X.shape[0], 1))
        Theta = -np.random.exponential(scale=7, size=(X.shape[0], 1))
        Vega = np.random.exponential(scale=6.6, size=(X.shape[0], 1))
        X = np.concatenate([X, OpenInterest, Delta, Gamma, Theta, Vega], axis=1)
        return X


def get_prices_call(X, option=None, generated=True):
    """
    Compute price of American option using Finite-Diff algorithm
    and add approximation of B-S model using Runge-Kutta (details in orig. paper) \\
    Parameters
    ----------
    X: matrix-like, time and stock prices
    Note: if X is pandas dataframe, use X.to_numpy() as parameter
    option: grid, result of ImplicitAmBer func (optional) -> only for generated data
    generated: bool, if True, it will use K, sigma and r same as generation
    """
    print(X.shape)
    Target = []
    if generated:
        for x in X:
            # generated data format is diff from real one
            price = american_call_option_price(
                x[1],
                K,
                x[0],
                RISK_FREE,
                SIGMA,
            )
            Target.append(price)
            if int(x[2]) != 999:
                continue
            else:
                time = int(x[0])
                stock = int(x[1])
                price = np.array([(np.round(option[stock, time], 3))])[0].astype(
                    np.float16
                )
                x[2] = price
                x[0] = time
    else:
        # real data, transformation explained in ../basic_data/data_analysis.ipynb
        # source: https://historicaloptiondata.com/sample-files/
        for x in X:
            # columns:
            # UnderlyingPrice	Type	Strike	OpenInterest	IV	Delta	Gamma	Theta	Vega	Time	TargetPrice
            price = american_call_option_price(
                S=x[0], K=x[2], T_days=x[9], r=RISK_FREE, sigma=x[4]
            )
            Target.append(price)

    Target = np.array(Target).reshape(-1, 1)
    X = np.concatenate([X, Target], axis=1)
    return X


class AmericanPutData:
    """
    Generating data for American Put options
    """

    def __init__(self, t_range, S_range, K, r=0.03, sigma=0.4):
        self.t_range = t_range  # time to expiration in years
        self.S_range = S_range  # stock price range
        self.K = K  # strike price
        self.r = r  # risk-free rate
        self.sigma = sigma  # volatility

    def _gs(self, x):
        return np.fmax(self.K - x, 0)  # boundary IVP condition

    def initialize_data(self, N_ivp, N_bvp, N_col, is_call=False):
        # colocation points (not boundary)
        X = np.concatenate(
            [
                np.random.uniform(
                    low=self.t_range[0], high=self.t_range[1], size=(N_col, 1)
                ),
                np.random.uniform(*self.S_range, size=(N_col, 1)),
            ],
            axis=1,
        )
        y = (
            np.ones(shape=(N_col, 1)) * 999
        )  # a trick to use F-D method only for this points
        X_col = np.concatenate([X, y], axis=1)

        # IVP points
        X = np.concatenate(
            [
                np.zeros((N_ivp, 1)) * self.t_range[1],  # all at expiry time
                np.random.uniform(*self.S_range, size=(N_ivp, 1)),
            ],
            axis=1,
        )
        y = self._gs(X[:, 1]).reshape(-1, 1)
        X_ivp = np.concatenate([X, y], axis=1)

        # bvp data
        # BVP1: price at lowest, payoff highest
        X = np.concatenate(
            [
                np.random.uniform(
                    low=self.t_range[0],
                    high=self.t_range[1],
                    size=(int(N_bvp / 4), 1),
                ),
                self.S_range[0] * np.ones((int(N_bvp / 4), 1)),
            ],
            axis=1,
        )
        y = self.K * np.ones((int(N_bvp / 4), 1))
        X_bvp1 = np.concatenate([X, y], axis=1)
        # BVP2: price at highest, payoff lowest
        X = np.concatenate(
            [
                np.random.uniform(
                    low=self.t_range[0],
                    high=self.t_range[1],
                    size=(int(3 * N_bvp / 4), 1),
                ),
                self.S_range[-1] * np.ones((int(3 * N_bvp / 4), 1)),
            ],
            axis=1,
        )
        y = np.zeros((int(3 * N_bvp / 4), 1))
        X_bvp2 = np.concatenate([X, y], axis=1)
        # make one matrix for all points
        X = np.concatenate([X_col, X_ivp, X_bvp1, X_bvp2], axis=0)
        # generate greeks and OpenInterest
        OpenInterest = np.random.exponential(scale=280, size=(X.shape[0], 1))
        Delta = np.random.uniform(low=-1, high=0, size=(X.shape[0], 1))
        Gamma = np.random.exponential(scale=0.022, size=(X.shape[0], 1))
        Theta = -np.random.exponential(scale=7.21, size=(X.shape[0], 1))
        Vega = np.random.exponential(scale=9.5, size=(X.shape[0], 1))
        X = np.concatenate([X, OpenInterest, Delta, Gamma, Theta, Vega], axis=1)
        return X


def get_prices_put(X, option=None, generated=True):
    """
    Compute price of American option using Finite-Diff algorithm
    and add approximation of B-S model using Runge-Kutta (details in orig. paper) \\
    Parameters
    ----------
    X: matrix-like, time and stock prices
    Note: if X is pandas dataframe, use X.to_numpy() as parameter
    option: grid, result of ImplicitAmBer func (optional) -> only for generated data
    generated: bool, if True, it will use K, sigma and r same as generation
    """
    print(X.shape)
    Target = []
    if generated:
        for x in X:
            # generated data format is diff from real one
            price = american_put_option_price(
                x[1],
                K,
                x[0],
                RISK_FREE,
                SIGMA,
            )
            Target.append(price)
            if int(x[2]) != 999:
                continue
            else:
                time = int(x[0])
                stock = int(x[1])
                price = np.array([(np.round(option[stock, time], 3))])[0].astype(
                    np.float16
                )
                x[2] = price
                x[0] = time
    else:
        # real data, transformation explained in ../basic_data/data_analysis.ipynb
        # source: https://historicaloptiondata.com/sample-files/
        for x in X:
            # columns:
            # UnderlyingPrice	Type	Strike	OpenInterest	IV	Delta	Gamma	Theta	Vega	Time	TargetPrice
            price = american_put_option_price(
                S=x[0], K=x[2], T_days=x[9], r=RISK_FREE, sigma=x[4]
            )
            Target.append(price)

    Target = np.array(Target).reshape(-1, 1)
    X = np.concatenate([X, Target], axis=1)
    return X
