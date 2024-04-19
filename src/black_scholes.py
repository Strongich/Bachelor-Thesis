import numpy as np
from scipy.stats import norm


def runge_kutta(f, t0, y0, tf, h):
    t = t0
    y = y0
    while t < tf:
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t = t + h
    return y


def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def american_put_option_price(S, K, T_days, r, sigma, N=50):
    T_years = T_days / 365.0

    def f(t, y):
        return max(
            K - S * np.exp(r * t - 0.5 * sigma**2 * t + sigma * np.sqrt(t) * y), 0
        )

    V = np.zeros(N + 1)
    V[-1] = max(K - S, 0)

    dt = T_years / N
    for i in range(N - 1, -1, -1):
        t = i * dt
        V[i] = max(K - S, runge_kutta(f, t, V[i + 1], T_years, dt))

    return V[0]


def american_call_option_price(S, K, T_days, r, sigma, N=50):
    T_years = T_days / 365.0

    def f(t, y):
        return np.maximum(
            S * np.exp(r * t - 0.5 * sigma**2 * t + sigma * np.sqrt(t) * y) - K, 0
        )

    V = np.zeros(N + 1)
    V[-1] = np.maximum(S - K, 0)

    dt = T_years / N
    for i in range(N - 1, -1, -1):
        t = i * dt
        V[i] = np.maximum(S - K, runge_kutta(f, t, V[i + 1], T_years, dt))

    return V[0]

