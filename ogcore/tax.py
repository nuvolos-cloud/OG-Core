"""
------------------------------------------------------------------------
Functions for taxes in the steady state and along the transition path.
------------------------------------------------------------------------
"""

# Packages
import numpy as np
from numba import jit
from ogcore import utils, pensions
from ogcore.txfunc import get_tax_rates

"""
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
"""

@jit(nopython=True)
def ETR_wealth(b, h_wealth, m_wealth, p_wealth):
    r"""
    Calculates the effective tax rate on wealth.

    .. math::
        \tau_{t}^{etr,w} =
        p^{w}\left(\frac{h^{w}b_{j,s,t}}{h^{w}b_{j,s,t} + m^{w}}\right)

    Args:
        b (Numpy array): savings
        h_wealth (scalar): parameter of wealth tax function
        p_wealth (scalar): parameter of wealth tax function
        m_wealth (scalar): parameter of wealth tax function

    Returns:
        tau_w (Numpy array): effective tax rate on wealth, size = SxJ

    """
    tau_w = (p_wealth * h_wealth * b) / (h_wealth * b + m_wealth)

    return tau_w

@jit(nopython=True)
def MTR_wealth(b, h_wealth, m_wealth, p_wealth):
    r"""
    Calculates the marginal tax rate on wealth from the wealth tax.

    .. math::
        \tau^{mtrw}_{t} = \tau^{etr,w}_{t}
        \left[2 - \left(\frac{h^w b_{j,s,t}}{h^w b_{j,s,t} + m^w}\right)\right]

    Args:
        b (Numpy array): savings
        h_wealth (scalar): parameter of wealth tax function
        p_wealth (scalar): parameter of wealth tax function
        m_wealth (scalar): parameter of wealth tax function

    Returns:
        tau_prime (Numpy array): marginal tax rate on wealth, size = SxJ

    """
    tau_prime = ETR_wealth(b, h_wealth, m_wealth, p_wealth) * 2 - (
        (h_wealth**2 * p_wealth * b**2) / ((b * h_wealth + m_wealth) ** 2)
    )

    return tau_prime

@jit(nopython=True)
def ETR_income(
    r: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    factor: float,
    e: np.ndarray,
    etr_params: list,
    labor_noncompliance_rate: np.ndarray,
    capital_noncompliance_rate: np.ndarray,
    tax_func_type: str,
) -> np.ndarray:
    """
    Calculates effective personal income tax rate.

    Args:
        r (array_like): real interest rate
        w (array_like): real wage rate
        b (Numpy array): savings
        n (Numpy array): labor supply
        factor (scalar): scaling factor converting model units to
            dollars
        e (Numpy array): effective labor units
        etr_params (list): list of effective tax rate function
            parameters or nonparametric function
        labor_noncompliance_rate (Numpy array): income tax noncompliance rate for labor income
        capital_noncompliance_rate (Numpy array): income tax noncompliance rate for capital income
        tax_func_type (str): type of tax function

    Returns:
        tau (Numpy array): effective tax rate on total income

    """
    X = (w * e * n) * factor
    Y = (r * b) * factor
    noncompliance_rate = (
        (X * labor_noncompliance_rate) + (Y * capital_noncompliance_rate)
    ) / (X + Y)

    tau = get_tax_rates(
        etr_params, X, Y, None, tax_func_type, "etr", for_estimation=False
    )

    return tau * (1 - noncompliance_rate)


@jit(nopython=True)
def MTR_income(
    r: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    factor: float,
    mtr_capital: bool,
    e: np.ndarray,
    etr_params: list,
    mtr_params: list,
    noncompliance_rate: np.ndarray,
    tax_func_type: str,
    analytical_mtrs: bool,
) -> np.ndarray:
    r"""
    Generates the marginal tax rate on labor income for households.

    Args:
        r (array_like): real interest rate
        w (array_like): real wage rate
        b (Numpy array): savings
        n (Numpy array): labor supply
        factor (scalar): scaling factor converting model units to
            dollars
        mtr_capital (bool): whether to compute the marginal tax rate on
            capital income or labor income
        e (Numpy array): effective labor units
        etr_params (list): list of effective tax rate function
            parameters or nonparametric function
        mtr_params (list): list of marginal tax rate function
            parameters or nonparametric function
        noncompliance_rate (Numpy array): income tax noncompliance rate
        tax_func_type (str): type of tax function
        analytical_mtrs (bool): whether to use analytical marginal tax rates

    Returns:
        tau (Numpy array): marginal tax rate on income source

    """
    X = (w * e * n) * factor
    Y = (r * b) * factor

    if analytical_mtrs:
        tau = get_tax_rates(
            etr_params,
            X,
            Y,
            None,
            tax_func_type,
            "mtr",
            analytical_mtrs,
            mtr_capital,
            for_estimation=False,
        )
    else:
        tau = get_tax_rates(
            mtr_params,
            X,
            Y,
            None,
            tax_func_type,
            "mtr",
            analytical_mtrs,
            mtr_capital,
            for_estimation=False,
        )

    return tau * (1 - noncompliance_rate)

@jit(nopython=True)
def get_biz_tax(
    w: np.ndarray,
    Y: np.ndarray,
    L: np.ndarray,
    K: np.ndarray,
    p_m: np.ndarray,
    delta_tau: np.ndarray,
    tau_b: np.ndarray,
    inv_tax_credit: np.ndarray,
    delta: float,
    T: int,
    M: int,
    method: str,
    m: int = None,
) -> np.ndarray:
    r"""
    Finds total business income tax revenue.

    .. math::
        R_{t}^{b} = \sum_{m=1}^{M}\tau_{m,t}^{b}(Y_{m,t} - w_{t}L_{m,t}) -
        \tau_{m,t}^{b}\delta_{m,t}^{\tau}K_{m,t}^{\tau} - \tau^{inv}_{m,t}I_{m,t}
    Args:
        w (array_like): real wage rate
        Y (array_like): aggregate output for each industry
        L (array_like): aggregate labor demand for each industry
        K (array_like): aggregate capital demand for each industry
        p_m (array_like): output prices
        delta_tau (array_like): depreciation rate for tax purposes
        tau_b (array_like): business income tax rate
        inv_tax_credit (array_like): investment tax credit rate
        delta (scalar): depreciation rate
        T (int): number of time periods
        M (int): number of industries
        method (str): 'SS' for steady state or 'TPI' for transition path
        m (int or None): index for production industry, if None, then
            compute for all industries

    Returns:
        business_revenue (array_like): aggregate business tax revenue

    """
    if m is not None:
        if method == "SS":
            delta_tau_m = delta_tau[-1, m]
            tau_b_m = tau_b[-1, m]
            tau_inv_m = inv_tax_credit[-1, m]
            price = p_m[m]
            Inv = delta * K[m]  # compute gross investment
        else:
            delta_tau_m = delta_tau[:T, m].reshape(T)
            tau_b_m = tau_b[:T, m].reshape(T)
            tau_inv_m = inv_tax_credit[:T, m].reshape(T)
            price = p_m[:T, m].reshape(T)
            w = w.reshape(T)
            Inv = delta * K
    else:
        if method == "SS":
            delta_tau_m = delta_tau[-1, :]
            tau_b_m = tau_b[-1, :]
            tau_inv_m = inv_tax_credit[-1, :]
            price = p_m
            Inv = delta * K
        else:
            delta_tau_m = delta_tau[:T, :].reshape(T, M)
            tau_b_m = tau_b[:T, :].reshape(T, M)
            tau_inv_m = inv_tax_credit[:T, :].reshape(T, M)
            price = p_m[:T, :].reshape(T, M)
            w = w.reshape(T, 1)
            Inv = delta * K

    business_revenue = (
        tau_b_m * (price * Y - w * L) - tau_b_m * delta_tau_m * K - tau_inv_m * Inv
    )
    return business_revenue

@jit(nopython=True)
def net_taxes(
    r: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    bq: np.ndarray,
    factor: float,
    tr: np.ndarray,
    ubi: np.ndarray,
    theta: np.ndarray,
    t: int,
    j: int,
    shift: bool,
    method: str,
    e: np.ndarray,
    etr_params: list,
    tau_payroll: np.ndarray,
    h_wealth: np.ndarray,
    m_wealth: np.ndarray,
    p_wealth: np.ndarray,
    lambdas: np.ndarray,
    tau_bq: np.ndarray,
    pension_params: dict,
    tax_func_type: str,
    labor_income_tax_noncompliance_rate: np.ndarray,
    capital_income_tax_noncompliance_rate: np.ndarray,
) -> np.ndarray:
    """
    Calculate net taxes paid for each household.

    Args:
        r (array_like): real interest rate
        w (array_like): real wage rate
        b (Numpy array): savings
        n (Numpy array): labor supply
        bq (Numpy array): bequests received
        factor (scalar): scaling factor converting model units to
            dollars
        tr (Numpy array): government transfers to the household
        ubi (Numpy array): universal basic income payments to households
        theta (Numpy array): social security replacement rate value for
            lifetime income group j
        t (int): time period
        j (int): index of lifetime income group
        shift (bool): whether computing for periods 0--s or 1--(s+1),
            =True for 1--(s+1)
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        e (Numpy array): effective labor units
        etr_params (list): list of effective tax rate function parameters
        tau_payroll (Numpy array): payroll tax rates
        h_wealth (Numpy array): parameter of wealth tax function
        m_wealth (Numpy array): parameter of wealth tax function
        p_wealth (Numpy array): parameter of wealth tax function
        lambdas (Numpy array): bequest distribution parameters
        tau_bq (array_like): bequest tax rate
        pension_params (dict): dictionary of pension parameters
        tax_func_type (str): type of tax function
        labor_income_tax_noncompliance_rate (Numpy array): noncompliance rate for labor income tax
        capital_income_tax_noncompliance_rate (Numpy array): noncompliance rate for capital income tax

    Returns:
        net_tax (Numpy array): net taxes paid for each household

    """
    T_I = income_tax_liab(
        r,
        w,
        b,
        n,
        factor,
        t,
        j,
        method,
        e,
        etr_params,
        tau_payroll,
        tax_func_type,
        labor_income_tax_noncompliance_rate,
        capital_income_tax_noncompliance_rate,
    )
    # TODO: replace "1" with Y in the args below when want NDC functions
    pension = pensions.pension_amount(
        r, w, n, 1, theta, t, j, shift, method, e, factor, pension_params
    )
    T_BQ = bequest_tax_liab(r, b, bq, t, j, method, lambdas, tau_bq)
    T_W = wealth_tax_liab(r, b, t, j, method, h_wealth, m_wealth, p_wealth)

    net_tax = T_I - pension + T_BQ + T_W - tr - ubi

    return net_tax

@jit(nopython=True)
def income_tax_liab(
    r: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    factor: float,
    t: int,
    j: int,
    method: str,
    e: np.ndarray,
    etr_params: list,
    tau_payroll: np.ndarray,
    tax_func_type: str,
    labor_income_tax_noncompliance_rate: np.ndarray,
    capital_income_tax_noncompliance_rate: np.ndarray,
) -> np.ndarray:
    """
    Calculate income and payroll tax liability for each household

    Args:
        r (array_like): real interest rate
        w (array_like): real wage rate
        b (Numpy array): savings
        n (Numpy array): labor supply
        factor (scalar): scaling factor converting model units to
            dollars
        t (int): time period
        j (int): index of lifetime income group
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        e (Numpy array): effective labor units
        etr_params (list): effective tax rate function parameters
        tau_payroll (Numpy array): payroll tax rates
        tax_func_type (str): type of tax function
        labor_income_tax_noncompliance_rate (Numpy array): noncompliance rate for labor income tax
        capital_income_tax_noncompliance_rate (Numpy array): noncompliance rate for capital income tax

    Returns:
        T_I (Numpy array): total income and payroll taxes paid for each
            household

    """
    if j is not None:
        if method == "TPI":
            if b.ndim == 2:
                r = r.reshape(r.shape[0], 1)
                w = w.reshape(w.shape[0], 1)
            labor_income_tax_compliance_rate = labor_income_tax_noncompliance_rate[t, j]
            capital_income_tax_compliance_rate = capital_income_tax_noncompliance_rate[t, j]
        else:
            labor_income_tax_compliance_rate = labor_income_tax_noncompliance_rate[-1, j]
            capital_income_tax_compliance_rate = capital_income_tax_noncompliance_rate[-1, j]
    else:
        if method == "TPI":
            r = utils.to_timepath_shape(r)
            w = utils.to_timepath_shape(w)
            labor_income_tax_compliance_rate = labor_income_tax_noncompliance_rate[t, :]
            capital_income_tax_compliance_rate = capital_income_tax_noncompliance_rate[t, :]
        else:
            labor_income_tax_compliance_rate = labor_income_tax_noncompliance_rate[-1, :]
            capital_income_tax_compliance_rate = capital_income_tax_noncompliance_rate[-1, :]
    income = r * b + w * e * n
    labor_income = w * e * n
    T_I = (
        ETR_income(
            r,
            w,
            b,
            n,
            factor,
            e,
            etr_params,
            labor_income_tax_compliance_rate,
            capital_income_tax_compliance_rate,
            tax_func_type,
        )
        * income
    )
    if method == "SS":
        T_P = tau_payroll[-1] * labor_income
    elif method == "TPI":
        length = w.shape[0]
        if len(b.shape) == 1:
            T_P = tau_payroll[t : t + length] * labor_income
        elif len(b.shape) == 2:
            T_P = tau_payroll[t : t + length].reshape(length, 1) * labor_income
        else:
            T_P = tau_payroll[t : t + length].reshape(length, 1, 1) * labor_income
    elif method == "TPI_scalar":
        T_P = tau_payroll[0] * labor_income

    income_payroll_tax_liab = T_I + T_P

    return income_payroll_tax_liab

@jit(nopython=True)
def wealth_tax_liab(
    r: np.ndarray,
    b: np.ndarray,
    t: int,
    j: int,
    method: str,
    h_wealth: np.ndarray,
    m_wealth: np.ndarray,
    p_wealth: np.ndarray,
) -> np.ndarray:
    """
    Calculate wealth tax liability for each household.

    Args:
        r (array_like): real interest rate
        b (Numpy array): savings
        t (int): time period
        j (int): index of lifetime income group
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        h_wealth (array_like): parameter of wealth tax function
        m_wealth (array_like): parameter of wealth tax function
        p_wealth (array_like): parameter of wealth tax function

    Returns:
        T_W (Numpy array): wealth tax liability for each household

    """
    if j is not None:
        if method == "TPI":
            if b.ndim == 2:
                r = r.reshape(r.shape[0], 1)
    else:
        if method == "TPI":
            r = utils.to_timepath_shape(r)

    if method == "SS":
        T_W = ETR_wealth(b, h_wealth[-1], m_wealth[-1], p_wealth[-1]) * b
    elif method == "TPI":
        length = r.shape[0]
        if len(b.shape) == 1:
            T_W = (
                ETR_wealth(
                    b,
                    h_wealth[t : t + length],
                    m_wealth[t : t + length],
                    p_wealth[t : t + length],
                )
                * b
            )
        elif len(b.shape) == 2:
            T_W = (
                ETR_wealth(
                    b,
                    h_wealth[t : t + length],
                    m_wealth[t : t + length],
                    p_wealth[t : t + length],
                )
                * b
            )
        else:
            T_W = (
                ETR_wealth(
                    b,
                    h_wealth[t : t + length].reshape(length, 1, 1),
                    m_wealth[t : t + length].reshape(length, 1, 1),
                    p_wealth[t : t + length].reshape(length, 1, 1),
                )
                * b
            )
    elif method == "TPI_scalar":
        T_W = ETR_wealth(b, h_wealth[0], m_wealth[0], p_wealth[0]) * b

    return T_W


@jit(nopython=True)
def bequest_tax_liab(
    r: np.ndarray,
    b: np.ndarray,
    bq: np.ndarray,
    t: int,
    j: int,
    method: str,
    lambdas: np.ndarray,
    tau_bq: np.ndarray,
) -> np.ndarray:
    """
    Calculate liability due from taxes on bequests for each household.

    Args:
        r (array_like): real interest rate
        b (Numpy array): savings
        bq (Numpy array): bequests received
        t (int): time period
        j (int): index of lifetime income group
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'
        lambdas (array_like): bequest distribution parameters
        tau_bq (array_like): bequest tax rate

    Returns:
        T_BQ (Numpy array): bequest tax liability for each household

    """
    if j is not None:
        lambdas_j = lambdas[j]
        if method == "TPI":
            if b.ndim == 2:
                r = r.reshape(r.shape[0], 1)
    else:
        lambdas_j = np.transpose(lambdas)
        if method == "TPI":
            r = utils.to_timepath_shape(r)

    if method == "SS":
        T_BQ = tau_bq[-1] * bq
    elif method == "TPI":
        length = r.shape[0]
        if len(b.shape) == 1:
            T_BQ = tau_bq[t : t + length] * bq
        elif len(b.shape) == 2:
            T_BQ = tau_bq[t : t + length].reshape(length, 1) * bq / lambdas_j
        else:
            T_BQ = tau_bq[t : t + length].reshape(length, 1, 1) * bq
    elif method == "TPI_scalar":
        T_BQ = tau_bq[0] * bq

    return T_BQ
