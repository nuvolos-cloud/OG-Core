"""
------------------------------------------------------------------------
Household functions.
------------------------------------------------------------------------
"""

# Packages
import numpy as np
from numba import jit
from ogcore import tax, utils
from ogcore.utils import to_timepath_shape
from ogcore.utils import tile_numba_1d, tile_numba_2d, tile_numba_3d

"""
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
"""


@jit(nopython=True)
def marg_ut_cons(c, sigma):
    r"""
    Compute the marginal utility of consumption.

    .. math::
        MU_{c} = c^{-\sigma}

    Args:
        c (array_like): household consumption
        sigma (scalar): coefficient of relative risk aversion

    Returns:
        output (array_like): marginal utility of consumption

    """
    if np.ndim(c) == 0:
        c = np.array([c])
    epsilon = 0.003
    cvec_cnstr = c < epsilon
    MU_c = np.zeros(c.shape)
    MU_c[~cvec_cnstr] = c[~cvec_cnstr] ** (-sigma)
    b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
    b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
    MU_c[cvec_cnstr] = 2 * b2 * c[cvec_cnstr] + b1
    output = MU_c
    new_shape = []
    arr = output
    for dim in arr.shape:
        if dim != 1:
            new_shape.append(dim)
    if len(new_shape) == 1:
        output =  arr.reshape(new_shape[0])
    elif len(new_shape) == 2:
        output = arr.reshape((new_shape[0], new_shape[1]))
    return output


@jit(nopython=True)
def marg_ut_labor(
    n: np.ndarray,
    chi_n: np.ndarray,
    b_ellipse: float,
    ltilde: float,
    upsilon: float
) -> np.ndarray:
    r"""
    Compute the marginal disutility of labor.

    .. math::
        MDU_{l} = \chi^n_{s}\biggl(\frac{b}{\tilde{l}}\biggr)
        \biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^{\upsilon-1}
        \Biggl[1-\biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^\upsilon
        \Biggr]^{\frac{1-\upsilon}{\upsilon}}

    Args:
        n (array_like): household labor supply
        chi_n (array_like): utility weights on disutility of labor
        b_ellipse (float): parameter for elliptical utility of labor
        ltilde (float): upper bound of household labor supply
        upsilon (float): curvature parameter for elliptical utility of labor

    Returns:
        output (array_like): marginal disutility of labor supply

    """
    nvec = n
    if np.ndim(nvec) == 0:
        nvec = np.array([nvec])
    eps_low = 0.000001
    eps_high = ltilde - 0.000001
    nvec_low = nvec < eps_low
    nvec_high = nvec > eps_high
    nvec_uncstr = np.logical_and(~nvec_low, ~nvec_high)
    MDU_n = np.zeros(nvec.shape)
    MDU_n[nvec_uncstr] = (
        (b_ellipse / ltilde)
        * ((nvec[nvec_uncstr] / ltilde) ** (upsilon - 1))
        * (
            (1 - ((nvec[nvec_uncstr] / ltilde) ** upsilon))
            ** ((1 - upsilon) / upsilon)
        )
    )
    b2 = (
        0.5
        * b_ellipse
        * (ltilde ** (-upsilon))
        * (upsilon - 1)
        * (eps_low ** (upsilon - 2))
        * (
            (1 - ((eps_low / ltilde) ** upsilon))
            ** ((1 - upsilon) / upsilon)
        )
        * (
            1
            + ((eps_low / ltilde) ** upsilon)
            * ((1 - ((eps_low / ltilde) ** upsilon)) ** (-1))
        )
    )
    b1 = (b_ellipse / ltilde) * (
        (eps_low / ltilde) ** (upsilon - 1)
    ) * (
        (1 - ((eps_low / ltilde) ** upsilon))
        ** ((1 - upsilon) / upsilon)
    ) - (
        2 * b2 * eps_low
    )
    MDU_n[nvec_low] = 2 * b2 * nvec[nvec_low] + b1
    d2 = (
        0.5
        * b_ellipse
        * (ltilde ** (-upsilon))
        * (upsilon - 1)
        * (eps_high ** (upsilon - 2))
        * (
            (1 - ((eps_high / ltilde) ** upsilon))
            ** ((1 - upsilon) / upsilon)
        )
        * (
            1
            + ((eps_high / ltilde) ** upsilon)
            * ((1 - ((eps_high / ltilde) ** upsilon)) ** (-1))
        )
    )
    d1 = (b_ellipse / ltilde) * (
        (eps_high / ltilde) ** (upsilon - 1)
    ) * (
        (1 - ((eps_high / ltilde) ** upsilon))
        ** ((1 - upsilon) / upsilon)
    ) - (
        2 * d2 * eps_high
    )
    MDU_n[nvec_high] = 2 * d2 * nvec[nvec_high] + d1
    new_shape = []
    arr = chi_n
    for dim in arr.shape:
        if dim != 1:
            new_shape.append(dim)
    if len(new_shape) == 1:
        chi_n =  arr.reshape(new_shape[0])
    elif len(new_shape) == 2:
        chi_n = arr.reshape((new_shape[0], new_shape[1]))

    output = MDU_n * chi_n
    new_shape = []
    arr = output
    for dim in arr.shape:
        if dim != 1:
            new_shape.append(dim)
    if len(new_shape) == 1:
        output =  arr.reshape(new_shape[0])
    elif len(new_shape) == 2:
        output = arr.reshape((new_shape[0], new_shape[1]))
    return output
    

@jit(nopython=True)
def get_bq_ss(
    BQ: np.ndarray,
    j: int,
    use_zeta: bool,
    zeta: np.ndarray,
    lambdas: np.ndarray,
    omega_SS: np.ndarray,
    S: int,
    J: int
) -> np.ndarray:
    r"""
    Calculate bequests to each household in the steady state.

    .. math::
        \hat{bq}_{j,s} = \zeta_{j,s}
        \frac{\hat{BQ}}{\lambda_{j}\hat{\omega}_{s}} \quad\forall j,s

    Args:
        BQ (array_like): aggregate bequests
        j (int): index of lifetime ability group
        use_zeta (bool): flag indicating whether to use zeta
        zeta (array_like): bequest distribution parameters
        lambdas (array_like): ability weights
        omega_SS (array_like): steady state population weights
        S (int): number of periods in a lifetime
        J (int): number of ability groups

    Returns:
        bq (array_like): bequests received by each household

    """
    if use_zeta:
        if j is not None:
            bq = (zeta[:, j] * BQ) / (lambdas[j] * omega_SS)
        else:
            bq = (zeta * BQ) / (
                lambdas.reshape((1, J)) * omega_SS.reshape((S, 1))
            )
    else:
        print("not use_zeta")
        if j is not None:
            bq = tile_numba_1d(BQ[j], S) / lambdas[j]
        else:
            lambdas_squeezed = lambdas
            new_shape = []
            for dim in lambdas_squeezed.shape:
                if dim != 1:
                    new_shape.append(dim)
            # Assuming that new shape is 1D
            lambdas_squeezed =  lambdas_squeezed.reshape(new_shape[0])
            BQ_per = BQ / lambdas_squeezed
            bq_reshaped = np.reshape(BQ_per, (1, J))
            print(f"bq_reshaped shape: {bq_reshaped.shape}")
            bq = tile_numba_2d(bq_reshaped, (S, 1))
    return bq


@jit(nopython=True)
def get_bq_tpi(
    BQ: np.ndarray,
    j: int,
    use_zeta: bool,
    zeta: np.ndarray,
    lambdas: np.ndarray,
    omega: np.ndarray,
    S: int,
    J: int
) -> np.ndarray:
    r"""
    Calculate bequests to each household along the transition path.

    .. math::
        \hat{bq}_{j,s,t} = \zeta_{j,s}
        \frac{\hat{BQ}_{t}}{\lambda_{j}\hat{\omega}_{s,t}} \quad\forall j,s,t

    Args:
        BQ (array_like): aggregate bequests
        j (int): index of lifetime ability group
        use_zeta (bool): flag indicating whether to use zeta
        zeta (array_like): bequest distribution parameters
        lambdas (array_like): ability weights
        omega (array_like): population weights
        S (int): number of periods in a lifetime
        J (int): number of ability groups

    Returns:
        bq (array_like): bequests received by each household

    """
    if use_zeta:
        if j is not None:
            len_T = BQ.shape[0]
            bq = (
                np.reshape(np.ascontiguousarray(zeta[:, j]), (1, S)) * np.reshape(np.ascontiguousarray(BQ), (len_T, 1))
            ) / (lambdas[j] * omega[:len_T, :])
        else:
            len_T = BQ.shape[0]
            bq = (
                np.reshape(np.ascontiguousarray(zeta), (1, S, J)) * to_timepath_shape(BQ)
            ) / (
                np.reshape(np.ascontiguousarray(lambdas), (1, 1, J))
                * np.reshape(np.ascontiguousarray(omega[:len_T, :]), (len_T, S, 1))
            )
    else:
        if j is not None:
            len_T = BQ.shape[0]
            bq = tile_numba_2d(
                np.reshape(np.ascontiguousarray(BQ[:, j] / lambdas[j]), (len_T, 1)), (1, S)
            )
        else:
            len_T = BQ.shape[0]
            BQ_per = BQ / np.reshape(np.ascontiguousarray(lambdas), (1, J))
            bq = tile_numba_3d(np.reshape(np.ascontiguousarray(BQ_per), (len_T, 1, J)), (1, S, 1))
    return bq


@jit(nopython=True)
def get_tr_ss(
    TR: np.ndarray,
    j: int,
    eta: np.ndarray,
    lambdas: np.ndarray,
    omega_SS: np.ndarray,
    S: int,
    J: int
) -> np.ndarray:
    r"""
    Calculate transfers to each household in the steady state.

    .. math::
        \hat{tr}_{j,s} = \eta_{j,s}
        \frac{\hat{TR}}{\lambda_{j}\hat{\omega}_{s}} \quad\forall j,s

    Args:
        TR (array_like): aggregate transfers
        j (int): index of lifetime ability group
        eta (array_like): transfer distribution parameters
        lambdas (array_like): ability weights
        omega_SS (array_like): steady state population weights
        S (int): number of periods in a lifetime
        J (int): number of ability groups

    Returns:
        tr (array_like): transfers received by household

    """
    if j is not None:
        tr = (eta[-1, :, j] * TR) / (lambdas[j] * omega_SS)
    else:
        tr = (eta[-1, :, :] * TR) / (
            lambdas.reshape((1, J)) * omega_SS.reshape((S, 1))
        )
    return tr


@jit(nopython=True)
def get_tr_tpi(
    TR: np.ndarray,
    j: int,
    eta: np.ndarray,
    lambdas: np.ndarray,
    omega: np.ndarray,
    S: int,
    J: int
) -> np.ndarray:
    r"""
    Calculate transfers to each household along the transition path.

    .. math::
        \hat{tr}_{j,s,t} = \eta_{j,s}
        \frac{\hat{TR}_{t}}{\lambda_{j}\hat{\omega}_{s,t}} \quad\forall j,s,t

    Args:
        TR (array_like): aggregate transfers
        j (int): index of lifetime ability group
        eta (array_like): transfer distribution parameters
        lambdas (array_like): ability weights
        omega (array_like): population weights
        S (int): number of periods in a lifetime
        J (int): number of ability groups

    Returns:
        tr (array_like): transfers received by household

    """
    len_T = TR.shape[0]
    if j is not None:
        tr = (eta[:len_T, :, j] * TR.reshape((len_T, 1))) / (
            lambdas[j] * omega[:len_T, :]
        )
    else:
        tr = (eta[:len_T, :, :] * utils.to_timepath_shape(TR)) / (
            lambdas.reshape((1, 1, J))
            * omega[:len_T, :].reshape((len_T, S, 1))
        )
    return tr


@jit(nopython=True)
def get_rm_ss(
    RM: np.ndarray,
    j: int,
    eta_RM: np.ndarray,
    lambdas: np.ndarray,
    omega_SS: np.ndarray,
    S: int,
    J: int
) -> np.ndarray:
    r"""
    Calculate remittances to each household in the steady state.

    .. math::
        \hat{rm}_{j,s} = \eta_{RM,j,s}
        \frac{\hat{RM}}{\lambda_{j}\hat{\omega}_{s}} \quad\forall j,s

    Args:
        RM (array_like): aggregate remittances
        j (int): index of lifetime ability group
        eta_RM (array_like): remittance distribution parameters
        lambdas (array_like): ability weights
        omega_SS (array_like): steady state population weights
        S (int): number of periods in a lifetime
        J (int): number of ability groups

    Returns:
        rm (array_like): remittances received by household

    """
    if j is not None:
        rm = (eta_RM[-1, :, j] * RM) / (lambdas[j] * omega_SS)
    else:
        rm = (eta_RM[-1, :, :] * RM) / (
            lambdas.reshape((1, J)) * omega_SS.reshape((S, 1))
        )
    return rm


@jit(nopython=True)
def get_rm_tpi(
    RM: np.ndarray,
    j: int,
    eta_RM: np.ndarray,
    lambdas: np.ndarray,
    omega: np.ndarray,
    S: int,
    J: int
) -> np.ndarray:
    r"""
    Calculate remittances to each household along the transition path.

    .. math::
        \hat{rm}_{j,s,t} = \eta_{RM,j,s,t}
        \frac{\hat{RM}_{t}}{\lambda_{j}\hat{\omega}_{s,t}} \quad\forall j,s,t

    Args:
        RM (array_like): aggregate remittances
        j (int): index of lifetime ability group
        eta_RM (array_like): remittance distribution parameters
        lambdas (array_like): ability weights
        omega (array_like): population weights
        S (int): number of periods in a lifetime
        J (int): number of ability groups

    Returns:
        rm (array_like): remittances received by household

    """
    len_T = RM.shape[0]
    if j is not None:
        rm = (eta_RM[:len_T, :, j] * RM.reshape((len_T, 1))) / (
            lambdas[j] * omega[:len_T, :]
        )
    else:
        rm = (eta_RM[:len_T, :, :] * utils.to_timepath_shape(RM)) / (
            lambdas.reshape((1, 1, J))
            * omega[:len_T, :].reshape((len_T, S, 1))
        )
    return rm


@jit(nopython=True)
def get_cons(
    r_p: np.ndarray,
    w: np.ndarray,
    p_tilde: np.ndarray,
    b: np.ndarray,
    b_splus1: np.ndarray,
    n: np.ndarray,
    bq: np.ndarray,
    rm: np.ndarray,
    net_tax: np.ndarray,
    e: np.ndarray,
    g_y: float
) -> np.ndarray:
    r"""
    Calculate household composite consumption.

    .. math::
        \hat{c}_{j,s,t} &= \biggl[(1 + r_{p,t})\hat{b}_{j,s,t}
        + \hat{w}_t e_{j,s}n_{j,s,t} + \hat{bq}_{j,s,t} + \hat{rm}_{j,s,t}
        + \hat{tr}_{j,s,t} + \hat{ubi}_{j,s,t} + \hat{pension}_{j,s,t}
        - \hat{tax}_{j,s,t} \\
        &\qquad - \sum_{i=1}^I\left(1 + \tau^c_{i,t}\right)p_{i,t}\hat{c}_{min,i}
        - e^{g_y}\hat{b}_{j,s+1,t+1}\biggr] / p_t \\
        &\qquad\qquad\forall j,t \quad\text{and}\quad E+1\leq s\leq E+S
        \quad\text{where}\quad \hat{b}_{j,E+1,t}=0

    Args:
        r_p (array_like): the real interest rate
        w (array_like): the real wage rate
        p_tilde (array_like): the ratio of real GDP to nominal GDP
        b (Numpy array): household savings
        b_splus1 (Numpy array): household savings one period ahead
        n (Numpy array): household labor supply
        bq (Numpy array): household bequests received
        rm (Numpy array): household remittances received
        net_tax (Numpy array): household net taxes paid
        e (Numpy array): effective labor units
        g_y (float): growth rate of technology

    Returns:
        cons (Numpy array): household consumption

    """
    cons = (
        (1 + r_p) * b
        + w * e * n
        + bq
        + rm
        - net_tax
        - b_splus1 * np.exp(g_y)
    ) / p_tilde

    return cons


@jit(nopython=True)
def get_ci(c_s, p_i, p_tilde, tau_c, alpha_c, method="SS"):
    r"""
    Compute consumption of good i given amount of composite consumption
    and prices.

    .. math::
        c_{i,j,s,t} = \frac{c_{s,j,t}}{\alpha_{i,j}p_{i,j}}

    Args:
        c_s (array_like): composite consumption
        p_i (array_like): prices for consumption good i
        p_tilde (array_like): composite good price
        tau_c (array_like): consumption tax rate
        alpha_c (array_like): consumption share parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        c_si (array_like): consumption of good i
    """
    if method == "SS":
        I = alpha_c.shape[0]
        S = c_s.shape[0]
        J = c_s.shape[1]
        tau_c = tau_c.reshape(I, 1, 1)
        alpha_c = alpha_c.reshape(I, 1, 1)
        p_tilde.reshape(1, 1, 1)
        p_i = p_i.reshape(I, 1, 1)
        c_s = c_s.reshape(1, S, J)
        c_si = alpha_c * (((1 + tau_c) * p_i) / p_tilde) ** (-1) * c_s
    else:  # Time path case
        I = alpha_c.shape[0]
        T = p_i.shape[0]
        S = c_s.shape[1]
        J = c_s.shape[2]
        tau_c = tau_c.reshape(T, I, 1, 1)
        alpha_c = alpha_c.reshape(1, I, 1, 1)
        p_tilde = p_tilde.reshape(T, 1, 1, 1)
        p_i = p_i.reshape(T, I, 1, 1)
        c_s = c_s.reshape(T, 1, S, J)
        c_si = alpha_c * (((1 + tau_c) * p_i) / p_tilde) ** (-1) * c_s
    return c_si


@jit(nopython=True)
def FOC_savings(
    r: np.ndarray,
    w: np.ndarray,
    p_tilde: np.ndarray,
    b: np.ndarray,
    b_splus1: np.ndarray,
    n: np.ndarray,
    bq: np.ndarray,
    rm: np.ndarray,
    factor: float,
    tr: np.ndarray,
    ubi: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    etr_params: list,
    mtry_params: list,
    t: int,
    j: int,
    sigma: float,
    g_y: float,
    chi_b: np.ndarray,
    beta: np.ndarray,
    capital_income_tax_noncompliance_rate: np.ndarray,
    labor_income_tax_noncompliance_rate: np.ndarray,
    e: np.ndarray,
    h_wealth: np.ndarray,
    m_wealth: np.ndarray,
    p_wealth: np.ndarray,
    tau_payroll: np.ndarray,
    lambdas: np.ndarray,
    tau_bq: np.ndarray,
    pension_params: np.ndarray,
    method: str,
) -> np.ndarray:
    r"""
    Computes Euler errors for the FOC for savings in the steady state.
    This function is usually looped through over J, so it does one
    lifetime income group at a time.

    .. math::
        \frac{c_{j,s,t}^{-\sigma}}{\tilde{p}_{t}} = e^{-\sigma g_y}
        \biggl[\chi^b_j\rho_s(b_{j,s+1,t+1})^{-\sigma} +
        \beta_j\bigl(1 - \rho_s\bigr)\Bigl(\frac{1 + r_{t+1}
        \bigl[1 - \tau^{mtry}_{s+1,t+1}\bigr]}{\tilde{p}_{t+1}}\Bigr)
        (c_{j,s+1,t+1})^{-\sigma}\biggr]

    Args:
        r (np.ndarray): the real interest rate
        w (np.ndarray): the real wage rate
        p_tilde (np.ndarray): composite good price
        b (np.ndarray): household savings
        b_splus1 (np.ndarray): household savings one period ahead
        n (np.ndarray): household labor supply
        bq (np.ndarray): household bequests received
        rm (np.ndarray): household remittances received
        factor (float): scaling factor converting model units to dollars
        tr (np.ndarray): government transfers to household
        ubi (np.ndarray): universal basic income payment
        theta (np.ndarray): social security replacement rate for each lifetime income group
        rho (np.ndarray): mortality rates
        etr_params (list): parameters of the effective tax rate functions
        mtry_params (list): parameters of the marginal tax rate on capital income functions
        t (int): model period
        j (int): index of ability type
        sigma (float): coefficient of relative risk aversion
        g_y (float): growth rate of technology
        chi_b (np.ndarray): utility weight on bequests
        beta (np.ndarray): discount factor
        capital_income_tax_noncompliance_rate (np.ndarray): noncompliance rate for capital income tax
        labor_income_tax_noncompliance_rate (np.ndarray): noncompliance rate for labor income tax
        e (np.ndarray): effective labor units
        h_wealth (np.ndarray): wealth tax parameter
        m_wealth (np.ndarray): wealth tax parameter
        p_wealth (np.ndarray): wealth tax parameter
        tau_payroll (np.ndarray): payroll tax rate
        lambdas (np.ndarray): ability weights
        tau_bq (np.ndarray): bequest tax rate
        pension_params (np.ndarray): pension parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        np.ndarray: Euler error from FOC for savings
    """
    if j is not None:
        chi_b = chi_b[j]
        beta = beta[j]
        if method == "SS":
            tax_noncompliance = capital_income_tax_noncompliance_rate[-1, j]
            new_shape = []
            arr = e[-1, :, j]
            for dim in arr.shape:
                if dim != 1:
                    new_shape.append(dim)
            if len(new_shape) == 1:
                e =  arr.reshape(new_shape[0])
            elif len(new_shape) == 2:
                e = arr.reshape((new_shape[0], new_shape[1]))
            else:
                e = arr
        elif method == "TPI_scalar":
            tax_noncompliance = capital_income_tax_noncompliance_rate[0, j]
            new_shape = []
            arr = e[0, :, j]
            for dim in arr.shape:
                if dim != 1:
                    new_shape.append(dim)
            if len(new_shape) == 1:
                e =  arr.reshape(new_shape[0])
            elif len(new_shape) == 2:
                e = arr.reshape((new_shape[0], new_shape[1]))
            else:
                e = arr
        else:
            length = r.shape[0]
            tax_noncompliance = capital_income_tax_noncompliance_rate[
                t : t + length, j
            ]
            e_long = np.concatenate(
                (
                    e,
                    tile_numba_3d(e[-1, :, :].reshape(1, e.shape[1], e.shape[2]), (e.shape[1], 1, 1)),
                ),
                axis=0,
            )
            e = np.diag(e_long[t : t + e.shape[1], :, j], max(e.shape[1] - length, 0))
    else:
        chi_b = chi_b
        beta = beta
        if method == "SS":
            tax_noncompliance = capital_income_tax_noncompliance_rate[-1, :]
            new_shape = []
            arr = e[-1, :, :]
            for dim in arr.shape:
                if dim != 1:
                    new_shape.append(dim)
            if len(new_shape) == 1:
                e =  arr.reshape(new_shape[0])
            elif len(new_shape) == 2:
                e = arr.reshape((new_shape[0], new_shape[1]))
            else:
                e = arr
        elif method == "TPI_scalar":
            tax_noncompliance = capital_income_tax_noncompliance_rate[0, :]
            new_shape = []
            arr = e[0, :, :]
            for dim in arr.shape:
                if dim != 1:
                    new_shape.append(dim)
            if len(new_shape) == 1:
                e =  arr.reshape(new_shape[0])
            elif len(new_shape) == 2:
                e = arr.reshape((new_shape[0], new_shape[1]))
            else:
                e = arr
        else:
            length = r.shape[0]
            tax_noncompliance = capital_income_tax_noncompliance_rate[
                t : t + length, :
            ]
            e_long = np.concatenate(
                (
                    e,
                    tile_numba_3d(e[-1, :, :].reshape(1, e.shape[1], e.shape[2]), (e.shape[1], 1, 1)),
                ),
                axis=0,
            )
            e = np.diag(e_long[t : t + e.shape[1], :, :], max(e.shape[1] - length, 0))
    new_shape = []
    arr = e
    for dim in arr.shape:
        if dim != 1:
            new_shape.append(dim)
    if len(new_shape) == 1:
        e =  arr.reshape(new_shape[0])
    elif len(new_shape) == 2:
        e = arr.reshape((new_shape[0], new_shape[1]))
    if method == "SS":
        h_wealth = h_wealth[-1]
        m_wealth = m_wealth[-1]
        p_wealth = p_wealth[-1]
        p_tilde = np.ones_like(rho[-1, :]) * p_tilde
    elif method == "TPI_scalar":
        h_wealth = h_wealth[0]
        m_wealth = m_wealth[0]
        p_wealth = p_wealth[0]
    else:
        h_wealth = h_wealth[t]
        m_wealth = m_wealth[t]
        p_wealth = p_wealth[t]
    taxes = tax.net_taxes(
        r,
        w,
        b,
        n,
        bq,
        factor,
        tr,
        ubi,
        theta,
        t,
        j,
        False,
        method,
        e,
        etr_params,
        tau_payroll,
        h_wealth,
        m_wealth,
        p_wealth,
        lambdas,
        tau_bq,
        pension_params,
        "DEP",
        labor_income_tax_noncompliance_rate,
        capital_income_tax_noncompliance_rate,
    )
    cons = get_cons(r, w, p_tilde, b, b_splus1, n, bq, rm, taxes, e, g_y)
    deriv = (
        (1 + r)
        - (
            r
            * tax.MTR_income(
                r,
                w,
                b,
                n,
                factor,
                True,
                e,
                etr_params,
                mtry_params,
                tax_noncompliance,
                "DEP",
                False,
            )
        )
        - tax.MTR_wealth(b, h_wealth, m_wealth, p_wealth)
    )
    savings_ut = (
        rho * np.exp(-sigma * g_y) * chi_b * b_splus1 ** (-sigma)
    )
    euler_error = np.zeros_like(n)
    if n.shape[0] > 1:
        euler_error[:-1] = (
            marg_ut_cons(cons[:-1], sigma) * (1 / p_tilde[:-1])
            - beta
            * (1 - rho[:-1])
            * deriv[1:]
            * marg_ut_cons(cons[1:], sigma)
            * (1 / p_tilde[1:])
            * np.exp(-sigma * g_y)
            - savings_ut[:-1]
        )
        euler_error[-1] = (
            marg_ut_cons(cons[-1], sigma) * (1 / p_tilde[-1])
            - savings_ut[-1]
        )
    else:
        euler_error[-1] = (
            marg_ut_cons(cons[-1], sigma) * (1 / p_tilde[-1])
            - savings_ut[-1]
        )

    return euler_error


@jit(nopython=True)
def FOC_labor(
    r: np.ndarray,
    w: np.ndarray,
    p_tilde: np.ndarray,
    b: np.ndarray,
    b_splus1: np.ndarray,
    n: np.ndarray,
    bq: np.ndarray,
    rm: np.ndarray,
    factor: float,
    tr: np.ndarray,
    ubi: np.ndarray,
    theta: np.ndarray,
    chi_n: np.ndarray,
    etr_params: list,
    mtrx_params: list,
    t: int,
    j: int,
    sigma: float,
    g_y: float,
    chi_b: np.ndarray,
    beta: np.ndarray,
    capital_income_tax_noncompliance_rate: np.ndarray,
    labor_income_tax_noncompliance_rate: np.ndarray,
    e: np.ndarray,
    h_wealth: np.ndarray,
    m_wealth: np.ndarray,
    p_wealth: np.ndarray,
    tau_payroll: np.ndarray,
    lambdas: np.ndarray,
    tau_bq: np.ndarray,
    pension_params: np.ndarray,
    b_ellipse: float,
    ltilde: float,
    upsilon: float,
    method: str,
) -> np.ndarray:
    r"""
    Computes errors for the FOC for labor supply in the steady
    state.  This function is usually looped through over J, so it does
    one lifetime income group at a time.

    .. math::
        w_t e_{j,s}\bigl(1 - \tau^{mtrx}_{s,t}\bigr)
       \frac{(c_{j,s,t})^{-\sigma}}{ \tilde{p}_{t}} = \chi^n_{s}
        \biggl(\frac{b}{\tilde{l}}\biggr)\biggl(\frac{n_{j,s,t}}
        {\tilde{l}}\biggr)^{\upsilon-1}\Biggl[1 -
        \biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^\upsilon\Biggr]
        ^{\frac{1-\upsilon}{\upsilon}}

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        p_tilde (array_like): composite good price
        b (Numpy array): household savings
        b_splus1 (Numpy array): household savings one period ahead
        n (Numpy array): household labor supply
        bq (Numpy array): household bequests received
        rm (Numpy array): household bequests received
        factor (scalar): scaling factor converting model units to dollars
        tr (Numpy array): government transfers to household
        ubi (Numpy array): universal basic income payment
        theta (Numpy array): social security replacement rate for each
            lifetime income group
        chi_n (Numpy array): utility weight on the disutility of labor
            supply
        e (Numpy array): effective labor units
        etr_params (list): parameters of the effective tax rate
            functions
        mtrx_params (list): parameters of the marginal tax rate
            on labor income functions
        t (int): model period
        j (int): index of ability type
        sigma (float): coefficient of relative risk aversion
        g_y (float): growth rate of technology
        chi_b (Numpy array): utility weight on bequests
        beta (Numpy array): discount factor
        capital_income_tax_noncompliance_rate (Numpy array): noncompliance rate for capital income tax
        labor_income_tax_noncompliance_rate (Numpy array): noncompliance rate for labor income tax
        h_wealth (Numpy array): wealth tax parameter
        m_wealth (Numpy array): wealth tax parameter
        p_wealth (Numpy array): wealth tax parameter
        tau_payroll (Numpy array): payroll tax rate
        lambdas (Numpy array): ability weights
        tau_bq (Numpy array): bequest tax rate
        pension_params (Numpy array): pension parameters
        b_ellipse (float): parameter for elliptical utility of labor
        ltilde (float): upper bound of household labor supply
        upsilon (float): curvature parameter for elliptical utility of labor
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        FOC_error (Numpy array): error from FOC for labor supply

    """
    if method == "SS":
        tau_payroll = tau_payroll[-1]
    elif method == "TPI_scalar":  # for 1st donut ring only
        tau_payroll = tau_payroll[0]
    else:
        length = r.shape[0]
        tau_payroll = tau_payroll[t : t + length]
    if j is not None:
        if method == "SS":
            tax_noncompliance = capital_income_tax_noncompliance_rate[-1, j]
            new_shape = []
            arr = e[-1, :, j]
            for dim in arr.shape:
                if dim != 1:
                    new_shape.append(dim)
            if len(new_shape) == 1:
                e =  arr.reshape(new_shape[0])
            elif len(new_shape) == 2:
                e = arr.reshape((new_shape[0], new_shape[1]))
            else:
                e = arr
        elif method == "TPI_scalar":
            tax_noncompliance = capital_income_tax_noncompliance_rate[0, j]
            new_shape = []
            arr = e[0, -1, j]
            for dim in arr.shape:
                if dim != 1:
                    new_shape.append(dim)
            if len(new_shape) == 1:
                e =  arr.reshape(new_shape[0])
            elif len(new_shape) == 2:
                e = arr.reshape((new_shape[0], new_shape[1]))
            else:
                e = arr
        else:
            tax_noncompliance = capital_income_tax_noncompliance_rate[
                t : t + length, j
            ]
            e_long = np.concatenate(
                (
                    e,
                    tile_numba_3d(e[-1, :, :].reshape(1, e.shape[1], e.shape[2]), (e.shape[1], 1, 1)),
                ),
                axis=0,
            )
            e = np.diag(e_long[t : t + e.shape[1], :, j], max(e.shape[1] - length, 0))
    else:
        if method == "SS":
            tax_noncompliance = capital_income_tax_noncompliance_rate[-1, :]
            new_shape = []
            arr = e[-1, :, :]
            for dim in arr.shape:
                if dim != 1:
                    new_shape.append(dim)
            if len(new_shape) == 1:
                e =  arr.reshape(new_shape[0])
            elif len(new_shape) == 2:
                e = arr.reshape((new_shape[0], new_shape[1]))
            else:
                e = arr
        elif method == "TPI_scalar":
            tax_noncompliance = capital_income_tax_noncompliance_rate[0, :]
            new_shape = []
            arr = e[0, -1, :]
            for dim in arr.shape:
                if dim != 1:
                    new_shape.append(dim)
            if len(new_shape) == 1:
                e =  arr.reshape(new_shape[0])
            elif len(new_shape) == 2:
                e = arr.reshape((new_shape[0], new_shape[1]))
            else:
                e = arr
        else:
            tax_noncompliance = capital_income_tax_noncompliance_rate[
                t : t + length, :
            ]
            e_long = np.concatenate(
                (
                    e,
                    tile_numba_3d(e[-1, :, :].reshape(1, e.shape[1], e.shape[2]), (e.shape[1], 1, 1)),
                ),
                axis=0,
            )
            e = np.diag(e_long[t : t + e.shape[1], :, :], max(e.shape[1] - length, 0))
    if method == "SS":
        tau_payroll = tau_payroll[-1]
    elif method == "TPI_scalar":  # for 1st donut ring only
        tau_payroll = tau_payroll[0]
    else:
        length = r.shape[0]
        tau_payroll = tau_payroll[t : t + length]
    if method == "TPI":
        if b.ndim == 2:
            r = r.reshape(r.shape[0], 1)
            w = w.reshape(w.shape[0], 1)
            tau_payroll = tau_payroll.reshape(tau_payroll.shape[0], 1)

    taxes = tax.net_taxes(
        r,
        w,
        b,
        n,
        bq,
        factor,
        tr,
        ubi,
        theta,
        t,
        j,
        False,
        method,
        e,
        etr_params,
        tau_payroll,
        h_wealth,
        m_wealth,
        p_wealth,
        lambdas,
        tau_bq,
        pension_params,
        "DEP",
        labor_income_tax_noncompliance_rate,
        capital_income_tax_noncompliance_rate,
    )
    cons = get_cons(r, w, p_tilde, b, b_splus1, n, bq, rm, taxes, e, g_y)
    deriv = (
        1
        - tau_payroll
        - tax.MTR_income(
            r,
            w,
            b,
            n,
            factor,
            False,
            e,
            etr_params,
            mtrx_params,
            tax_noncompliance,
            "DEP",
            False,
        )
    )
    FOC_error = marg_ut_cons(cons, sigma) * (
        1 / p_tilde
    ) * w * deriv * e - marg_ut_labor(n, chi_n, b_ellipse, ltilde, upsilon)

    return FOC_error


@jit(nopython=True)
def get_y(
    r_p: np.ndarray,
    w: np.ndarray,
    b_s: np.ndarray,
    n: np.ndarray,
    e: np.ndarray,
    method: str
) -> np.ndarray:
    r"""
    Compute household income before taxes.

    .. math::
        y_{j,s,t} = r_{p,t}b_{j,s,t} + w_{t}e_{j,s}n_{j,s,t}

    Args:
        r_p (array_like): real interest rate on the household portfolio
        w (array_like): real wage rate
        b_s (Numpy array): household savings coming into the period
        n (Numpy array): household labor supply
        e (Numpy array): effective labor units
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        y (Numpy array): household income before taxes
    """
    if method == "SS":
        new_shape = []
        arr = e[-1, :, :]
        for dim in arr.shape:
            if dim != 1:
                new_shape.append(dim)
        if len(new_shape) == 1:
            e =  arr.reshape(new_shape[0])
        elif len(new_shape) == 2:
            e = arr.reshape((new_shape[0], new_shape[1]))
        else:
            e = arr
    y = r_p * b_s + w * e * n

    return y


def constraint_checker_SS(bssmat, nssmat, cssmat, ltilde):
    """
    Checks constraints on consumption, savings, and labor supply in the
    steady state.

    Args:
        bssmat (Numpy array): steady state distribution of capital
        nssmat (Numpy array): steady state distribution of labor
        cssmat (Numpy array): steady state distribution of consumption
        ltilde (scalar): upper bound of household labor supply

    Returns:
        None

    Raises:
        Warnings: if constraints are violated, warnings printed

    """
    print("Checking constraints on capital, labor, and consumption.")

    if (bssmat < 0).any():
        print("\tWARNING: There is negative capital stock")
    flag2 = False
    if (nssmat < 0).any():
        print(
            "\tWARNING: Labor supply violates nonnegativity ", "constraints."
        )
        flag2 = True
    if (nssmat > ltilde).any():
        print("\tWARNING: Labor supply violates the ltilde constraint.")
        flag2 = True
    if flag2 is False:
        print(
            "\tThere were no violations of the constraints on labor",
            " supply.",
        )
    if (cssmat < 0).any():
        print("\tWARNING: Consumption violates nonnegativity", " constraints.")
    else:
        print(
            "\tThere were no violations of the constraints on", " consumption."
        )


def constraint_checker_TPI(b_dist, n_dist, c_dist, t, ltilde):
    """
    Checks constraints on consumption, savings, and labor supply along
    the transition path. Does this for each period t separately.

    Args:
        b_dist (Numpy array): distribution of capital at time t
        n_dist (Numpy array): distribution of labor at time t
        c_dist (Numpy array): distribution of consumption at time t
        t (int): time period
        ltilde (scalar): upper bound of household labor supply

    Returns:
        None

    Raises:
        Warnings: if constraints are violated, warnings printed

    """
    if (b_dist <= 0).any():
        print(
            "\tWARNING: Aggregate capital is less than or equal to ",
            "zero in period %.f." % t,
        )
    if (n_dist < 0).any():
        print(
            "\tWARNING: Labor supply violates nonnegativity",
            " constraints in period %.f." % t,
        )
    if (n_dist > ltilde).any():
        print(
            "\tWARNING: Labor suppy violates the ltilde constraint",
            " in period %.f." % t,
        )
    if (c_dist < 0).any():
        print(
            "\tWARNING: Consumption violates nonnegativity",
            " constraints in period %.f." % t,
        )
