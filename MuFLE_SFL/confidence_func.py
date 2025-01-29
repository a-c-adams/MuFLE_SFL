import numpy as np
from scipy import signal

def Hess_matrix(Spline_int, Spline_lt, Time,  Y_pm, Fit_hist, H_pm):
    """ A function that returns matrix to simplify the hessian equations
    :param Spline_int: Intensity splines coefficient
    :param Spline_lt: Lifetime splines coefficient
    :param Time: Time Histogram
    :param Y_pm: Observed Histogram
    :param Fit_hist: Fitted Histogram
    :param H_pm: Measured IRF Histogram
    :return: Tau and gamma matrix, + matrix to simplfy the hessian calculations
    """

    Gamma_p = np.reshape(np.tile(Spline_int, len(Time.T)), np.shape(Time), order='F')
    Tau_p = np.reshape(np.tile(Spline_lt, len(Time.T)), np.shape(Time), order='F')

    O_p_m = Y_pm - Fit_hist

    # these are three matrices that will simplify the proceeding processes
    # V is the greek lower case for Nu
    V_p_m_l = []

    for index, pixel in enumerate(Time):
        eq = np.exp(-Time / Tau_p)
        ModelConv = signal.convolve(eq[index], H_pm[index], mode='full', method='direct')
        V_p_m_l.append(ModelConv[:len(Time.T)])

    # C is the greek lower case for sigma
    C_p_m_l = []

    for index, pixel in enumerate(Time):
        eq = Time * np.exp(-Time / Tau_p)
        ModelConv = signal.convolve(eq[index], H_pm[index], mode='full', method='direct')
        C_p_m_l.append(ModelConv[:len(Time.T)])

    # Q is the greek lower case for rho
    Q_p_m_l = []

    for index, pixel in enumerate(Time):
        eq = (Time ** 2) * np.exp(-Time / Tau_p)
        ModelConv = signal.convolve(eq[index], H_pm[index], mode='full', method='direct')
        Q_p_m_l.append(ModelConv[:len(Time.T)])

    return Gamma_p, Tau_p, O_p_m, V_p_m_l, C_p_m_l, Q_p_m_l

def J_gamma_gamma(Bfunc_int, m, V_matrix):
    """ Function that calculates the second order derivative of gamma in a single exponential function
    :param Bfunc_int: Intensity B-Spliens basis function
    :param m: Time Histogram
    :param V_matrix: V_p_m_l -- histogram simplifying the calculation
    :return: The second order derivative of gamma
    """
    A = []
    B = []

    for i in Bfunc_int:
        gamma_B_fuc = np.reshape(np.tile(i, len(m.T)), np.shape(m), order='F')

        A.append(V_matrix * np.array(gamma_B_fuc))
        B.append(V_matrix * np.array(gamma_B_fuc))

    ABSum = np.tensordot(A, B, axes=([1, 2], [1, 2]))

    J_gamma_gamma = 2 * ABSum

    return J_gamma_gamma

def J_tau_gamma(Bfunc_int, Bfunc_lt, m, Tau_matrix, Gamma_matrix, C_matrix, O_matrix, V_matrix):
    """ Function that calculates the second order derivative of tau gamma in a single exponential function
    :param Bfunc_int: Intensity B-splines basis function
    :param Bfunc_lt: Lifetime B-splinesn basis function
    :param m: Time Histogram
    :param Tau_matrix: Tau matrix -- lifetime the same size as the histogram
    :param Gamma_matrix: Gamma matrix -- intensity the same size as the histogram
    :param C_matrix: C_p_m_l
    :param O_matrix: O_p_m
    :param V_matrix: V_p_m_l
    :return: The second order derivative of tau gamma
    """

    AA = []
    AB = []

    BA = []
    BB = []

    for i in Bfunc_lt:
        tau_B_fuc = np.reshape(np.tile(i, len(m.T)), np.shape(m), order='F')

        AA.append(C_matrix * (Gamma_matrix / (Tau_matrix ** 2)) * np.array(tau_B_fuc))
        BA.append(O_matrix * C_matrix * (tau_B_fuc / (Tau_matrix ** 2)))

    for i in Bfunc_int:
        gamma_B_fuc = np.reshape(np.tile(i, len(m.T)), np.shape(m), order='F')

        AB.append(V_matrix * np.array(gamma_B_fuc))
        BB.append(gamma_B_fuc)

    ASum = np.tensordot(AA, AB, axes=([1, 2], [1, 2]))
    BSum = np.tensordot(BA, BB, axes=([1, 2], [1, 2]))

    J_tau_gamma = 2 * ASum - 2 * BSum

    return J_tau_gamma

def J_gamma_b(Bfunc_int, m, V_matrix, bias):
    """ Function that calculates the second order derivative of gamma bias in a single exponential function
    :param Bfunc_int: Intensity B-splines basis function
    :param m: Time Histogram
    :param V_matrix: V_p_m_l
    :param bias: Fitted Bias
    :return: The second order derivative of gamma bias
    """

    AA = []

    for i in Bfunc_int:
        tau_B_fuc = np.reshape(np.tile(i, len(m.T)), np.shape(m), order='F')
        AA.append(np.sum((V_matrix * tau_B_fuc), axis=1))

    J_gamma_b = 2 * np.exp(bias) * AA

    return J_gamma_b

def J_tau_tau(Bfunc_lt, m, Gamma_matrix, Tau_matrix, C_matrix, Q_matrix, O_matrix):
    """Function that calculates the second order derivative of tau tau in a single exponential function
    :param Bfunc_lt: Lifetime B-splines basis function
    :param m: Time Histogram
    :param Gamma_matrix: Gamma matrix -- intensity the same size as the histogram
    :param Tau_matrix: Tau matrix -- lifetime the same size as the histogram
    :param C_matrix: C_p_m_l
    :param Q_matrix: Q_p_m_l
    :param O_matrix: O_p_m
    :return: The second order derivative of tau tau
    """
    AA = []
    AB = []

    BA = []
    BB = []

    for i in Bfunc_lt:
        tau_B_fuc = np.reshape(np.tile(i, len(m.T)), np.shape(m), order='F')

        AA.append((C_matrix * (Gamma_matrix / (Tau_matrix ** 2)) ** 2) * tau_B_fuc)
        AB.append(tau_B_fuc)

        BA.append(O_matrix * Q_matrix * (tau_B_fuc / (Tau_matrix ** 2)) * 1 / (Tau_matrix ** 2) - O_matrix * C_matrix * ((2 * tau_B_fuc) / (Tau_matrix ** 3)))
        BB.append(Gamma_matrix * tau_B_fuc)

    ASum = np.tensordot(AA, AB, axes=([1, 2], [1, 2]))
    BSum = np.tensordot(BA, BB, axes=([1, 2], [1, 2]))

    J_tau_tau = 2 * ASum - 2 * BSum

    return J_tau_tau

def J_tau_b(Bfunc_lt, m, Gamma_matrix, Tau_matrix, C_matrix, bias):
    """ Function that calculates the second order derivative of tau bias in a single exponential function
    :param Bfunc_lt: Lifetime B-splines basis function
    :param m: Time Histogram
    :param Gamma_matrix: Gamma matrix -- intensity the same size as the histogram
    :param Tau_matrix: Tau matrix -- lifetime the same size as the histogram
    :param C_matrix: C_p_m_l
    :param bias: Fitted Bias
    :return: The second order derivative of tau bias
    """
    AA = []

    for i in Bfunc_lt:
        tau_B_fuc = np.reshape(np.tile(i, len(m.T)), np.shape(m), order='F')
        AA.append(np.sum((C_matrix * (Gamma_matrix / (Tau_matrix ** 2)) * tau_B_fuc), axis=1))

    J_tau_b = 2 * np.exp(bias) * AA

    return J_tau_b

def J_b_b(bias, O_matrix):
    """Function that calculates the second order derivative of bias bias in a single exponential function
    :param bias: Fitted bias
    :param O_matrix: O_p_m
    :return: The second order derivative of bias bias
    """
    J_b_b = np.diag(- 2 * np.exp(bias) * np.sum(O_matrix, axis=1) + 2 * np.exp(bias) * np.exp(bias))

    return J_b_b


def SE_CovHess(J_gamma_gamma, J_tau_gamma, J_gamma_b, J_tau_tau, J_tau_b, J_b_b, HessLen, O_matrix):
    """ Function that pieces together the individual derivative matrix into a hessian
    :param J_gamma_gamma: The second order derivative of gamma gamma in a single exponential
    :param J_tau_gamma: The second order derivative of tau gamma in a single exponential
    :param J_gamma_b: The second order derivative of gamma bias in a single exponential
    :param J_tau_tau: The second order derivative of tau tau in a single exponential
    :param J_tau_b: The second order derivative of tau bias in a single exponential
    :param J_b_b: The second order derivative of bias bias in a single exponential
    :param HessLen: length on the hessian matrix
    :param O_matrix: O_p_m
    :return: The covariance matrix
    """
    Hess = np.zeros((HessLen, HessLen))

    Hess[:len(J_gamma_gamma), :len(J_gamma_gamma)] = J_gamma_gamma
    Hess[len(J_gamma_gamma):(len(J_gamma_gamma) + len(J_tau_gamma)), :len(J_gamma_gamma)] = J_tau_gamma
    Hess[(len(J_gamma_gamma) + len(J_tau_gamma)):HessLen, :len(J_gamma_gamma)] = J_gamma_b.T
    Hess[:len(J_gamma_b), (len(J_gamma_b) + len(J_tau_gamma)):HessLen] = J_gamma_b
    Hess[:len(J_gamma_gamma), len(J_gamma_gamma):(len(J_gamma_b) + len(J_tau_gamma))] = J_tau_gamma.T
    Hess[len(J_gamma_gamma):(len(J_gamma_gamma) + len(J_tau_tau)), len(J_gamma_gamma):(len(J_gamma_gamma) + len(J_tau_tau))] = J_tau_tau
    Hess[(len(J_gamma_gamma) + len(J_tau_b)):HessLen, len(J_gamma_gamma):(len(J_gamma_gamma) + len(J_tau_b))] = J_tau_b.T
    Hess[len(J_gamma_gamma):(len(J_gamma_gamma) + len(J_tau_b)), (len(J_gamma_gamma) + len(J_tau_b)):HessLen] = J_tau_b
    Hess[(len(J_gamma_gamma) + len(J_tau_b)):HessLen, (len(J_gamma_gamma) + len(J_tau_b)):HessLen] = J_b_b

    return np.mean(pow(O_matrix, 2)) * np.linalg.inv(Hess)

# Confidence intervals
def Confidence_Intervals(Bfunc, Cov):
    """ Function that uses the delta method to calculate the confidence intervals
    :param Bfunc: BSplines Basis function of the parameters
    :param Cov: Covariance matrix of the parameters
    :return: The confidence intervals
    """
    TotalCI = []

    for i in Bfunc.T:
        inew = i[np.newaxis, :]
        varh = inew @ Cov @ inew.T
        TotalCI.append(np.sqrt(varh))

    TotalCI = np.array(TotalCI)
    TotalCI = np.squeeze(TotalCI)

    return TotalCI
