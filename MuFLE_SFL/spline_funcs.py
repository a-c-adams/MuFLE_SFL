import numpy as np
from scipy.interpolate import BSpline, splrep
from scipy import signal
from scipy.optimize import least_squares

def Basisfunc_Knots_IdealCoef(Wave_axis, Decay_feature, knot_num, degree):
    ''' Function that gives the B-splones basis function, coefficients and knot vector for a 2D array
    (i.e. either the interpolation of the spectral intensity or spectral lifetime)
    :param Wave_axis: X axis for the spline calculation
    :param Decay_feature: Either intensity or lifetime for the initial coefficients to be estimated through
    :param knot_num: The number of knots needed for the splines
    :param degree: Splines degree
    :return: The splines basis function, coefficients and knot vector
    '''

    internal_knots = np.arange(1, knot_num, 1) / knot_num

    # t = the internal knots needed for the task
    # k = the degree of the spline fit -- not the order

    # splrep finds the B-Spline representation of a 1-D curve y = f(X)
    tck = splrep(Wave_axis, Decay_feature, t=internal_knots, k=degree)

    Coefficients = tck[1]
    knots = tck[0]

    # the wave_axis - x is from 0.01 to 0.99, to get the correct knots these need to change
    # back to be set to 0 and 1
    knots[knots == 0.01] = 0
    knots[knots == 0.99] = 1

    # calculation of the basis function
    Bfunc = []

    Start = np.min(np.nonzero(knots)) + 1
    Stop = len(knots) + 1

    for i in np.arange(Start, Stop):
        #  basis_element returns a basis element for th knot vector
        b = BSpline.basis_element(knots[i - Start:i])
        mask = np.zeros(Wave_axis.shape)
        mask[(Wave_axis >= knots[i - Start]) & (Wave_axis <= knots[i - 1])] = 1
        # when WaveCol has a 1 at the end, the last basis function is 0 where it should be 1
        basis = b(Wave_axis) * mask
        Bfunc.append(basis)

    return Coefficients, knots, Bfunc

def SE_Spline_residuals(Params, Data, Time, IRF, BfuncInt, BfuncLt):
    """ Function that calculates the residuals of a single exponential MuFLE model, using splines to fit both
    the spectral intensity and spectral lifetime, and a single exponential decay to fit the decays at each channel
    :param Params: Starting parameters
    :param Data: Observed histogram
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param BfuncInt: Intensity basis function
    :param BfuncLt: Lifetime basis function
    :return: The residuals between the observed data and fitted data
    """

    IntCoef = Params[:len(BfuncInt)]
    LtCoef = Params[len(BfuncInt): (len(BfuncInt) + len(BfuncLt))]
    Bias = Params[(len(BfuncInt) + len(BfuncLt)):]

    Int = IntCoef @ BfuncInt
    Int = np.reshape(np.tile(Int, len(Time.T)), np.shape(Time), order='F')

    Lifetime = LtCoef @ BfuncLt
    Lifetime = np.reshape(np.tile(Lifetime, len(Time.T)), np.shape(Time), order='F')

    Model = Int * np.exp(-Time/Lifetime)

    Conv = []

    for index, pixel in enumerate(Model):
        ModelConv = signal.convolve(Model[index], IRF[index], mode='full', method='direct')
        Conv.append(ModelConv[:len(Time.T)])

    Conv = np.array(Conv)

    Bias = np.exp(Bias)[:, np.newaxis]

    Conv = Conv + Bias

    return np.ravel(Data - Conv)

def SE_LS_fittingfunc(InitialGuess, Data, Time, IRF, BfuncInt, BfuncLt):
    """ MuFlE single exponential fitting function
    :param InitialGuess: Starting parameters
    :param Data: Observed histogram
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param BfuncInt: Intensity basis function
    :param BfuncLt: Lifetime basis function
    :return: The estimated intensity, lifetime and bias values + the results message
    """

    # least_squares function uses lm gradient descent to find the local minima
    results = least_squares(fun=SE_Spline_residuals, x0=InitialGuess, args=(Data, Time, IRF, BfuncInt, BfuncLt), ftol=1e-15, method='lm', loss='linear', max_nfev=100000)
    IntCoef = results.x[:len(BfuncInt)]
    LtCoef = results.x[len(BfuncInt):len(BfuncInt) + len(BfuncLt)]
    Bias = results.x[len(BfuncInt) + len(BfuncLt):]

    return IntCoef, LtCoef,  Bias, results

def JacobFunc(Coef_matrix, Data, Time, IRF, BfuncInt, BfuncLt):

    gammaCoef = Coef_matrix[:len(BfuncInt)]
    tauCoef = Coef_matrix[len(BfuncInt):len(BfuncInt) + len(BfuncLt)]
    Bias = Coef_matrix[len(BfuncInt) + len(BfuncLt):]

    Int = np.array(gammaCoef) @ BfuncInt
    Lt = np.array(tauCoef) @ BfuncLt

    Lifetime = np.reshape(np.tile(Lt, len(Time.T)), np.shape(Time), order='F')

    Int1 = np.reshape(np.tile(Int, len(Time.T)), np.shape(Time), order='F')

    FitModel = Int1 * np.exp(-Time / Lifetime)

    FitConv = []

    for index, pixel in enumerate(FitModel):
        ModelConv = signal.convolve(FitModel[index], IRF[index], mode='full', method='direct')
        FitConv.append(ModelConv[:len(Time.T)])

    Bias1 = np.exp(Bias)[:, np.newaxis]

    FitHist = FitConv + Bias1

    m = Time
    O = Data - FitHist

    Jac_gamma = []

    for func in BfuncInt:

        gamma_B_fuc = np.reshape(np.tile(func, len(m.T)), np.shape(m), order='F')

        gamma_exp = gamma_B_fuc * np.exp(-m / Lifetime)

        conv = []

        for index, pixel in enumerate(m):
            sConv = signal.convolve(gamma_exp[index], (IRF)[index], mode='full', method='direct')
            conv.append(sConv[:len(m.T)])

        conv = np.array(conv)

        Jac_gamma.append(- np.sum(O * conv))

    Jac_tau = []

    for func in BfuncLt:

        func_matrix = np.reshape(np.tile(func, len(m.T)), np.shape(m), order='F')

        Numer = FitModel * (m * func_matrix)
        Denom = Lifetime ** 2

        Frac = Numer/Denom

        conv = []

        for index, pixel in enumerate(m):
            sConv = signal.convolve(Frac[index], IRF[index], mode='full', method='direct')
            conv.append(sConv[:len(m.T)])

        conv = np.array(conv)

        Jac_tau.append(- np.sum(O * conv))

    Jac_bias = []

    for b, resid in zip(Bias, O):
        Jac_bias.append((- np.exp(b)) * (np.sum(resid)))

    Jacobian = Jac_gamma + Jac_tau + Jac_bias

    return Jacobian

def Gradient(Coef_matrix, Data, Time, IRF, BfuncInt, BfuncLt):
    """ Function that calculates the derivatives between the coefficients and the observed data
    :param Coef_matrix: Coefficient results from minimising the loss functions above (single MuFLE exponential)
    :param Data: Observed histogram
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param BfuncInt: Intensity basis function
    :param BfuncLt: Lifetime basis function
    :return: Gradient
    """

    gammaCoef = Coef_matrix[:len(BfuncInt)]
    tauCoef = Coef_matrix[len(BfuncInt):len(BfuncInt) + len(BfuncLt)]
    Bias = Coef_matrix[len(BfuncInt) + len(BfuncLt):]

    Int = np.array(gammaCoef) @ BfuncInt
    Lt = np.array(tauCoef) @ BfuncLt

    Lifetime = np.reshape(np.tile(Lt, len(Time.T)), np.shape(Time), order='F')

    Int1 = np.reshape(np.tile(Int, len(Time.T)), np.shape(Time), order='F')

    FitModel = Int1 * np.exp(-Time / Lifetime)

    FitConv = []

    for index, pixel in enumerate(FitModel):
        ModelConv = signal.convolve(FitModel[index], IRF[index], mode='full', method='direct')
        FitConv.append(ModelConv[:len(Time.T)])

    Bias1 = np.exp(Bias)[:, np.newaxis]

    FitHist = FitConv + Bias1

    m = Time
    O = Data - FitHist

    Jac_gamma = []

    for func in BfuncInt:

        gamma_B_fuc = np.reshape(np.tile(func, len(m.T)), np.shape(m), order='F')

        gamma_exp = gamma_B_fuc * np.exp(-m / Lifetime)

        conv = []

        for index, pixel in enumerate(m):
            sConv = signal.convolve(gamma_exp[index], (IRF)[index], mode='full', method='direct')
            conv.append(sConv[:len(m.T)])

        conv = np.array(conv)

        Jac_gamma.append(-(O * conv))

    Jac_tau = []

    for func in BfuncLt:

        func_matrix = np.reshape(np.tile(func, len(m.T)), np.shape(m), order='F')

        Numer = FitModel * (m * func_matrix)
        Denom = Lifetime ** 2

        Frac = Numer / Denom

        conv = []

        for index, pixel in enumerate(m):
            sConv = signal.convolve(Frac[index], IRF[index], mode='full', method='direct')
            conv.append(sConv[:len(m.T)])

        conv = np.array(conv)

        Jac_tau.append(-(O * conv))

    Jac_bias = []

    for b, resid in zip(Bias, O):
        Jac_bias.append((- np.exp(b)) * (resid))

    Grad = Jac_gamma + Jac_tau + Jac_bias

    return Grad

def FitHistFunc(tauCoef, Bfunctau, gammaCoef, Bfuncgamma, Time, IRF, Bias):
    """ Function that calculates the fitted histogram from the optimal coefficients
    :param tauCoef: Lifetime/tau coefficient
    :param Bfunctau: lifetime basis function
    :param gammaCoef: Intensity/gamma coefficient
    :param Bfuncgamma: intensity basis function
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param Bias: Bias value
    :return: Fitted histogram
    """

    Int = np.array(gammaCoef) @ Bfuncgamma
    Lt = np.array(tauCoef) @ Bfunctau

    Lifetime = np.reshape(np.tile(Lt, len(Time.T)), np.shape(Time), order='F')
    Int = np.reshape(np.tile(Int, len(Time.T)), np.shape(Time), order='F')

    Model = Int * np.exp(-Time / Lifetime)

    Conv = []

    for index, pixel in enumerate(Model):
        ModelConv = signal.convolve(Model[index], IRF[index], mode='full', method='direct')
        Conv.append(ModelConv[:len(Time.T)])

    Bias = np.exp(Bias)[:, np.newaxis]

    return np.array(Conv + Bias)

# Re worded based on MuFLE paper nomenclature:

def SFL_residuals(Params, Y_pm, Time, H_pm, cubic_func):
    """ Function that calculates the residuals of a single exponential MuFLE model, using splines to fit both
    the spectral intensity and spectral lifetime, and a single exponential decay to fit the decays at each channel
    :param Params: Starting parameters
    :param Data: Observed histogram
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param BfuncInt: Intensity basis function
    :param BfuncLt: Lifetime basis function
    :return: The residuals between the observed data and fitted data
    """

    gamma_coef = Params[:len(cubic_func)]
    tau_coef = Params[len(cubic_func): (len(cubic_func) * 2)]
    Bias = Params[(len(cubic_func) + len(cubic_func)):]

    gamma = gamma_coef @ cubic_func
    # resize gamma to be shape p x m
    gamma_pm = np.reshape(np.tile(gamma, len(Time.T)), np.shape(Time), order='F')

    tau = tau_coef @ cubic_func
    # resize tau to be shape p x m
    tau_pm = np.reshape(np.tile(tau, len(Time.T)), np.shape(Time), order='F')

    f_pm = gamma_pm * np.exp(-Time/tau_pm)

    s_pm = []

    for index, pixel in enumerate(f_pm):
        ModelConv = signal.convolve(f_pm[index], H_pm[index], mode='full', method='direct')
        s_pm.append(ModelConv[:len(Time.T)])

    s_pm = np.array(s_pm)

    b_p = np.exp(Bias)[:, np.newaxis]

    S_pm = s_pm + b_p

    return np.ravel(Y_pm - S_pm)

def SFL_fittingfunc(InitialGuess, Y_pm, Time, H_pm, cubic_func):
    """ MuFlE single exponential fitting function
    :param InitialGuess: Starting parameters
    :param Data: Observed histogram
    :param Time: Time histogram
    :param IRF: Measured IRF
    :param BfuncInt: Intensity basis function
    :param BfuncLt: Lifetime basis function
    :return: The estimated intensity, lifetime and bias values + the results message
    """

    # least_squares function uses lm gradient descent to find the local minima
    results = least_squares(fun=SFL_residuals, x0=InitialGuess, args=(Y_pm, Time, H_pm, cubic_func), ftol=1e-15,
                            method='lm', loss='linear', max_nfev=100000)
    gamma_coef = results.x[:len(cubic_func)]
    tau_coef = results.x[len(cubic_func):len(cubic_func) + len(cubic_func)]
    bias = results.x[len(cubic_func) + len(cubic_func):]

    return gamma_coef, tau_coef,  bias, results

def fit_hist_func(tauCoef, cubic_func, gammaCoef, Time, H_pm, Bias):
    """
    :param tauCoef: Lifetime/tau coefficient
    :param cubic_func: cubic basis function
    :param gammaCoef: Intensity/gamma coefficient
    :param Time: Time matrix
    :param H_pm: Measured IRF
    :param Bias: Bias value
    :return: Fitted histogram
    """
    Int = np.array(gammaCoef) @ cubic_func
    Lt = np.array(tauCoef) @ cubic_func

    Lifetime = np.reshape(np.tile(Lt, len(Time.T)), np.shape(Time), order='F')
    Int = np.reshape(np.tile(Int, len(Time.T)), np.shape(Time), order='F')

    Model = Int * np.exp(-Time / Lifetime)

    Conv = []

    for index, pixel in enumerate(Model):
        ModelConv = signal.convolve(Model[index], H_pm[index], mode='full', method='direct')
        Conv.append(ModelConv[:len(Time.T)])

    Bias = np.exp(Bias)[:, np.newaxis]

    return np.array(Conv + Bias)
