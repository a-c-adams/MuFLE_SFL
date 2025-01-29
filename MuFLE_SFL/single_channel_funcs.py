# individual channel functions
from lmfit import Parameters, Minimizer, report_fit
from scipy import signal
import numpy as np

# Single Exponential
# Initial Guess to be stored in LMfit Parameter()
def SE_initial_guess(Int_initial_guess, Lt_initial_guess, Bias_initial_guess):
    """ Function that takes the initial individual channel starting parameters and stores them
    in the parameters function to be given to LMfit
    :param Int_initial_guess: Intensity initial guess
    :param Lt_initial_guess: Lifetime initial guess
    :param Bias_initial_guess: Bias initial guess
    :return: The Parameters in the lmfit function/as a dictionary to be used in the minimiser
    """
    SE_params = Parameters()
    SE_params.add('Int', value=Int_initial_guess)
    SE_params.add('Lifetime', value=Lt_initial_guess)
    SE_params.add('Bias', value=Bias_initial_guess)

    return SE_params

# residuals
def SE_single_pixel_residuals(params, Time, data, IRF):
    """ Function that calculates the residuals between the observed and fit decays in the individual channels
    :param params: Initial guess parameters from the SE_initial_guess function
    :param Time: Time axis
    :param data: individual channel decay
    :param IRF: IRF from that specific channel
    :return: the residuals from a convolved exponential decay using the parameters to start
    """
    Int = params['Int']
    Lifetime = params['Lifetime']
    Bias = params['Bias']

    Decay = Int * np.exp(-(Time/Lifetime))
    Conv = signal.convolve(Decay, IRF, mode='full', method='direct')

    model = Conv[:len(Decay)] + np.exp(Bias)

    return model - data

# Fitting function
def single_pixel_fitting_func(params, Residuals, Time, Data, IRF):
    """ Function that takes the starting parameters, residuals function, time data and IRF and returns
    the parameters that minimise the loss function using the LM gradient descent method
    :param params: Starting parameters
    :param Residuals: Function that calculates the residuals (single, double or triple exponential fit)
    :param Time: Time axis for one channel
    :param Data: Decay from the one channel
    :param IRF: IRF from that one channel
    :return: The fitted decay, the value of the residuals and the confidence interval
    """

    # fit with the default lv algorithm
    minner = Minimizer(Residuals, params, fcn_args=(Time, Data, IRF), nan_policy='propagate')
    result = minner.leastsq()

    # calculate final result
    LMFitDecay = Data + result.residual

    # write error report
    # report = report_fit(result)

    # Add results into data frame
    Title = []
    Value = []
    STD = []

    for name, param in result.params.items():
        Title.append(name)
        Value.append(param.value)
        STD.append(param.stderr)

    return LMFitDecay, Title, Value, STD