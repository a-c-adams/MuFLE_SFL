#  This is the execution script used to run MuFLE in single exponential model
#  The structure of this script is as follows:
#           - Histograms of the instrument response function and observed fluorescence are loaded and preprocessed
#           - The individual channel fluorescence intensity and fluorescence lifetime are next calculated using LMfit
#           - MuFLE then calculates the spectral fluorescence lifetime and fluorescence intensity in single exponential
#           mode. The results are then plotted for observation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TCSPC_fileload import FileLoad, TCSPCtime
from preprocessing import hist_preprocess, hist_plot, Int_Muf, Lt_Muf_SC, Decay, Bias

from single_channel_funcs import SE_single_pixel_residuals, single_pixel_fitting_func, SE_initial_guess
from spline_funcs import Basisfunc_Knots_IdealCoef, SFL_fittingfunc, fit_hist_func
from confidence_func import Hess_matrix, J_gamma_gamma, J_tau_gamma, J_gamma_b, \
    J_tau_tau, J_tau_b, J_b_b, SE_CovHess, Confidence_Intervals

""" File loading """
# IRF data loading
IRF_file_num = np.arange(1, 4)

IRF_all = []

for f in IRF_file_num:
    file_dict, tech_dict = FileLoad(FolderPath='/Users/alexadams/Documents/Flamingo/JupyterNotebooks/new_structure/all_EPTRFS_Data',
                                    FilePath='240417',
                                    FileType=f'/histogram_50ps_Q_RB_{f}_1.mat',
                                   WaveFilePath='/calibration_TRFS_system/lambdaMap.mat',
                                   TimeFilePath='/calibration_TRFS_system/TDCres.mat')


    Hist = file_dict['HistData']
    IRF_hist = Hist[0]
    IRF_all.append(np.flip(IRF_hist))

IRF_TCSPC_hist = np.sum(IRF_all, axis=0)

# Hist data loading
file_num = np.arange(1, 4)

TCSPC_all = []

for f in file_num:
    file_dict4, tech_dict4 = FileLoad(FolderPath='/Users/alexadams/Documents/Flamingo/JupyterNotebooks/new_structure/all_EPTRFS_Data',
                                    FilePath='240417',
                                    FileType=f'/histogram_50ps_T240417_CoCl2_W3A_loc1_{f}_1.mat',
                                   WaveFilePath='/calibration_TRFS_system/lambdaMap.mat',
                                   TimeFilePath='/calibration_TRFS_system/TDCres.mat')

    Hist = file_dict4['HistData']
    TCSPC_hist1 = Hist[0]
    TCSPC_all.append(np.flip(TCSPC_hist1))

TCSPC_time = TCSPCtime(TechDict=tech_dict4)

wave = [0.51 * i + 474 for i in np.arange(1, 513, 1)]

TCSPC_wave = np.reshape(np.repeat(wave, 1200), (512, 1200), order='C')

TCSPC_hist = np.sum(TCSPC_all, axis=0)

"""Peak Intensity"""
# just to find the peak of the histograms
Histcrop = TCSPC_hist[162:332, 700:-10]
HistPeak = np.unravel_index(np.argmax(TCSPC_hist[162:332, 700:-10]), np.shape(TCSPC_hist[162:332, 700:-10]))

print('--------------------')
print(f'Peak intensity of the hist is {Histcrop[HistPeak]}a.u.')

Peak_photon_count = Histcrop[HistPeak]

if Peak_photon_count < 300:
    print("Peak Histogram is less than 300. Exiting...")
    exit()

# Add a line that says if the peak is < 300 stop the script running and produce the message saying there isnt
# sufficient photon counts in the histograms

""" Histogram preprocess """
# number_of_channels: how many time channels to include in the final histogram,
# filter, a variable in hist_preprocess function
# allows you to change where the histogram is cropped from at the beginning ie at 500nm or 520nm -- default is 552 nm
# A filter at 152 and 150 pixel numbers means the preprocessed histogram goes from 552 nm - 628nm
number_of_channels = 200

Hist_pro, Time_pro, Wave_pro, IRF_pro, Wave_col = hist_preprocess(IRFRaw=IRF_TCSPC_hist, Hist=TCSPC_hist, Wave=TCSPC_wave,
                                                                 Time=TCSPC_time, Crop=number_of_channels, Filter=100)

HistPeak = np.unravel_index(np.argmax(Hist_pro), np.shape(Hist_pro))

# plotting the preprocessed histogram
Observed_histogram = hist_plot(Wave=Wave_pro, Time=Time_pro, Hist=Hist_pro)

# check for saturation and IRF alignment
plt.plot(Time_pro[80, 0:80], Hist_pro[80, 0:80], linewidth=4, color='#986beb', label='Obs Hist')
plt.plot(Time_pro[80, 0:80], IRF_pro[80, 0:80] + 0.01, linewidth=4, color='#cdcd00', label='IRF')
plt.xlabel('Time (ns)', fontsize=15)
plt.ylabel('Intensity (a.u.)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()

""" Single Channel Fit Section """
Single_channel_all = []

for Pixel, PixelIRF, decayTime, Wavei in zip(Hist_pro, IRF_pro, Time_pro, Wave_pro.T[1]):
        Params = SE_initial_guess(Int_initial_guess=1, Lt_initial_guess=4, Bias_initial_guess=-4)
        FitDecay, Title, Value, STD = single_pixel_fitting_func(params=Params, Residuals=SE_single_pixel_residuals,
                                                                Time=decayTime, Data=Pixel, IRF=PixelIRF)
        WaveNM = []

        WaveNM.append(Wavei)

        Single_channel_all.append(Value + STD + WaveNM)

MuFLE_final_df = pd.DataFrame(Single_channel_all, columns=['SP Intensity', 'SP Lifetime', 'SP Bias', 'SP Intensity CI',
                                                       'SP Lifetime CI', 'SP Variance', 'Wavelength'])

""" The Splines Section """
# Function which fits the knots and Basis functions calculated from scipy with a given internal knot number and degree
gamma_Coef, gamma_Knots, cubic_Bfunc = Basisfunc_Knots_IdealCoef(Wave_axis=Wave_col, Decay_feature=MuFLE_final_df['SP Intensity'],
                                                        knot_num=3, degree=3)

# Creating initial parameters
rng = np.random.default_rng(123145)

# gamma: random values from 0-1, tau: random values from 0-4
Rand_gamma = rng.uniform(0, 1)
Rand_tau = rng.uniform(0, 4)

gammaCoef_startingparam = [Rand_gamma] * len(cubic_Bfunc)
tauCoef_startingparam = [Rand_tau] * len(cubic_Bfunc)

# starting parameters are placed into a 1D array for the fitting routine
InitialGuess = np.append(gammaCoef_startingparam, tauCoef_startingparam)
InitialGuess = np.append(InitialGuess, [-4] * number_of_channels)

# fitting functions
MuFLE_gamma_Coef, MuFLE_tau_Coef, MuFLE_bias, results = SFL_fittingfunc(InitialGuess=InitialGuess, Y_pm=Hist_pro, Time=Time_pro,
                                                             H_pm=IRF_pro, cubic_func=cubic_Bfunc)

# spline results for wavelength range:
Spline_gamma_results = np.array(MuFLE_gamma_Coef) @ cubic_Bfunc
Spline_tau_results = np.array(MuFLE_tau_Coef) @ cubic_Bfunc

MuFLE_final_df['MuFLE_int'] = Spline_gamma_results
MuFLE_final_df['MuFLE_lifetime'] = Spline_tau_results

""" The Hessian section """
Fit_hist = fit_hist_func(tauCoef=MuFLE_tau_Coef, cubic_func=cubic_Bfunc, gammaCoef=MuFLE_gamma_Coef, Time=Time_pro,
                         H_pm=IRF_pro, Bias=MuFLE_bias)


Gamma_p, Tau_p, O_p_m, V_p_m_l, C_p_m_l, Q_p_m_l = Hess_matrix(Spline_int=Spline_gamma_results, Spline_lt=Spline_tau_results,
                                                               Time=Time_pro, Y_pm=Hist_pro, Fit_hist=Fit_hist, H_pm=IRF_pro)

# second order derivatives
J_gamma_gamma = J_gamma_gamma(Bfunc_int=cubic_Bfunc, m=Time_pro, V_matrix=V_p_m_l)

J_tau_gamma = J_tau_gamma(Bfunc_int=cubic_Bfunc, Bfunc_lt=cubic_Bfunc, m=Time_pro, Tau_matrix=Tau_p, Gamma_matrix=Gamma_p,
                          C_matrix=C_p_m_l, O_matrix=O_p_m, V_matrix=V_p_m_l)

J_gamma_b = J_gamma_b(Bfunc_int=cubic_Bfunc, m=Time_pro, V_matrix=V_p_m_l, bias=MuFLE_bias)

J_tau_tau = J_tau_tau(Bfunc_lt=cubic_Bfunc, m=Time_pro, Gamma_matrix=Gamma_p, Tau_matrix=Tau_p, C_matrix=C_p_m_l,
                      Q_matrix=Q_p_m_l, O_matrix=O_p_m)

J_tau_b = J_tau_b(Bfunc_lt=cubic_Bfunc, m=Time_pro, Gamma_matrix=Gamma_p, Tau_matrix=Tau_p,
                  C_matrix=C_p_m_l, bias=MuFLE_bias)

J_b_b = J_b_b(bias=MuFLE_bias, O_matrix=O_p_m)

HessLen = len(InitialGuess)

"""Covariance and Confidence Intervals"""
Cov = SE_CovHess(J_gamma_gamma=J_gamma_gamma, J_tau_gamma=J_tau_gamma, J_gamma_b=J_gamma_b, J_tau_tau=J_tau_tau,
                 J_tau_b=J_tau_b, J_b_b=J_b_b, HessLen=HessLen, O_matrix=O_p_m)

CovInt = Cov[:len(cubic_Bfunc), :len(cubic_Bfunc)]
CovLt = Cov[len(cubic_Bfunc):len(cubic_Bfunc) + len(cubic_Bfunc), len(cubic_Bfunc):len(cubic_Bfunc) + len(cubic_Bfunc)]

BfuncLt = np.array(cubic_Bfunc)
BfuncInt = np.array(cubic_Bfunc)

# Using the delta method to calculate the confidence intervals
TotalLTCI = Confidence_Intervals(Bfunc=BfuncLt, Cov=CovLt)
TotalIntCI = Confidence_Intervals(Bfunc=BfuncInt, Cov=CovInt)

MuFLE_final_df['Spline Int CI'] = TotalIntCI
MuFLE_final_df['Spline Lifetime CI'] = TotalLTCI

""" Plotting """
Intensity_plot = Int_Muf(Single_channel=MuFLE_final_df, Hist=Hist_pro, Spline_int=Spline_gamma_results, Int_CI=TotalIntCI,
                         Muf_colour='#FF5733')

# LT: Raw hist and mufle (full axis)
Lifetime_plot = Lt_Muf_SC(Single_channel=MuFLE_final_df, Spline_lt=Spline_tau_results, Lt_CI=TotalLTCI, Muf_colour='#FF5733')

# Decay
Decay_fig = Decay(Spline_int=Spline_gamma_results[75], Spline_lt=Spline_tau_results[75], m=Time_pro[75], IRF=IRF_pro[75],
                  Obs_hist=Hist_pro[75], bias=MuFLE_bias[75])

# Bias
Bias_fig = Bias(Single_channel=MuFLE_final_df, bias=MuFLE_bias)

# option to save the dataframe as a csv file