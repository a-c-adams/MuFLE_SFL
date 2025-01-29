# Preprocessing functions
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt

def IRF_back(IRFHist):
    """ Function to remove the dark count rates of the IRF to 0 -- setting a 0 bias
    :param IRFHist: The 2D IRF histogram measured from the quenched Rose-bengal solution
    This function calculated the mean of the first 2 time bins (containing no signal)
    The function then tiles the average dark count rate and subtracts it from the IRF histogram
    :return: The IRF with the dark counts removed
    """
    DarkCount = np.mean(IRFHist[:, 0:2], axis=1)
    DarkCountRep = np.transpose(np.stack([DarkCount] * len(IRFHist.T)))
    Sub = np.array(np.subtract(IRFHist, DarkCountRep))
    Sub[Sub < 0] = 0
    return Sub

def sum_norm(Decay):
    """ Function to normalise a decay using the summation normalisation method
    :param Decay: One decay from the original histogram
    :return: The normalised decay
    """
    a = Decay - np.min(Decay)
    return(a/np.sum(a))

def max_norm(Decay):
    """ Function to normalise a decay or histogram using the maximum normalisation method
    :param: One decay (or the entire histogram) from the original histogram
    :return: The normalised decay (or histogram)
    """
    a = Decay - np.min(Decay)
    b = np.max(Decay) - np.min(Decay)
    return a / b

def hist_preprocess(IRFRaw, Hist, Wave, Time, Crop, Filter):
    """ Function that preprocesses the IRF, Time, Wave and histogram, preparing them for further analysis
    :param IRFRaw: The measured IRF histogram
    :param Hist: The measured data histogram
    :param Wave: The total wavelength variable
    :param Time: The total time variable
    :param Crop: The higher wavelength range to crop the histogram at-- also dictates the number of channels
    the histogram is cropped into
    :param Filter: Which lower wavelength range to crop this histogram -- removing the area where no
    photons are collected due to the band-pass filter but is adjustable for fluorophores like
    Fluorescien -- traditionally a filter of 152 channels is used removing the first 152 decays
    :return: Cropped and normalised IRF, histogram, equilivent time and wavelengths. In addition
    to the total knots used in the splines analysis
    """
    # IRF background removed
    IRF = IRF_back(IRFRaw)

    # Removing the last 10 time bins
    # Crop the histogram and wavelength so only the decay is present
    # Remove the time bins where no spectrum is either (ie before the peak)
    Crop = Filter + Crop
    Wavecrop, Timecrop, Histcrop, IRFcrop = Wave[Filter:Crop, 700:-10], Time[Filter:Crop, :490], \
                                            Hist[Filter:Crop, 700:-10], IRF[Filter:Crop, 700:-10]

    # Wavecrop, Timecrop, Histcrop, IRFcrop = Wave[Filter:Crop, :400], Time[Filter:Crop, :400], \
    #                                         Hist[Filter:Crop, :400], IRF[Filter:Crop, :400]

    # Normalise the IRF using the summed norm method
    NormIRF = []

    for pix in IRFcrop:
        NormIRF.append(sum_norm(Decay=pix))

    NormIRF = np.array(NormIRF)
    # NormIRF = max_norm(IRFcrop)

    # Normalize the histogram
    NormHist = max_norm(Histcrop)

    Conv = []

    for row in range(NormHist.shape[0]):
        ModelConv = convolve(NormIRF[row], NormHist[row], mode='full', method='direct')
        Conv.append(ModelConv[:len(NormHist.T)])

    Conv = np.array(Conv)

    IRFcropFinal = NormIRF/np.max(Conv)
    IRFNew = IRFcropFinal

    # Splines Wavedummy
    WaveDummy = np.reshape(np.repeat(np.linspace(0.01, 0.99, num=len(NormHist)), len(NormHist.T)),
                           (len(NormHist), len(NormHist.T)), order='C')

    # Joint Estimation
    for i in WaveDummy.T:
        WaveCol = i

    return NormHist, Timecrop, Wavecrop, IRFNew, WaveCol

def sim_hist_preprocess(IRF, Hist, Wave, Time, Crop, Filter):
    """ Function that preprocesses the IRF, Time, Wave and histogram, preparing them for further analysis
    :param IRFRaw: The measured IRF histogram
    :param Hist: The measured data histogram
    :param Wave: The total wavelength variable
    :param Time: The total time variable
    :param Crop: The higher wavelength range to crop the histogram at-- also dictates the number of channels
    the histogram is cropped into
    :param Filter: Which lower wavelength range to crop this histogram -- removing the area where no
    photons are collected due to the band-pass filter but is adjustable for fluorophores like
    Fluorescien -- traditionally a filter of 152 channels is used removing the first 152 decays
    :return: Cropped and normalised IRF, histogram, equilivent time and wavelengths. In addition
    to the total knots used in the splines analysis
    """

    # Removing the last 10 time bins
    # Crop the histogram and wavelength so only the decay is present
    # Remove the time bins where no spectrum is either (ie before the peak)
    Crop = Filter + Crop
    # Wavecrop, Timecrop, Histcrop, IRFcrop = Wave[Filter:Crop, 700:-10], Time[Filter:Crop, :490], \
    #                                         Hist[Filter:Crop, 700:-10], IRF[Filter:Crop, 700:-10]

    Wavecrop, Timecrop, Histcrop, IRFcrop = Wave[Filter:Crop, :400], Time[Filter:Crop, :400], \
                                            Hist[Filter:Crop, :400], IRF[Filter:Crop, :400]


    # Wavecrop, Timecrop, Histcrop, IRFcrop = Wave[Filter:Crop, :2000], Time[Filter:Crop, :2000], \
    #                                         Hist[Filter:Crop, :2000], IRF[Filter:Crop, :2000]

    # Normalize the histogram
    NormHist = max_norm(Histcrop)

    Conv = []

    for row in range(NormHist.shape[0]):
        ModelConv = convolve(IRFcrop[row], NormHist[row], mode='full', method='direct')
        Conv.append(ModelConv[:len(NormHist.T)])

    Conv = np.array(Conv)

    IRFcropFinal = IRFcrop/np.max(Conv)
    IRFNew = IRFcropFinal

    # Splines Wavedummy
    WaveDummy = np.reshape(np.repeat(np.linspace(0.01, 0.99, num=len(NormHist)), len(NormHist.T)),
                           (len(NormHist), len(NormHist.T)), order='C')

    # Joint Estimation
    for i in WaveDummy.T:
        WaveCol = i

    return NormHist, Timecrop, Wavecrop, IRFNew, WaveCol


# plotting functions
def hist_plot(Wave, Time, Hist):
    """ Function to surfave 3D plot a histogram
    :param Wave: Wavelength histogram
    :param Time: Time histogram
    :param Hist: Data histogram
    :return: The 3D surface plot figure
    """
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(Wave, Time, Hist, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Time (ns)', fontsize=14)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel('Intensity (a.u.)', fontsize=14, rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    ax.view_init(30, 300)
    # plt.savefig('LottoC_ep1_hist.png', format='png', dpi=600)
    plt.show()

def Int_Muf(Single_channel, Hist, Spline_int, Int_CI, Muf_colour):
    """ Function to plot the MuFLE intensity on one y axis and the plater reader intensity on another
    :param Single_channel: Single channel fit data frame
    :param Hist: Pre-processed histogram data
    :param Spline_int: Splines intensity result
    :param Int_CI: Splines intensity confidence interval
    :param Muf_colour: Mufle colour
    :return: The plotted figure
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(Single_channel['Wavelength'], max_norm(np.sum(Hist, axis=1)), color='#636363', alpha=0.6, linewidth=2)
    ax2.plot(Single_channel['Wavelength'], Spline_int, color=Muf_colour, label='MuFLE', linewidth=4)
    ax2.fill_between(Single_channel['Wavelength'], (Spline_int - Int_CI), (Spline_int + Int_CI),
                     color=Muf_colour, alpha=.1)
    ax1.set_xlabel('Wavelength (nm)', fontsize=18)
    ax1.set_ylabel('Observed Intensity (a.u.)', fontsize=18, color='#636363')
    ax2.set_ylabel('Fitted Intensity (a.u.)', fontsize=18, color=Muf_colour)
    # plt.legend(fontsize=15)
    ax1.tick_params(axis='y', width=3, colors='#636363', labelsize='large')
    ax2.tick_params(axis='y', width=3, colors=Muf_colour, labelsize='large')
    plt.xticks(fontsize=18)
    plt.tight_layout()
    ax1.set_ylim([(np.min(max_norm(np.sum(Hist, axis=1)))), (np.max(max_norm(np.sum(Hist, axis=1)) * 0.91))])
    ax2.set_ylim([(np.min(Spline_int) * 0.9), (np.max(Spline_int) * 1.06)])
    plt.show()
    plt.close()

def Lt_Muf_SC(Single_channel, Spline_lt, Lt_CI, Muf_colour):
    """ Function to plot the mufle lifetime result compared to the single channel -- with no y axis limit
    :param Single_channel: Single channel results
    :param Spline_lt: Spline lifetime results
    :param Lt_CI: Spline lifetime confidence intervals
    :param Muf_colour: MuFLE colour
    :return: The plotted figure
    """

    plt.scatter(Single_channel['Wavelength'], Single_channel['SP Lifetime'], color='#636363', s=50, label='Single Pixel Fit',
                alpha=0.3)
    plt.plot(Single_channel['Wavelength'], Spline_lt, color=Muf_colour, label='MuFLE', linewidth=4)
    plt.fill_between(Single_channel['Wavelength'], (Spline_lt - Lt_CI), (Spline_lt + Lt_CI),
                     color=Muf_colour, alpha=.4)
    plt.xlabel('Wavelength (nm)', fontsize=18)
    plt.ylabel('Fluorescence lifetime (ns)', fontsize=18)
    # plt.legend(fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
    plt.ylim([0, 6])
    # plt.savefig('LottoC_Lt.png', format='png', dpi=1200)
    plt.show()
    plt.close()

def Decay(Spline_int, Spline_lt, m, IRF, Obs_hist, bias):
    """ Function to plot a single channel decay
    :param Spline_int: A single channel splines intensity result
    :param Spline_lt: A single channel splines lifetime result
    :param m: A single channel time axis
    :param IRF: A single channel IRF
    :param Obs_hist: The single channel observed decay
    :param bias: The single channel bias term
    :return: The plotted figure
    """

    Decay = Spline_int * np.exp(-m / Spline_lt)
    Conv = convolve(Decay, IRF, mode='full', method='direct')

    plt.plot(m, Obs_hist, color='#377eb8')
    plt.plot(m, (Conv[:len(m)] + np.exp(bias)), color='#e41a1c')
    plt.xlabel('Time (ns)', fontsize=15)
    plt.ylabel('Intensity (a.u.)', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.savefig('RhBWater_Fit577nm.png', format='png', dpi=1200)
    plt.show()

def Decay_IRF(Spline_int, Spline_lt, m, IRF, Obs_hist, bias):
    """ Function to plot a single channel decay
    :param Spline_int: A single channel splines intensity result
    :param Spline_lt: A single channel splines lifetime result
    :param m: A single channel time axis
    :param IRF: A single channel IRF
    :param Obs_hist: The single channel observed decay
    :param bias: The single channel bias term
    :return: The plotted figure
    """

    Decay = Spline_int * np.exp(-m / Spline_lt)
    Conv = convolve(Decay, IRF, mode='full', method='direct')

    plt.plot(m, Obs_hist, linewidth=3, color='#377eb8')
    plt.plot(m, IRF, linewidth=3, color='#ab8b04')
    plt.plot(m, Conv[:len(m)] + np.exp(bias), linewidth=3,color='#e41a1c')
    plt.xlabel('Time (ns)', fontsize=15)
    plt.ylabel('Intensity (a.u.)', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.savefig('RhBWater_Fit577nm.png', format='png', dpi=1200)
    plt.show()

def Bias(Single_channel, bias):
    """ Function to plot the Splines bias values
    :param Single_channel: Single channel data frame results
    :param bias: Splines bias values
    :return: The plotted figure
    """
    plt.scatter(Single_channel['Wavelength'], np.exp(bias), color='#66c2a4', label='MuFLE', s=40)
    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('Intensity (a.u.)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title("Bias")
    plt.tight_layout()
    # plt.savefig('Fluo_Bias.png', format='png', dpi=1200)
    plt.show()
