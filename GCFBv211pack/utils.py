# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import wave as wave
import time
from scipy.interpolate import UnivariateSpline
from scipy import signal
from functools import lru_cache


def audioread(filepath):
    """Reads a wavfile as a float 
        
        Parameters
        ----------
        filepath: string
            Filepath to the input wav file

        Returns
        ----------
        snd: float
            Sound signal as a float and normalized scale (-1 ~ +1) 
        fs: 
            Sampling frequency
    """
    wav = wave.open(filepath)
    fs = wav.getframerate() # sampling frequency
    snd = wav.readframes(wav.getnframes())
    snd = np.frombuffer(snd, dtype=np.int16) # int16 (-32768 ~ +32767)
    wav.close()
    snd = snd/abs(np.iinfo(np.int16).min) # float (-1 ~ +1)
    return snd, fs


def rms(x):
    """
        Caliculates a root-mean-squared (RMS) value of input signal (1D)

        Parameters
        ----------
        x: float
            Input signal (1D)

        Returns
        ----------
        y: RMS value

    """
    y = np.sqrt(np.mean(x * x))
    return y


def nextpow2(n):
    """Find exponent of next higher power of 2

    Args:
        n (array-like): Input values

    Returns:
        p (array-like): Exponent of next higher power of 2
    """
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    p = int(np.log2(2**m_i))

    return p


def eqlz2meddis_hc_level(snd, out_level_db, *args):
    """ Equalizing Signal RMS Level to the Level for MeddisHairCell

    Args:
        snd (float): Input sound
        out_level_db (float): Output level (No default value,  RMS level)

    Returns:
        snd_eq_meddis (float): Equalized Sound (rms value of 1 is 30 dB SPL)
        amp_db (array_like): 3 values in dB, [out_level_dB, compensation_value_dB, source_level_dB]

    Matlab examples:
        rms(s(t)) == sqrt(mean(s.^2)) == 1   --> 30 dB SPL
        rms(s(t)) == sqrt(mean(s.^2)) == 10  --> 50 dB SPL
        rms(s(t)) == sqrt(mean(s.^2)) == 100 --> 70 dB SPL  

    Reference:
        Meddis (1986), JASA, 79(3),pp.702-711.
    """
    source_level = np.sqrt(np.mean(snd**2)) * 10**(30/20) # level in terms of Meddis HC Level

    # amplifiy the source snd based on the Meddis HC lavel
    amp = (10**(out_level_db/20))/source_level
    snd_eq_meddis = amp * snd

    # summarize information
    amp_db = [out_level_db, 20*np.log10(amp), 20*np.log10(source_level)]

    return snd_eq_meddis, amp_db


def equal_freq_scale(name_scale, num_ch, range_freq):
    """Calculation of Equal Frequency scale on ERB/Mel/Log/Linear scale

    Args:
        name_scale (string): 'ERB', 'mel', 'log', 'linear'
        num_ch (int): Number of channels
        range_freq (array_like): Frequency Range

    Returns:
        frs (array_like): Fr vector
        wf_val (array_like): Wraped freq. value
    """
    eps = np.finfo(float).eps # epsilon

    if name_scale == 'linear':
        range_wf = range_freq
        diff_wf = np.diff(range_wf) / (num_ch-1)
        wf_val = np.linspace(range_wf[0], range_wf[1]+eps*1000, diff_wf)
        frs = wf_val

    elif name_scale == 'mel':
        range_wf = freq2mel(range_freq)
        diff_wf = np.diff(range_wf) / (num_ch-1)
        wf_val = np.linspace(range_wf[0], range_wf[1]+eps*1000, diff_wf)
        frs = mel2freq(wf_val)

    elif name_scale == 'ERB':
        range_wf, _ = freq2erb(range_freq)
        diff_wf = np.diff(range_wf) / (num_ch-1)
        wf_val = np.arange(range_wf[0], range_wf[1]+eps*1000, diff_wf)
        frs, _ = erb2freq(wf_val)

    elif name_scale == 'log':
        if min(range_freq) < 50:
            print("min(range_freq) < 50. Rplaced by 50.")
            range_freq[0] = 50
        range_wf = np.log10(range_freq)
        diff_wf = np.diff(range_wf) / (num_ch-1)
        wf_val = np.linspace(range_wf[0], range_wf[1]+eps*1000, diff_wf)
        frs = 10**(wf_val)
    else:
        help(equal_freq_scale)
        print("Specify name_scale correctly", file=sys.stderr)
        sys.exit(1)
    
    return frs, wf_val


def freq2mel(freq):
    """Convert mel to linear frequency

    Args:
        freq (array_like): linaer-scale frequency [Hz] 

    Returns:
        mel (array_like): mel-scale frequency [mel]

    Note:
        The function was made by the GammachirPy project because there is not original code of "mel2freq" in GCFBv211pack 
    """
    mel = 2595 * np.log10(1+freq/700)
    return mel


def mel2freq(mel):
    """Convert mel to linear frequency

    Args:
        mel (array_like): mel-scale frequency [mel] 

    Returns:
        freq (array_like): linear-scale frequency [Hz]

    Note:
        The function was made by the GammachirPy project because there is not original code of "mel2freq" in GCFBv211pack 
    """
    freq = 700 * ((10**(mel/2595))-1)
    return freq


def freq2erb(cf=None, warning=0):
    """Convert linear frequency to ERB

    Args:
        cf (array_like): center frequency in linaer-scale [Hz]. Default is None.
        warning (int): check frequency range. Default is 0.

    Returns:
        erb_rate (array_like): ERB_N rate [ERB_N] or [cam] 
        erb_width (array_like): ERB_N Bandwidth [Hz]
    """

    if warning == 1:
        # Warnig for frequency range
        cfmin = 50
        cfmax = 12000

        if np.min(cf) < cfmin or np.max(cf) > cfmax:
            print("Warning : Min or max frequency exceeds the proper ERB range: "
                +"{} (Hz) <= Fc <= {} (Hz)".format(cfmin, cfmax), file=sys.stderr)
            sys.exit(1)

    erb_rate = 21.4 * np.log10(4.37*cf/1000+1)
    erb_width = 24.7 * (4.37*cf/1000+1)

    return erb_rate, erb_width


def erb2freq(erb_rate):
    """Convert erb_rate to linear frequency

    Args:
        erb_rate (array_like): ERB_N rate [ERB_N] or [cam] 
    
    Returns:
        cf (array_like): center frequency in linaer-scale [Hz] 
        erb_width (array_like): ERB_N Bandwidth [Hz]
    """
    cf = (10**(erb_rate/21.4) - 1) / 4.37 * 1000
    erb_width = 24.7 * (4.37*cf/1000 + 1)

    return cf, erb_width


def fr2fpeak(n, b, c, fr):
    """Estimate fpeak from fr

    Args:
        n (float): a parameter of the gammachirp
        b (float): a parameter of the gammachirp
        c (float): a parameter of the gammachirp
        fr (float): fr

    Returns:
        fpeak (float): peak frequency
        erbw (float): erbwidth at fr
    """
    _, erb_width = freq2erb(fr)
    fpeak = fr + c*erb_width*b/n

    return fpeak, erb_width


@lru_cache(maxsize=None)
def out_mid_crct_filt(str_crct, sr, sw_plot=0, sw_filter=0):
    """Outer/middle ear compensation filter

    Args:
        str_crct (string): String for Correction ELC/MAF/MAP
        sr (int): Sampling rate
        sw_plot (int): Switch of plot (0:OFF/1:ON) (default:0)
        sw_filter (int): Switch of filter type
            0: FIR linear phase filter (default)
            1: FIR linear phase inverse filter filter
            2: FIR mimimum phase filter (length: half of linear phase filter)

    Returns:
        fir_coef (array_like): FIR filter coefficients
        str_filt (string): Filter infomation

    Notes:
        In the original Matlab code of out_mid_crct_filt.m, persistent variables 
        are called by "persistent" function. The GammachirPy uses the "lru_cache"  
        instead of the persistent variables to call outputs if args are same 
        to previous one. 

    Reference:
        https://docs.python.org/3/library/functools.html
    """
    if sr > 48000:
        print("OutMidCrctFilt : Sampling rate of {} (Hz) (> 48000 (Hz) is not recommended)".format(sr))
        print("<-- ELC etc. is only defined below 16000 (Hz)")

    if sw_filter == 0:
        str_filt = "FIR linear phase filter"
    elif sw_filter == 1:
        str_filt = "FIR linear phase inverse filter"
    elif sw_filter == 2:
        str_filt = "FIR minimum phase filter"
    else:
        help(out_mid_crct_filt)
        print("Specify filter type", file=sys.stderr)
        sys.exit(1)        

    if not str_crct in ['ELC', 'MAF', 'MAP', 'MidEar']:
        help(out_mid_crct_filt)
        print("Specifiy correction: ELC/MAF/MAP/MidEar", file=sys.stderr)
        sys.exit(1)

    """
    Generating filter at the first time
    """
    print("*** OutMidCrctFilt: Generating {} {} ***".format(str_crct, str_filt))
    n_int = 1024
    # n_int = 0 # No spline interpolation:  NG no convergence at remez

    crct_pwr, freq, _ = out_mid_crct(str_crct, n_int, sr, 0)
    crct = np.sqrt(crct_pwr[:,0])
    freq = freq[:,0]

    len_coef = 200 # ( -45 dB) <- 300 (-55 dB)
    n_coef= int(np.fix(len_coef/16000*sr/2)*2) # even number only

    if sw_filter == 1:
        crct = 1 / np.max(np.sqrt(crct_pwr), 0.1) # Giving up less tan -20 dB : f>15000 Hz
                                                 # if requered, the response becomes worse
    
    len_coef = 200 # ( -45 dB) <- 300 (-55 dB)
    n_coef= int(np.fix(len_coef/16000*sr/2)*2) # even number only
    
    """ 
    Calculate the minimax optimal filter with a frequency response
    instead of "fir_coef = firpm(n_coef,freq/sr*2,crct)" in the original code out_mid_crct_filt.m
    """
    x1 = np.array(np.arange(len(freq))).T * 2
    x2 = np.array(np.arange(len(freq)*2)).T
    freq_interp = np.interp(x2, x1, freq)
    fir_coef = signal.remez(n_coef+1, freq_interp, crct, fs=sr) # len(freq_interp) must be twice of len(crct)

    win, _ = taper_window(len(fir_coef),'HAN',len_coef/10)
    fir_coef = win * fir_coef

    """
    Minimum phase reconstruction
    """
    if sw_filter == 2: 
        _, x_mp = rceps(fir_coef)
        fir_coef = x_mp[0:int(np.fix(len(x_mp)/2))]

    """
    Plot
    """
    if sw_plot == 1:
        N_rsl = 1024
        freq2, frsp = signal.freqz(fir_coef, 1, N_rsl, fs=sr)

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        plt.plot(fir_coef)
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlim([0, 300])
        ax1.set_ylim([-0.3, 0.3])
        
        ax2 = fig.add_subplot(2, 1, 2)
        plt.plot(freq2, abs(frsp), freq, crct, '--')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude (linear term)')
        ax2.set_xlim([0, 25000])
        ax2.set_ylim([0, 1.8])
        
    return fir_coef, str_filt



def out_mid_crct(str_crct, n_frq_rsl=0, fs=32000, sw_plot=1):
    """Correction of ELC, MAF, MAP, MID. 
    It produces interpolated points for the ELC/MAF/MAP/MidEar correction.

    Args:
        str_crct (string): Correction ELC/MAF/MAP/MidEar
        n_frq_rsl (int): Number of data points, if zero, then direct out (default: 0)
        fs (int): Sampling frequency (default: 32000)
        sw_plot (int): Switch for plot (0/1, default:1)

    Returns:
        crct_pwr_lin (array_like): Correction value in LINEAR POWER. 
            This is defined as:  CrctLiPwr =10^(-freq_char_db_to_be_cmpnstd/10)
        freq (array_like): Corresponding Frequency at the data point
        freq_char_db_to_be_cmpnstd (array_like): Frequency char of ELC/MAP dB 
            to be compensated for filterbank (defined By Glassberg and Moore.)

    Note: 
        "ER4B" option in str_crct was omitted because the option uses a special 
        measurement data. 
    """

     
    """  ER4B: Omitted
    if str_crct == 'ER4B':
        crct_pwr_lin, freq, freq_char_db_to_be_cmpnstd = OutMidCrct_ER4B(n_frq_rsl, fs, sw_plot)
        return crct_pwr_lin, freq, freq_char_db_to_be_cmpnstd
    """

    """
    Conventional ELC/MAF/MAP/MidEar
    """

    f1 = [	20,   25,  30,     35,  40,    45,  50,   55,   60,   70,  # 1-10
            80,   90,  100,   125,  150,   177, 200,  250,  300,  350,  # 11-20
            400,  450, 500,   550,  600,   700, 800,  900,  1000, 1500,  # 21-30
            2000, 2500, 2828, 3000, 3500, 4000, 4500, 5000, 5500, 6000,  # 31-40
            7000, 8000, 9000, 10000, 12748, 15000]   # 41-46

    elc = [ 31.8, 26.0, 21.7, 18.8, 17.2, 15.4, 14.0, 12.6, 11.6, 10.6, 
            9.2, 8.2, 7.7, 6.7, 5.3, 4.6, 3.9, 2.9, 2.7, 2.3, 
            2.2, 2.3, 2.5, 2.7, 2.9, 3.4, 3.9, 3.9, 3.9, 2.7, 
            0.9, -1.3, -2.5, -3.2, -4.4, -4.1, -2.5, -0.5, 2.0, 5.0, 
            10.2, 15.0, 17.0, 15.5, 11.0, 22.0]

    maf = [ 73.4, 65.2, 57.9, 52.7, 48.0, 45.0, 41.9, 39.3, 36.8, 33.0, 
            29.7, 27.1, 25.0, 22.0, 18.2, 16.0, 14.0, 11.4, 9.2, 8.0, 
            6.9,  6.2,  5.7,  5.1,  5.0,  5.0,  4.4,  4.3, 3.9, 2.7, 
            0.9, -1.3, -2.5, -3.2, -4.4, -4.1, -2.5, -0.5, 2.0, 5.0, 
            10.2, 15.0, 17.0, 15.5, 11.0, 22.0]

    f2  = [  125,  250,  500, 1000, 1500, 2000, 3000, 
            4000, 6000, 8000,10000,12000,14000,16000]

    map = [ 30.0, 19.0, 12.0,  9.0, 11.0, 16.0, 16.0, 
            14.0, 14.0,  9.9, 24.7, 32.7, 44.1, 63.7]

    # MidEar Correction (little modification at 17000:1000:20000)
    f3 =  [   1,  20,  25, 31.5,   40,   50,   63,   80,  100,  125, 
            160, 200, 250,  315,  400,  500,  630,  750,  800, 1000,
            1250, 1500, 1600,  2000,  2500,  3000,  3150,  4000,  5000,  6000, 
            6300, 8000, 9000, 10000, 11200, 12500, 14000, 15000, 16000,  20000]

    mid =  [  50,  39.15, 31.4, 25.4, 20.9,  18, 16.1, 14.2, 12.5, 11.13,
            9.71,   8.42,  7.2,  6.1,  4.7, 3.7,  3.0,  2.7,  2.6,   2.6,
             2.7,    3.7,  4.6,  8.5, 10.8, 7.3,  6.7,  5.7,  5.7,   7.6,
             8.4,   11.3, 10.6,  9.9, 11.9, 13.9, 16.0, 17.3, 17.8,  20.0] 


    frq_tbl = []
    tbl_freq_char = []
    if str_crct == 'ELC':
        frq_tbl = np.array([f1]).T
        tbl_freq_char = np.array([elc]).T
        val_half_fs = 130
    elif str_crct == 'MAF':
        frq_tbl = np.array([f1]).T
        tbl_freq_char = np.array([maf]).T
        val_half_fs = 130
    elif str_crct == 'MAF':
        frq_tbl = np.array([f2]).T
        tbl_freq_char = np.array([map]).T
        val_half_fs = 180
    elif str_crct == 'MidEar':
        frq_tbl = np.array([f3]).T
        tbl_freq_char = np.array([mid]).T
        val_half_fs = 23
    elif str_crct == 'NO':
        pass
    else:
        print("Specifiy correction: ELC/MAF/MAP/MidEar or NO correction", \
              file=sys.stderr)
        sys.exit(1)

    """
    Additional dummy data for high sampling frequency
    """
    if fs > 32000:
        frq_tbl = np.vstack([frq_tbl, fs/2])
        tbl_freq_char = np.vstack([tbl_freq_char, val_half_fs])
        frq_tbl, indx = np.unique(frq_tbl, return_index=True)
        frq_tbl = np.array([frq_tbl]).T
        tbl_freq_char = tbl_freq_char[indx]

    str1 = ''
    if n_frq_rsl <= 0:
        str1 = 'No interpolation. Output: values in original table.'
        freq = frq_tbl
        freq_char_db_to_be_cmpnstd = tbl_freq_char
    else:
        freq = np.array([np.arange(n_frq_rsl)/n_frq_rsl * fs/2]).T
        if str_crct == 'NO':
            freq_char_db_to_be_cmpnstd = np.zeros(freq.shape)
        else:
            str1 = 'Spline interpolated value in equal frequency spacing.'
            freq_1d = freq.T[0,:]
            frq_tbl_1d = frq_tbl.T[0,:]
            tbl_freq_char_1d = tbl_freq_char.T[0,:]
            spl = UnivariateSpline(frq_tbl_1d, tbl_freq_char_1d, s=0)
            freq_char_db_to_be_cmpnstd = spl(freq_1d)
            freq_char_db_to_be_cmpnstd = np.array([freq_char_db_to_be_cmpnstd]).T
    
    if sw_plot == 1:
        str = "*** Frequency Characteristics (" + str_crct + "): Its inverse will be corrected. ***"
        print(str) 
        print("{}".format(str1))
        fig, ax = plt.subplots()
        plt.plot(frq_tbl, tbl_freq_char, 'b-',freq, freq_char_db_to_be_cmpnstd, 'r--')
        plt.xlim(0, 25000)
        plt.ylim(-20,140)
        ax.set_title(str)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Level (dB)')
        plt.show()

    crct_pwr_lin = 10**(-freq_char_db_to_be_cmpnstd/10) # in Linear Power. Checked 19 Apr 2016

    return crct_pwr_lin, freq, freq_char_db_to_be_cmpnstd


def taper_window(len_win, type_taper, len_taper=None, range_sigma=3, sw_plot=0):
    """Taper Window Generator for signal onset/offset

    Args:
        len_win (int): Length of window (number of points)
        type_taper (string): Type of Taper (KeyWords of 3 letters)
            - HAM: Hamming
            - HAN/COS: Hanning/Cosine
            - BLA: Blackman
            - GAU: Gaussian
            - (other): Linear
        len_taper (int, optional): Length of taper. Defaults to None.
        range_sigma (int, optional): Range in sigma. Defaults to 3.
        sw_plot (int, optional): OFF/ON. Defaults to 0.

    Returns:
        taper_win (array_like): Taper window points (max: 1)
        type_taper (string): Type of taper (full name)
    """

    if len_taper == None:
        len_taper = int(np.fix(len_win/2))
    
    elif len_taper*2+1 >= len_win:
        print("Caution (taper_window) : No flat part. ")
        
        if not len_taper == np.fix(len_win/2):
            print("Caution (taper_window) : len_taper <-- fix(len_win/2)")
            
        len_taper = int(np.fix(len_win/2))

    len_taper= int(len_taper)

    if type_taper == 'HAM':
        taper = np.hamming(len_taper*2+1)
        type_taper = 'Hamming'

    elif type_taper == 'HAN' or type_taper == 'COS':
        taper = np.hamming(len_taper*2+1)
        type_taper = 'Hanning/Cosine'

    elif type_taper == 'BLA':
        taper = np.blackman(len_taper*2+1)
        type_taper = 'Blackman'

    elif type_taper == 'GAU':
        if len(range_sigma) == 0:
            range_sigma = 3
        nn = np.arange(-len_taper, len_taper, 1)
        taper = np.exp(-(range_sigma/len_taper)**2 / 2)
        type_taper == 'Gauss'

    else:
        taper = np.array(list(np.arange(1,len_taper+1,1)) + list([len_taper+1]) + \
                         list(np.arange(len_taper,1-1,-1))) / (len_taper+1)
        type_taper = 'Line'

    len_taper = int(np.fix(len_taper))
    taper_win = list(taper[0:len_taper]) + list(np.ones(len_win-len_taper*2)) + \
               list(taper[(len_taper+1):(len_taper*2+1)])

    if sw_plot == 1:
        _, ax = plt.subplots()
        plt.plot(taper_win)
        ax.set_xlabel('Points')
        ax.set_ylabel('Amplitude')
        plt.title(f"TypeTaper: {type_taper}")

    return taper_win, type_taper


def rceps(x):
    """returns the real cepstrum of the real sequence X

    Args:
        x (array_like): input signal

    Returns:
        x_hat: real cepstrum
        y_hat: a unique minimum-phase sequence that has the reame real cepstrum as x

    Note:
        This code is based on "rceps.m" in MATLAB and is under-construction. 

    Examples:
        x = [4 1 5]; % Non-minimum phase sequence
        x_hat = array([1.62251148, 0.3400368 , 0.3400368 ])
        y_hat = array([5.33205452, 3.49033278, 1.1776127 ])

    References:
    - A.V. Oppenheim and R.W. Schafer, Digital Signal Processing, Prentice-Hall, 1975.
    - Programs for Digital Signal Processing, IEEE Press, John Wiley & Sons, 1979, algorithm 7.2.
    - https://mathworks.com/help/signal/ref/rceps.html
    """

    if isrow(x):
        x_t = np.array([x]).T
    else:
        x_t = x

    fft_x_abs = np.abs(np.fft.fft(x_t, n=None, axis=0))

    x_hat_t = np.real(np.fft.ifft(np.log(fft_x_abs), n=None, axis=0))

    # x_hat
    if isrow(x):
        # transform the result to a row vector
        x_hat = x_hat_t[:,0]
    else:
        x_hat = x_hat_t

    # y_hat
    n_rows = x_hat_t.shape[0]
    n_cols = x_hat_t.shape[1]
    odd = n_rows % 2
    a1 = np.array([1])
    a2 = 2*np.ones((int((n_rows+odd)/2)-1, 1))
    a3 = np.zeros((int((n_rows+odd)/2)-1,1))
    wn = np.kron(np.ones((1, n_cols)), np.vstack((a1, a2, a3)))
    """
    Matlab can use zero and negative numbers for args of ones function, 
    but the np.ones cannot. So, an internal array is removed. 
    The original code is: 
    wn = np.kron(np.ones((1, n_cols)), np.array([[1], 2*np.ones((int((n_rows+odd)/2)-1, 1)), 
         np.ones(1-odd, 1), np.zeros((int((n_rows+odd)/2)-1,1))]))
    """
    y_hat_t = np.real(np.fft.ifft(np.exp(np.fft.fft((wn*x_hat_t),n=None, axis=0)), n=None, axis=0))
    if isrow(x):
        # transform the result to a row vector
        y_hat = y_hat_t[:,0]
    else:
        y_hat = y_hat_t

    return x_hat, y_hat


def isrow(x):
    """returns True if x is a row vector, False otherwise.

    Args:
        x (array_like): verctors

    Returns:
        logical (bool): True/False
    """

    if np.size(np.shape(x)) == 1:
        logical = True
    else:
        logical = False

    return logical
    

def iscolumn(x):
    """returns True if x is a column vector, False otherwise.

    Args:
        x (array_like): verctors

    Returns:
        logical (bool): True/False
    """
    if np.size(np.shape(x)) == 2:
        if np.shape(x)[1] == 1:
            logical = True
        else:
            logical = False
    else:
        logical = False
        
    return logical


def make_asym_cmp_filters_v2(fs, frs, b, c):
    """Computes the coefficients for a bank of Asymmetric Compensation Filters
    This is a modified version to fix the round off problem at low freqs
    Use this with ACFilterBank.m
    See also asym_cmp_frsp_v2 for frequency response

    Args:
        fs (int): Sampling frequency
        frs (array_like): array of the center frequencies, frs
        b (array_like): array or scalar of a bandwidth coefficient, b
        c (float): array or scalar of asymmetric parameters, c

    Returns:
        ACFcoef: 
        - fs (int): Sampling frequency
        - bz (array_like): MA coefficients  (num_ch*3*num_filt)
        - ap (array_like): AR coefficients  (num_ch*3*num_filt)

    Notes:
        [1] Ref for p1-p4: Unoki,M , Irino,T. , and Patterson, R.D. , "Improvement of an IIR asymmetric compensation gammachirp filter," Acost. Sci. & Tech. (ed. by the Acoustical Society of Japan ), 22 (6), pp. 426-430, Nov. 2001.
        [2] Conventional setting was removed.
            fn = frs + nfilt* p3 .*c .*b .*erbw/n;
            This frequency fn is for normalizing GC(=GT*Hacf) filter to be unity at the peak, frequnecy. But now we use Hacf as a highpass filter as well. cGC = pGC *Hacf. In this case, this normalization is useless. 
            So, it was set as the gain at frs is unity.  (4. Jun 2004 )
        [3] Removed
            ACFcoef.fn(:,nff) = fn;
            n : scalar of order t^(n-1) % used only in normalization 
    """

    class ACFcoef:
        fs = []
        ap = np.array([])
        bz = np.array([])


    num_ch, len_frs = np.shape(frs)
    if len_frs > 1:
        print("frs should be a column vector frs.", file=sys.stderr)
        sys.exit(1)
    
    _, erbw = freq2erb(frs)
    ACFcoef.fs = fs

    # New coefficients. See [1]
    num_filt = 4
    p0 = 2
    p1 = 1.7818 * (1-0.0791*b) * (1-0.1655*np.abs(c))
    p2 = 0.5689 * (1-0.1620*b) * (1-0.0857*np.abs(c))
    p3 = 0.2523 * (1-0.0244*b) * (1+0.0574*np.abs(c))
    p4 = 1.0724

    if num_filt > 4:
        print("num_filt > 4", file=sys.stderr)
        sys.exit(1) 

    ACFcoef.ap = np.zeros((num_ch, 3, num_filt))
    ACFcoef.bz = np.zeros((num_ch, 3, num_filt))

    for nfilt in range(num_filt):
        r  = np.exp(-p1*(p0/p4)**(nfilt) * 2*np.pi*b*erbw / fs)
        del_frs = (p0*p4)**(nfilt)*p2*c*b*erbw;  
        phi = 2*np.pi*(frs+del_frs).clip(0)/fs
        psi = 2*np.pi*(frs-del_frs).clip(0)/fs
        fn = frs # see [2]

        # second order filter
        ap = np.concatenate([np.ones(np.shape(r)), -2*r*np.cos(phi), r**2], axis=1)
        bz = np.concatenate([np.ones(np.shape(r)), -2*r*np.cos(psi), r**2], axis=1)

        vwr = np.exp(1j*2*np.pi*fn/fs)
        vwrs = np.concatenate([np.ones(np.shape(vwr)), vwr, vwr**2], axis=1)
        nrm = np.array([np.abs(np.sum(vwrs*ap, axis=1) / np.sum(vwrs*bz, axis=1))]).T
        bz = bz * (nrm*np.ones((1, 3)))

        ACFcoef.ap[:,:,nfilt] = ap
        ACFcoef.bz[:,:,nfilt] = bz

    return ACFcoef



def fr1_to_fp2(n, b1, c1, b2, c2, frat, fr1, sr=24000, n_fft=2048, sw_plot=0):
    """Convert fr1 (for passive GC; pGC) to fp2 (for compressive GC; cGC)

    Args:
        n (int): Parameter defining the envelope of the gamma distribution (for pGC)
        b1 (float): Parameter defining the envelope of the gamma distribution (for pGC)
        c1 (float): Chirp factor (for pGC)
        b2 (float): Parameter defining the envelope of the gamma distribution (for cGC)
        c2 (float): Chirp factor  (for cGC)
        frat (float): Frequency ratio, the main level-dependent variable
        fr1 (float): Center frequency (for pGC)
        sr (int, optional): Sampling rate. Defaults to 24000.
        n_fft (int, optional): Size of FFT. Defaults to 2048.
        sw_plot (int, optional): Show plot of cgc_frsp and pGC_frsp. Defaults to 0.

    Returns:
        fp2 (float): Peak frequency (for compressive GC)
        fr2 (float): Center Frequency (for compressive GC)
    """
    _, erbw1 = freq2erb(fr1)
    fp1, _ = fr2fpeak(n, b1, c1, fr1)
    fr2 = frat * fp1
    _, erbw2 = freq2erb(fr2)

    bw1 = b1 * erbw1
    bw2 = b2 * erbw2

    # coef1*fp2^3 + coef2*fp2^2 + coef3*fp2 + coef4 = 0 
    coef1 = -n
    coef2 = c1*bw1 + c2*bw2 + n*fr1 + 2*n*fr2
    coef3 = -2*fr2*(c1*bw1+n*fr1) - n*((bw2)**2+fr2**2) - 2*c2*bw2*fr1
    coef4 = c2*bw2*((bw1)**2+fr1**2) + (c1*bw1+n*fr1)*(bw2**2+fr2**2)
    coefs = [coef1, coef2, coef3, coef4]

    p = np.roots(coefs)
    fp2cand = p[np.imag(p)==0]
    if len(fp2cand) == 1:
        fp2 = fp2cand
    else:
        val, ncl = np.min(np.abs(fp2cand - fp1))
        fp2 = fp2cand(ncl) # in usual cGC range, fp2 is close to fp1

    # sw_plot = 1
    if sw_plot == 1: # Check
        fs = 48000
        n_frq_rsl = 2048
        cgc_rsp = cmprs_gc_frsp(fr1, fs, n, b1, c1, frat, b2, c2, n_frq_rsl)

        nfr2 = np.zeros((len(fp2cand), 1))
        for nn in range(len(fp2cand)):
            nfr2[nn] = np.argmin(abs(cgc_rsp.freq - fp2cand[nn]))
        
        fig, ax = plt.subplots()
        plt_freq = np.array(cgc_rsp.freq).T
        plt_cgc_frsp = np.array(cgc_rsp.cGCFrsp/np.max(cgc_rsp.cGCFrsp)).T
        plt_pgc_frsp = np.array(cgc_rsp.pGCFrsp).T

        ax.plot(plt_freq, plt_cgc_frsp, label="cGCFrsp") # compressive GC
        ax.plot(plt_freq, plt_pgc_frsp, label="pGCFrsp") # passive GC
        ax.set_xlim([0, np.max(fp2cand)*2])
        ax.set_ylim([0, 1])
        ax.legend()
        plt.show()

    return fp2, fr2


def cmprs_gc_frsp(fr1, fs=48000, n=4, b1=1.81, c1=-2.96, frat=1, b2=2.01, c2=2.20, n_frq_rsl=1024):
    """Frequency Response of Compressive GammaChirp

    Args:
        fr1 (array-like): Resonance Freqs.
        fs (int, optional): Sampling Freq. Defaults to 48000.
        n (int, optional): Order of Gamma function, t**(n-1). Defaults to 4.
        b1 (float, optional): b1 for exp(-2*pi*b1*erb(f)). Defaults to 1.81.
        c1 (float, optional): c1 for exp(j*2*pi*fr + c1*ln(t)). Defaults to -2.96.
        frat (int, optional): Frequency ratio. fr2 = frat*fp1. Defaults to 1.
        b2 (float, optional): _description_. Defaults to 2.01.
        c2 (float, optional): _description_. Defaults to 2.20.
        n_frq_rsl (int, optional): _description_. Defaults to 1024.

    Returns:
        cGCresp: Struct of cGC response
            pGCFrsp (array-like): Passive GC freq. resp. (num_ch*n_frq_rsl matrix)
            cGCFrsp (array-like): Comressive GC freq. resp. (num_ch*n_frq_rsl matrix)
            cGCNrmFrsp (array-like): Normalized cGCFrsp (num_ch*n_frq_rsl matrix)
            ACFrsp: Asym (array-like). Compensation Filter freq. resp.
            AsymFunc (array-like): Asym Func
            freq (array-like): Frequency (1*n_frq_rsl)
            Fp2 (array-like): Peak freq.
            ValFp2 (array-like): Peak Value
    """
    
    class cGCresp:
        Fr1 = []
        n = []
        b1 = []
        c1 = []
        frat = []
        b2 = []
        c2 = []
        NfrqRsl = []
        pGCFrsp = []
        cGCFrsp = []
        cGCNrmFrsp = []
        ACFFrsp = []
        AsymFunc = []
        Fp1 = []
        Fr2 = []
        Fp2 = []
        ValFp2 = []
        NormFctFp2 = []
        freq = []

    if isrow(fr1):
        fr1 = np.array([fr1]).T

    num_ch = len(fr1)

    if isinstance(n, (int, float)):
        n = n * np.ones((num_ch, 1))
    if isinstance(b1, (int, float)):
        b1 = b1 * np.ones((num_ch, 1))
    if isinstance(c1, (int, float)):
        c1 = c1 * np.ones((num_ch, 1))
    if isinstance(frat, (int, float)):
        frat = frat * np.ones((num_ch, 1))
    if isinstance(b2, (int, float)):
        b2 = b2 * np.ones((num_ch, 1))
    if isinstance(c2, (int, float)):
        c2 = c2 * np.ones((num_ch, 1))

    pgc_frsp, freq, _, _, _ = gammachirp_frsp(fr1, fs, n, b1, c1, 0.0, n_frq_rsl)
    fp1, _ = fr2fpeak(n, b1, c1, fr1)
    fr2 = frat * fp1
    acf_frsp, freq, asym_func = asym_cmp_frsp_v2(fr2, fs, b2, c2, n_frq_rsl)
    cgc_frsp = pgc_frsp * asym_func # cgc_frsp = pgc_frsp * acf_frsp
    
    val_fp2 = np.max(cgc_frsp, axis=1)
    nchFp2 = np.argmax(cgc_frsp, axis=1)
    if isrow(val_fp2):
        val_fp2 = np.array([val_fp2]).T
    
    norm_fact_fp2 = 1/val_fp2

    # function cGCresp = CmprsGCFrsp(fr1,fs,n,b1,c1,frat,b2,c2,n_frq_rsl)
    cGCresp.Fr1 = fr1
    cGCresp.n = n
    cGCresp.b1 = b1
    cGCresp.c1 = c1
    cGCresp.frat = frat
    cGCresp.b2 = b2
    cGCresp.c2 = c2
    cGCresp.NfrqRsl = n_frq_rsl
    cGCresp.pGCFrsp = pgc_frsp
    cGCresp.cGCFrsp = cgc_frsp
    cGCresp.cGCNrmFrsp = cgc_frsp * (norm_fact_fp2 * np.ones((1,n_frq_rsl)))
    cGCresp.ACFFrsp = acf_frsp
    cGCresp.AsymFunc = asym_func
    cGCresp.Fp1 = fp1
    cGCresp.Fr2 = fr2
    cGCresp.Fp2 = freq[nchFp2]
    cGCresp.ValFp2 = val_fp2
    cGCresp.NormFctFp2 = norm_fact_fp2
    cGCresp.freq = [freq]

    return cGCresp


def gammachirp_frsp(frs, sr=48000, order_g=4, coef_erbw=1.019, coef_c=0.0, phase=0.0, n_frq_rsl=1024):
    """Frequency Response of GammaChirp

    Args:
        frs (array_like, optional): Resonance freq. Defaults to None.
        sr (int, optional): Sampling freq. Defaults to 48000.
        order_g (int, optional): Order of Gamma function t**(order_g-1). Defaults to 4.
        coef_erbw (float, optional): Coeficient -> exp(-2*pi*coef_erbw*ERB(f)). Defaults to 1.019.
        coef_c (int, optional): Coeficient -> exp(j*2*pi*Fr + coef_c*ln(t)). Defaults to 0.0.
        phase (int, optional): Coeficient -> exp(j*2*pi*Fr + coef_c*ln(t)). Defaults to 0.9.
        n_frq_rsl (int, optional): Freq. resolution. Defaults to 1024.

    Returns:
        amp_frsp (array_like): Absolute of freq. resp. (num_ch*n_frq_rsl matrix)
        freq (array_like): Frequency (1*n_frq_rsl)
        f_peak (array_like): Peak frequency (num_ch * 1)
        grp_dly (array_like): Group delay (num_ch*n_frq_rsl matrix)
        phs_frsp (array_like): Angle of freq. resp. (num_ch*n_frq_rsl matrix)
    """
    if isrow(frs):
        frs = np.array([frs]).T

    num_ch = len(frs)

    if isinstance(order_g, (int, float)) or len(order_g) == 1:
        order_g = order_g * np.ones((num_ch, 1))
    if isinstance(coef_erbw, (int, float)) or len(coef_erbw) == 1:
        coef_erbw = coef_erbw * np.ones((num_ch, 1))
    if isinstance(coef_c, (int, float)) or len(coef_c) == 1:
        coef_c = coef_c * np.ones((num_ch, 1))
    if isinstance(phase, (int, float)) or len(phase) == 1:
        phase = phase * np.ones((num_ch, 1))

    if n_frq_rsl < 256:
        print("n_frq_rsl < 256", file=sys.stderr)
        sys.exit(1)

    _, erbw = freq2erb(frs)
    freq = np.arange(n_frq_rsl) / n_frq_rsl * sr / 2
    freq = np.array([freq]).T

    one1 = np.ones((1, n_frq_rsl))
    bh = (coef_erbw * erbw) * one1
    fd = (np.ones((num_ch, 1)) * freq[:,0]) - frs * one1
    cn = (coef_c / order_g) * one1
    n = order_g * one1
    c = coef_c * one1
    phase = phase * one1

    # Analytic form (normalized at f_peak)
    amp_frsp = ((1+cn**2) / (1+(fd/bh)**2))**(n/2) \
                * np.exp(c * (np.arctan(fd/bh)-np.arctan(cn)))
    
    f_peak = frs + coef_erbw * erbw * coef_c / order_g
    grp_dly = 1/(2*np.pi) * (n*bh + c*fd) / (bh**2 + fd**2)
    phs_frsp = -n * np.arctan(fd/bh) - c / 2*np.log((2*np.pi*bh)**2 + (2*np.pi*fd)**2) + phase

    return amp_frsp, freq, f_peak, grp_dly, phs_frsp
    

def asym_cmp_frsp_v2(frs, fs=48000, b=None, c=None, n_frq_rsl=1024, num_filt=4):
    """Amplitude spectrum of Asymmetric compensation IIR filter (ACF) for the gammachirp 
    corresponding to make_asym_cmp_filters_v2

    Args:
        frs (array_like, optional): Center freqs. Defaults to None.
        fs (int, optional): Sampling freq. Defaults to 48000.
        b (array_like, optional): Bandwidth coefficient. Defaults to None.
        c (array_like, optional): Asymmetric paramters. Defaults to None.
        n_frq_rsl (int, optional): Freq. resolution for linear freq. scale for specify renponse at frs
                                (n_frq_rsl>64). Defaults to 1024.
        num_filt (int, optional): Number of 2nd-order filters. Defaults to 4.

    Returns:
        acf_frsp: Absolute values of frequency response of ACF (num_ch * n_frq_rsl)
        freq: freq. (1 * n_frq_rsl)
        asym_func: Original asymmetric function (num_ch * n_frq_rsl)
    """

    if isrow(frs):
        frs = np.array([frs]).T
    if isrow(b):
        b = np.array([b]).T
    if isrow(c):
        c = np.array([c]).T
    num_ch = len(frs)

    if n_frq_rsl >= 64:
        freq = np.arange(n_frq_rsl) / n_frq_rsl * fs/2
    elif n_frq_rsl == 0:
        freq = frs
        n_frq_rsl = len(freq)
    else:
        help(asym_cmp_frsp_v2)
        print("Specify n_frq_rsl 0) for frs or N>=64 for linear-freq scale", file=sys.stderr)
        sys.exit(1)

    # coef.
    sw_coef = 0 # self consistency
    # sw_coef = 1 # reference to make_asym_cmp_filters_v2

    if sw_coef == 0:
        # New Coefficients. num_filt = 4; See [1]
        p0 = 2
        p1 = 1.7818 * (1 - 0.0791*b) * (1 - 0.1655*np.abs(c))
        p2 = 0.5689 * (1 - 0.1620*b) * (1 - 0.0857*np.abs(c))
        p3 = 0.2523 * (1 - 0.0244*b) * (1 + 0.0574*np.abs(c))
        p4 = 1.0724
    else:
        ACFcoef = make_asym_cmp_filters_v2(fs, frs, b, c)

    # filter coef.
    _, erbw = freq2erb(frs)
    acf_frsp = np.ones((num_ch, n_frq_rsl))
    freq2 = np.concatenate([np.ones((num_ch,1))*freq, frs], axis=1)

    for nfilt in range(num_filt):

        if sw_coef == 0:
            r = np.exp(-p1 * (p0/p4)**nfilt * 2 * np.pi * b * erbw / fs)
            delfr = (p0*p4)**nfilt * p2 * c * b * erbw
            phi = 2*np.pi*np.maximum(frs + delfr, 0)/fs
            psi = 2*np.pi*np.maximum(frs - delfr, 0)/fs
            fn = frs
            ap = np.concatenate([np.ones((num_ch, 1)), -2*r*np.cos(phi), r**2], axis=1)
            bz = np.concatenate([np.ones((num_ch, 1)), -2*r*np.cos(psi), r**2], axis=1)
        else:
            ap = ACFcoef.ap[:, :, nfilt]
            bz = ACFcoef.bz[:, :, nfilt]

        cs1 = np.cos(2*np.pi*freq2/fs)
        cs2 = np.cos(4*np.pi*freq2/fs)
        bzz0 = np.array([bz[:, 0]**2 + bz[:, 1]**2 + bz[:, 2]**2]).T * np.ones((1, n_frq_rsl+1))
        bzz1 = np.array([2 * bz[:, 1] * (bz[:, 0] + bz[:, 2])]).T * np.ones((1, n_frq_rsl+1))
        bzz2 = np.array([2 * bz[:, 0] * bz[:, 2]]).T * np.ones((1, n_frq_rsl+1))
        hb = bzz0 + bzz1*cs1 + bzz2*cs2

        app0 = np.array([ap[:, 0]**2 + ap[:, 1]**2 + ap[:, 2]**2]).T * np.ones((1, n_frq_rsl+1))
        app1 = np.array([2 * ap[:, 1] * (ap[:, 0] + ap[:, 2])]).T * np.ones((1, n_frq_rsl+1))
        app2 = np.array([2 * ap[:, 0] * ap[:, 2]]).T * np.ones((1, n_frq_rsl+1))
        ha = app0 + app1*cs1 + app2*cs2

        h = np.sqrt(hb/ha)
        h_norm = np.array([h[:, n_frq_rsl]]).T * np.ones((1, n_frq_rsl)) # Normalizatoin by fn value

        acf_frsp = acf_frsp * h[:,0:n_frq_rsl] / h_norm

    # original Asymmetric Function without shift centering
    fd = np.ones((num_ch, 1))*freq - frs*np.ones((1, n_frq_rsl))
    be = (b * erbw) * np.ones((1, n_frq_rsl))
    cc = (c * np.ones((num_ch, 1)) * np.ones((1, n_frq_rsl))) # in case when c is scalar
    asym_func = np.exp(cc * np.arctan2(fd, be))

    return acf_frsp, freq, asym_func


class classACFstatus:
        NumCh = []
        NumFilt = []
        Lbz = []
        Lap = []
        SigInPrev = []
        SigOutPrev = []
        Count = []


def acfilterbank(ACFcoef, ACFstatus, sig_in=[], sw_ordr=0):
    """IIR ACF time-slice filtering for time-varing filter

    Args:
        ACFcoef (structure): ACFcoef: coef from make_asym_cmp_filters_v2
            ap: AR coefficents (==a ~= pole) num_ch*lap*num_filt
            fs : sampling rate  (also switch for verbose)
                (The variables named 'a' and 'b' are not used to avoid the
                confusion to the gammachirp parameters.)
            verbose : Not specified) quiet   1) verbose
        ACFstatus (structure):
            NumCh: Number of channels (Set by initialization
            Lbz: size of MA
            Lap: size of AR
            NumFilt: Length of filters
            SigInPrev: Previous status of sig_in
            SigOutPrev: Previous status of SigOut
        sig_in (array_like, optional): Input signal. Defaults to [].
        sw_ordr (int, optional): Switch filtering order. Defaults to 0.

    Returns:
        SigOut (array_like): Filtered signal (num_ch * 1)
        ACFstatus: Current status
    """    

    if len(sig_in) == 0 and len(ACFstatus) != 0:
        help(acfilterbank)
        sys.exit()

    if not hasattr(ACFstatus, 'NumCh'):
        ACFstatus = classACFstatus()

        num_ch, lbz, num_filt = np.shape(ACFcoef.bz)
        num_ch, lap, _ = np.shape(ACFcoef.ap)

        if lbz != 3 or lap !=3:
            print("No gaurantee for usual IIR filters except for AsymCmpFilter.\n"\
                + "Please check make_asym_cmp_filters_v2.")
    
        ACFstatus.NumCh = num_ch
        ACFstatus.NumFilt = num_filt
        ACFstatus.Lbz = lbz # size of MA
        ACFstatus.Lap = lap # size of AR
        ACFstatus.SigInPrev = np.zeros((num_ch, lbz))
        ACFstatus.SigOutPrev = np.zeros((num_ch, lap, num_filt))
        ACFstatus.Count = 0
        print("ACFilterBank: Initialization of ACFstatus")
        sig_out = []

        return sig_out, ACFstatus
    
    if isrow(sig_in):
        sig_in = np.array([sig_in]).T
    
    num_ch_sig, len_sig = np.shape(sig_in)
    if len_sig != 1:
        print("Input signal sould be num_ch*1 vector (1 sample time-slice)", file=sys.stderr)
        sys.exit(1)
    if num_ch_sig != ACFstatus.NumCh:
        print(f"num_ch_sig ({num_ch_sig}) != ACFstatus.NumCh ({ACFstatus.NumCh})")

    # time stamp
    if hasattr(ACFcoef, 'verbose'):
        if ACFcoef.verbose == 1: # verbose when ACFcoef.verbose is specified to 1
            t_disp = 50 # ms
            t_cnt = ACFstatus.Count/(np.fix(ACFcoef.fs/1000)) # ms

            if ACFstatus.Count == 0:
                print("ACFilterBank: Start processing")
                tic = time.time()

            elif np.mod(t_cnt, t_disp) == 0:
                toc = time.time()
                print("ACFilterBank: Processed {} (ms). elapsed Time = {} (sec)"\
                    .format(t_cnt, np.round(tic-toc, 1)))
    
    ACFstatus.Count = ACFstatus.Count+1
    
    """
    Processing
    """
    ACFstatus.SigInPrev = np.concatenate([ACFstatus.SigInPrev[:, 1:ACFstatus.Lbz], sig_in], axis=1)

    x = ACFstatus.SigInPrev.copy()
    nfilt_list = np.arange(ACFstatus.NumFilt)

    if sw_ordr == 1:
        nfilt_list = np.flip(nfilt_list)

    for nfilt in nfilt_list:

        forward = ACFcoef.bz[:, ACFstatus.Lbz::-1, nfilt] * x
        feedback = ACFcoef.ap[:, ACFstatus.Lap:0:-1, nfilt] * \
            ACFstatus.SigOutPrev[:, 1:ACFstatus.Lap, nfilt]

        fwdSum = np.sum(forward, axis=1)
        fbkSum = np.sum(feedback, axis=1)

        y = np.array([(fwdSum - fbkSum) / ACFcoef.ap[:, 0, nfilt]]).T
        ACFstatus.SigOutPrev[:, :, nfilt] = \
            np.concatenate([ACFstatus.SigOutPrev[:, 1:ACFstatus.Lap, nfilt], y], axis=1)
        x = ACFstatus.SigOutPrev[:, :, nfilt].copy()

    sig_out = y

    return sig_out, ACFstatus


def fftfilt(b, x):
    """FFTFILT Overlap-add method for FIR filtering using FFT.

    Args:
        b (array_like): Impulse response of the filter
        x (array_like): Input signal

    Returns:
        y (array_like): Output signal filtered

    Note: 
        This code is based on the "fftfilt" fuction of Matlab.
    """    

    if isrow(x):
        xCol = np.array([x]).T
    else:
        xCol = x.copy()
    nx, mx = np.shape(xCol)
    
    if isrow(b):
        bCol = np.array([b]).T
    else:
        bCol = b.copy()
    nb, mb = np.shape(bCol)

    # figure out which nfft and L to use
    if nb >= nx or nb > 2**20:
        nfft = 2<<(nb + nx -1).bit_length()
        L = nx
    else:
        fftflops = np.array([18, 59, 138, 303, 660, 1441, 3150, 6875, 14952, 32373,\
                             69762, 149647, 319644, 680105, 1441974, 3047619, 6422736, \
                             13500637, 28311786, 59244791, 59244791*2.09])
        n = 2**np.arange(1,22)
        nValid = n[n > nb-1]
        fftflopsValid = np.extract([n > nb-1], fftflops)
        # minimize (number of blocks) * (number of flops per fft)
        L1 = nValid - (nb - 1)
        ind = np.argmin(np.ceil(nx/L1)*fftflopsValid)
        nfft = nValid[ind] # must have nfft > (nb-1)
        L = L1[ind]
    
    B = np.fft.fft(bCol, n=nfft, axis=0)
    if iscolumn(bCol):
        B1 = B[:, 0]
    else:
        B1 = B
    if iscolumn(xCol):
        xCol1 = xCol[:, 0]
    else:
        xCol1 = xCol

    y1 = np.zeros(np.shape(xCol), dtype=np.complex)
    istart = 0
    while istart <= nx:
        iend = np.minimum(istart+L, nx)
        if (iend - (istart-1)) == 0:
            X = xCol1[istart[np.ones((nfft, 1))]] # need to fft a scalar
        else:
            X = np.fft.fft(xCol1[istart:iend], n=nfft, axis=0)
        Y = np.fft.ifft(X*B1, n=nfft, axis=0)
        yend = np.minimum(nx, istart+nfft)
        y1[istart:yend, 0] = y1[istart:yend, 0] + Y[0:(yend-istart)]
        istart += L
    
    if not any(np.imag(b)) or not any(np.imag(x)):
        y1 = np.real(y1)
    
    if isrow(x) and iscolumn(y1):
        y = y1[:,0]
    else:
        y = y1

    return y