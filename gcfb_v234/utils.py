# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import wave as wave
from scipy.interpolate import UnivariateSpline, interp1d
from scipy import signal
from functools import lru_cache
from dataclasses import dataclass, field
from typing import List

@dataclass
class param_trans_func:
    fs: int = 48000
    n_frq_rsl: int = 2048
    freq_calib: int = 1000 # SPL at ear drum is calibrated at 1000 Hz
    type_field2eardrum: str = ''
    type_midear2cochlea: str = ''
    type_field2cochlea_db: str = ''
    name_filter: str = ''
    sw_plot: int = 0
    sw_get_table: int = 0

@dataclass
class out_trans_func:
    fs: int = 48000
    freq_calib: int = 1000  # SPL at ear drum is calibrated at 1000 Hz
    freq: List[float] = field(default_factory=list)
    field2eardrum_db: List[float] = field(default_factory=list)
    field2eardrum_db_at_freq_calib: List[float] = field(default_factory=list)
    field2eardrum_db_cmpnst_db: List[float] = field(default_factory=list)
    field2coclea: List[float] = field(default_factory=list)
    field2cochlea_db: List[float] = field(default_factory=list)
    field2cochlea_db_at_freq_calib: List[float] = field(default_factory=list)
    midear2cochlea_db: List[float] = field(default_factory=list)
    midear2cochlea_db_at_freq_calib: List[float] = field(default_factory=list)
    type_field2cochlea_db: str = ''
    type_field2eardrum: str = 'FreeField'
    type_midear2cochlea: str = 'MiddleEar_Moore16'

@dataclass
class SPLatHL0dB():
    freq: List[float] = field(default_factory=list)
    spl_db_at_hl_0db: List[float] = field(default_factory=list)
    speech: float = 20.0
    standard: str = ''
    earphone = ''
    artifial_ear = ''


def audioread(filepath):
    """Reads a wavfile as a float

    Args:
        filepath (string): Filepath to the input wav file

    Returns:
        snd (array_like): Sound signal as a float and normalized scale (-1 ~ +1) 
        fs (int): Sampling frequency
    """    
    with wave.open(filepath, 'rb') as wav:
        fs = wav.getframerate() # sampling frequency
        snd = wav.readframes(wav.getnframes()) # audio sound

    snd = np.frombuffer(snd, dtype=np.int16) # int16 (-32768 ~ +32767)
    snd = snd/abs(np.iinfo(np.int16).min) # float (-1 ~ +1)

    return snd, fs


def rms(x):
    """Caliculates a root-mean-squared (RMS) value of input signal (1D)

    Args:
        x (array_like): Input signal (1D)

    Returns:
        y (float): RMS value
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


def eqlz2meddis_hc_level(snd_in, out_level_db=None, input_rms1_dbspl=None):
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

    if input_rms1_dbspl == None:
        # conventional methods 2004~2021 (v231)
        source_level = np.sqrt(np.mean(snd_in**2)) * 10**(30/20) # level in terms of Meddis HC Level

        # amplifiy the source snd based on the Meddis HC lavel
        amp_cmpnst = (10**(out_level_db/20))/source_level
        snd_eq_meddis = amp_cmpnst * snd_in

        source_level_db = 20*np.log10(source_level)
        cmpnst_db = 20*np.log10(amp_cmpnst)

    else:
        # Alternative method 2022(v233)~, more pricise
        source_level_db = 20*np.log10(np.sqrt(np.mean(snd_in**2))) + input_rms1_dbspl
        out_level_db = source_level_db # It is invarient. Just signal level becomes Meddis HC level.

        cmpnst_db = input_rms1_dbspl - 30 # rms(s(t)) == 1 should become 30 dB SPL
        amp_cmpnst = 10**(cmpnst_db/20)
        snd_eq_meddis = amp_cmpnst * snd_in

    # summarize information
    amp_db = [out_level_db, cmpnst_db, source_level_db]

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


def freq2erb(cf, warning=0):
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


# @lru_cache(maxsize=None)
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

    win, _ = taper_window(len(fir_coef), 'HAN', len_coef/10)
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
        n_rsl = 1024
        freq2, frsp = signal.freqz(fir_coef, 1, n_rsl, fs=sr)

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
        # np.hanning: does not match to Matlab 'hannig()'
        # https://stackoverflow.com/questions/56485663/hanning-window-values-doesnt-match-in-python-and-matlab
        taper = np.hanning(len_taper*2+1 +2)[1:-1]
        type_taper = 'Hanning/Cosine'

    elif type_taper == 'BLA':
        taper = np.blackman(len_taper*2+1)
        type_taper = 'Blackman'

    elif type_taper == 'GAU':
        nn = np.arange(-len_taper, len_taper)
        taper = np.exp(-(range_sigma*nn/len_taper)**2 / 2)
        type_taper == 'Gauss'

    else:
        taper = np.array(list(np.arange(1,len_taper+1,1)) + list([len_taper+1]) \
                        + list(np.arange(len_taper,1-1,-1))) / (len_taper+1)
        type_taper = 'Line'

    len_taper = int(np.fix(len_taper))
    taper_win = list(taper[0:len_taper]) + list(np.ones(len_win-len_taper*2)) \
               + list(taper[(len_taper+1):(len_taper*2+1)])

    if sw_plot == 1:
        _, ax = plt.subplots()
        plt.plot(taper_win)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('Points')
        ax.set_ylabel('Amplitude')
        plt.grid()
        plt.title(f"TypeTaper: {type_taper}")

    return taper_win, type_taper


def rceps(x):
    """returns the real cepstrum of the real sequence X

    Args:
        x (array_like): input signal

    Returns:
        cep (array_like): real cepstrum
        min_phase (array_like): a unique minimum-phase sequence that has the real cepstrum as x

    Reference:
        Oppenheim & Schafer (2009) Discrete-Time Signal Processing, 3rd ed. Pearson.
    """
    if isrow(x):
        x_t = np.array([x]).T

    # Cepstrum
    # Compute the Fourier Transform of the signal
    x_spec = np.fft.fft(x_t, n=None, axis=0)
    
    # Take the logarithm of the magnitude of the Fourier Transform
    log_x_spec = np.log(np.abs(x_spec))
    
    # Compute the real cepstrum by taking the inverse Fourier Transform of the log magnitude
    cep = np.real(np.fft.ifft(log_x_spec, n=None, axis=0))
    
    # Minimum-phase reconstruction by Homomorphic filtering (Oppenheim & , )
    # Calculate the asymmetric part of the cepstrum to construct the minimum-phase signal
    n_rows = cep.shape[0]
    n_cols = cep.shape[1]
    odd = n_rows % 2

    # Construct the window function
    a1 = np.array([1])
    a2 = 2*np.ones((int((n_rows+odd)/2)-1, 1))
    a3 = np.zeros((int((n_rows+odd)/2)-1, 1))
    win = np.kron(np.ones((1, n_cols)), np.vstack((a1, a2, a3)))

    # Calculate the minimum-phase signal
    min_phase = np.real(np.fft.ifft(np.exp(np.fft.fft((win * cep),n=None, axis=0)), n=None, axis=0))

    if isrow(x):
        cep = cep[:,0]
        min_phase = min_phase[:,0]

    return cep, min_phase


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


def fftfilt(b, x):
    """Overlap-add method for FIR filtering using FFT.

    Args:
        b (array_like): Impulse response of the filter
        x (array_like): Input signal

    Returns:
        y (array_like): Output signal filtered

    Note:
        This code is implimented based on pambox.utils.fftfilt() in the pambox package.
        https://github.com/achabotl/pambox/blob/develop/pambox/utils.py#L246
    """

    if isrow(x):
        x_t = np.array([x]).T
    if isrow(b):
        b_t = np.array([b]).T

    n_x = len(x_t)
    n_b = len(b_t)

    # figure out which n_fft and l_x to use
    if n_b >= n_x:
        n_fft = 2<<(n_b + n_x -1).bit_length()
    else:
        fft_flops = 2**np.arange(np.ceil(np.log2(n_b)), 27)
        cost = np.ceil(n_x/(fft_flops - n_b + 1)) * fft_flops * (np.log2(fft_flops) + 1)
        n_fft = fft_flops[np.argmin(cost)]
    n_fft = int(n_fft)
    l = int(n_fft - n_b + 1)
    
    # spectral representation of the filter
    b_spec = np.fft.fft(b_t, n=n_fft, axis=0)

    # filtering using overlap-add method
    y = np.zeros(np.shape(x_t), dtype=np.complex128)
    i = 0
    while i <= n_x:
        il = np.min([i + l, n_x])
        k = np.min([i + n_fft, n_x])
        y_t = np.fft.ifft(b_spec * np.fft.fft(x_t[i:il], n=n_fft, axis=0), n=n_fft, axis=0)
        y[i:k] = y[i:k] + y_t[:(k-i)]
        i += l

    if isrow(x):
        y = y[:, 0]

    return np.real(y)


@lru_cache(maxsize=None)
def mk_filter_field2cochlea(str_crct, fs, sw_fwd_bwd=1, sw_plot=0):
    """ Filter of Field to Cochlea with compativility of out_mid_crct_filt

    Args:
        str_crct (string): Filter type (FreeField (FF)/DiffuseField (DF)/ITU/ELC)
        fs (int): Sampling rate
        sw_fwd_bwd (int, optional): _description_. Defaults to 1.
        sw_plot (int, optional): _description_. Defaults to 0.

    Returns:
        fir_coef (array_like): FIR filter coefficients
        param (structure): Filter infomation
            .type_field2eardrum (string): filter type (field~eardrum)
            .type_midear2cochear (string): filter type (midear~coclear)

    Notes:
        In the original Matlab code of MkFilterField2Cochlea.m, persistent variables 
        are called by "FIRCoefFwd_Keep" function. The GammachirPy uses the "lru_cache"  
        instead of the persistent variables to call outputs if args are same 
        to previous one. 

    Reference:
        https://docs.python.org/3/library/functools.html
    """
    if fs > 48000:
        print("OutMidCrctFilt : Sampling rate of {} (Hz) (> 48000 (Hz) is not recommended)".format(fs))
        print("<-- ELC etc. is only defined below 16000 (Hz)")

    param = param_trans_func()
    param.fs = fs

    if str_crct == 'FreeField' or str_crct == 'FF':
        sw_type = 1
        param.type_field2eardrum = 'FreeField'
        param.type_midear2cochlea = 'MiddleEar' # default but specify here for clarity
    elif str_crct == 'DiffuseField' or str_crct == 'DF':
        sw_type = 2
        param.type_field2eardrum = 'DiffuseField'
        param.type_midear2cochlea = 'MiddleEar'
    elif str_crct == 'ITU':
        sw_type = 3
        param.type_field2eardrum = 'ITU'
        param.type_midear2cochlea = 'MiddleEar'
    elif str_crct == 'EarDrum' or str_crct == 'ED':
        sw_type = 4
        # level at EarDrum: NO transfer function of Outer Ear
        param.type_field2eardrum = 'NoField2EarDrum'
        param.type_midear2cochlea = 'MiddleEar'
    elif str_crct == 'ELC': # for backward compativility
        sw_type = 10
        param.type_field2cochlea_db = 'ELC' # for backward compativility
        param.type_field2eardrum = 'NoUse_ELC'
        param.type_midear2cochlea = 'NoUse_ELC'
    else:
        print("Specify: FreeField(FF)/DiffuseField(DF)/ITU/EarDrum(ED)/ELC", \
              file=sys.stderr)
        sys.exit(1)

    if sw_fwd_bwd == 1:
        param.name_filter = "(Forward) FIR minimum phase filter"
    elif sw_fwd_bwd == -1:
        param.name_filter = "(Backward) FIR minimum phase inverse filter"
    else:
        help(mk_filter_field2cochlea)
        print("Specify sw_fwd_bwd: (1) Forward, (-1) Backward", file=sys.stderr)
        sys.exit(1)

    param.name_filter = '[' + str_crct + '] ' + param.name_filter

    # Genarate filter at the first time
    msg = '*** mk_filter_field2cochlea: Generating ' + param.name_filter + ' ***'
    print(msg)

    if sw_type <= 4:
        trans_func, _ = trans_func_field2cochlea(param)
        frsp_crct = 10 ** (trans_func.field2cochlea_db/20)
        freq = trans_func.freq
        param.type_field2cochlea_db = trans_func.type_field2cochlea_db
    elif sw_type == 10:
        # ELC for backward compativility
        n_rslt = 2048
        crct_pwr, freq = out_mid_crct_filt(str_crct, n_rslt, fs, 0)
        frsp_crct = np.sqrt(crct_pwr)
    
    if sw_fwd_bwd == -1: # Backward filter
        frsp_crct = 1 / (np.max(frsp_crct, 0.1))
        # Giving up less then -20dB : f>15000Hz. If required, the response becomes worse.
        # from out_mid_crct_filt()
    
    len_coef = 200 # (-45 dB) <- 300 (-55 dB)
    n_coef = int(np.fix(len_coef / 16000 * fs/2) * 2) # fs dependent length, even number only
    
    """ 
    Calculate the minimax optimal filter with a frequency response
    instead of "fir_coef = firpm(n_coef,freq/1600*2,frsp_crct)" 
    in the original code out_mid_crct_filt.m
    """
    x1 = np.array(np.arange(len(freq))).T * 2
    x2 = np.array(np.arange(len(freq)*2)).T
    freq_interp = np.interp(x2, x1, freq)
    fir_coef = signal.remez(n_coef+1, freq_interp, frsp_crct, fs=fs)

    win, _ = taper_window(len(fir_coef), 'han', len_coef/10) # necessary to avoid sprious
    fir_coef = win * fir_coef

    # Minimum phase reconstruction -- important to avoid pre-echo
    _, x_mp = rceps(fir_coef)
    fir_coef = x_mp[0:int(np.fix(len(x_mp)/2))] # half length is sufficicent

    """
    Plot
    """
    if sw_plot == 1:
        n_rsl = len(frsp_crct)
        freq2, frsp = signal.freqz(fir_coef, 1, n_rsl, fs=fs)

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        plt.plot(fir_coef)
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        #ax1.set_xlim([0, 300])
        #ax1.set_ylim([-0.3, 0.3])
        ax1.set_title('Type: ' + param.type_field2eardrom)
        
        ax2 = fig.add_subplot(2, 1, 2)
        plt.plot(freq2, abs(frsp), freq, frsp_crct, '--')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude (linear term)')
        #ax2.set_xlim([0, 25000])
        #ax2.set_ylim([0, 1.8])
        elc_error = np.mean(((np.abs(frsp) - frsp_crct)**2) / np.mean(frsp_crct ** 2))
        elc_error_db = 10*np.log10(elc_error) # corrected

        print(f"Fitting Error: {elc_error_db} (dB)")
        if elc_error_db > -30:
            print("Warning: Error in ELC correction = {} dB > -30 dB".format(elc_error_db))

    return fir_coef, param


def trans_func_field2cochlea(param_in):
    """ Transfer function from field to cochlear input

    Args:
        param_in (struct): input parameters
            .type_field2eardrum (str): string for transfer function / headphones
            .type_midear2coclea (str): string for middle ear transfer fuction from the ear drum to coclear
            .n_frq_rsl (int): number of data points, if zero, then direct out
            .fs (int): sampling frequency
            .freq_calib (int): frequency at which SPL at the ear drum is calibrated
            .sw_plot: swicth for plot

    Returns:
        trans_func (struct): structure of transfer function
            .freq: corresponding frequency at the data point
            .field2cochlea_db: transfer functions in dB from field to coclea
            .field2cochlea_db_at_freq_calib: transfer fuction at the calibration frequency
            .type_field2cochlea_db: 

    Note:
        "*_AMLAB15/*_AMLAB16" option in type_field2eardrum_list was omitted
        because the options use special measurement data. 
    """
    """ *_AMLAB15/*_AMLAB16: Omitted
    if sw_crct >= 5:
        ...
    """
    trans_func = out_trans_func()

    param_out = param_in

    freq = np.arange(0, param_in.n_frq_rsl, 1) / param_in.n_frq_rsl * param_in.fs/2
    trans_func.freq = freq

    # Field to eardrum
    type_field2eardrum_list = [
        'FreeField2EarDrum_Moore16', 
        'DiffuseField2EarDrum_Moore16',  
        'ITUField2EarDrum', 
        'NoField2EarDrum',           # EarDrum direct 6 Feb 2022
        ]
    
    # Find a type in the list
    crct = [(i, s) for i, s in enumerate(type_field2eardrum_list) if param_in.type_field2eardrum in s]

    if len(crct) == 0:
        print('Select "type_field2eardrum" from one of: \n')
        print('\n'.join(type_field2eardrum_list), file=sys.stderr)
        sys.exit(1)

    sw_crct = crct[0][0] # index
    str_crct = crct[0][1] # type
    trans_func.type_field2eardrum = str_crct

    # str_interp1 = 'cubic' # cublic spline ('spline' in Matlab)
    str_interp1 = 'linear' # better than spline (26 Oct '21)

    if sw_crct <= 2:
        freq_tbl, frsp_db_tbl, _ = trans_func_field2eardrum_set(trans_func.type_field2eardrum)

        if param_in.fs/2 > np.max(freq_tbl):
            freq_tbl = np.append(freq_tbl, param_in.fs/2)
            frsp_db_tbl = np.append(frsp_db_tbl, frsp_db_tbl[-1]) # insert the final value at fs/2
        # interporate frsp_db_tbl in erb-scale
        freq_tbl_erb, _ = freq2erb(freq_tbl)
        freq_erb, _ = freq2erb(freq)
        func_interp1d = interp1d(freq_tbl_erb, frsp_db_tbl, kind=str_interp1, fill_value='extrapolate')
        field2eardrum_db = func_interp1d(freq_erb)

    elif sw_crct == 3: # NoField2EarDrum
        field2eardrum_db = np.zeros(np.size(freq)) # == 0 dB

    else:
        print('Select "sw_crct" from 1 to 4 (sw_crct <= 4): \n', file=sys.stderr)
        sys.exit(1)
    
    # Compensate to 0 dB at param_in.freq_calib
    # find bin number of param_in.freq_calib
    idx_freq_calib = np.argmin(np.abs(freq - param_in.freq_calib))

    trans_func.freq_calib = freq[idx_freq_calib]
    trans_func.field2eardrum_db = field2eardrum_db - field2eardrum_db[idx_freq_calib]
    trans_func.field2eardrum_db_at_freq_calib = trans_func.field2eardrum_db[idx_freq_calib]
    trans_func.field2eardrum_db_cmpnst_db = field2eardrum_db[idx_freq_calib]

    """
    Ear Drum to cochlea
    """
    if not param_in.type_midear2cochlea == 'MiddleEar':
        print(f'Not prepared yet: {param_in.type_midear2cochlea}', file=sys.stderr)
        sys.exit(1)
    else:
        freq_tbl2, frsp_db_tbl2 = trans_func_middle_ear_moore16()
        if param_in.fs/2 > np.max(freq_tbl2):
            freq_tbl2 = np.append(freq_tbl2, param_in.fs/2)
            frsp_db_tbl2 = np.append(frsp_db_tbl2, frsp_db_tbl2[-1]) # insert the final value at fs/2

        # interporate frsp_db_tbl in erb-scale
        freq_tbl2_erb, _ = freq2erb(freq_tbl2)
        freq_erb2, _ = freq2erb(freq)
        func_interp1d = interp1d(freq_tbl2_erb, frsp_db_tbl2, kind=str_interp1, fill_value='extrapolate')
        midear2coclea_db = func_interp1d(freq_erb2)

        trans_func.midear2cochlea_db = midear2coclea_db
        trans_func.midear2cochlea_db_at_freq_calib = midear2coclea_db[idx_freq_calib]
        trans_func.type_midear2cochlea = 'MiddleEar_Moore16'

    """
    Total: field to coclea
    """
    trans_func.field2cochlea_db = trans_func.field2eardrum_db + trans_func.midear2cochlea_db
    trans_func.field2cochlea_db_at_freq_calib = trans_func.field2cochlea_db[idx_freq_calib]
    trans_func.type_field2cochlea_db = trans_func.type_field2eardrum + ' + ' + trans_func.type_field2cochlea_db
    
    print('type_field2cochlea_db: ' + trans_func.type_field2eardrum)

    print(f'trans_func.freq_at_freq_calib = {trans_func.freq_calib} Hz ( <-- {param_in.freq_calib} Hz )')
    print(f'trans_func.field2eardrum_db_at_freq_calib = {trans_func.field2eardrum_db_at_freq_calib:.3f} dB')
    print(f'                            (Compensated for {trans_func.field2eardrum_db_cmpnst_db:.3f} dB)')
    print(f'trans_func.midear2cochlea_db_at_freq_calib = {trans_func.midear2cochlea_db_at_freq_calib:.3f} dB')
    print(f'trans_func.field2cochlea_db_at_freq_calib = {trans_func.field2cochlea_db_at_freq_calib:.3f} dB')

    """
    Plot data
    """
    if param_in.sw_plot == 1:
        fig = plt.figure(figsize=[7.5, 12], tight_layout=True)

        # frequency response: from field to eardrum
        ax1 = fig.add_subplot(311)
        ax1.semilogx(freq, trans_func.field2eardrum_db, 
                     trans_func.freq_calib, trans_func.field2eardrum_db_at_freq_calib, 'rx')
        if sw_crct <= 2:
            ax1.semilogx(freq_tbl, frsp_db_tbl, 'o')
            ax1.autoscale(enable=True, axis='both', tight=True)
        ax1.text(trans_func.freq_calib*0.9, trans_func.field2eardrum_db_at_freq_calib-3, 
                 f'{trans_func.field2eardrum_db_at_freq_calib:5.2f} dB')
        ax1.grid(b=True, which='both')
        ax1.set_xlabel('Freqnency (Hz)')
        ax1.set_ylabel('Gain (dB)')
        ax1.set_xlim(10, 30000)
        ax1.set_ylim(-20, 20)
        str_title = f'Frequeny response: {trans_func.type_field2eardrum}, ' \
                    + f'Gain normalized at {trans_func.freq_calib:.0f} (Hz)'
        ax1.set_title(str_title)

        # freqnency respose: from midear to coclear
        ax2 = fig.add_subplot(312)
        ax2.semilogx(freq, trans_func.midear2cochlea_db, freq_tbl2, frsp_db_tbl2, 'ro')
        ax2.semilogx(trans_func.freq_calib, -0.3, 'r-v')
        ax2.semilogx(trans_func.freq_calib, trans_func.field2cochlea_db_at_freq_calib+1, 'r-v')
        ax2.autoscale(enable=True, axis='both', tight=True)
        ax2.text(trans_func.freq_calib*1.08, trans_func.field2cochlea_db_at_freq_calib/2, 
                 f'{trans_func.field2cochlea_db_at_freq_calib:5.2f} dB')
        ax2.grid(b=True, which='both')
        ax2.set_xlabel('Freqnency (Hz)')
        ax2.set_ylabel('Gain (dB)')
        ax2.set_xlim(10, 30000)
        ax2.set_ylim(-30, 10)
        ax2.set_title(f'Frequeny response: {trans_func.type_midear2cochlea}')

        # frequency response: field2cochlea
        ax3 = fig.add_subplot(313)
        ax3.semilogx(freq, trans_func.field2cochlea_db)
        ax3.grid(b=True, which='both')
        ax3.set_xlabel('Freqnency (Hz)')
        ax3.set_ylabel('Gain (dB)')
        ax3.set_xlim(10, 30000)
        ax3.set_ylim(-30, 10)
        ax3.set_title('Frequency response: total transfer function from field to coclea')

    return trans_func, param_out


def trans_func_field2eardrum_set(str_crct, freq_list=None):
    """ Various set of transfer functions from field to ear drum

    Args:
        str_crct (str): Name of transer function from field to eardrum
        freq_list (array, optional): Frequency list to pick up. Defaults to None.

    Returns:
        freq_tbl(array): table of frequencies
        frsp_db_tbl (array): table of frequency responses
        type_field2eardcum (str): type of transfer function from field to eardrum
    """    
    if str_crct == 'FreeField2EarDrum_Moore16':
        type_field2eardrum = 'FreeField'
        freq_tbl, frsp_db_tbl = trans_func_free_field2eardrum_moore16(type_field2eardrum)

    elif str_crct == 'DiffuseField2EarDrum_Moore16':
        type_field2eardrum = 'DiffuseField'
        freq_tbl, frsp_db_tbl = trans_func_free_field2eardrum_moore16(type_field2eardrum)

    elif str_crct == 'ITUField2EarDrum':
        type_field2eardrum = 'ITU'
        freq_tbl, frsp_db_tbl = trans_func_free_field2eardrum_itu(type_field2eardrum)

    else:
        print('Specify: FreeField(FF)/DiffuseField(DF)/ITU', file=sys.stderr)
        sys.exit(1)
    
    if not freq_list == None:
       # selection of freq_list
       for i_freq, freq in enumerate(freq_list):
            try:
                j_freq = freq_list.index(freq)
            except ValueError:
                print("Freq {} is not listed on the table".format(freq), file=sys.stderr)
                sys.exit(1)
            freq_tbl[i_freq] = freq_tbl[j_freq]
            frsp_db_tbl[i_freq] = frsp_db_tbl[j_freq]
                    
    return freq_tbl, frsp_db_tbl, type_field2eardrum


def trans_func_free_field2eardrum_moore16(type_field2eardrum):
    """ Transfer function from field to ear drum based on [Moore16]

    Args:
        type_field2eardrum (str): type of field to eardrum (FreeField/DiffuseField)

    Returns:
        freq_tbl (array): table of frequencies
        frsp_tbl (array): table of frequency responses
    
    Notes and Referencfes:
        Information about the middle ear transfer function from BJC Moore
        Puria, S., Rosowski, J. J., Peake, W. T., 1997. Sound-pressure measurements 
        in the cochlear vestibule of human-cadaver ears. J. Acoust. Soc. Am. 101, 2754-2770.
        Aibara, R., Welsh, J. T., Puria, S., Goode, R. L., 2001.
        Human middle-ear sound transfer function and cochlear input impedance.
        Hear. Res. 152, 100-109.
        
        However, its exact form was chosen so that our model of loudness perception would
        give accurate predictions of the absolute threshold, as described in:
        Glasberg, B. R., Moore, B. C. J., 2006.
        Prediction of absolute thresholds and equal-loudness contours using
        a modified loudness model. J. Acoust. Soc. Am. 120, 585-588.
    """
    if type_field2eardrum == 'FreeField': 
        table = np.array([
            [20, 0.0],
            [25, 0.0],
            [31.5, 0.0],
            [40, 0.0],
            [50, 0.0],
            [63, 0.0],
            [80, 0.0],
            [100, 0.0],
            [125, 0.1],
            [160, 0.3],
            [200, 0.5],
            [250, 0.9],
            [315, 1.4],
            [400, 1.6],
            [500, 1.7],
            [630, 2.5],
            [750, 2.7],
            [800, 2.6],
            [1000, 2.6],
            [1250, 3.2],
            [1500, 5.2],
            [1600, 6.6],
            [2000, 12.0],
            [2500, 16.8],
            [3000, 15.3],
            [3150, 15.2],
            [4000, 14.2],
            [5000, 10.7],
            [6000, 7.1],
            [6300, 6.4],
            [8000, 1.8],
            [9000, -0.9],
            [10000, -1.6],
            [11200, 1.9],
            [12500, 4.9],
            [14000, 2.0],
            [15000, -2.0],
            [16000, 2.5]
            ])

    elif type_field2eardrum == 'DiffuseField':
        table = np.array([
            [20, 0.0],
            [25, 0.0],
            [31.5, 0.0],
            [40, 0.0],
            [50, 0.0],
            [63, 0.0],
            [80, 0.0],
            [100, 0.0],
            [125, 0.1],
            [160, 0.3],
            [200, 0.4],
            [250, 0.5],
            [315, 1.0],
            [400, 1.6],
            [500, 1.7],
            [630, 2.2],
            [750, 2.7],
            [800, 2.9],
            [1000, 3.8],
            [1250, 5.3],
            [1500, 6.8],
            [1600, 7.2],
            [2000, 10.2],
            [2500, 14.9],
            [3000, 14.5],
            [3150, 14.4],
            [4000, 12.7],
            [5000, 10.8],
            [6000, 8.9],
            [6300, 8.7],
            [8000, 8.5],
            [9000, 6.2],
            [10000, 5.0],
            [11200, 4.5],
            [12500, 4.0],
            [14000, 3.3],
            [15000, 2.6],
            [16000, 2.0]
            ])        
    else:
        help(trans_func_free_field2eardrum_moore16)
        print('Specify type_field2eardrum, "FreeField"/"DiffuseField"', file=sys.stderr)
        sys.exit(1)

    freq_tbl = table[:, 0]
    frsp_db_tbl = table[:, 1]

    return freq_tbl, frsp_db_tbl


def trans_func_free_field2eardrum_itu(type_field2eardrum):
    """ ITU Rec P 58 08/96 Head and Torso Simulator transfer fns. from Peter Hugher BTRL, 4-June-2001.
    Args:
        type_field2eardrum (str): type of field to eardrum ("ITU")

    Returns:
        freq_tbl (array): table of frequencies
        frsp_tbl (array): table of frequency responses

    Notes: 
        Negative of values in Table 14a of ITU P58 (05/2013), accesible at  http://www.itu.int/rec/T-REC-P.58-201305-I/en
        Freely available. Converts from ear reference point (ERP) to eardrum reference point (DRP)
        EXCEPT extra 2 points added for 20k & 48k by MAS, MAr 2012
    """    
    if type_field2eardrum == 'ITU':
        # remove 20000 Hz and 48000 Hz
        freq_tbl = np.array([0, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 
                             1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000])
        frsp_db_tbl = np.array([0, 0, 0, 0, 0, 0.3, 0.2, 0.5, 0.6, 0.7, 1.1, 1.7, 
                                2.6, 4.2, 6.5, 9.4, 10.3, 6.6, 3.2, 3.3, 16, 14.4])

    else:
        help(trans_func_free_field2eardrum_itu)
        print('Specify type_field2eardrum, "ITU"', file=sys.stderr)
        sys.exit(1)

    return freq_tbl, frsp_db_tbl


def trans_func_middle_ear_moore16(freq_list=None):
    """ Transfer function from middle ear to cochlear input

    Args:
        freq_list (array, optional): Frequency list to pick up. Defaults to None.

    Returns:
        freq_tbl (array): table of frequencies
        frsp_tbl (array): table of frequency responses

    Notes and References:
        Information about the middle ear transfer function from BJC Moore
        Puria, S., Rosowski, J. J., Peake, W. T., 1997. Sound-pressure measurements 
        in the cochlear vestibule of human-cadaver ears. J. Acoust. Soc. Am. 101, 2754-2770.
        Aibara, R., Welsh, J. T., Puria, S., Goode, R. L., 2001.
        Human middle-ear sound transfer function and cochlear input impedance.
        Hear. Res. 152, 100-109.
        
        However, its exact form was chosen so that our model of loudness perception would
        give accurate predictions of the absolute threshold, as described in:
        Glasberg, B. R., Moore, B. C. J., 2006.
        Prediction of absolute thresholds and equal-loudness contours using
        a modified loudness model. J. Acoust. Soc. Am. 120, 585-588.
    """    
    table = np.array([
        [20, -39.6],
        [25, -32.0],
        [31.5, -25.85],
        [40, -21.4],
        [50, -18.5],
        [63, -15.9],
        [80, -14.1],
        [100, -12.4],
        [125, -11.0],
        [160, -9.6],
        [200, -8.3],
        [250, -7.4],
        [315, -6.2],
        [400, -4.8],
        [500, -3.8],
        [630, -3.3],
        [750, -2.9],
        [800, -2.6],
        [1000, -2.6],
        [1250, -4.5],
        [1500, -5.4],
        [1600, -6.1],
        [2000, -8.5],
        [2500, -10.4],
        [3000, -7.3],
        [3150, -7.0],
        [4000, -6.6],
        [5000, -7.0],
        [6000, -9.2],
        [6300, -10.2],
        [8000, -12.2],
        [9000, -10.8],
        [10000, -10.1],
        [11200, -12.7],
        [12500, -15.0],
        [14000, -18.2],
        [15000, -23.8],
        [16000, -32.3],
        [18000, -45.5],
        [20000, -50.0],
        ])

    if freq_list == None:
        freq_tbl = table[:, 0]
        frsp_db_tbl = table[:, 1]
    elif freq_list.ndim == 0:
        # scalar value
        try:
            j_freq = np.where(table[:, 0] == freq_list)
        except ValueError:
            print(f"freq {freq} is not listed on the table", file=sys.stderr)
            sys.exit(1)
        freq_tbl = freq_list
        frsp_db_tbl = table[j_freq, 1][0][0]
    else:
        # vector
        # selection from freq_table
        freq_tbl = np.zeros(1, len(freq_list))
        freq_db_tbl = np.zeros(1, len(freq_list))
        for i_freq, freq in enumerate(freq_list):
            try:
                j_freq = np.where(table[:, 0] == freq)
            except ValueError:
                print(f"freq {freq} is not listed on the table", file=sys.stderr)
                sys.exit(1)
            freq_tbl[i_freq] = freq_tbl[j_freq, 0][0][0]
            frsp_db_tbl[i_freq] = frsp_db_tbl[j_freq, 1][0][0]
    return freq_tbl, frsp_db_tbl


def hl2pin_cochlea(freq, hl_db):
    """ HL to  Pinput level at cochlea

    Args:
        freq (int): input frequency
        hl_db (int/float): hearing lelvel (dB)

    Returns:
        pin_cohlea_db (float): cochlea input level (db)
    """    
    spl_db = hl2spl(freq, hl_db)
    _, frsp_me_db = trans_func_middle_ear_moore16(freq)
    pin_cochlea_db = spl_db + frsp_me_db

    return pin_cochlea_db


def hl2spl(freq, hl_db):
    """ Convert HL level to SPL

    Args:
        freq (array_like): input frequency
        hl_db (array_like): hearing levels in dB

    Returns:
        spl_db (array_like): SPL in dB
    """    
    table1 = spl_at_hl_0db_table()
    freq_ref = table1.freq
    spl_db_at_hl_0db = table1.spl_db_at_hl_0db

    # search correspond frequency
    i_freq = np.where(freq_ref == freq)
    if len(i_freq) == 0:
        print(f"Frequency should be one of 125*2^n & 750*n (Hz) <= 8000.", file=sys.stderr)
        sys.exit(1)

    spl_db = hl_db + spl_db_at_hl_0db[i_freq]

    return spl_db


def spl_at_hl_0db_table():
    """ Table of corresponding between HL and SPL

    Returns:
        table (struct): corresponding data 
    """    
    table = SPLatHL0dB()
    
    # ANSI-S3.6_2010 (1996)
    freq_spl_db_at_hl0db_table = np.array([
        [125, 45.0], 
        [160, 38.5], 
        [200, 32.5], 
        [250, 27.0], 
        [315, 22.0], 
        [400, 17.0], 
        [500, 13.5], 
        [630, 10.5], 
        [750, 9.0], 
        [800, 8.5], 
        [1000, 7.5], 
        [1250, 7.5], 
        [1500, 7.5], 
        [1600, 8.0], 
        [2000, 9.0], 
        [2500, 10.5], 
        [3000, 11.5], 
        [3150, 11.5], 
        [4000, 12.0], 
        [5000, 11.0], 
        [6000, 16.0], 
        [6300, 21.0], 
        [8000, 15.5]
        ])
    speech = 20.0 # ANSI-S3.6_2010

    table.freq = freq_spl_db_at_hl0db_table[:, 0]
    table.spl_db_at_hl_0db = freq_spl_db_at_hl0db_table[:, 1]
    table.speech = speech
    table.standard = 'ANSI-S3.6_2010'
    table.earphone = 'Any supra aural earphone having the characteristics described \
        in clause 9.1.1 or ISO 389-1'
    table.artifial_ear = 'IEC 60318-1'

    return table


def interp1(x, y, x_new, method='linear', extrapolate=False):
    """ 1-dim interporation like the 'interp1' function of MATLAB. 

    Args:
        x (array_like): original values on x-axis
        y (array_like): original values on y-axis
        x_new (array_like): new values on x-axis
        method (str, optional): 'kind' in 'interp1d'. Defaults to 'linear'.
        extrapolate (logic, optional): 'fill value' in 'interp1d'. Defaults to False.

    Returns:
        y_new (array_like): new values on y-axis
    """
    if method == 'spline':
        method = 'cubic'

    func_interp1d = interp1d(x, y, kind=method, fill_value='extrapolate' if extrapolate else None)
    y_new = func_interp1d(x_new)

    return y_new


def eqlz_gcfb2rms1_at_0db(gc_val, str_floor=None):
    """ Nomalizing output level of dcGC (absolute threshold 0 dB == rms of 1)

    Args:
        gc_val (array_like): output of gcfb_v23* (rms(snd) == 1 --> 30 dB)
        str_floor (str, optional): flooring. Defaults to None.
            'NoiseFlooe': adding Gauss noise (rms(randn)==1)
            'ZeroFloor': set 0 for the value less than 1

    Raises:
        ValueError: Specify "str_floor" properly: "NoiseFloor" or "ZeroFloor

    Returns:
        gc_re_at : GC relative to AbsThreshold 0dB (rms(snd) == 1 --> 0 dB)

    Note:
        snd --> eqlz2meddis_hc_level --> gcfb
            GC output level is the same as the meddis_hc_level as shown below
        This function converts the level from meddis_hc_level to 
            rms(s[t]) == np.sqrt(mean(s**2)) == 1 --> 0 dB
        Use this when the absolute threshold is set to 0 dB as in gcfb_v23*
        gcfb --> eqlz_gcfb2rms1_at_0db --> gcfb_eqlz
    """    
    meddis_hc_level_db_rms1 = 30 # used in gcfb level set
    gc_re_at = 10**(meddis_hc_level_db_rms1/20) * gc_val

    if not str_floor == None:
        if str_floor == 'NoiseFloor':
            gc_re_at = gc_re_at + np.random.randn(gc_re_at.shape) # add Gauss noise
        elif str_floor == 'ZeroFloor':
            gc_re_at = np.maximum(gc_re_at-1, 0) # cut-off value less than 1 
        else:
            raise ValueError('Specify "str_floor" properly: "NoiseFloor" or "ZeroFloor')

    return gc_re_at