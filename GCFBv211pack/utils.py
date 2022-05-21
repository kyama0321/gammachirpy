# -*- coding: utf-8 -*-
import sys
from tkinter import SW
import numpy as np
import matplotlib.pyplot as plt
import wave as wave
from scipy.interpolate import UnivariateSpline
from scipy import signal
from functools import lru_cache

from sklearn.mixture import GaussianMixture


def audioread(filepath):
    """Reads a wavfile as a float 
        
        Parameters
        ----------
        filepath: string
            Filepath to the input wav file

        Returns
        ----------
        wavSnd: float
            Sound signal as a float and normalized scale (-1 ~ +1) 
        wavFs: 
            Sampling frequency
    """
    wav = wave.open(filepath)
    wavFs = wav.getframerate() # sampling frequency
    wavSnd = wav.readframes(wav.getnframes())
    wavSnd = np.frombuffer(wavSnd, dtype=np.int16) # int16 (-32768 ~ +32767)
    wav.close()
    wavSnd = wavSnd/abs(np.iinfo(np.int16).min) # float (-1 ~ +1)
    return wavSnd, wavFs


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


def Eqlz2MeddisHCLevel(Snd, OutLeveldB, *args):
    """ Equalizing Signal RMS Level to the Level for MeddisHairCell

    Args:
        Snd (float): Input sound
        OutLeveldB (float): Output level (No default value,  RMS level)

    Returns:
        SndEqM (float): Equalized Sound (rms value of 1 is 30 dB SPL)
        AmpdB (array_like): 3 values in dB, [OutputLevel_dB, CompensationValue_dB, SourceLevel_dB]

    Matlab examples:
        rms(s(t)) == sqrt(mean(s.^2)) == 1   --> 30 dB SPL
        rms(s(t)) == sqrt(mean(s.^2)) == 10  --> 50 dB SPL
        rms(s(t)) == sqrt(mean(s.^2)) == 100 --> 70 dB SPL  

    Reference:
        Meddis (1986), JASA, 79(3),pp.702-711.
    """
    SourceLevel = np.sqrt(np.mean(Snd**2)) * 10**(30/20) # level in terms of Meddis Level

    Amp = (10**(OutLeveldB/20))/SourceLevel
    SndEqM = Amp * Snd

    AmpdB = [OutLeveldB, 20*np.log10(Amp), 20*np.log10(SourceLevel)]

    return SndEqM, AmpdB


def EqualFreqScale(NameScale, NumCh, RangeFreq):
    """Calculation of Equal Frequency scale on ERB/Mel/Log/Linear scale

    Args:
        NameScale (string): 'ERB', 'mel', 'log', 'linear'
        NumCh (int): Number of channels
        RangeFreq (array_like): Frequency Range

    Returns:
        Frs (array_like): Fr vector
        WFval (array_like): Wraped freq. value
    """
    eps = np.finfo(float).eps # epsilon

    if NameScale == 'linear':
        RangeWF = RangeFreq
        dWF = np.diff(RangeWF) / (NumCh-1)
        WFvals = np.linspace(RangeWF[0], RangeWF[1]+eps*1000, dWF)
        Frs = WFvals

    elif NameScale == 'mel':
        RangeWF = Freq2FMel(RangeFreq)
        dWF = np.diff(RangeWF) / (NumCh-1)
        WFvals = np.linspace(RangeWF[0], RangeWF[1]+eps*1000, dWF)
        Frs = Mel2Freq(WFvals)

    elif NameScale == 'ERB':
        RangeWF, _ = Freq2ERB(RangeFreq)
        dWF = np.diff(RangeWF) / (NumCh-1)
        WFvals = np.arange(RangeWF[0], RangeWF[1]+eps*1000, dWF)
        Frs, _ = ERB2Freq(WFvals)

    elif NameScale == 'log':
        if min(RangeFreq) < 50:
            print("min(RangeFreq) < 50. Rplaced by 50.")
            RangeFreq[0] = 50
        RangeWF = np.log10(RangeFreq)
        dWF = np.diff(RangeWF) / (NumCh-1)
        WFvals = np.linspace(RangeWF[0], RangeWF[1]+eps*1000, dWF)
        Frs = 10**(WFvals)
    else:
        help(EqualFreqScale)
        print("Specify NameScale correctly", file=sys.stderr)
        sys.exit(1)
    
    return Frs, WFvals


def Freq2FMel(freq):
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


def Mel2Freq(mel):
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


def Freq2ERB(cf):
    """Convert linear frequency to ERB

    Args:
        cf (array_like): center frequency in linaer-scale [Hz] 

    Returns:
        ERBrate (array_like): ERB_N rate [ERB_N] or [cam] 
        ERBwidth (array_like): ERB_N Bandwidth [Hz]
    """
    # Warnig for frequency range
    cfmin = 50
    cfmax = 12000
    if np.min(cf) < cfmin or np.max(cf) > cfmax:
        print("Warning : Min or max frequency exceeds the proper ERB range: "
            +"{} (Hz) <= Fc <= {} (Hz)".format(cfmin, cfmax), file=sys.stderr)
        sys.exit(1)

    ERBrate = 21.4 * np.log10(4.37*cf/1000+1)
    ERBwidth = 24.7 * (4.37*cf/1000+1)

    return ERBrate, ERBwidth


def ERB2Freq(ERBrate):
    """Convert ERBrate to linear frequency

    Args:
        ERBrate (array_like): ERB_N rate [ERB_N] or [cam] 
    
    Returns:
        cf (array_like): center frequency in linaer-scale [Hz] 
        ERBwidth (array_like): ERB_N Bandwidth [Hz]
    """
    cf = (10**(ERBrate/21.4)-1) / 4.37 * 1000
    ERBwidth = 24.7 * (4.37*cf/1000+1)

    return cf, ERBwidth


def Fr2Fpeak(n, b, c, fr):
    """Estimate fpeak from fr

    Args:
        n (float): a parameter of the gammachirp
        b (float): a parameter of the gammachirp
        c (float): a parameter of the gammachirp
        fr (float): fr

    Returns:
        fpeak (float): peak frequency
        ERBw (float): ERBwidth at fr
    """

    _, ERBw = Freq2ERB(fr)
    fpeak = fr + c*ERBw*b/n

    return fpeak, ERBw


@lru_cache(maxsize=None)
def OutMidCrctFilt(StrCrct, SR, SwPlot=0, SwFilter=0):
    """Outer/middle ear compensation filter

    Args:
        StrCrct (string): String for Correction ELC/MAF/MAP
        SR (int): Sampling rate
        SwPlot (int): Switch of plot (0:OFF/1:ON) (default:0)
        SwFilter (int): Switch of filter type
            0: FIR linear phase filter (default)
            1: FIR linear phase inverse filter filter
            2: FIR mimimum phase filter (length: half of linear phase filter)

    Returns:
        FIRCoef (array_like): FIR filter coefficients
        StrFilt (string): Filter infomation

    Notes:
        In the original Matlab code of OutMidCrctFilt.m, persistent variables 
        are called by "persistent" function. The GammachirPy uses the "lru_cache"  
        instead of the persistent variables to call outputs if args are same 
        to previous one. 

    Reference:
        https://docs.python.org/3/library/functools.html
    """

    if SR > 48000:
        print("OutMidCrctFilt : Sampling rate of {} (Hz) (> 48000 (Hz) is not recommended)".format(SR))
        print("<-- ELC etc. is only defined below 16000 (Hz)")

    if SwFilter == 0:
        StrFilt = "FIR linear phase filter"
    elif SwFilter == 1:
        StrFilt = "FIR linear phase inverse filter"
    elif SwFilter == 2:
        StrFilt = "FIR minimum phase filter"
    else:
        help(OutMidCrctFilt)
        print("Specify filter type", file=sys.stderr)
        sys.exit(1)        

    if not StrCrct in ['ELC', 'MAF', 'MAP', 'MidEar']:
        help(OutMidCrctFilt)
        print("Specifiy correction: ELC/MAF/MAP/MidEar", file=sys.stderr)
        sys.exit(1)

    """
    Generating filter at the first time
    """
    print("*** OutMidCrctFilt: Generating {} {} ***".format(StrCrct, StrFilt))
    Nint = 1024
    # Nint = 0 # No spline interpolation:  NG no convergence at remez

    crctPwr, freq, _ = OutMidCrct(StrCrct, Nint, SR, 0)
    crct = np.sqrt(crctPwr[:,0])
    freq = freq[:,0]

    LenCoef = 200 # ( -45 dB) <- 300 (-55 dB)
    NCoef = int(np.fix(LenCoef/16000*SR/2)*2) # even number only

    if SwFilter == 1:
        crct = 1 / np.max(np.sqrt(crctPwr), 0.1) # Giving up less tan -20 dB : f>15000 Hz
                                                 # if requered, the response becomes worse
    
    LenCoef = 200 # ( -45 dB) <- 300 (-55 dB)
    NCoef = int(np.fix(LenCoef/16000*SR/2)*2) # even number only
    
    """ 
    Calculate the minimax optimal filter with a frequency response
    instead of "FIRCoef = firpm(NCoef,freq/SR*2,crct)" in the original code OutMidCrctFilt.m
    """
    x1 = np.array(np.arange(len(freq))).T * 2
    x2 = np.array(np.arange(len(freq)*2)).T
    freq_interp = np.interp(x2, x1, freq)
    FIRCoef = signal.remez(NCoef+1, freq_interp, crct, fs=SR) # len(freq_interp) must be twice of len(crct)

    Win, _ = TaperWindow(len(FIRCoef),'HAN',LenCoef/10)
    FIRCoef = Win * FIRCoef

    """
    Minimum phase reconstruction
    """
    if SwFilter == 2: 
        _, x_mp = rceps(FIRCoef)
        FIRCoef = x_mp[0:int(np.fix(len(x_mp)/2))]

    """
    Plot
    """
    if SwPlot == 1:
        Nrsl = 1024
        freq2, frsp = signal.freqz(FIRCoef, 1, Nrsl, fs=SR)

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        plt.plot(FIRCoef)
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
        
    return FIRCoef, StrFilt



def OutMidCrct(StrCrct, NfrqRsl=0, fs=32000, SwPlot=1):
    """Correction of ELC, MAF, MAP, MID. 
    It produces interpolated points for the ELC/MAF/MAP/MidEar correction.

    Args:
        StrCrct (string): Correction ELC/MAF/MAP/MidEar
        NfrqRsl (int): Number of data points, if zero, then direct out (default: 0)
        fs (int): Sampling frequency (default: 32000)
        SwPlot (int): Switch for plot (0/1, default:1)

    Returns:
        CrctLinPwr (array_like): Correction value in LINEAR POWER. 
            This is defined as:  CrctLiPwr =10^(-FreqChardB_toBeCmpnstd/10)
        freq (array_like): Corresponding Frequency at the data point
        FreqChardB_toBeCmpnstd (array_like): Frequency char of ELC/MAP dB 
            to be compensated for filterbank (defined By Glassberg and Moore.)

    Note: 
        "ER4B" option in StrCrct was omitted because the option uses a special measurement data. 
    """

    # ER4B: Omitted 
    """  
    if StrCrct == 'ER4B':
        CrctLinPwr, freq, FreqChardB_toBeCmpnstd = OutMidCrct_ER4B(NfrqRsl, fs, SwPlot)
        return CrctLinPwr, freq, FreqChardB_toBeCmpnstd
    """

    """
    Conventional ELC/MAF/MAP/MidEar
    """

    f1 = [	20,   25,  30,     35,  40,    45,  50,   55,   60,   70,  # 1-10
            80,   90,  100,   125,  150,   177, 200,  250,  300,  350,  # 11-20
            400,  450, 500,   550,  600,   700, 800,  900,  1000, 1500,  # 21-30
            2000, 2500, 2828, 3000, 3500, 4000, 4500, 5000, 5500, 6000,  # 31-40
            7000, 8000, 9000, 10000, 12748, 15000]   # 41-46

    ELC = [ 31.8, 26.0, 21.7, 18.8, 17.2, 15.4, 14.0, 12.6, 11.6, 10.6, 
            9.2, 8.2, 7.7, 6.7, 5.3, 4.6, 3.9, 2.9, 2.7, 2.3, 
            2.2, 2.3, 2.5, 2.7, 2.9, 3.4, 3.9, 3.9, 3.9, 2.7, 
            0.9, -1.3, -2.5, -3.2, -4.4, -4.1, -2.5, -0.5, 2.0, 5.0, 
            10.2, 15.0, 17.0, 15.5, 11.0, 22.0]

    MAF = [ 73.4, 65.2, 57.9, 52.7, 48.0, 45.0, 41.9, 39.3, 36.8, 33.0, 
            29.7, 27.1, 25.0, 22.0, 18.2, 16.0, 14.0, 11.4, 9.2, 8.0, 
            6.9,  6.2,  5.7,  5.1,  5.0,  5.0,  4.4,  4.3, 3.9, 2.7, 
            0.9, -1.3, -2.5, -3.2, -4.4, -4.1, -2.5, -0.5, 2.0, 5.0, 
            10.2, 15.0, 17.0, 15.5, 11.0, 22.0]

    f2  = [  125,  250,  500, 1000, 1500, 2000, 3000, 
            4000, 6000, 8000,10000,12000,14000,16000]

    MAP = [ 30.0, 19.0, 12.0,  9.0, 11.0, 16.0, 16.0, 
            14.0, 14.0,  9.9, 24.7, 32.7, 44.1, 63.7]

    # MidEar Correction (little modification at 17000:1000:20000)
    f3 =  [   1,  20,  25, 31.5,   40,   50,   63,   80,  100,  125, 
            160, 200, 250,  315,  400,  500,  630,  750,  800, 1000,
            1250, 1500, 1600,  2000,  2500,  3000,  3150,  4000,  5000,  6000, 
            6300, 8000, 9000, 10000, 11200, 12500, 14000, 15000, 16000,  20000]

    MID =  [  50,  39.15, 31.4, 25.4, 20.9,  18, 16.1, 14.2, 12.5, 11.13,
            9.71,   8.42,  7.2,  6.1,  4.7, 3.7,  3.0,  2.7,  2.6,   2.6,
             2.7,    3.7,  4.6,  8.5, 10.8, 7.3,  6.7,  5.7,  5.7,   7.6,
             8.4,   11.3, 10.6,  9.9, 11.9, 13.9, 16.0, 17.3, 17.8,  20.0] 


    frqTbl = []
    TblFreqChar = []
    if StrCrct == 'ELC':
        frqTbl = np.array([f1]).T
        TblFreqChar = np.array([ELC]).T
        ValHalfFs = 130
    elif StrCrct == 'MAF':
        frqTbl = np.array([f1]).T
        TblFreqChar = np.array([MAF]).T
        ValHalfFs = 130
    elif StrCrct == 'MAF':
        frqTbl = np.array([f2]).T
        TblFreqChar = np.array([MAP]).T
        ValHalfFs = 180
    elif StrCrct == 'MidEar':
        frqTbl = np.array([f3]).T
        TblFreqChar = np.array([MID]).T
        ValHalfFs = 23
    elif StrCrct == 'NO':
        pass
    else:
        print("Specifiy correction: ELC/MAF/MAP/MidEar or NO correction", file=sys.stderr)
        sys.exit(1)

    """
    Additional dummy data for high sampling frequency
    """
    if fs > 32000:
        frqTbl = np.vstack([frqTbl, fs/2])
        TblFreqChar = np.vstack([TblFreqChar, ValHalfFs])
        frqTbl, indx = np.unique(frqTbl, return_index=True)
        frqTbl = np.array([frqTbl]).T
        TblFreqChar = TblFreqChar[indx]

    str1 = ''
    if NfrqRsl <= 0:
        str1 = 'No interpolation. Output: values in original table.'
        freq = frqTbl
        FreqChardB_toBeCmpnstd = TblFreqChar
    else:
        freq = np.array([np.arange(NfrqRsl)/NfrqRsl * fs/2]).T
        if StrCrct == 'NO':
            FreqChardB_toBeCmpnstd = np.zeros(freq.shape)
        else:
            str1 = 'Spline interpolated value in equal frequency spacing.'
            freq_1d = freq.T[0,:]
            frqTbl_1d = frqTbl.T[0,:]
            TblFreqChar_1d = TblFreqChar.T[0,:]
            spl = UnivariateSpline(frqTbl_1d, TblFreqChar_1d, s=0)
            FreqChardB_toBeCmpnstd = spl(freq_1d)
            FreqChardB_toBeCmpnstd = np.array([FreqChardB_toBeCmpnstd]).T
    
    if SwPlot == 1:
        str = "*** Frequency Characteristics (" + StrCrct + "): Its inverse will be corrected. ***"
        print(str) 
        print("{}".format(str1))
        fig, ax = plt.subplots()
        plt.plot(frqTbl, TblFreqChar, 'b-',freq, FreqChardB_toBeCmpnstd, 'r--')
        plt.xlim(0, 25000)
        plt.ylim(-20,140)
        ax.set_title(str)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Level (dB)')
        plt.show()

    CrctLinPwr = 10**(-FreqChardB_toBeCmpnstd/10) # in Linear Power. Checked 19 Apr 2016

    return CrctLinPwr, freq, FreqChardB_toBeCmpnstd


def TaperWindow(LenWin, TypeTaper, LenTaper=None, RangeSigma=3, SwPlot=0):
    """Taper Window Generator for signal onset/offset

    Args:
        LenWin (int): Length of window (number of points)
        TypeTaper (string): Type of Taper (KeyWords of 3 letters)
            - HAM: Hamming
            - HAN/COS: Hanning/Cosine
            - BLA: Blackman
            - GAU: Gaussian
            - (other): Linear
        LenTaper (int, optional): Length of taper. Defaults to None.
        RangeSigma (int, optional): Range in sigma. Defaults to 3.
        SwPlot (int, optional): OFF/ON. Defaults to 0.

    Returns:
        TaperWin (array_like): Taper window points (max: 1)
        TypeTaper (string): Type of taper (full name)
    """

    if LenTaper == None:
        LenTaper = int(np.fix(LenWin/2))
    
    elif LenTaper*2+1 >= LenWin:
        print("Caution (TaperWindow) : No flat part. ")
        
        if not LenTaper == np.fix(LenWin/2):
            print("Caution (TaperWindow) : LenTaper <-- fix(LenWin/2)")
            
        LenTaper = int(np.fix(LenWin/2))

    LenTaper= int(LenTaper)

    if TypeTaper == 'HAM':
        Taper = np.hamming(LenTaper*2+1)
        TypeTaper = 'Hamming'

    elif TypeTaper == 'HAN' or TypeTaper == 'COS':
        Taper = np.hamming(LenTaper*2+1)
        TypeTaper = 'Hanning/Cosine'

    elif TypeTaper == 'BLA':
        Taper = np.blackman(LenTaper*2+1)
        TypeTaper = 'Blackman'

    elif TypeTaper == 'GAU':
        if len(RangeSigma) == 0:
            RangeSigma = 3
        nn = np.arange(-LenTaper, LenTaper, 1)
        Taper = np.exp(-(RangeSigma/LenTaper)**2 / 2)
        TypeTaper == 'Gauss'

    else:
        Taper = np.array(list(np.arange(1,LenTaper+1,1)) + list([LenTaper+1]) + list(np.arange(LenTaper,1-1,-1))) / (LenTaper+1)
        TypeTaper = 'Line'

    LenTaper = int(np.fix(LenTaper))
    TaperWin = list(Taper[0:LenTaper]) + list(np.ones(LenWin-LenTaper*2)) + list(Taper[(LenTaper+1):(LenTaper*2+1)])

    if SwPlot == 1:
        fig, ax = plt.subplots()
        plt.plot(TaperWin)
        ax.set_xlabel('Points')
        ax.set_ylabel('Amplitude')
        plt.title('TypeTaper: {}'.format(TypeTaper))

    return TaperWin, TypeTaper


def rceps(x):
    """returns the real cepstrum of the real sequence X

    Args:
        x (array_like): input signal

    Returns:
        xhat: real cepstrum
        yhat: a unique minimum-phase sequence that has the reame real cepstrum as x

    Note:
        This code is based on "rceps.m" in MATLAB and is under-construction. 

    Examples:
        x = [4 1 5]; % Non-minimum phase sequence
        xhat = array([1.62251148, 0.3400368 , 0.3400368 ])
        yhat = array([5.33205452, 3.49033278, 1.1776127 ])

    References:
    - A.V. Oppenheim and R.W. Schafer, Digital Signal Processing, Prentice-Hall, 1975.
    - Programs for Digital Signal Processing, IEEE Press, John Wiley & Sons, 1979, algorithm 7.2.
    - https://mathworks.com/help/signal/ref/rceps.html
    """

    if isrow(x):
        xT = np.array([x]).T
    else:
        xT = x

    fftxabs = np.abs(np.fft.fft(xT, n=None, axis=0))

    xhatT = np.real(np.fft.ifft(np.log(fftxabs), n=None, axis=0))

    # xhat
    if isrow(x):
        # transform the result to a row vector
        xhat = xhatT[:,0]
    else:
        xhat = xhatT

    # yhat
    nRows = xhatT.shape[0]
    nCols = xhatT.shape[1]
    odd = nRows % 2
    a1 = np.array([1])
    a2 = 2*np.ones((int((nRows+odd)/2)-1, 1))
    a3 = np.zeros((int((nRows+odd)/2)-1,1))
    wn = np.kron(np.ones((1, nCols)), np.vstack((a1,a2,a3)))
    """
    Matlab can use zero and negative numbers for args of ones function, but the np.ones cannot. So, an internal array is removed. The original code is: 
    wn = np.kron(np.ones((1, nCols)), np.array([[1], 2*np.ones((int((nRows+odd)/2)-1, 1)), np.ones(1-odd, 1), np.zeros((int((nRows+odd)/2)-1,1))]))
    """
    yhatT = np.real(np.fft.ifft(np.exp(np.fft.fft((wn*xhatT),n=None, axis=0)), n=None, axis=0))
    if isrow(x):
        # transform the result to a row vector
        yhat = yhatT[:,0]
    else:
        yhat = yhatT

    return xhat, yhat


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
    